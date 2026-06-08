"""Stage 2 Temporal Memory Distillation — DDP training loop.

Launch:
    torchrun --nproc_per_node=2 stage2/train.py \\
        --cfg stage2/configs/sav_repvit_m0_9.yaml \\
        --data-path /mnt/exoshdd/datasets/SA-V/ \\
        --student-ckpt checkpoints/efficient_sam3_repvit_s.pt \\
        --teacher-ckpt /mnt/exoshdd/checkpoints/sam3/sam3.pt \\
        --output output --tag stage2_run0

Production loop: full memory-bank teacher (Step 7 done), checkpoint
save+resume, TensorBoard logging, periodic held-out SA-V val eval.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2.config import get_config, MAX_FRAMES
from stage2.data.build import build_loader
from stage2.eval import evaluate_distill
from stage2.loss import build_distill_loss, build_mask_loss
from stage2.models import ModelEma, build_student_model, build_teacher_model
from stage2.optim import CosineWarmupScheduler, build_optimizer
from stage2.utils import (
    auto_resume_helper,
    load_checkpoint,
    save_checkpoint,
    save_running_checkpoint,
)


# ---------------------------------------------------------------------------
# argparse + DDP boilerplate
# ---------------------------------------------------------------------------

def parse_option():
    parser = argparse.ArgumentParser("EfficientSAM3 Stage-2 training", add_help=True)
    parser.add_argument('--cfg', type=str, required=False, metavar="FILE")
    parser.add_argument('--opts', default=None, nargs='+')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--student-ckpt', type=str)
    parser.add_argument('--teacher-ckpt', type=str)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--tag', type=str, default='default')
    parser.add_argument('--accumulation-steps', type=int)
    parser.add_argument('--use-checkpoint', action='store_true')
    parser.add_argument('--disable_amp', action='store_true')
    parser.add_argument('--only-cpu', action='store_true')
    parser.add_argument('--eval', action='store_true', help='run eval only, no training')
    parser.add_argument('--eval-on-train', action='store_true',
                        help='in --eval mode, use the train loader instead of val. '
                             'Diagnostic: does the model reproduce training-time loss on its '
                             'own train distribution? If yes, the train/val gap is a real '
                             'generalization failure rather than an eval-path bug.')
    parser.add_argument('--no-val', action='store_true', help='skip held-out val eval during training')
    parser.add_argument('--val-max-batches', type=int, default=-1,
                        help='cap val batches per eval; -1 = full val set')
    parser.add_argument('--calibrate-bn-batches', type=int, default=0,
                        help='run N batches in train mode before --eval to warm up backbone BN '
                             'running stats (needed for checkpoints saved before the '
                             'StudentTemporalModel.train() override, i.e. ep1-17)')
    parser.add_argument('--backbone-bn-train-mode', action='store_true',
                        help='keep backbone BN in train mode during eval (per-batch stats). '
                             'Required for ep1-ep17 checkpoints where backbone ran in train '
                             'mode throughout training and Perceiver learned batch-normalised '
                             'features. Without this flag, eval mode BN changes the feature '
                             'distribution enough to push cosine similarity to ~0.')
    parser.add_argument('--throughput', action='store_true')
    parser.add_argument('--smoke-test', action='store_true',
                        help='exit after a few iters to validate forward/backward')
    parser.add_argument('--smoke-iters', type=int, default=2)
    parser.add_argument('--local-rank', type=int, default=None)
    args = parser.parse_args()
    config = get_config(args)
    return args, config


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank, world_size, local_rank = 0, 1, 0
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '29500')
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    dist.barrier()
    return rank, world_size, local_rank


def set_seed(seed: int, rank: int):
    seed = seed + rank
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def log(msg: str):
    if is_main_process():
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args, config = parse_option()
    rank, world_size, local_rank = setup_distributed()
    set_seed(config.SEED, rank)
    cudnn.benchmark = True

    device = torch.device('cuda', local_rank)

    log(f"[stage2] rank={rank} world={world_size} local={local_rank}")
    log(f"[stage2] MAX_FRAMES={MAX_FRAMES} bs/gpu={config.DATA.BATCH_SIZE} "
        f"workers={config.DATA.NUM_WORKERS} img_size={config.DATA.IMG_SIZE}")

    # -- data ---------------------------------------------------------------
    build_val = (not args.no_val) and (config.EVAL.EVERY_N_EPOCHS > 0)
    _, _, loader_train, loader_val = build_loader(
        config, rank=rank, world_size=world_size, build_val=build_val
    )
    log(f"[stage2] train ds len={len(loader_train.dataset)} steps/epoch={len(loader_train)}")
    if loader_val is not None:
        log(f"[stage2] val   ds len={len(loader_val.dataset)} steps={len(loader_val)}")

    # -- models -------------------------------------------------------------
    student = build_student_model(config).to(device)
    teacher = build_teacher_model(config).to(device)
    teacher.eval()
    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in student.parameters())
    log(f"[stage2] student trainable={n_train/1e6:.2f}M total={n_total/1e6:.2f}M  "
        f"teacher total={sum(p.numel() for p in teacher.parameters())/1e6:.2f}M (frozen)")

    # Prompt conditioning leaves some PromptEncoder params (point/box embeddings)
    # unused when the batch prompt is mask-only, so DDP needs unused-param detection.
    find_unused = config.TRAIN.FIND_UNUSED_PARAMETERS or bool(
        getattr(config.MODEL, 'PROMPT_CONDITIONING', False))
    student = DDP(
        student,
        device_ids=[local_rank],
        find_unused_parameters=find_unused,
    )

    # -- loss / optim / sched -----------------------------------------------
    loss_fn = build_distill_loss(config).to(device)

    # mask-prompted (Milestone 1) flags + mask loss
    prompt_conditioning = bool(getattr(config.MODEL, 'PROMPT_CONDITIONING', False))
    mask_loss_weight = float(getattr(config.DISTILL, 'MASK_LOSS_WEIGHT', 0.0))
    teacher_frame0_only = bool(getattr(config.DISTILL, 'TEACHER_FRAME0_ONLY', False))
    mask_loss_fn = build_mask_loss(config).to(device) if (prompt_conditioning and mask_loss_weight > 0) else None
    if prompt_conditioning:
        log(f"[stage2] PROMPT_CONDITIONING on: mask_loss_w={mask_loss_weight} "
            f"teacher_frame0_only={teacher_frame0_only}")

    # Lazy teacher-target cache: frozen teacher is deterministic per (video_id,
    # obj_idx), so cache its output once and replay it (skips ~81% of iter compute).
    teacher_cache = None
    tc_dir = str(getattr(config.DISTILL, 'TEACHER_CACHE_DIR', '') or '')
    if tc_dir:
        from stage2.teacher_cache import TeacherTargetCache
        teacher_cache = TeacherTargetCache(tc_dir, enabled=True)
        log(f"[stage2] teacher-target cache ON -> {tc_dir}")

    optimizer = build_optimizer(config, student.module)

    iters_per_epoch = max(1, len(loader_train))
    total_iters = config.TRAIN.EPOCHS * iters_per_epoch
    warmup_iters = config.TRAIN.WARMUP_EPOCHS * iters_per_epoch
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_iters=warmup_iters,
        total_iters=total_iters,
        base_lr=config.TRAIN.BASE_LR,
        min_lr=config.TRAIN.MIN_LR,
        warmup_lr=config.TRAIN.WARMUP_LR,
    )

    amp_dtype = torch.bfloat16 if config.TRAIN.AMP_DTYPE == 'bfloat16' else torch.float16
    use_amp = config.AMP_ENABLE and not args.only_cpu
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))

    # -- EMA (Perceiver only) ----------------------------------------------
    ema_perceiver = None
    if config.TRAIN.EMA_ENABLE:
        ema_perceiver = ModelEma(
            student.module.perceiver, decay=config.TRAIN.EMA_DECAY, device=device,
        )
        n_ema = sum(p.numel() for p in ema_perceiver.module.parameters())
        log(f"[stage2] EMA enabled on Perceiver: decay={config.TRAIN.EMA_DECAY} "
            f"shadow={n_ema/1e6:.2f}M")

    # -- resume -------------------------------------------------------------
    start_epoch = config.TRAIN.START_EPOCH
    step_in_epoch_resume = 0
    global_step = 0
    best_val: float | None = None
    # early-stopping counter — incremented at each val epoch that does NOT set
    # a new best. Restored from checkpoint so resume preserves it.
    epochs_since_best: int = 0

    resume_path = config.MODEL.RESUME
    if not resume_path and config.TRAIN.AUTO_RESUME:
        resume_path = auto_resume_helper(config) or ''
    if resume_path and os.path.exists(resume_path):
        log(f"[stage2] resuming from {resume_path}")
        meta = load_checkpoint(
            config=config, path=resume_path, model=student.module,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            ema=ema_perceiver,
            map_location=f'cuda:{local_rank}',
        )
        global_step = meta['global_step']
        best_val = meta['best_val']
        epochs_since_best = int(meta.get('epochs_since_best', 0))
        if meta['step_in_epoch'] > 0:
            # mid-epoch running checkpoint: resume in the same epoch, skip
            # the first N batches (sampler is deterministic via set_epoch).
            start_epoch = meta['epoch']
            step_in_epoch_resume = meta['step_in_epoch']
            log(f"[stage2] resumed MID-EPOCH: ep={start_epoch} "
                f"skip_first={step_in_epoch_resume} step={global_step} best_val={best_val}")
        else:
            start_epoch = meta['epoch'] + 1
            log(f"[stage2] resumed: epoch_done={meta['epoch']} next={start_epoch} "
                f"step={global_step} best_val={best_val}")
        if ema_perceiver is not None and not meta.get('has_ema'):
            log("[stage2] ckpt has no EMA state — initialized fresh from current weights")

    # -- tb writer (rank 0 only) -------------------------------------------
    writer = None
    if is_main_process():
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG, 'tb')
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir, flush_secs=10)
        log(f"[stage2] tensorboard logs -> {tb_dir}")

    # -- eval-only mode -----------------------------------------------------
    if args.eval:
        if args.eval_on_train:
            eval_loader = loader_train
            # set_epoch on the sampler so shuffle is deterministic across runs
            eval_loader.sampler.set_epoch(0)
            eval_split_label = 'train'
        else:
            eval_loader = loader_val
            eval_split_label = 'val'
        if eval_loader is None:
            log(f"[stage2] --eval set but no {eval_split_label} loader (use without --no-val)")
            return
        result = evaluate_distill(
            student=student.module, teacher=teacher, loader=eval_loader,
            loss_fn=loss_fn, device=device, amp_dtype=amp_dtype,
            use_amp=use_amp, max_batches=args.val_max_batches,
            calibrate_bn_batches=args.calibrate_bn_batches,
            backbone_bn_train_mode=args.backbone_bn_train_mode,
            prompt_conditioning=prompt_conditioning,
            teacher_frame0_only=teacher_frame0_only,
        )
        log(f"[stage2][eval:{eval_split_label}] total={result.total:.4f} mse={result.mse:.4f} "
            f"cos={result.cosine:.4f} norm={result.norm:.4f} n={result.n_batches}")
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    # -- training loop ------------------------------------------------------
    log(f"[stage2] entering training loop (start_epoch={start_epoch} step={global_step} total_epochs={config.TRAIN.EPOCHS})")
    accum = max(1, config.TRAIN.ACCUMULATION_STEPS)

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        loader_train.sampler.set_epoch(epoch)
        student.train()
        t_epoch = time.time()
        t_window = time.time()
        n_window = 0

        # mid-epoch resume: skip batches we already processed in the prior
        # interrupted run (sampler permutation is deterministic per-epoch).
        skip_n = step_in_epoch_resume if epoch == start_epoch else 0
        if skip_n > 0:
            log(f"[stage2] ep{epoch}: skipping first {skip_n} batches (mid-epoch resume)")

        for step, batch in enumerate(loader_train):
            if step < skip_n:
                continue
            frames = batch['frames'].to(device, non_blocking=True)             # [B,T,3,H,W]
            attn_mask = batch['attention_mask'].to(device, non_blocking=True)  # [B,T]
            gt_masks = batch['masks'].to(device, non_blocking=True)            # [B,T,H,W]
            mask_valid = batch['mask_valid'].to(device, non_blocking=True)     # [B,T]

            assert frames.shape[1] == MAX_FRAMES
            assert frames.shape[-1] == config.DATA.IMG_SIZE  # 1008 — RoPE constraint

            # mask-prompted (Milestone 1): frame-0 prompt steers the memory toward
            # the selected object; teacher conditions on GT mask only at frame 0.
            prompt = None
            if prompt_conditioning and 'prompt_mask' in batch:
                prompt = {'mask': batch['prompt_mask'].to(device, non_blocking=True)}

            # cache_enabled=False — critical for BF16 training. Without it,
            # autocast's weight cache returns STALE bf16 casts of fp32 params
            # even after optimizer.step() (cache is only invalidated by
            # GradScaler.unscale_(), which we don't call in BF16 mode). The
            # stale cache causes the optimizer to descend a different objective
            # than the one produced by a fresh load of the same weights.
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp,
                                    cache_enabled=False):
                if prompt_conditioning:
                    s_out, hires = student(frames, attn_mask, prompt=prompt,
                                           return_high_res=(mask_loss_weight > 0))
                else:
                    s_out = student(frames, attn_mask)          # [B,T,256,H',W']

                # Teacher target: replay from cache when available (frozen teacher
                # is deterministic per (video_id, obj_idx)); else run it live and
                # back-fill the cache. The teacher is ~81% of per-iter compute, so
                # a warm cache makes iters ~5x faster (tools/profile_iter.py).
                t_out = None
                cache_hit = False
                if teacher_cache is not None:
                    per_shape = (s_out.shape[1], s_out.shape[2], s_out.shape[3], s_out.shape[4])
                    t_out, cache_hit, _miss = teacher_cache.get_batch(
                        batch['video_ids'], batch['obj_idx'], per_shape, device)
                if not cache_hit:
                    with torch.no_grad():
                        t_out = teacher(frames, attn_mask, gt_masks, mask_valid,
                                        frame0_only=teacher_frame0_only)  # [B,T,256,H',W']
                    if teacher_cache is not None:
                        teacher_cache.put_batch(batch['video_ids'], batch['obj_idx'], t_out)
                losses = loss_fn(s_out, t_out.to(s_out.dtype), attn_mask)
                loss = losses.total
                # mask supervision on the decoded masks vs GT
                mask_bce_val = mask_dice_val = None
                if prompt_conditioning and mask_loss_weight > 0:
                    student_mod = student.module if hasattr(student, 'module') else student
                    s_logits = student_mod.decode_masks(s_out, hires)
                    ml = mask_loss_fn(s_logits, gt_masks, mask_valid)
                    loss = loss + mask_loss_weight * ml.total
                    mask_bce_val = ml.bce.item()
                    mask_dice_val = ml.dice.item()
                loss = loss / accum

            loss_val = loss.item()
            # mask BCE can legitimately be large early on; raise the guard when on.
            bad_thresh = 100.0 if (prompt_conditioning and mask_loss_weight > 0) else 10.0
            if not (loss_val == loss_val) or loss_val > bad_thresh:  # NaN or explosion
                log(f"[stage2] BAD BATCH ep{epoch} it{step}: loss={loss_val:.2f} — skipping update")
                optimizer.zero_grad(set_to_none=True)
                n_window += frames.shape[0]
                continue

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            n_window += frames.shape[0]

            grad_norm_val: float | None = None
            mlp_out_w_std_val: float | None = None
            if (step + 1) % accum == 0:
                if config.TRAIN.CLIP_GRAD > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    grad_norm_t = torch.nn.utils.clip_grad_norm_(
                        [p for p in student.parameters() if p.requires_grad],
                        max_norm=config.TRAIN.CLIP_GRAD,
                    )
                    grad_norm_val = float(grad_norm_t.item()) if grad_norm_t is not None else None
                if scaler.is_enabled():
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                scheduler.step(global_step)
                if ema_perceiver is not None:
                    ema_perceiver.update(student.module.perceiver)
                # collapse early-warning: snapshot mlp_out.2.weight magnitude
                with torch.no_grad():
                    mlp_out_w_std_val = float(
                        student.module.perceiver.mlp_out[2].weight.float().std().item()
                    )

            # mid-epoch checkpoint (rank 0 only); guards against SIGHUP/preempt
            save_every = int(config.TRAIN.SAVE_EVERY_ITERS)
            if (save_every > 0 and is_main_process()
                    and (step + 1) % save_every == 0):
                rpath = save_running_checkpoint(
                    config=config, epoch=epoch, global_step=global_step,
                    step_in_epoch=step + 1,
                    model=student.module, optimizer=optimizer, scheduler=scheduler,
                    scaler=scaler, best_val=best_val, ema=ema_perceiver,
                    epochs_since_best=epochs_since_best,
                )
                log(f"[stage2] ep{epoch} it{step+1}: running ckpt -> {rpath}")

            if step % config.PRINT_FREQ == 0:
                dt = max(1e-6, time.time() - t_window)
                ips = n_window / dt
                # collapse early-warning: avg per-token L2 norm of student output
                with torch.no_grad():
                    s_token_norm_val = float(
                        s_out.detach().float().norm(p=2, dim=2).mean().item()
                    )
                gn_str = f"gn={grad_norm_val:.3f}" if grad_norm_val is not None else "gn=--"
                mw_str = f"mw={mlp_out_w_std_val:.3f}" if mlp_out_w_std_val is not None else "mw=--"
                mask_str = (f" mbce={mask_bce_val:.3f} mdice={mask_dice_val:.3f}"
                            if mask_bce_val is not None else "")
                log(f"[stage2] ep{epoch} it{step}/{len(loader_train)} "
                    f"loss={losses.total.item():.4f} mse={losses.mse.item():.4f} "
                    f"cos={losses.cosine.item():.4f} norm={losses.norm.item():.4f} "
                    f"lr={scheduler.get_lr():.2e} "
                    f"ips={ips:.2f} mask_sum={attn_mask.sum().item()} "
                    f"{gn_str} {mw_str} s_norm={s_token_norm_val:.2f}{mask_str}")
                # note: ips = samples/sec (n_window += batch_size); iter/s = ips / batch_size
                if writer is not None:
                    writer.add_scalar('train/loss_total', losses.total.item(), global_step)
                    writer.add_scalar('train/loss_mse', losses.mse.item(), global_step)
                    writer.add_scalar('train/loss_cosine', losses.cosine.item(), global_step)
                    writer.add_scalar('train/loss_norm', losses.norm.item(), global_step)
                    writer.add_scalar('train/lr', scheduler.get_lr(), global_step)
                    writer.add_scalar('train/ips', ips, global_step)
                    if grad_norm_val is not None:
                        writer.add_scalar('train/grad_norm', grad_norm_val, global_step)
                    if mlp_out_w_std_val is not None:
                        writer.add_scalar('train/mlp_out_w_std', mlp_out_w_std_val, global_step)
                    writer.add_scalar('train/s_token_norm', s_token_norm_val, global_step)
                t_window = time.time(); n_window = 0

            if args.smoke_test and step + 1 >= args.smoke_iters:
                log(f"[stage2] smoke test OK ({args.smoke_iters} iters); exiting")
                if dist.is_initialized():
                    dist.barrier(); dist.destroy_process_group()
                if writer is not None:
                    writer.close()
                return

        log(f"[stage2] epoch {epoch} done in {time.time()-t_epoch:.1f}s")

        # epoch boundary crossed — subsequent epochs start at batch 0
        step_in_epoch_resume = 0

        # -- periodic val eval --------------------------------------------
        do_eval = (loader_val is not None
                   and config.EVAL.EVERY_N_EPOCHS > 0
                   and (epoch + 1) % config.EVAL.EVERY_N_EPOCHS == 0)
        is_best = False
        if do_eval:
            result = evaluate_distill(
                student=student.module, teacher=teacher, loader=loader_val,
                loss_fn=loss_fn, device=device, amp_dtype=amp_dtype,
                use_amp=use_amp, max_batches=args.val_max_batches,
                prompt_conditioning=prompt_conditioning,
                teacher_frame0_only=teacher_frame0_only,
            )
            log(f"[stage2][val] ep{epoch} total={result.total:.4f} mse={result.mse:.4f} "
                f"cos={result.cosine:.4f} norm={result.norm:.4f} n={result.n_batches}")
            if writer is not None:
                writer.add_scalar('val/loss_total', result.total, epoch)
                writer.add_scalar('val/loss_mse', result.mse, epoch)
                writer.add_scalar('val/loss_cosine', result.cosine, epoch)
                writer.add_scalar('val/loss_norm', result.norm, epoch)
            if best_val is None or result.total < best_val:
                best_val = result.total
                is_best = True
                epochs_since_best = 0
                log(f"[stage2] new best val={best_val:.4f}")
            else:
                epochs_since_best += 1
                log(f"[stage2] no new best (since_best={epochs_since_best}, current={result.total:.4f} vs best={best_val:.4f})")

            # second val pass with EMA weights swapped in (if enabled)
            if ema_perceiver is not None:
                perceiver = student.module.perceiver
                snapshot = copy.deepcopy(perceiver.state_dict())
                ema_perceiver.copy_to(perceiver)
                try:
                    ema_result = evaluate_distill(
                        student=student.module, teacher=teacher, loader=loader_val,
                        loss_fn=loss_fn, device=device, amp_dtype=amp_dtype,
                        use_amp=use_amp, max_batches=args.val_max_batches,
                    )
                finally:
                    perceiver.load_state_dict(snapshot)
                log(f"[stage2][val-ema] ep{epoch} total={ema_result.total:.4f} "
                    f"mse={ema_result.mse:.4f} cos={ema_result.cosine:.4f} norm={ema_result.norm:.4f}")
                if writer is not None:
                    writer.add_scalar('val/ema_loss_total', ema_result.total, epoch)
                    writer.add_scalar('val/ema_loss_mse', ema_result.mse, epoch)
                    writer.add_scalar('val/ema_loss_cosine', ema_result.cosine, epoch)
                    writer.add_scalar('val/ema_loss_norm', ema_result.norm, epoch)

        # -- checkpoint (rank 0) ------------------------------------------
        if is_main_process() and ((epoch + 1) % config.SAVE_FREQ == 0 or is_best
                                  or epoch + 1 == config.TRAIN.EPOCHS):
            path = save_checkpoint(
                config=config, epoch=epoch, global_step=global_step,
                model=student.module, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, best_val=best_val, is_best=is_best,
                ema=ema_perceiver,
                epochs_since_best=epochs_since_best,
            )
            log(f"[stage2] saved {path}{'  [best]' if is_best else ''}")
        if dist.is_initialized():
            dist.barrier()

        # -- early stopping check ----------------------------------------
        patience = int(getattr(config.TRAIN, 'EARLY_STOP_PATIENCE', 0))
        if patience > 0 and epochs_since_best >= patience:
            log(f"[stage2] EARLY STOP at epoch {epoch}: "
                f"{epochs_since_best} consecutive val epochs without a new "
                f"best (patience={patience}); best_val={best_val:.4f}")
            break

    if writer is not None:
        writer.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
