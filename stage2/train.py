"""Stage 2 Temporal Memory Distillation — DDP training loop.

Launch:
    torchrun --nproc_per_node=2 stage2/train.py \\
        --cfg stage2/configs/sav_repvit_m0_9.yaml \\
        --data-path /mnt/hdd/datasets/SA-V/ \\
        --student-ckpt checkpoints/efficient_sam3_repvit_s.pt \\
        --teacher-ckpt /mnt/hdd/checkpoints/sam3/sam3.pt \\
        --output output --tag stage2_run0

Production loop: full memory-bank teacher (Step 7 done), checkpoint
save+resume, TensorBoard logging, periodic held-out SA-V val eval.
"""

from __future__ import annotations

import argparse
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
from stage2.loss import build_distill_loss
from stage2.models import build_student_model, build_teacher_model
from stage2.optim import CosineWarmupScheduler, build_optimizer
from stage2.utils import auto_resume_helper, load_checkpoint, save_checkpoint


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
    parser.add_argument('--no-val', action='store_true', help='skip held-out val eval during training')
    parser.add_argument('--val-max-batches', type=int, default=-1,
                        help='cap val batches per eval; -1 = full val set')
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

    student = DDP(
        student,
        device_ids=[local_rank],
        find_unused_parameters=config.TRAIN.FIND_UNUSED_PARAMETERS,
    )

    # -- loss / optim / sched -----------------------------------------------
    loss_fn = build_distill_loss(config).to(device)
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

    # -- resume -------------------------------------------------------------
    start_epoch = config.TRAIN.START_EPOCH
    global_step = 0
    best_val: float | None = None

    resume_path = config.MODEL.RESUME
    if not resume_path and config.TRAIN.AUTO_RESUME:
        resume_path = auto_resume_helper(config) or ''
    if resume_path and os.path.exists(resume_path):
        log(f"[stage2] resuming from {resume_path}")
        meta = load_checkpoint(
            config=config, path=resume_path, model=student.module,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            map_location=f'cuda:{local_rank}',
        )
        start_epoch = meta['epoch'] + 1
        global_step = meta['global_step']
        best_val = meta['best_val']
        log(f"[stage2] resumed: epoch={meta['epoch']} step={global_step} best_val={best_val}")

    # -- tb writer (rank 0 only) -------------------------------------------
    writer = None
    if is_main_process():
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG, 'tb')
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        log(f"[stage2] tensorboard logs -> {tb_dir}")

    # -- eval-only mode -----------------------------------------------------
    if args.eval:
        if loader_val is None:
            log("[stage2] --eval set but no val loader (use without --no-val)")
            return
        result = evaluate_distill(
            student=student.module, teacher=teacher, loader=loader_val,
            loss_fn=loss_fn, device=device, amp_dtype=amp_dtype,
            use_amp=use_amp, max_batches=args.val_max_batches,
        )
        log(f"[stage2][eval] total={result.total:.4f} mse={result.mse:.4f} "
            f"cos={result.cosine:.4f} n={result.n_batches}")
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    # -- training loop ------------------------------------------------------
    log(f"[stage2] entering training loop (start_epoch={start_epoch} step={global_step})")
    accum = max(1, config.TRAIN.ACCUMULATION_STEPS)

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        loader_train.sampler.set_epoch(epoch)
        student.train()
        t_epoch = time.time()
        t_window = time.time()
        n_window = 0

        for step, batch in enumerate(loader_train):
            frames = batch['frames'].to(device, non_blocking=True)             # [B,T,3,H,W]
            attn_mask = batch['attention_mask'].to(device, non_blocking=True)  # [B,T]
            gt_masks = batch['masks'].to(device, non_blocking=True)            # [B,T,H,W]
            mask_valid = batch['mask_valid'].to(device, non_blocking=True)     # [B,T]

            assert frames.shape[1] == MAX_FRAMES
            assert frames.shape[-1] == config.DATA.IMG_SIZE  # 1008 — RoPE constraint

            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                s_out = student(frames, attn_mask)              # [B,T,256,H',W']
                with torch.no_grad():
                    t_out = teacher(frames, attn_mask, gt_masks, mask_valid)  # [B,T,256,H',W']
                losses = loss_fn(s_out, t_out, attn_mask)
                loss = losses.total / accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            n_window += frames.shape[0]

            if (step + 1) % accum == 0:
                if config.TRAIN.CLIP_GRAD > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in student.parameters() if p.requires_grad],
                        max_norm=config.TRAIN.CLIP_GRAD,
                    )
                if scaler.is_enabled():
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                scheduler.step(global_step)

            if step % config.PRINT_FREQ == 0:
                dt = max(1e-6, time.time() - t_window)
                ips = n_window / dt
                log(f"[stage2] ep{epoch} it{step}/{len(loader_train)} "
                    f"loss={losses.total.item():.4f} mse={losses.mse.item():.4f} "
                    f"cos={losses.cosine.item():.4f} lr={scheduler.get_lr():.2e} "
                    f"ips={ips:.2f} mask_sum={attn_mask.sum().item()}")
                if writer is not None:
                    writer.add_scalar('train/loss_total', losses.total.item(), global_step)
                    writer.add_scalar('train/loss_mse', losses.mse.item(), global_step)
                    writer.add_scalar('train/loss_cosine', losses.cosine.item(), global_step)
                    writer.add_scalar('train/lr', scheduler.get_lr(), global_step)
                    writer.add_scalar('train/ips', ips, global_step)
                t_window = time.time(); n_window = 0

            if args.smoke_test and step + 1 >= args.smoke_iters:
                log(f"[stage2] smoke test OK ({args.smoke_iters} iters); exiting")
                if dist.is_initialized():
                    dist.barrier(); dist.destroy_process_group()
                if writer is not None:
                    writer.close()
                return

        log(f"[stage2] epoch {epoch} done in {time.time()-t_epoch:.1f}s")

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
            )
            log(f"[stage2][val] ep{epoch} total={result.total:.4f} mse={result.mse:.4f} "
                f"cos={result.cosine:.4f} n={result.n_batches}")
            if writer is not None:
                writer.add_scalar('val/loss_total', result.total, epoch)
                writer.add_scalar('val/loss_mse', result.mse, epoch)
                writer.add_scalar('val/loss_cosine', result.cosine, epoch)
            if best_val is None or result.total < best_val:
                best_val = result.total
                is_best = True
                log(f"[stage2] new best val={best_val:.4f}")

        # -- checkpoint (rank 0) ------------------------------------------
        if is_main_process() and ((epoch + 1) % config.SAVE_FREQ == 0 or is_best
                                  or epoch + 1 == config.TRAIN.EPOCHS):
            path = save_checkpoint(
                config=config, epoch=epoch, global_step=global_step,
                model=student.module, optimizer=optimizer, scheduler=scheduler,
                scaler=scaler, best_val=best_val, is_best=is_best,
            )
            log(f"[stage2] saved {path}{'  [best]' if is_best else ''}")
        if dist.is_initialized():
            dist.barrier()

    if writer is not None:
        writer.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
