"""Held-out distillation-loss evaluation on SA-V val split.

Runs student + teacher over the val loader, AllReduces sum-of-loss / count
across DDP ranks, returns mean MSE + cosine + total.

DAVIS J&F is deferred: it requires (a) DAVIS 2017 trainval (only test-dev is
locally available — first-frame anno only) and (b) plugging the student into
the SAM3 video predictor's tracking pipeline, which is Stage 3 territory.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass
class DistillEvalResult:
    total: float
    mse: float
    cosine: float
    norm: float
    n_batches: int


@torch.no_grad()
def evaluate_distill(
    *,
    student: nn.Module,
    teacher: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    max_batches: int = -1,
    calibrate_bn_batches: int = 0,
    backbone_bn_train_mode: bool = False,
) -> DistillEvalResult:
    """
    calibrate_bn_batches: if > 0, run this many loader batches in train() mode
    first to warm up backbone BN running stats before switching to eval().

    backbone_bn_train_mode: keep backbone BN layers in train() mode during the
    eval pass (uses per-batch statistics instead of accumulated running stats).
    Required for ep1-ep17 checkpoints: backbone BN was in train mode throughout
    those epochs, so the Perceiver was trained on batch-normalised features.
    Switching to eval mode (even with calibrated running stats) changes the
    feature distribution enough to push cosine similarity to ~0.
    """
    def _force_backbone_bn_train(model: nn.Module):
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                               nn.SyncBatchNorm)):
                m.train()

    if calibrate_bn_batches > 0:
        # Force ALL BN layers into train mode directly, bypassing
        # StudentTemporalModel.train() override (which keeps backbone in eval).
        # This lets BN accumulate SA-V running stats before the eval pass.
        _force_backbone_bn_train(student)
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= calibrate_bn_batches:
                    break
                frames = batch['frames'].to(device, non_blocking=True)
                attn = batch['attention_mask'].to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp,
                                        cache_enabled=False):
                    student(frames, attn)

    import os as _os
    skip_eval_mode = _os.environ.get('STAGE2_EVAL_NO_EVAL_MODE', '0') == '1'
    if skip_eval_mode:
        # Diagnostic: keep student in train mode (Perceiver in train mode,
        # backbone still in eval via StudentTemporalModel.train() override).
        # This is what the training loop sees per step.
        student.train()
    else:
        student.eval()
    # For legacy ep1-ep17 checkpoints: backbone BN was always in train mode
    # during training. Re-force train mode so eval uses per-batch stats,
    # matching the distribution the Perceiver was trained on.
    if backbone_bn_train_mode:
        _force_backbone_bn_train(student)

    sums = torch.zeros(4, device=device, dtype=torch.float64)  # total, mse, cos, norm
    n = torch.zeros((), device=device, dtype=torch.float64)

    import os
    debug_per_batch = os.environ.get('STAGE2_EVAL_DEBUG', '0') == '1'
    no_amp = os.environ.get('STAGE2_EVAL_NO_AMP', '0') == '1'
    if no_amp:
        use_amp = False
        print('[eval-debug] AMP disabled for diagnostic', flush=True)

    for i, batch in enumerate(loader):
        if 0 < max_batches <= i:
            break
        frames = batch['frames'].to(device, non_blocking=True)
        attn = batch['attention_mask'].to(device, non_blocking=True)
        gt_masks = batch['masks'].to(device, non_blocking=True)
        mask_valid = batch['mask_valid'].to(device, non_blocking=True)

        # cache_enabled=False — match the production training context. Stale
        # bf16 weight cache otherwise gives different forward output than the
        # training-time forward. See debug_definitive.py for the proof.
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp,
                                cache_enabled=False):
            s_out = student(frames, attn)
            t_out = teacher(frames, attn, gt_masks, mask_valid)
            losses = loss_fn(s_out, t_out, attn)

        if debug_per_batch:
            attn_sum = int(attn.sum().item())
            mv_sum = int(mask_valid.sum().item())
            s_norm = float(s_out.float().norm(p=2, dim=2).mean().item())
            t_norm = float(t_out.float().norm(p=2, dim=2).mean().item())
            print(f"[eval-debug] batch={i} attn_sum={attn_sum} mv_sum={mv_sum} "
                  f"s_norm={s_norm:.4f} t_norm={t_norm:.4f} "
                  f"total={losses.total.item():.4f} mse={losses.mse.item():.4f} "
                  f"cos={losses.cosine.item():.4f} norm={losses.norm.item():.4f}",
                  flush=True)

        sums[0] += losses.total.detach().double()
        sums[1] += losses.mse.detach().double()
        sums[2] += losses.cosine.detach().double()
        sums[3] += losses.norm.detach().double()
        n += 1

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(n, op=dist.ReduceOp.SUM)

    n_val = max(1.0, n.item())
    student.train()
    return DistillEvalResult(
        total=(sums[0].item() / n_val),
        mse=(sums[1].item() / n_val),
        cosine=(sums[2].item() / n_val),
        norm=(sums[3].item() / n_val),
        n_batches=int(n.item()),
    )
