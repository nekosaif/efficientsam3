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
) -> DistillEvalResult:
    student.eval()
    sums = torch.zeros(3, device=device, dtype=torch.float64)  # total, mse, cos
    n = torch.zeros((), device=device, dtype=torch.float64)

    for i, batch in enumerate(loader):
        if 0 < max_batches <= i:
            break
        frames = batch['frames'].to(device, non_blocking=True)
        attn = batch['attention_mask'].to(device, non_blocking=True)
        gt_masks = batch['masks'].to(device, non_blocking=True)
        mask_valid = batch['mask_valid'].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            s_out = student(frames, attn)
            t_out = teacher(frames, attn, gt_masks, mask_valid)
            losses = loss_fn(s_out, t_out, attn)

        sums[0] += losses.total.detach().double()
        sums[1] += losses.mse.detach().double()
        sums[2] += losses.cosine.detach().double()
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
        n_batches=int(n.item()),
    )
