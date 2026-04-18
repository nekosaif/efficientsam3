"""Checkpoint save/load + auto-resume for Stage 2.

Persists only trainable params (Perceiver) — the frozen RepViT/SAM backbones
come from their own pretrained checkpoints at construction time, so duplicating
them here would 10x the file size for no gain.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Dict, Optional

import torch


def _trainable_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    full = model.state_dict()
    return {k: v for k, v in full.items() if k in trainable_names}


def save_checkpoint(
    *,
    config,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[torch.amp.GradScaler],
    best_val: Optional[float],
    is_best: bool = False,
):
    out_dir = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
    os.makedirs(out_dir, exist_ok=True)

    payload = {
        'epoch': epoch,
        'global_step': global_step,
        'model_trainable': _trainable_state_dict(model),
        'optimizer': optimizer.state_dict(),
        'scheduler_iter': getattr(scheduler, '_iter', global_step),
        'scaler': scaler.state_dict() if (scaler is not None and scaler.is_enabled()) else None,
        'best_val': best_val,
        'config': config.dump() if hasattr(config, 'dump') else str(config),
    }

    path = os.path.join(out_dir, f'ckpt_epoch_{epoch}.pth')
    torch.save(payload, path)

    latest = os.path.join(out_dir, 'ckpt_latest.pth')
    if os.path.islink(latest) or os.path.exists(latest):
        try:
            os.remove(latest)
        except OSError:
            pass
    try:
        os.symlink(os.path.basename(path), latest)
    except OSError:
        torch.save(payload, latest)

    if is_best:
        best_path = os.path.join(out_dir, 'ckpt_best.pth')
        torch.save(payload, best_path)

    return path


def load_checkpoint(
    *,
    config,
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler: Optional[torch.amp.GradScaler] = None,
    map_location='cpu',
) -> Dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)

    sd = payload.get('model_trainable') or payload.get('model')
    msg = model.load_state_dict(sd, strict=False)
    missing = [k for k in msg.missing_keys if k in {n for n, p in model.named_parameters() if p.requires_grad}]
    if missing:
        raise RuntimeError(f"Missing trainable keys when loading {path}: {missing[:8]}")

    if optimizer is not None and 'optimizer' in payload:
        optimizer.load_state_dict(payload['optimizer'])
    if scheduler is not None and 'scheduler_iter' in payload:
        scheduler.step(int(payload['scheduler_iter']))
    if scaler is not None and payload.get('scaler') is not None and scaler.is_enabled():
        scaler.load_state_dict(payload['scaler'])

    return {
        'epoch': int(payload.get('epoch', 0)),
        'global_step': int(payload.get('global_step', 0)),
        'best_val': payload.get('best_val'),
    }


def auto_resume_helper(config) -> Optional[str]:
    out_dir = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
    latest = os.path.join(out_dir, 'ckpt_latest.pth')
    if os.path.exists(latest):
        return latest
    cands = glob.glob(os.path.join(out_dir, 'ckpt_epoch_*.pth'))
    if not cands:
        return None
    def _epoch(p):
        m = re.search(r'ckpt_epoch_(\d+)\.pth', os.path.basename(p))
        return int(m.group(1)) if m else -1
    return max(cands, key=_epoch)
