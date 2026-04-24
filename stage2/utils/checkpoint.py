"""Checkpoint save/load + auto-resume for Stage 2.

Persists only trainable params (Perceiver) — the frozen RepViT/SAM backbones
come from their own pretrained checkpoints at construction time, so duplicating
them here would 10x the file size for no gain.

Two flavors of save:
  - save_checkpoint()         epoch-end: ckpt_epoch_N.pth (+ best/latest)
  - save_running_checkpoint() mid-epoch: ckpt_running.pth (single overwrite)

Both honor optional EMA shadow. auto_resume_helper prefers ckpt_running.pth
when newer than the latest epoch ckpt — so a SIGHUP mid-epoch loses minutes,
not hours.
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


def _build_payload(
    *,
    config,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    best_val,
    ema=None,
    step_in_epoch: int = 0,
) -> Dict:
    return {
        'epoch': epoch,
        'global_step': global_step,
        'step_in_epoch': step_in_epoch,
        'model_trainable': _trainable_state_dict(model),
        'optimizer': optimizer.state_dict(),
        'scheduler_iter': getattr(scheduler, '_iter', global_step),
        'scaler': scaler.state_dict() if (scaler is not None and scaler.is_enabled()) else None,
        'ema': ema.state_dict() if ema is not None else None,
        'best_val': best_val,
        'config': config.dump() if hasattr(config, 'dump') else str(config),
    }


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
    ema=None,
):
    out_dir = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
    os.makedirs(out_dir, exist_ok=True)

    payload = _build_payload(
        config=config, epoch=epoch, global_step=global_step,
        model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        best_val=best_val, ema=ema, step_in_epoch=0,
    )

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

    # epoch checkpoint supersedes any stale running ckpt — drop it
    running = os.path.join(out_dir, 'ckpt_running.pth')
    if os.path.exists(running):
        try:
            os.remove(running)
        except OSError:
            pass

    return path


def save_running_checkpoint(
    *,
    config,
    epoch: int,
    global_step: int,
    step_in_epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[torch.amp.GradScaler],
    best_val: Optional[float],
    ema=None,
) -> str:
    """Mid-epoch save. Single overwriting file (ckpt_running.pth).

    Atomic write via tmp + rename so an interrupted save doesn't corrupt the
    previous good running ckpt.
    """
    out_dir = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
    os.makedirs(out_dir, exist_ok=True)

    payload = _build_payload(
        config=config, epoch=epoch, global_step=global_step,
        model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        best_val=best_val, ema=ema, step_in_epoch=step_in_epoch,
    )

    path = os.path.join(out_dir, 'ckpt_running.pth')
    tmp = path + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(
    *,
    config,
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler: Optional[torch.amp.GradScaler] = None,
    ema=None,
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
    if ema is not None and payload.get('ema') is not None:
        ema.load_state_dict(payload['ema'])

    return {
        'epoch': int(payload.get('epoch', 0)),
        'global_step': int(payload.get('global_step', 0)),
        'step_in_epoch': int(payload.get('step_in_epoch', 0)),
        'best_val': payload.get('best_val'),
        'has_ema': payload.get('ema') is not None,
    }


def auto_resume_helper(config) -> Optional[str]:
    """Pick newest viable resume checkpoint.

    Order of preference:
      1. ckpt_running.pth if it exists AND is newer than the latest epoch ckpt
         (mid-epoch save protecting against SIGHUP/preemption).
      2. ckpt_latest.pth symlink → most recent epoch ckpt.
      3. Highest-numbered ckpt_epoch_N.pth as fallback.
    """
    out_dir = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    running = os.path.join(out_dir, 'ckpt_running.pth')
    latest = os.path.join(out_dir, 'ckpt_latest.pth')

    have_running = os.path.exists(running)
    have_latest = os.path.exists(latest)

    if have_running and have_latest:
        # mtime compare — running wins iff strictly newer
        if os.path.getmtime(running) > os.path.getmtime(latest):
            return running
        return latest
    if have_running:
        return running
    if have_latest:
        return latest

    cands = glob.glob(os.path.join(out_dir, 'ckpt_epoch_*.pth'))
    if not cands:
        return None
    def _epoch(p):
        m = re.search(r'ckpt_epoch_(\d+)\.pth', os.path.basename(p))
        return int(m.group(1)) if m else -1
    return max(cands, key=_epoch)
