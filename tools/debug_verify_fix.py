"""Verify the cache_enabled=False fix: forward A and B should match
WITHOUT any explicit clear_autocast_cache() call."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sam3'))

from stage2.config import get_config
from stage2.loss import build_distill_loss
from stage2.models import build_student_model
from stage2.optim import build_optimizer


class _Args:
    cfg = 'stage2/configs/sav_repvit_m0_9.yaml'
    opts = None
    batch_size = None; data_path = None; student_ckpt = None
    teacher_ckpt = None; resume = None; output = None; tag = None
    accumulation_steps = None; use_checkpoint = False; disable_amp = False
    only_cpu = False; eval = False; throughput = False; local_rank = 0


def main():
    config = get_config(_Args)
    device = torch.device('cuda')
    torch.manual_seed(42)

    model_A = build_student_model(config).to(device)
    optimizer = build_optimizer(config, model_A)
    for pg in optimizer.param_groups: pg['lr'] *= 10000
    loss_fn = build_distill_loss(config).to(device)
    amp_dtype = torch.bfloat16

    B, T, H, W = 1, 8, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE
    torch.manual_seed(0)
    frames_tr = torch.randn(B, T, 3, H, W, device=device)
    attn_tr = torch.ones(B, T, dtype=torch.bool, device=device)
    target = torch.randn(B, T, 256, 72, 72, device=device)
    model_A.train()
    # Use cache_enabled=False here to mirror the production training-loop fix
    for _ in range(10):
        with torch.amp.autocast('cuda', dtype=amp_dtype, cache_enabled=False):
            s = model_A(frames_tr, attn_tr)
            l = loss_fn(s, target, attn_tr).total
        l.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model_A.parameters() if p.requires_grad], 1.0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)

    model_B = build_student_model(config).to(device)
    model_B.load_state_dict(model_A.state_dict(), strict=True)
    model_A.eval(); model_B.eval()

    torch.manual_seed(123)
    frames_ev = torch.randn(B, T, 3, H, W, device=device)
    attn_ev = torch.ones(B, T, dtype=torch.bool, device=device)

    def fwd(m):
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=amp_dtype, cache_enabled=False):
                return m(frames_ev, attn_ev)

    def n(t): return float(t.float().norm(p=2, dim=2).mean().item())

    print('=== Forward A → B (NO clear, both use cache_enabled=False) ===')
    s_A = fwd(model_A)
    s_B = fwd(model_B)
    print(f'  s_A norm = {n(s_A):.4f}')
    print(f'  s_B norm = {n(s_B):.4f}')
    diff = float((s_A.float() - s_B.float()).abs().max())
    print(f'  max_diff = {diff:.6e}')
    print()
    print('  VERDICT:', '✓ FIX CONFIRMED' if diff < 1e-3 else '✗ Still diverges')


if __name__ == '__main__':
    main()
