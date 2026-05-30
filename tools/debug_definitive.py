"""Definitive: is `torch.clear_autocast_cache()` the fix?

Trains model_A 10 steps, builds model_B fresh + load_state_dict, then runs
forward on both A and B with and without clearing the autocast cache between
A and B. Same input, same weights, same dtype, same device.
"""

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
    for _ in range(10):
        with torch.amp.autocast('cuda', dtype=amp_dtype):
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
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                return m(frames_ev, attn_ev)

    def norm(t):
        return float(t.float().norm(p=2, dim=2).mean().item())

    # --- Scenario 1: NO CLEAR between training and either forward ---
    print('=== Scenario 1: forward A → forward B (no cache clear) ===')
    s_A_noclr = fwd(model_A)
    s_B_noclr = fwd(model_B)
    print(f'  s_A norm = {norm(s_A_noclr):.4f}')
    print(f'  s_B norm = {norm(s_B_noclr):.4f}')
    print(f'  max_diff = {float((s_A_noclr.float() - s_B_noclr.float()).abs().max()):.6e}')

    # --- Scenario 2: clear, then forward both ---
    print('\n=== Scenario 2: clear → forward A → forward B ===')
    torch.clear_autocast_cache()
    s_A_clr = fwd(model_A)
    s_B_clr = fwd(model_B)
    print(f'  s_A norm = {norm(s_A_clr):.4f}')
    print(f'  s_B norm = {norm(s_B_clr):.4f}')
    print(f'  max_diff = {float((s_A_clr.float() - s_B_clr.float()).abs().max()):.6e}')

    # --- Scenario 3: clear, forward A, clear AGAIN, forward B ---
    print('\n=== Scenario 3: clear → A → clear → B ===')
    torch.clear_autocast_cache()
    s_A_3 = fwd(model_A)
    torch.clear_autocast_cache()
    s_B_3 = fwd(model_B)
    print(f'  s_A norm = {norm(s_A_3):.4f}')
    print(f'  s_B norm = {norm(s_B_3):.4f}')
    print(f'  max_diff = {float((s_A_3.float() - s_B_3.float()).abs().max()):.6e}')

    # --- Scenario 4: forward B FIRST, then A ---
    print('\n=== Scenario 4: clear → B → A (reversed order) ===')
    torch.clear_autocast_cache()
    s_B_4 = fwd(model_B)
    s_A_4 = fwd(model_A)
    print(f'  s_B norm = {norm(s_B_4):.4f}')
    print(f'  s_A norm = {norm(s_A_4):.4f}')
    print(f'  max_diff = {float((s_A_4.float() - s_B_4.float()).abs().max()):.6e}')

    # --- Compare A across scenarios (does A's output change?) ---
    print('\n=== Same model_A under different cache states ===')
    print(f'  A_noclr  norm = {norm(s_A_noclr):.4f}')
    print(f'  A_clr    norm = {norm(s_A_clr):.4f}')
    print(f'  A_3      norm = {norm(s_A_3):.4f}')
    print(f'  A_4      norm = {norm(s_A_4):.4f}')
    print(f'  A_noclr vs A_clr: {float((s_A_noclr.float() - s_A_clr.float()).abs().max()):.6e}')


if __name__ == '__main__':
    main()
