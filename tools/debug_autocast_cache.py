"""Test if the bug is stale autocast bf16 cache on model_A's parameters."""

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


def _v(p):
    """Storage version (bumps on in-place mutation)."""
    try:
        return p._version
    except AttributeError:
        return p.data._version


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

    # Check tensor versions BEFORE first eval forward
    p_A = model_A.perceiver.blocks[0].cross_attn.in_proj_weight
    p_B = model_B.perceiver.blocks[0].cross_attn.in_proj_weight
    print('=== Param state (before any eval forward) ===')
    print(f'  A.in_proj_weight: data_ptr={p_A.data_ptr():#x}  ._version={_v(p_A)}  '
          f'shape={tuple(p_A.shape)}  std={p_A.float().std():.4f}')
    print(f'  B.in_proj_weight: data_ptr={p_B.data_ptr():#x}  ._version={_v(p_B)}  '
          f'shape={tuple(p_B.shape)}  std={p_B.float().std():.4f}')
    print(f'  byte-equal: {torch.equal(p_A, p_B)}')

    torch.manual_seed(123)
    frames_ev = torch.randn(B, T, 3, H, W, device=device)
    attn_ev = torch.ones(B, T, dtype=torch.bool, device=device)

    # Hypothesis 1: stale autocast cache — clear it before A's forward
    print('\n=== A forward WITH clear_autocast_cache() before ===')
    torch.clear_autocast_cache()
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            s_A_cleared = model_A(frames_ev, attn_ev)
    a_norm_cleared = s_A_cleared.float().norm(p=2, dim=2).mean().item()
    print(f'  s_A norm (cleared cache) = {a_norm_cleared:.4f}')

    print('\n=== A forward WITHOUT clearing cache (just open new autocast context) ===')
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            s_A_default = model_A(frames_ev, attn_ev)
    a_norm_default = s_A_default.float().norm(p=2, dim=2).mean().item()
    print(f'  s_A norm (default)        = {a_norm_default:.4f}')

    diff_AA = float((s_A_cleared.float() - s_A_default.float()).abs().max())
    print(f'  diff between cleared and default A: {diff_AA:.6e}')

    print('\n=== B forward (same context) ===')
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            s_B = model_B(frames_ev, attn_ev)
    b_norm = s_B.float().norm(p=2, dim=2).mean().item()
    print(f'  s_B norm = {b_norm:.4f}')

    diff_AB_cleared = float((s_A_cleared.float() - s_B.float()).abs().max())
    diff_AB_default = float((s_A_default.float() - s_B.float()).abs().max())
    print(f'\n  A_cleared vs B: max_diff = {diff_AB_cleared:.6e}')
    print(f'  A_default vs B: max_diff = {diff_AB_default:.6e}')

    # Hypothesis 2: maybe MEAN-cast tensor differs due to `.data.copy_()` mutation
    # of B's params having a different storage layout than A's after-training params.
    print('\n=== Storage diagnostics ===')
    print(f'  A.in_proj_weight is_contiguous: {p_A.is_contiguous()}')
    print(f'  B.in_proj_weight is_contiguous: {p_B.is_contiguous()}')
    print(f'  A.in_proj_weight stride: {p_A.stride()}')
    print(f'  B.in_proj_weight stride: {p_B.stride()}')

    # Hypothesis 3: maybe optimizer.state has bias_correction state etc. that
    # affects future forwards somehow. Already cleared via optimizer.zero_grad.

    # Hypothesis 4: maybe model_A has GRADIENTS that affect autocast somehow
    print(f'\n  A.in_proj_weight.grad: {p_A.grad}')
    print(f'  B.in_proj_weight.grad: {p_B.grad}')


if __name__ == '__main__':
    main()
