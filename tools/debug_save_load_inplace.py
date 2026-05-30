"""Localize hidden state by reloading state_dict INTO THE SAME model_A.

If model_A's forward output changes after `model_A.load_state_dict(model_A.state_dict())`,
the bug isn't about building two models — it's literally that load_state_dict
disturbs/erases state that lives outside the state_dict.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sam3'))

from stage2.config import get_config  # noqa: E402
from stage2.loss import build_distill_loss  # noqa: E402
from stage2.models import build_student_model  # noqa: E402
from stage2.optim import build_optimizer  # noqa: E402


class _Args:
    cfg = 'stage2/configs/sav_repvit_m0_9.yaml'
    opts = None
    batch_size = None; data_path = None; student_ckpt = None
    teacher_ckpt = None; resume = None; output = None; tag = None
    accumulation_steps = None; use_checkpoint = False; disable_amp = False
    only_cpu = False; eval = False; throughput = False; local_rank = 0


def _summary(t):
    pt = t.float().norm(p=2, dim=2)
    return f"mean={pt.mean():.4f} std={pt.std():.4f} max={pt.max():.4f}"


def main():
    config = get_config(_Args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)

    model = build_student_model(config).to(device)
    optimizer = build_optimizer(config, model)
    for pg in optimizer.param_groups:
        pg['lr'] *= 10000  # blow up to see big weight changes quickly
    loss_fn = build_distill_loss(config).to(device)
    amp_dtype = torch.bfloat16

    # 10 train steps so we leave init regime
    B, T, H, W = 1, 8, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE
    torch.manual_seed(0)
    frames_tr = torch.randn(B, T, 3, H, W, device=device)
    attn_tr = torch.ones(B, T, dtype=torch.bool, device=device)
    target = torch.randn(B, T, 256, 72, 72, device=device)
    model.train()
    for _ in range(10):
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            s = model(frames_tr, attn_tr)
            l = loss_fn(s, target, attn_tr).total
        l.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    print(f'after 10 steps: mlp_out.2.weight std = '
          f'{model.perceiver.mlp_out[2].weight.float().std():.4f}, loss = {l.item():.3f}')

    # Same eval input across all forwards
    torch.manual_seed(123)
    frames_ev = torch.randn(B, T, 3, H, W, device=device)
    attn_ev = torch.ones(B, T, dtype=torch.bool, device=device)

    # Reference: forward A
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            s_A = model(frames_ev, attn_ev)
    print(f's_A (before reload):           {_summary(s_A)}')

    # In-place reload — state_dict captured at THIS instant, immediately loaded back.
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    msg = model.load_state_dict(sd, strict=True)
    print(f'  reload missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}')
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            s_A2 = model(frames_ev, attn_ev)
    print(f's_A2 (after reload, same model):{_summary(s_A2)}')
    diff_inplace = float((s_A.float() - s_A2.float()).abs().max())
    print(f'  max_abs_diff between s_A and s_A2: {diff_inplace:.6f}')

    # Run THREE more times without any reload, see drift between repeated forwards
    print('\nRepeated forward without reload (test forward-determinism):')
    for i in range(3):
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                s_i = model(frames_ev, attn_ev)
        diff_i = float((s_A2.float() - s_i.float()).abs().max())
        print(f'  forward {i+1}: {_summary(s_i)}  max_diff_vs_s_A2={diff_i:.6f}')

    print('\nVERDICT:')
    if diff_inplace > 1e-3:
        print('  → load_state_dict() ON THE SAME MODEL changes forward output.')
        print('     Hidden state outside state_dict is being clobbered by load.')
        print('     Candidates: MultiheadAttention internal buffers, autocast cache invalidation,')
        print('     or an attribute that gets mutated and re-init by setattr in load.')
    else:
        print('  → load_state_dict() in-place is bit-exact.')
        print('     The cross-model divergence must be in build_student_model nondeterminism.')


if __name__ == '__main__':
    main()
