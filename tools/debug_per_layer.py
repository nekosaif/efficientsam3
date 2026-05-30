"""Find the first layer whose output diverges between model_A (trained, same process)
and model_B (freshly built + load_state_dict from A).

Confirms tensor-identity / byte-identity at every parameter and buffer first.
Then walks the model's forward step-by-step via hooks and reports the first
layer with a divergent output.
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


def _hook_all(model, store: dict):
    handles = []
    for name, mod in model.named_modules():
        if name == '':
            continue
        def make(name):
            def hook(m, _i, o):
                store[name] = o
            return hook
        handles.append(mod.register_forward_hook(make(name)))
    return handles


def _tensor_summary(t):
    if isinstance(t, tuple):
        return [_tensor_summary(x) for x in t]
    if isinstance(t, dict):
        return {k: _tensor_summary(v) for k, v in t.items()}
    if not torch.is_tensor(t):
        return f"non-tensor: {type(t).__name__}"
    return f"shape={tuple(t.shape)} dtype={t.dtype} norm={t.float().norm().item():.4f}"


def _max_abs(a, b):
    if isinstance(a, tuple):
        return max(_max_abs(x, y) for x, y in zip(a, b))
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        return -1.0
    if a.shape != b.shape:
        return float('inf')
    return float((a.float() - b.float()).abs().max().item())


def main():
    config = get_config(_Args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)

    print('=== Build + train model A ===')
    model_A = build_student_model(config).to(device)
    optimizer = build_optimizer(config, model_A)
    for pg in optimizer.param_groups:
        pg['lr'] *= 10000
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
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    print(f'  ep10 mlp_out.2.weight std = {model_A.perceiver.mlp_out[2].weight.float().std():.4f}')

    print('\n=== Build model B (fresh) ===')
    model_B = build_student_model(config).to(device)
    sd = model_A.state_dict()
    msg = model_B.load_state_dict(sd, strict=True)
    print(f'  load strict=True missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}')

    # Byte-identity check on every named_parameter + named_buffer
    print('\n=== Byte-identity check ===')
    diffs = []
    for (name_a, p_a), (name_b, p_b) in zip(model_A.named_parameters(), model_B.named_parameters()):
        assert name_a == name_b
        if not torch.equal(p_a.data, p_b.data):
            d = float((p_a.float() - p_b.float()).abs().max().item())
            diffs.append((name_a, 'param', d, tuple(p_a.shape)))
    for (name_a, b_a), (name_b, b_b) in zip(model_A.named_buffers(), model_B.named_buffers()):
        assert name_a == name_b
        if b_a.dtype != b_b.dtype:
            diffs.append((name_a, 'buffer-dtype', float('nan'), (b_a.dtype, b_b.dtype)))
            continue
        if not torch.equal(b_a.data, b_b.data):
            d = float((b_a.float() - b_b.float()).abs().max().item()) if b_a.is_floating_point() else float('nan')
            diffs.append((name_a, 'buffer', d, tuple(b_a.shape)))
    if not diffs:
        print('  ✓ All params + buffers byte-identical')
    else:
        print(f'  ⚠ {len(diffs)} differing tensors:')
        for name, kind, d, sh in diffs[:25]:
            print(f'      [{kind}] {name}: max_diff={d}  shape={sh}')

    # Compare module-mode flags
    print('\n=== self.training flag comparison ===')
    model_A.eval(); model_B.eval()
    flag_diffs = []
    for (name_a, m_a), (name_b, m_b) in zip(model_A.named_modules(), model_B.named_modules()):
        if m_a.training != m_b.training:
            flag_diffs.append((name_a, m_a.training, m_b.training))
    print(f'  {"OK" if not flag_diffs else "⚠ " + str(len(flag_diffs)) + " differing"}')
    for n, a, b in flag_diffs[:10]:
        print(f'      {n}: A.training={a}  B.training={b}')

    # Compare __dict__ attribute sets between corresponding modules
    print('\n=== __dict__ attribute key comparison ===')
    attr_diffs = []
    for (name_a, m_a), (name_b, m_b) in zip(model_A.named_modules(), model_B.named_modules()):
        a_keys = set(m_a.__dict__.keys())
        b_keys = set(m_b.__dict__.keys())
        if a_keys != b_keys:
            attr_diffs.append((name_a, a_keys - b_keys, b_keys - a_keys))
    print(f'  {"OK" if not attr_diffs else "⚠ " + str(len(attr_diffs)) + " modules with different attr sets"}')
    for n, a_only, b_only in attr_diffs[:5]:
        print(f'      {n}: A-only={a_only}  B-only={b_only}')

    # Now: forward both, capture per-layer outputs via hook, find first divergence
    print('\n=== Per-layer forward divergence walk ===')
    torch.manual_seed(123)
    frames_ev = torch.randn(B, T, 3, H, W, device=device)
    attn_ev = torch.ones(B, T, dtype=torch.bool, device=device)

    store_A: dict = {}
    store_B: dict = {}
    h_A = _hook_all(model_A, store_A)
    h_B = _hook_all(model_B, store_B)
    try:
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                s_A = model_A(frames_ev, attn_ev)
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                s_B = model_B(frames_ev, attn_ev)
    finally:
        for h in h_A: h.remove()
        for h in h_B: h.remove()

    print(f'  s_A summary: {_tensor_summary(s_A)}')
    print(f'  s_B summary: {_tensor_summary(s_B)}')

    # Walk hooks in registration order, find first divergence
    print('\n  --- first 20 diverging modules (in module-graph order) ---')
    first_divergences = []
    for name in store_A.keys():
        if name not in store_B:
            continue
        try:
            d = _max_abs(store_A[name], store_B[name])
        except Exception as e:
            d = float('nan')
        if d > 1e-3 or d != d:
            first_divergences.append((name, d))
        if len(first_divergences) >= 20:
            break
    for name, d in first_divergences:
        print(f'      [diff={d:.4e}]  {name}')


if __name__ == '__main__':
    main()
