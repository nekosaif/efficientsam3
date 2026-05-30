"""Pinpoint the Stage-2 save/load corruption.

Hypothesis chain:
  H1 — `model.state_dict()` captures full forward-deterministic state. If a
        fresh model loaded with `.load_state_dict(state)` reproduces the same
        forward output, then the bug is purely that `_trainable_state_dict`
        drops buffers / shared params.
  H2 — `state_dict()` does NOT capture full state. Then we have hidden state
        outside state_dict (non-registered tensors, RNG state, autocast cache,
        etc.). Need to find which layer's output diverges first.

Outputs:
  - s_out per-token L2 norm for: original (model_A), full-load (model_B_full),
    partial-load (model_B_partial)
  - key-set diffs between full and partial state_dict
  - per-layer divergence point if model_B != model_A under full load

Run:
  python3 tools/debug_save_load.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sam3'))

from stage2.config import get_config  # noqa: E402
from stage2.models import build_student_model  # noqa: E402


class _Args:
    cfg = 'stage2/configs/sav_repvit_m0_9.yaml'
    opts = None
    batch_size = None
    data_path = None
    student_ckpt = None
    teacher_ckpt = None
    resume = None
    output = None
    tag = None
    accumulation_steps = None
    use_checkpoint = False
    disable_amp = False
    only_cpu = False
    eval = False
    throughput = False
    local_rank = 0


def _norms(t: torch.Tensor) -> dict:
    """Per-token L2 norm summary."""
    if t.ndim == 5:
        # [B, T, C, H, W]
        per_token = t.float().norm(p=2, dim=2)
    else:
        per_token = t.float().norm(p=2, dim=-1)
    return {
        "mean": float(per_token.mean().item()),
        "max": float(per_token.max().item()),
        "min": float(per_token.min().item()),
        "std": float(per_token.std().item()),
    }


def main():
    config = get_config(_Args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Need deterministic input on a fixed seed so any drift comes from the model
    torch.manual_seed(42)

    # ---- build model A ---------------------------------------------------------
    print('=== Build model A (fresh from Stage 1) ===', flush=True)
    model_A = build_student_model(config).to(device).eval()
    n_train = sum(p.numel() for p in model_A.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model_A.parameters())
    n_buf = sum(b.numel() for b in model_A.buffers())
    print(f'  trainable={n_train/1e6:.2f}M total_params={n_total/1e6:.2f}M  '
          f'buffers={n_buf/1e6:.2f}M', flush=True)

    # ---- inspect what state_dict() contains -----------------------------------
    full_sd = model_A.state_dict()
    param_names = {n for n, _ in model_A.named_parameters()}
    buf_names = {n for n, _ in model_A.named_buffers()}
    trainable_names = {n for n, p in model_A.named_parameters() if p.requires_grad}
    sd_keys = set(full_sd.keys())
    print(f'\n=== state_dict coverage ===')
    print(f'  state_dict keys     : {len(sd_keys)}')
    print(f'  named_parameters    : {len(param_names)}')
    print(f'  named_buffers       : {len(buf_names)}')
    print(f'  trainable subset    : {len(trainable_names)}')
    not_in_sd = (param_names | buf_names) - sd_keys
    if not_in_sd:
        print(f'  ⚠ in named_* but NOT in state_dict: {len(not_in_sd)}')
        for k in list(not_in_sd)[:10]:
            print(f'      - {k}')
    sd_only = sd_keys - (param_names | buf_names)
    if sd_only:
        print(f'  ⚠ in state_dict but NOT in named_*: {len(sd_only)}')
        for k in list(sd_only)[:10]:
            print(f'      - {k}')

    # ---- generate fixed input -------------------------------------------------
    print('\n=== Reference forward (model A) ===', flush=True)
    torch.manual_seed(123)
    B, T, H, W = 1, 8, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE
    frames = torch.randn(B, T, 3, H, W, device=device)
    attn = torch.ones(B, T, dtype=torch.bool, device=device)
    with torch.no_grad():
        s_A = model_A(frames, attn)
    print(f'  s_A shape={tuple(s_A.shape)}  norm={_norms(s_A)}', flush=True)

    # ---- Test H1: round-trip with FULL state_dict -----------------------------
    print('\n=== Round-trip with FULL state_dict ===', flush=True)
    torch.save({'state': model_A.state_dict()}, '/tmp/diag_full_sd.pth')
    model_B_full = build_student_model(config).to(device).eval()
    full_sd_loaded = torch.load('/tmp/diag_full_sd.pth', weights_only=False)['state']
    msg = model_B_full.load_state_dict(full_sd_loaded, strict=True)
    print(f'  load strict=True OK (missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)})')
    with torch.no_grad():
        s_Bf = model_B_full(frames, attn)
    print(f'  s_B (full) norm={_norms(s_Bf)}')
    match_full = torch.allclose(s_A, s_Bf, atol=1e-3, rtol=1e-3)
    max_abs_diff_full = float((s_A - s_Bf).abs().max())
    print(f'  match_full={match_full}  max_abs_diff={max_abs_diff_full:.6f}')

    # ---- Test H2: round-trip with PARTIAL state_dict (current scheme) ----------
    print('\n=== Round-trip with PARTIAL state_dict (trainable only) ===', flush=True)
    partial_sd = {k: v for k, v in model_A.state_dict().items() if k in trainable_names}
    torch.save({'state': partial_sd}, '/tmp/diag_partial_sd.pth')
    print(f'  partial_sd: {len(partial_sd)} keys vs full {len(full_sd)} keys '
          f'(dropped {len(full_sd) - len(partial_sd)})')
    # show which keys are in full but NOT in partial
    dropped = sd_keys - set(partial_sd.keys())
    print(f'  ~~ first 20 dropped keys ~~')
    for k in sorted(dropped)[:20]:
        print(f'      - {k}')
    print(f'  ~~ last 5 dropped keys ~~')
    for k in sorted(dropped)[-5:]:
        print(f'      - {k}')

    model_B_partial = build_student_model(config).to(device).eval()
    partial_sd_loaded = torch.load('/tmp/diag_partial_sd.pth', weights_only=False)['state']
    msg = model_B_partial.load_state_dict(partial_sd_loaded, strict=False)
    print(f'  load strict=False (missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)})')
    if msg.missing_keys:
        print(f'  first 5 missing: {msg.missing_keys[:5]}')
    with torch.no_grad():
        s_Bp = model_B_partial(frames, attn)
    print(f'  s_B (partial) norm={_norms(s_Bp)}')
    match_partial = torch.allclose(s_A, s_Bp, atol=1e-3, rtol=1e-3)
    max_abs_diff_partial = float((s_A - s_Bp).abs().max())
    print(f'  match_partial={match_partial}  max_abs_diff={max_abs_diff_partial:.6f}')

    # ---- Verdict ---------------------------------------------------------------
    print('\n=== VERDICT ===')
    if match_full and not match_partial:
        print('  → state_dict captures full state, but PARTIAL save drops critical keys')
        print('  → Fix: save full state_dict (or whitelisted-set including missing keys)')
    elif match_full and match_partial:
        print('  → Both full and partial round-trip reproduce s_A. Bug is elsewhere')
        print('     (build_student_model nondeterminism? autocast cache? RNG?)')
    elif not match_full:
        print('  → state_dict() does NOT capture full forward state')
        print(f'     full-load max diff: {max_abs_diff_full:.4f}')
        print(f'     partial-load max diff: {max_abs_diff_partial:.4f}')
        print('  → Look for non-registered tensor attributes or build-time randomness')
    else:
        print('  → Inconsistent state, partial fails but full passes — investigate')


if __name__ == '__main__':
    main()
