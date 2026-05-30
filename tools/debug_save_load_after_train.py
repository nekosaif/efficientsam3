"""Reproduce the save/load bug under realistic conditions.

Steps:
  1. Build model + optimizer + AMP scaler exactly like train.py.
  2. Run N optimizer steps with random inputs (so weights mutate).
  3. Forward → s_A (model in training-loop state).
  4. Save FULL state_dict.
  5. Save the PARTIAL state_dict we currently use in production.
  6. In the SAME process: build a fresh model, load each variant, forward → s_B.
  7. Compare s_A vs s_B for both variants.

If full-load matches and partial doesn't: bug = drop in our _trainable_state_dict.
If even full-load diverges after training: hidden state outside state_dict.

Run:
  python3 tools/debug_save_load_after_train.py [--steps 20]
"""

from __future__ import annotations

import argparse
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


def _summary(t: torch.Tensor) -> str:
    pt = t.float().norm(p=2, dim=2)
    return (f"per-token L2 norm: mean={pt.mean():.4f} max={pt.max():.4f} "
            f"min={pt.min():.4f} std={pt.std():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--lr-mult', type=float, default=100.0,
                    help='Multiply BASE_LR by this to make weights move visibly in a few steps')
    args = ap.parse_args()

    config = get_config(_Args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(42)

    # ---- build model + optimizer (single GPU, no DDP) ------------------------
    print(f'=== Build model A and step it {args.steps} times (lr × {args.lr_mult}) ===', flush=True)
    model_A = build_student_model(config).to(device)
    # Mimic prod: backbone frozen, only perceiver trainable
    optimizer = build_optimizer(config, model_A)
    # Bump LR so weights move visibly in a few iters
    for pg in optimizer.param_groups:
        pg['lr'] = pg['lr'] * args.lr_mult * 100  # 100× extra for visibility
    loss_fn = build_distill_loss(config).to(device)

    use_amp = config.AMP_ENABLE
    amp_dtype = torch.bfloat16 if config.TRAIN.AMP_DTYPE == 'bfloat16' else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))

    B, T, H, W = 1, 8, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE

    # Fix the training input + a synthetic target ("teacher") so optimizer has signal
    torch.manual_seed(0)
    frames_train = torch.randn(B, T, 3, H, W, device=device)
    attn_train = torch.ones(B, T, dtype=torch.bool, device=device)
    # Fake teacher target — random fixed tensor of the expected output shape
    target = torch.randn(B, T, 256, 72, 72, device=device)

    # Print initial weight stats so we can see drift
    def _w_std(m):
        return float(m.perceiver.mlp_out[2].weight.float().std().item())

    print(f'  ep0 init mlp_out.2.weight std = {_w_std(model_A):.4f}')

    model_A.train()  # via override: perceiver train, backbone eval
    for step in range(args.steps):
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            s_out = model_A(frames_train, attn_train)
            losses = loss_fn(s_out, target, attn_train)
            loss = losses.total
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model_A.parameters() if p.requires_grad], max_norm=1.0,
        )
        if scaler.is_enabled():
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    print(f'  ep{args.steps} mlp_out.2.weight std = {_w_std(model_A):.4f}  '
          f'(loss={loss.item():.3f})')

    # ---- now: forward (eval mode) on a FIXED input → reference s_A -----------
    print('\n=== Reference forward s_A (eval mode, post-training) ===')
    torch.manual_seed(123)
    frames_eval = torch.randn(B, T, 3, H, W, device=device)
    attn_eval = torch.ones(B, T, dtype=torch.bool, device=device)
    model_A.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            s_A = model_A(frames_eval, attn_eval)
    print(f'  s_A: {_summary(s_A)}')

    # ---- Save FULL state_dict and PARTIAL ------------------------------------
    full_sd = model_A.state_dict()
    trainable_names = {n for n, p in model_A.named_parameters() if p.requires_grad}
    partial_sd = {k: v for k, v in full_sd.items() if k in trainable_names}
    torch.save({'state': full_sd}, '/tmp/diag_trained_full.pth')
    torch.save({'state': partial_sd}, '/tmp/diag_trained_partial.pth')
    print(f'  saved: full={len(full_sd)} keys, partial={len(partial_sd)} keys')

    # ---- Test FULL load -------------------------------------------------------
    print('\n=== FULL round-trip ===')
    model_B = build_student_model(config).to(device).eval()
    sd = torch.load('/tmp/diag_trained_full.pth', weights_only=False)['state']
    msg = model_B.load_state_dict(sd, strict=True)
    print(f'  load missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}')
    # Confirm weights match
    same_w = torch.equal(model_A.perceiver.mlp_out[2].weight, model_B.perceiver.mlp_out[2].weight)
    print(f'  mlp_out.2.weight identical after load: {same_w}')
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            s_Bf = model_B(frames_eval, attn_eval)
    print(f'  s_B full: {_summary(s_Bf)}')
    max_diff_full = float((s_A.float() - s_Bf.float()).abs().max())
    rel_diff_full = float(((s_A.float() - s_Bf.float()).abs() / s_A.float().abs().clamp_min(1e-6)).mean())
    print(f'  max_abs_diff_full = {max_diff_full:.6f}  mean_rel_diff = {rel_diff_full:.6f}')

    # ---- Test PARTIAL load (with prod-style backbone BN buffer reload) -------
    print('\n=== PARTIAL round-trip (mimicking production save) ===')
    model_C = build_student_model(config).to(device).eval()
    sd_p = torch.load('/tmp/diag_trained_partial.pth', weights_only=False)['state']
    msg = model_C.load_state_dict(sd_p, strict=False)
    print(f'  load missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}')
    # Mirror production: also save+load backbone BN buffers
    from stage2.utils.checkpoint import _backbone_bn_buffers, _load_backbone_bn_buffers
    bn_buf = _backbone_bn_buffers(model_A)
    _load_backbone_bn_buffers(model_C, bn_buf)
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            s_Bp = model_C(frames_eval, attn_eval)
    print(f'  s_B partial: {_summary(s_Bp)}')
    max_diff_p = float((s_A.float() - s_Bp.float()).abs().max())
    print(f'  max_abs_diff_partial = {max_diff_p:.6f}')

    # ---- Verdict -------------------------------------------------------------
    print('\n=== VERDICT ===')
    if max_diff_full < 1e-3 and max_diff_p < 1e-3:
        print('  → BOTH match. Bug NOT in save/load of post-training model.')
        print('     Likely culprit: build_student_model nondeterminism between processes,')
        print('     or runtime state outside state_dict (autocast cache, randomness).')
    elif max_diff_full < 1e-3 and max_diff_p > 1e-3:
        print('  → FULL matches but PARTIAL diverges → fix is to save FULL state_dict.')
    elif max_diff_full > 1e-3:
        print('  → Even FULL state_dict load diverges. Hidden state outside state_dict.')


if __name__ == '__main__':
    main()
