"""Force the SDPA math backend for the perceiver MHA — confirm or refute that
the divergence is a backend-dispatch issue."""

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

    blk_A = model_A.perceiver.blocks[0]
    blk_B = model_B.perceiver.blocks[0]

    print('=== Default backend dispatch (current behavior) ===')
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            pix_A = model_A.encode_frame(frames_ev[:, 0])
            pix_B = model_B.encode_frame(frames_ev[:, 0])
            tokens_A = pix_A.flatten(2).transpose(1, 2)
            tokens_B = pix_B.flatten(2).transpose(1, 2)
            lat = model_A.perceiver.init_memory(B, device=device, dtype=frames_ev.dtype)
            tpe = model_A.perceiver.temporal_embed[0]
            latents = lat + tpe

            q_A = blk_A.norm_q1(latents)
            kv_A = blk_A.norm_kv1(tokens_A)
            q_B = blk_B.norm_q1(latents)
            kv_B = blk_B.norm_kv1(tokens_B)
            assert torch.equal(q_A, q_B) and torch.equal(kv_A, kv_B), "inputs already differ"
            xA, _ = blk_A.cross_attn(q_A, kv_A, kv_A, need_weights=False)
            xB, _ = blk_B.cross_attn(q_B, kv_B, kv_B, need_weights=False)
    d_default = float((xA.float() - xB.float()).abs().max())
    print(f'  default backend max_diff = {d_default:.6e}')

    # Force math backend
    print('\n=== Forced MATH backend (deterministic) ===')
    try:
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                    xAm, _ = blk_A.cross_attn(q_A, kv_A, kv_A, need_weights=False)
                    xBm, _ = blk_B.cross_attn(q_B, kv_B, kv_B, need_weights=False)
        d_math = float((xAm.float() - xBm.float()).abs().max())
        print(f'  math backend max_diff = {d_math:.6e}')
    except AttributeError:
        # fallback for older PyTorch
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
                    xAm, _ = blk_A.cross_attn(q_A, kv_A, kv_A, need_weights=False)
                    xBm, _ = blk_B.cross_attn(q_B, kv_B, kv_B, need_weights=False)
        d_math = float((xAm.float() - xBm.float()).abs().max())
        print(f'  math backend max_diff = {d_math:.6e}')

    # Force flash backend
    print('\n=== Forced FLASH backend ===')
    try:
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                    xAf, _ = blk_A.cross_attn(q_A, kv_A, kv_A, need_weights=False)
                    xBf, _ = blk_B.cross_attn(q_B, kv_B, kv_B, need_weights=False)
        d_flash = float((xAf.float() - xBf.float()).abs().max())
        print(f'  flash backend max_diff = {d_flash:.6e}')
    except Exception as e:
        print(f'  flash unsupported: {e}')

    # Force mem-efficient backend
    print('\n=== Forced EFFICIENT_ATTENTION backend ===')
    try:
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
                    xAe, _ = blk_A.cross_attn(q_A, kv_A, kv_A, need_weights=False)
                    xBe, _ = blk_B.cross_attn(q_B, kv_B, kv_B, need_weights=False)
        d_eff = float((xAe.float() - xBe.float()).abs().max())
        print(f'  efficient backend max_diff = {d_eff:.6e}')
    except Exception as e:
        print(f'  efficient unsupported: {e}')

    # Compare full-forward s_norm with math backend forced
    print('\n=== Full s_out with MATH backend forced ===')
    try:
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                    s_A_full = model_A(frames_ev, torch.ones(B, T, dtype=torch.bool, device=device))
                    s_B_full = model_B(frames_ev, torch.ones(B, T, dtype=torch.bool, device=device))
        d_full = float((s_A_full.float() - s_B_full.float()).abs().max())
        a_norm = s_A_full.float().norm(p=2, dim=2).mean().item()
        b_norm = s_B_full.float().norm(p=2, dim=2).mean().item()
        print(f'  s_A norm={a_norm:.4f}  s_B norm={b_norm:.4f}  max_diff={d_full:.6e}')
    except Exception as e:
        print(f'  failed: {e}')


if __name__ == '__main__':
    main()
