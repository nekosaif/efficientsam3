"""Find divergence on the FIRST forward call (frame 0), with per-call capture."""

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


def _maxabs(a, b):
    if isinstance(a, tuple):
        return max(_maxabs(x, y) for x, y in zip(a, b))
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        return -1.0
    return float((a.float() - b.float()).abs().max().item())


def main():
    config = get_config(_Args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)

    print('=== Build + train model A 10 iters ===')
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

    # Same input
    torch.manual_seed(123)
    frames_ev = torch.randn(B, T, 3, H, W, device=device)

    # Pick FRAME 0 only — manually walk the forward
    print('\n=== Frame-0 manual walk ===')
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            # 1. Backbone
            pix_A = model_A.encode_frame(frames_ev[:, 0])
            pix_B = model_B.encode_frame(frames_ev[:, 0])
        print(f'  backbone pix_feat: max_diff = {_maxabs(pix_A, pix_B):.6e}')

        tokens_A = pix_A.flatten(2).transpose(1, 2)
        tokens_B = pix_B.flatten(2).transpose(1, 2)
        print(f'  tokens (flat+transpose): max_diff = {_maxabs(tokens_A, tokens_B):.6e}')

        lat_A = model_A.perceiver.init_memory(B, device=device, dtype=frames_ev.dtype)
        lat_B = model_B.perceiver.init_memory(B, device=device, dtype=frames_ev.dtype)
        print(f'  init_memory:       max_diff = {_maxabs(lat_A, lat_B):.6e}')

        # add temporal embed for frame 0
        tpe_A = model_A.perceiver.temporal_embed[0]
        tpe_B = model_B.perceiver.temporal_embed[0]
        print(f'  temporal_embed[0]: max_diff = {_maxabs(tpe_A, tpe_B):.6e}')

        latents_A = lat_A + tpe_A
        latents_B = lat_B + tpe_B
        print(f'  latents=init+tpe:  max_diff = {_maxabs(latents_A, latents_B):.6e}')

        # Block 0
        blk_A = model_A.perceiver.blocks[0]
        blk_B = model_B.perceiver.blocks[0]

        with torch.amp.autocast('cuda', dtype=amp_dtype):
            q_A = blk_A.norm_q1(latents_A)
            q_B = blk_B.norm_q1(latents_B)
            print(f'  norm_q1(latents): max_diff = {_maxabs(q_A, q_B):.6e}')
            print(f'    A dtype={q_A.dtype} shape={tuple(q_A.shape)}')
            print(f'    B dtype={q_B.dtype} shape={tuple(q_B.shape)}')
            # Direct LayerNorm call via functional, side-by-side
            import torch.nn.functional as F
            ln_A_w = blk_A.norm_q1.weight
            ln_A_b = blk_A.norm_q1.bias
            ln_B_w = blk_B.norm_q1.weight
            ln_B_b = blk_B.norm_q1.bias
            print(f'    weight identical: {torch.equal(ln_A_w, ln_B_w)}  bias identical: {torch.equal(ln_A_b, ln_B_b)}')

            # Cross-check: run norm_q1_A on latents_B and vice versa
            q_AB = blk_A.norm_q1(latents_B)
            q_BA = blk_B.norm_q1(latents_A)
            print(f'    norm_q1_A(latents_B) vs norm_q1_A(latents_A): {_maxabs(q_AB, q_A):.6e}')
            print(f'    norm_q1_B(latents_A) vs norm_q1_B(latents_B): {_maxabs(q_BA, q_B):.6e}')

            # Inputs A and B byte-identical?
            print(f'    latents_A vs latents_B byte-identical: {torch.equal(latents_A, latents_B)}')
            # In bf16 too
            la = latents_A.to(torch.bfloat16)
            lb = latents_B.to(torch.bfloat16)
            print(f'    latents (bf16) byte-identical:        {torch.equal(la, lb)}')

            # And the dtype of latents inside autocast region
            print(f'    latents_A dtype (entering norm_q1): {latents_A.dtype}')

            # Manual layer_norm via functional
            mq_A = F.layer_norm(latents_A, ln_A_w.shape, ln_A_w, ln_A_b, eps=blk_A.norm_q1.eps)
            mq_B = F.layer_norm(latents_B, ln_B_w.shape, ln_B_w, ln_B_b, eps=blk_B.norm_q1.eps)
            print(f'    manual F.layer_norm:               max_diff = {_maxabs(mq_A, mq_B):.6e}')


if __name__ == '__main__':
    main()
