"""Walk frame 0 step-by-step through every operation in the perceiver.

Find the FIRST operation that produces different output between model_A
(trained) and model_B (fresh + load_state_dict from A).
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

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


def _diff(a, b, name):
    if isinstance(a, tuple):
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f'{name}[{i}]')
        return
    if not torch.is_tensor(a):
        print(f'  {name:50}  non-tensor')
        return
    d = float((a.float() - b.float()).abs().max().item())
    eq = torch.equal(a, b) if a.dtype == b.dtype else False
    flag = '✓' if eq else '✗'
    print(f'  {name:50}  max_diff = {d:.6e}  equal={flag}  dtype={a.dtype} shape={tuple(a.shape)}')


def main():
    config = get_config(_Args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)

    print('=== Build + train model A ===')
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

    print('\n=== Walking perceiver frame 0 step-by-step ===')
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            # backbone
            pix_A = model_A.encode_frame(frames_ev[:, 0])
            pix_B = model_B.encode_frame(frames_ev[:, 0])
            _diff(pix_A, pix_B, 'backbone.pix_feat')

            tokens_A = pix_A.flatten(2).transpose(1, 2)
            tokens_B = pix_B.flatten(2).transpose(1, 2)
            _diff(tokens_A, tokens_B, 'tokens')

            lat_A = model_A.perceiver.init_memory(B, device=device, dtype=frames_ev.dtype)
            lat_B = model_B.perceiver.init_memory(B, device=device, dtype=frames_ev.dtype)
            tpe_A = model_A.perceiver.temporal_embed[0]
            tpe_B = model_B.perceiver.temporal_embed[0]
            latents_A = lat_A + tpe_A
            latents_B = lat_B + tpe_B
            _diff(latents_A, latents_B, 'init latents + tpe')

            blk_A = model_A.perceiver.blocks[0]
            blk_B = model_B.perceiver.blocks[0]

            # 1. norm_q1
            q_A = blk_A.norm_q1(latents_A)
            q_B = blk_B.norm_q1(latents_B)
            _diff(q_A, q_B, 'blk0.norm_q1(latents)')

            # 2. norm_kv1
            kv_A = blk_A.norm_kv1(tokens_A)
            kv_B = blk_B.norm_kv1(tokens_B)
            _diff(kv_A, kv_B, 'blk0.norm_kv1(tokens)')

            # 3. cross_attn
            x_A, _ = blk_A.cross_attn(q_A, kv_A, kv_A, need_weights=False)
            x_B, _ = blk_B.cross_attn(q_B, kv_B, kv_B, need_weights=False)
            _diff(x_A, x_B, 'blk0.cross_attn(q,kv,kv)')

            # 4. residual
            latents1_A = latents_A + x_A
            latents1_B = latents_B + x_B
            _diff(latents1_A, latents1_B, 'latents += cross_attn_out')

            # 5. self_attn path
            s_A = blk_A.norm2(latents1_A)
            s_B = blk_B.norm2(latents1_B)
            _diff(s_A, s_B, 'blk0.norm2')
            sa_A, _ = blk_A.self_attn(s_A, s_A, s_A, need_weights=False)
            sa_B, _ = blk_B.self_attn(s_B, s_B, s_B, need_weights=False)
            _diff(sa_A, sa_B, 'blk0.self_attn')

            latents2_A = latents1_A + sa_A
            latents2_B = latents1_B + sa_B
            _diff(latents2_A, latents2_B, 'latents += self_attn_out')

            # 6. mlp path
            m_A = blk_A.norm3(latents2_A)
            m_B = blk_B.norm3(latents2_B)
            _diff(m_A, m_B, 'blk0.norm3')
            mlp_A = blk_A.mlp(m_A)
            mlp_B = blk_B.mlp(m_B)
            _diff(mlp_A, mlp_B, 'blk0.mlp(m)')

            latents3_A = latents2_A + mlp_A
            latents3_B = latents2_B + mlp_B
            _diff(latents3_A, latents3_B, 'final latents after block 0')


if __name__ == '__main__':
    main()
