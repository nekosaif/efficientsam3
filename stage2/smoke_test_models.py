"""Step 2 smoke test: construct student + teacher, forward one fake batch.

Run:
    python3 stage2/smoke_test_models.py

Validates:
  - Student loads RepViT M0.9 from checkpoints/efficient_sam3_repvit_s.pt
  - Teacher loads SAM3 ViT-H from /mnt/hdd/checkpoints/sam3/sam3.pt
  - Perceiver is wired in on the student
  - Forward over [1, 8, 3, 1024, 1024] produces matching shapes on both
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2.config import get_config, MAX_FRAMES
from stage2.models import build_student_model, build_teacher_model


def main():
    class _Args:
        cfg = None
        opts = None
        batch_size = 1
        data_path = None
        student_ckpt = 'checkpoints/efficient_sam3_repvit_s.pt'
        teacher_ckpt = '/mnt/hdd/checkpoints/sam3/sam3.pt'
        resume = None
        output = 'output'
        tag = 'smoke'
        accumulation_steps = None
        use_checkpoint = False
        disable_amp = False
        only_cpu = False
        eval = False
        throughput = False
        local_rank = 0

    config = get_config(_Args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}  MAX_FRAMES={MAX_FRAMES}")

    # --- student ---
    t0 = time.time()
    student = build_student_model(config).to(device)
    n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in student.parameters())
    print(f"[student] built in {time.time()-t0:.1f}s  trainable={n_trainable/1e6:.2f}M "
          f"total={n_total/1e6:.2f}M")

    # --- teacher ---
    t0 = time.time()
    teacher = build_teacher_model(config).to(device)
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"[teacher] built in {time.time()-t0:.1f}s  total={n_teacher/1e6:.2f}M (frozen)")

    # --- forward ---
    H = W = config.DATA.IMG_SIZE  # 1008 — teacher RoPE constraint, student is conv
    frames = torch.randn(1, MAX_FRAMES, 3, H, W, device=device)
    mask = torch.ones(1, MAX_FRAMES, dtype=torch.bool, device=device)

    autocast_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    with torch.amp.autocast('cuda', dtype=autocast_dtype, enabled=(device.type == 'cuda')):
        t0 = time.time()
        s_out = student(frames, mask)
        print(f"[student] forward: {time.time()-t0:.2f}s  out={tuple(s_out.shape)}")

        t0 = time.time()
        with torch.no_grad():
            t_out = teacher(frames, mask)
        print(f"[teacher] forward: {time.time()-t0:.2f}s  out={tuple(t_out.shape)}")

    assert s_out.shape == t_out.shape, f"shape mismatch: student {s_out.shape} vs teacher {t_out.shape}"
    assert s_out.shape[2] == config.DISTILL.FEATURE_DIM, \
        f"channels {s_out.shape[2]} != FEATURE_DIM {config.DISTILL.FEATURE_DIM}"
    print(f"[OK] shapes match. feat_dim={s_out.shape[2]} spatial={tuple(s_out.shape[-2:])}")


if __name__ == '__main__':
    main()
