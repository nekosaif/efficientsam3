#!/usr/bin/env python3
"""Profile a Milestone-1 training iter: data / student fwd / teacher fwd / mask-decode / backward.

Confirms whether the (frozen) teacher forward is the per-iter bottleneck — i.e.
whether caching teacher targets is worth it — and by how much.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sam3"))
import torch
from stage2.config import get_config
from stage2.models import build_student_model, build_teacher_model
from stage2.loss import build_distill_loss, build_mask_loss


class _Args:
    cfg = "stage2/configs/sav_repvit_m0_9_prompt.yaml"
    student_ckpt = "checkpoints/efficient_sam3_repvit_s.pt"
    teacher_ckpt = "/mnt/exoshdd/checkpoints/sam3/sam3.pt"
    opts = None
    batch_size = 2
    data_path = resume = output = tag = None
    accumulation_steps = None
    use_checkpoint = disable_amp = only_cpu = eval = throughput = False
    local_rank = 0


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    cfg = get_config(_Args)
    dev = torch.device("cuda")
    student = build_student_model(cfg).to(dev); student.train()
    teacher = build_teacher_model(cfg).to(dev)
    distill = build_distill_loss(cfg); mloss = build_mask_loss(cfg)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-4)

    B, T, H = 2, 8, 1008
    # fixed fake batch (we're timing compute, not data here)
    frames = torch.randn(B, T, 3, H, H, device=dev)
    attn = torch.ones(B, T, dtype=torch.bool, device=dev)
    gt = torch.zeros(B, T, H, H, device=dev)
    for t in range(T):
        a = 100 + t * 20; gt[:, t, a:a+250, a:a+250] = 1.0
    valid = torch.ones(B, T, dtype=torch.bool, device=dev)
    prompt = {"mask": gt[:, 0:1]}

    times = {"student_fwd": [], "teacher_fwd": [], "mask_decode": [], "loss_bwd": []}
    N = 8
    for it in range(N + 2):  # 2 warmup
        rec = (it >= 2)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            sync(); t0 = time.time()
            s_out, hires = student(frames, attn, prompt=prompt, return_high_res=True)
            sync(); t1 = time.time()
            with torch.no_grad():
                t_out = teacher(frames, attn, gt, valid, frame0_only=True)
            sync(); t2 = time.time()
            dl = distill(s_out, t_out, attn)
            s_logits = student.decode_masks(s_out, hires)
            sync(); t3 = time.time()
            ml = mloss(s_logits, gt, valid)
            loss = dl.total + ml.total
        loss.backward(); opt.step()
        sync(); t4 = time.time()
        if rec:
            times["student_fwd"].append(t1 - t0)
            times["teacher_fwd"].append(t2 - t1)
            times["mask_decode"].append(t3 - t2)
            times["loss_bwd"].append(t4 - t3)

    print(f"\n=== per-iter compute breakdown (batch={B}, avg of {N}) ===")
    tot = 0.0
    for k, v in times.items():
        avg = sum(v) / len(v); tot += avg
        print(f"  {k:14s}: {avg*1000:7.1f} ms")
    print(f"  {'TOTAL':14s}: {tot*1000:7.1f} ms   ({B/tot:.2f} samples/s compute-only)")
    tf = sum(times['teacher_fwd'])/len(times['teacher_fwd'])
    print(f"\n  teacher is {tf/tot*100:.0f}% of compute → caching it would cut iter to ~{(tot-tf)*1000:.0f} ms "
          f"(~{B/(tot-tf):.2f} samples/s, {tot/(tot-tf):.2f}x faster compute)")


if __name__ == "__main__":
    main()
