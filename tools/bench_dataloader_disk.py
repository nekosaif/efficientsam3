#!/usr/bin/env python3
"""Micro-benchmark: SA-V dataloader throughput from a given tar dir.

Builds SAVVideoDataset over whatever *.tar files are in --tar-dir, then times
how long it takes to pull N batches through the SAME pipeline training uses
(random-access tar member read + decord decode + resize/pad). No model — this
isolates the DATA path, which is the training bottleneck.

Run it once pointed at the HDD tar dir and once at an NVMe copy to compare.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sam3"))

import torch
from torch.utils.data import DataLoader
from stage2.data.sav_dataset import SAVVideoDataset, sav_collate_fn, sav_worker_init_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar-dir", required=True)
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3, help="batches to skip before timing")
    args = ap.parse_args()

    ds = SAVVideoDataset(
        tar_dir=args.tar_dir,
        img_size=1008,
        index_cache=None,          # build index over just this dir's tars
        split="train",
        select_first_object=False,  # match M1 (arbitrary object + prompt)
    )
    print(f"[bench] {args.tar_dir}: {len(ds)} videos")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=sav_collate_fn, worker_init_fn=sav_worker_init_fn,
        persistent_workers=True, prefetch_factor=2, drop_last=True,
    )

    t0 = None
    n = 0
    seen = 0
    for i, batch in enumerate(loader):
        # touch the tensors so lazy work is realized
        _ = batch["frames"].shape, batch["prompt_mask"].shape
        if i == args.warmup:
            t0 = time.time()
        if i >= args.warmup:
            n += 1
            seen += batch["frames"].shape[0]
        if i >= args.warmup + args.batches:
            break
    dt = time.time() - t0
    ips = seen / dt
    print(f"[bench] {args.tar_dir}")
    print(f"  batches timed={n}  samples={seen}  wall={dt:.1f}s")
    print(f"  throughput = {ips:.3f} samples/s   ({dt/max(1,seen):.2f} s/sample)")


if __name__ == "__main__":
    main()
