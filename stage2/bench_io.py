"""Data-only throughput bench: measure pure dataloader iters/sec."""
import argparse
import time

import torch
from torch.utils.data import DataLoader

from stage2.data.sav_dataset import SAVVideoDataset, sav_collate_fn, sav_worker_init_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--prefetch', type=int, default=4)
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--img-size', type=int, default=1008)
    args = ap.parse_args()

    ds = SAVVideoDataset(
        '/mnt/hdd/datasets/SA-V/',
        index_cache='data/sav_index.pkl',
        img_size=args.img_size,
    )
    print(f'dataset: {len(ds)} videos')

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=args.prefetch,
        collate_fn=sav_collate_fn,
        worker_init_fn=sav_worker_init_fn,
        shuffle=True,
        drop_last=True,
    )

    print(f'workers={args.workers} bs={args.batch_size} prefetch={args.prefetch}')
    it = iter(loader)
    for _ in range(args.warmup):
        next(it)
    print(f'warmup {args.warmup} done; timing {args.iters}...')

    t0 = time.perf_counter()
    n = 0
    for _ in range(args.iters):
        b = next(it)
        n += b['frames'].shape[0]
    dt = time.perf_counter() - t0

    iters_s = args.iters / dt
    samples_s = n / dt
    print(f'iters/s={iters_s:.3f}  samples/s={samples_s:.3f}  '
          f'sec/iter={dt/args.iters:.3f}  total={dt:.1f}s')


if __name__ == '__main__':
    main()
