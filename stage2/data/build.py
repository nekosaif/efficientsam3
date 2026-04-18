"""DataLoader builder for Stage 2 (SA-V video)."""

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .sav_dataset import (
    SAVVideoDataset,
    sav_collate_fn,
    sav_worker_init_fn,
)


def build_dataset(config, split: str = 'train'):
    if config.DATA.DATASET == 'sav':
        ds = SAVVideoDataset(
            tar_dir=config.DATA.SAV_TAR_DIR or config.DATA.DATA_PATH,
            max_frames=config.DATA.MAX_FRAMES,
            img_size=config.DATA.IMG_SIZE,
            pixel_mean=tuple(config.DATA.MEAN),
            pixel_std=tuple(config.DATA.STD),
            index_cache=config.DATA.SAV_INDEX_CACHE,
            num_samples=(100 if config.DATA.DEBUG else config.DATA.NUM_SAMPLES),
            split=split,
        )
        return ds
    raise NotImplementedError(f"Unknown DATASET: {config.DATA.DATASET}")


def build_loader(config, rank: int, world_size: int, build_val: bool = False):
    dataset_train = build_dataset(config, split='train')

    sampler_train = DistributedSampler(
        dataset_train,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    loader_train = DataLoader(
        dataset_train,
        batch_size=config.DATA.BATCH_SIZE,
        sampler=sampler_train,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        persistent_workers=(config.DATA.PERSISTENT_WORKERS and config.DATA.NUM_WORKERS > 0),
        prefetch_factor=(config.DATA.PREFETCH_FACTOR if config.DATA.NUM_WORKERS > 0 else None),
        collate_fn=sav_collate_fn,
        worker_init_fn=sav_worker_init_fn,
        drop_last=True,
    )

    dataset_val, loader_val = None, None
    if build_val:
        dataset_val = build_dataset(config, split='val')
        sampler_val = DistributedSampler(
            dataset_val,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        loader_val = DataLoader(
            dataset_val,
            batch_size=config.DATA.BATCH_SIZE,
            sampler=sampler_val,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            persistent_workers=(config.DATA.PERSISTENT_WORKERS and config.DATA.NUM_WORKERS > 0),
            prefetch_factor=(config.DATA.PREFETCH_FACTOR if config.DATA.NUM_WORKERS > 0 else None),
            collate_fn=sav_collate_fn,
            worker_init_fn=sav_worker_init_fn,
            drop_last=False,
        )

    return dataset_train, dataset_val, loader_train, loader_val
