"""Stage 2 optimizer + LR scheduler builders.

Trainable params live only in `student.perceiver` (backbone + projections are
frozen). Cosine LR with linear warmup, computed in epochs but stepped per
iteration for smooth curves.
"""

from __future__ import annotations

import math
from typing import Iterable, List

import torch
from torch import optim


def _split_param_groups(named_params: Iterable, weight_decay: float):
    decay, no_decay = [], []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        # 1-D params (norms, biases) and explicit bias suffix → no weight decay
        if p.ndim == 1 or name.endswith('.bias'):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


def build_optimizer(config, model: torch.nn.Module) -> optim.Optimizer:
    groups = _split_param_groups(model.named_parameters(), config.TRAIN.WEIGHT_DECAY)
    name = config.TRAIN.OPTIMIZER.NAME.lower()
    if name == 'adamw':
        return optim.AdamW(
            groups,
            lr=config.TRAIN.BASE_LR,
            eps=config.TRAIN.OPTIMIZER.EPS,
            betas=tuple(config.TRAIN.OPTIMIZER.BETAS),
        )
    if name == 'sgd':
        return optim.SGD(
            groups,
            lr=config.TRAIN.BASE_LR,
            momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
            nesterov=True,
        )
    raise ValueError(f"Unknown optimizer: {name}")


class CosineWarmupScheduler:
    """Per-iteration cosine LR schedule with linear warmup."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_iters: int,
        total_iters: int,
        base_lr: float,
        min_lr: float,
        warmup_lr: float,
    ):
        self.optimizer = optimizer
        self.warmup_iters = max(1, warmup_iters)
        self.total_iters = max(self.warmup_iters + 1, total_iters)
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_lr = warmup_lr
        self._iter = 0
        self.step(0)

    def step(self, iter_idx: int):
        self._iter = iter_idx
        if iter_idx < self.warmup_iters:
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * iter_idx / self.warmup_iters
        else:
            t = (iter_idx - self.warmup_iters) / max(1, (self.total_iters - self.warmup_iters))
            t = min(1.0, max(0.0, t))
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * t))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']
