"""EMA-of-weights for the TemporalPerceiver only.

Scoped to Perceiver (~5M params) — not the frozen 388M backbone — so the
EMA shadow stays cheap to keep on GPU and small in checkpoints. Used for
val-time smoothing, not as a teacher (teacher is the frozen SAM3 ViT-H).
"""

from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn


class ModelEma:
    """Exponential moving average of parameters + buffers.

    Standard formulation: shadow = decay * shadow + (1 - decay) * model.
    Operates in-place on the shadow module, in float32 for stability under
    bf16 training.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, device=None):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        # promote shadow to fp32 (stable EMA under bf16 training)
        self.module.float()
        if device is not None:
            self.module.to(device)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if not torch.is_floating_point(v):
                v.copy_(msd[k])  # ints/bools (e.g. num_batches_tracked) just track
                continue
            new = msd[k].detach().to(dtype=v.dtype, device=v.device)
            v.mul_(d).add_(new, alpha=1.0 - d)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {'decay': self.decay, 'shadow': self.module.state_dict()}

    def load_state_dict(self, sd: Dict):
        self.decay = float(sd.get('decay', self.decay))
        self.module.load_state_dict(sd['shadow'])

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        """Overwrite `model` parameters/buffers with EMA shadow."""
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if k in msd:
                msd[k].copy_(v.to(dtype=msd[k].dtype, device=msd[k].device))
