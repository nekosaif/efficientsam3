"""Lazy, self-populating cache for frozen-teacher distillation targets.

The teacher is frozen and its output `pix_feat_with_mem` [T,256,72,72] is a
deterministic function of (frames, frame-0 mask, selected object). Over a 50-epoch
run we recompute the SAME target 50x — and the teacher forward is ~81% of per-iter
compute (tools/profile_iter.py). This cache computes each target ONCE (epoch 0),
stores it, and replays it on later epochs — skipping the teacher entirely.

Storage: per-sample `.npz`, int8 per-channel symmetric quantization (cos_sim with
the fp32 target ~0.99995; ~1% L2 error — negligible for a distill target whose loss
is cosine-direction + scale-invariant log-ratio norm). ~10 MB/sample, ~490 GB full
train set — fits the NVMe. Key = sha1(video_id + obj_idx).

Correctness/safety:
  - Atomic writes (tmp + os.replace) so an interrupted write never yields a
    half-file that a later epoch would read as valid.
  - On any read error / shape mismatch / cache miss → returns None; caller falls
    back to the live teacher (and may re-cache).
  - Per-SAMPLE (not per-batch): handles drop_last / ragged batches and lets the
    cache fill regardless of batch composition.
"""

from __future__ import annotations

import hashlib
import os
from typing import List, Optional

import numpy as np
import torch


class TeacherTargetCache:
    def __init__(self, root: str, enabled: bool = True, dtype_store: str = "int8"):
        self.root = root
        self.enabled = enabled
        self.dtype_store = dtype_store
        if enabled:
            os.makedirs(root, exist_ok=True)

    # ---- keying -----------------------------------------------------------
    @staticmethod
    def key(video_id: str, obj_idx: int) -> str:
        h = hashlib.sha1(f"{video_id}|{int(obj_idx)}".encode()).hexdigest()
        return h

    def _path(self, key: str) -> str:
        # shard into 256 subdirs to keep directory sizes sane
        return os.path.join(self.root, key[:2], key + ".npz")

    # ---- quantize / dequantize -------------------------------------------
    @staticmethod
    def _quant_int8(t: torch.Tensor):
        """Per-channel symmetric int8. t: [T,C,H,W] fp32/bf16 -> (int8[T,C,H,W], scale[C])."""
        t = t.float()
        amax = t.abs().amax(dim=(0, 2, 3)).clamp_min(1e-8)   # [C]
        scale = (amax / 127.0)
        q = (t / scale.view(1, -1, 1, 1)).round().clamp(-127, 127).to(torch.int8)
        return q, scale

    @staticmethod
    def _dequant_int8(q: np.ndarray, scale: np.ndarray) -> torch.Tensor:
        qt = torch.from_numpy(q).float()
        st = torch.from_numpy(scale).float().view(1, -1, 1, 1)
        return qt * st

    # ---- single-sample IO -------------------------------------------------
    def get(self, video_id: str, obj_idx: int, expected_shape) -> Optional[torch.Tensor]:
        if not self.enabled:
            return None
        p = self._path(self.key(video_id, int(obj_idx)))
        if not os.path.exists(p):
            return None
        try:
            with np.load(p) as z:
                q = z["q"]; scale = z["scale"]
            t = self._dequant_int8(q, scale)         # [T,C,H,W]
            if tuple(t.shape) != tuple(expected_shape):
                return None
            return t
        except Exception:
            return None

    def put(self, video_id: str, obj_idx: int, target: torch.Tensor) -> None:
        """target: [T,C,H,W] for ONE sample."""
        if not self.enabled:
            return
        key = self.key(video_id, int(obj_idx))
        p = self._path(key)
        if os.path.exists(p):
            return
        os.makedirs(os.path.dirname(p), exist_ok=True)
        # np.savez appends ".npz" to a path without that suffix, so write to an
        # explicit tmp filename ending in .npz, then atomically rename.
        tmp = p + f".tmp{os.getpid()}.npz"
        try:
            q, scale = self._quant_int8(target.detach().cpu())
            with open(tmp, "wb") as f:
                np.savez(f, q=q.numpy(), scale=scale.numpy())
            os.replace(tmp, p)
        except Exception:
            # caching is best-effort; never break training on a write error
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass

    # ---- batch helpers ----------------------------------------------------
    def get_batch(self, video_ids: List[str], obj_idxs, per_sample_shape, device):
        """Return (target[B,T,C,H,W], all_hit: bool, miss_idx: List[int]).

        If every sample hits, returns the stacked tensor and all_hit=True.
        Otherwise returns (None, False, [indices that missed]) so the caller can
        run the live teacher (for the whole batch) and then back-fill misses.
        """
        if not self.enabled:
            return None, False, list(range(len(video_ids)))
        outs = []
        miss = []
        for i, (vid, oi) in enumerate(zip(video_ids, _to_list(obj_idxs))):
            t = self.get(vid, oi, per_sample_shape)
            if t is None:
                miss.append(i)
                outs.append(None)
            else:
                outs.append(t)
        if miss:
            return None, False, miss
        stacked = torch.stack(outs, dim=0).to(device)
        return stacked, True, []

    def put_batch(self, video_ids: List[str], obj_idxs, target: torch.Tensor) -> None:
        """target: [B,T,C,H,W]. Skips samples with obj_idx < 0 (empty samples)."""
        if not self.enabled:
            return
        for i, (vid, oi) in enumerate(zip(video_ids, _to_list(obj_idxs))):
            if int(oi) < 0:
                continue
            self.put(vid, oi, target[i])


def _to_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return list(x)
