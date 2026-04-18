"""SA-V video dataset for Stage 2 temporal memory distillation.

Videos are streamed from TAR archives without extraction. Each worker opens
its own tar handles via `worker_init_fn` to avoid lock contention.

Decoding: prefer `decord.VideoReader` (random-access, decode-only-needed-frames)
falling back to `cv2.VideoCapture` (decodes whole video). Both run against a
`/dev/shm` tempfile (RAM-backed). torchvision.io video reader is broken in
torchvision 0.25.0.

Annotations
-----------
Each `<vid>.mp4` is paired with `<vid>_manual.json` (preferred) and/or
`<vid>_auto.json`. The JSON has:
  - masklet_num: int — number of objects
  - masklet: list[T_annot][num_obj] of {size:[H,W], counts: RLE}
  - masklet_first_appeared_frame: list[num_obj]
  - SA-V annotates every 4th video frame (6Hz on 24Hz video).

Sampling restricts MAX_FRAMES picks to annotated frame indices so the
teacher's memory bank can be conditioned on real GT masks (Stage 2 #7).
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import tarfile
import tempfile
import uuid
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

from ..config import MAX_FRAMES

try:
    from decord import VideoReader, cpu as _decord_cpu
    _HAS_DECORD = True
except ImportError:
    _HAS_DECORD = False


def _decode_rle(rle: Dict) -> np.ndarray:
    """COCO RLE → uint8 [H, W] binary mask."""
    counts = rle['counts']
    if isinstance(counts, str):
        counts = counts.encode('utf-8')
    return mask_utils.decode({'size': rle['size'], 'counts': counts})


class SAVVideoDataset(Dataset):
    """Streams SA-V videos + masklets from TAR archives, returns fixed-shape clips.

    Output per sample:
      - frames:         FloatTensor[MAX_FRAMES, 3, H, W]  (normalized, padded)
      - attention_mask: BoolTensor[MAX_FRAMES]            (True for real frames)
      - masks:          FloatTensor[MAX_FRAMES, H, W]     ({0,1} GT object mask)
      - mask_valid:     BoolTensor[MAX_FRAMES]            (True iff mask is non-empty)
      - video_id:       str
    """

    INDEX_VERSION = 2  # bump when index schema changes

    def __init__(
        self,
        tar_dir: str,
        max_frames: int = MAX_FRAMES,
        img_size: int = 1024,
        pixel_mean: Tuple[float, float, float] = (123.675, 116.28, 103.53),
        pixel_std: Tuple[float, float, float] = (58.395, 57.12, 57.375),
        index_cache: Optional[str] = None,
        num_samples: int = -1,
        split: str = 'train',
        val_fraction: float = 0.02,
        prefer_manual: bool = True,
    ):
        assert max_frames == MAX_FRAMES, \
            f"max_frames must equal MAX_FRAMES ({MAX_FRAMES}) for ONNX static shape"

        self.tar_dir = tar_dir
        self.max_frames = max_frames
        self.img_size = img_size
        self.mean = np.array(pixel_mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(pixel_std, dtype=np.float32).reshape(1, 1, 3)
        self.split = split
        self.prefer_manual = prefer_manual

        # bump cache name with version so v1 caches don't collide with v2 schema
        if index_cache:
            base, ext = os.path.splitext(index_cache)
            index_cache = f"{base}_v{self.INDEX_VERSION}{ext}"

        # Each entry: (tar_path, mp4_member, json_member)
        full_index: List[Tuple[str, str, str]] = self._build_or_load_index(index_cache)

        # Deterministic train/val split by hashing video stem. Stable across runs
        # regardless of tar listing order or future index rebuilds.
        import hashlib
        bucket_mod = 10000
        val_cut = int(val_fraction * bucket_mod)
        def _bucket(entry):
            stem = os.path.basename(entry[1])
            h = int(hashlib.sha1(stem.encode()).hexdigest()[:8], 16)
            return h % bucket_mod
        if split == 'train':
            self.index = [e for e in full_index if _bucket(e) >= val_cut]
        elif split == 'val':
            self.index = [e for e in full_index if _bucket(e) < val_cut]
        else:
            raise ValueError(f"Unknown split: {split}")

        if num_samples > 0:
            self.index = self.index[:num_samples]

        self._tar_handles: Dict[str, tarfile.TarFile] = {}

    # ---- index construction ------------------------------------------------

    def _build_or_load_index(self, cache_path: Optional[str]) -> List[Tuple[str, str, str]]:
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                idx = pickle.load(f)
            print(f"[SAVVideoDataset] loaded index cache {cache_path}: {len(idx)} videos")
            return idx

        tar_paths = sorted(glob.glob(os.path.join(self.tar_dir, '*.tar')))
        if not tar_paths:
            raise FileNotFoundError(f"No *.tar files in {self.tar_dir}")

        idx: List[Tuple[str, str, str]] = []
        for tp in tar_paths:
            try:
                with tarfile.open(tp, 'r') as tf:
                    members = tf.getnames()
            except (tarfile.TarError, OSError) as e:
                print(f"[SAVVideoDataset] skip {tp}: {e}")
                continue

            mp4s, manuals, autos = {}, {}, {}
            for m in members:
                base = os.path.basename(m)
                if base.endswith('.mp4'):
                    mp4s[base[:-4]] = m
                elif base.endswith('_manual.json'):
                    manuals[base[: -len('_manual.json')]] = m
                elif base.endswith('_auto.json'):
                    autos[base[: -len('_auto.json')]] = m

            for stem, mp4_member in mp4s.items():
                json_member = (
                    manuals.get(stem) if self.prefer_manual else None
                ) or autos.get(stem) or manuals.get(stem)
                if json_member is None:
                    continue
                idx.append((tp, mp4_member, json_member))

        print(f"[SAVVideoDataset] built index: {len(idx)} videos across {len(tar_paths)} tars")

        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(idx, f)
            print(f"[SAVVideoDataset] cached index to {cache_path}")

        return idx

    # ---- worker tar handle management --------------------------------------

    def _get_tar(self, tar_path: str) -> tarfile.TarFile:
        tf = self._tar_handles.get(tar_path)
        if tf is None:
            tf = tarfile.open(tar_path, 'r')
            self._tar_handles[tar_path] = tf
        return tf

    def _read_member(self, tar_path: str, member_name: str) -> bytes:
        tf = self._get_tar(tar_path)
        member = tf.getmember(member_name)
        f = tf.extractfile(member)
        if f is None:
            raise RuntimeError(f"extractfile returned None for {member_name}")
        return f.read()

    def close_handles(self):
        for tf in self._tar_handles.values():
            try:
                tf.close()
            except Exception:
                pass
        self._tar_handles.clear()

    def __del__(self):
        self.close_handles()

    # ---- video decode ------------------------------------------------------

    def _decode_video_len(self, mp4_bytes: bytes) -> Tuple[int, object]:
        """Open video, return (T_video, decoder_handle). Caller must call _decode_close."""
        shm = '/dev/shm' if os.path.isdir('/dev/shm') else tempfile.gettempdir()
        tmp = os.path.join(shm, f'sav_{uuid.uuid4().hex}.mp4')
        with open(tmp, 'wb') as f:
            f.write(mp4_bytes)

        if _HAS_DECORD:
            try:
                vr = VideoReader(tmp, ctx=_decord_cpu(0))
                return len(vr), ('decord', vr, tmp)
            except Exception:
                pass  # fall through to cv2

        cap = cv2.VideoCapture(tmp)
        T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return T, ('cv2', cap, tmp)

    def _decode_frames(self, handle, indices: np.ndarray) -> np.ndarray:
        """Decode only `indices` frames. Returns ndarray[k, H0, W0, 3] uint8 RGB."""
        kind, dec, _ = handle
        if kind == 'decord':
            arr = dec.get_batch(list(indices)).asnumpy()  # already RGB
            return arr
        # cv2 fallback: must seek + read sequentially
        cap = dec
        out = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame_bgr = cap.read()
            if not ok:
                raise RuntimeError(f"cv2 read failed at frame {i}")
            out.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        return np.stack(out, axis=0)

    def _decode_close(self, handle):
        kind, dec, tmp = handle
        if kind == 'cv2':
            try:
                dec.release()
            except Exception:
                pass
        # decord VideoReader has no explicit close; gc handles it
        try:
            os.remove(tmp)
        except OSError:
            pass

    # ---- preprocessing -----------------------------------------------------

    def _resize_pad_image(self, rgb: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """Long-side resize to img_size, pad bottom-right, normalize.
        Returns (CHW float32, scale, new_h, new_w) — scale/new_h/w reused for masks.
        """
        h, w = rgb.shape[:2]
        scale = self.img_size / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = self.img_size - new_h
        pad_w = self.img_size - new_w
        if pad_h or pad_w:
            resized = cv2.copyMakeBorder(
                resized, 0, pad_h, 0, pad_w,
                borderType=cv2.BORDER_CONSTANT, value=0,
            )

        x = resized.astype(np.float32)
        x = (x - self.mean) / self.std
        return np.transpose(x, (2, 0, 1)), scale, new_h, new_w

    def _resize_pad_mask(self, m: np.ndarray, scale: float, new_h: int, new_w: int) -> np.ndarray:
        """Apply same resize+pad as the paired image, nearest-neighbor on the mask."""
        resized = cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        pad_h = self.img_size - new_h
        pad_w = self.img_size - new_w
        if pad_h or pad_w:
            resized = cv2.copyMakeBorder(
                resized, 0, pad_h, 0, pad_w,
                borderType=cv2.BORDER_CONSTANT, value=0,
            )
        return resized.astype(np.float32)

    # ---- frame/mask sampling ----------------------------------------------

    def _sample_indices(self, annot_video_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pick MAX_FRAMES (or fewer) annotated frame indices, evenly spaced.
        Returns (video_frame_picks, masklet_local_picks).
        """
        n = len(annot_video_idx)
        k = min(n, self.max_frames)
        if k == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        if k == 1:
            local = np.array([0], dtype=np.int64)
        else:
            local = np.linspace(0, n - 1, num=k).round().astype(np.int64)
        return annot_video_idx[local], local

    def _empty_sample(self, video_id: str) -> Dict:
        H = self.img_size
        return {
            'frames': torch.zeros(self.max_frames, 3, H, H, dtype=torch.float32),
            'attention_mask': torch.zeros(self.max_frames, dtype=torch.bool),
            'masks': torch.zeros(self.max_frames, H, H, dtype=torch.float32),
            'mask_valid': torch.zeros(self.max_frames, dtype=torch.bool),
            'video_id': video_id,
        }

    # ---- Dataset API -------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict:
        tar_path, mp4_member, json_member = self.index[idx]

        handle = None
        try:
            mp4_bytes = self._read_member(tar_path, mp4_member)
            json_bytes = self._read_member(tar_path, json_member)
            T_video, handle = self._decode_video_len(mp4_bytes)
            meta = json.loads(json_bytes)
            masklet = meta['masklet']                                 # [T_a][num_obj]
            num_obj = int(meta.get('masklet_num', 0))
        except Exception as e:
            print(f"[SAVVideoDataset] idx={idx} {mp4_member} failed: {e}")
            if handle is not None:
                self._decode_close(handle)
            return self._empty_sample(mp4_member)

        T_annot = len(masklet)
        if num_obj == 0 or T_annot == 0:
            self._decode_close(handle)
            return self._empty_sample(mp4_member)

        # SA-V annotates every 4th video frame: annotated[i] ↔ video frame 4*i
        annot_video_idx = (np.arange(T_annot) * 4).astype(np.int64)
        annot_video_idx = annot_video_idx[annot_video_idx < T_video]
        if len(annot_video_idx) == 0:
            self._decode_close(handle)
            return self._empty_sample(mp4_member)

        video_picks, local_picks = self._sample_indices(annot_video_idx)
        if len(video_picks) == 0:
            self._decode_close(handle)
            return self._empty_sample(mp4_member)

        # Decode only the picked frames (decord random-access; cv2 seek+read).
        try:
            picked_frames = self._decode_frames(handle, video_picks)  # [k, H0, W0, 3]
        except Exception as e:
            print(f"[SAVVideoDataset] idx={idx} {mp4_member} decode failed: {e}")
            self._decode_close(handle)
            return self._empty_sample(mp4_member)
        finally:
            self._decode_close(handle)

        # Pick the first object (deterministic for now; multi-object later if needed).
        obj_idx = 0

        H = self.img_size
        frames_out = torch.zeros(self.max_frames, 3, H, H, dtype=torch.float32)
        masks_out = torch.zeros(self.max_frames, H, H, dtype=torch.float32)
        attn_out = torch.zeros(self.max_frames, dtype=torch.bool)
        valid_out = torch.zeros(self.max_frames, dtype=torch.bool)

        for slot, (rgb, li) in enumerate(zip(picked_frames, local_picks)):
            img_chw, scale, nh, nw = self._resize_pad_image(rgb)
            frames_out[slot] = torch.from_numpy(img_chw)
            attn_out[slot] = True

            try:
                m_raw = _decode_rle(masklet[int(li)][obj_idx])        # [H0, W0] uint8
            except Exception:
                continue
            m_proc = self._resize_pad_mask(m_raw, scale, nh, nw)
            masks_out[slot] = torch.from_numpy(m_proc)
            valid_out[slot] = bool(m_raw.any())

        return {
            'frames': frames_out,
            'attention_mask': attn_out,
            'masks': masks_out,
            'mask_valid': valid_out,
            'video_id': mp4_member,
        }


def sav_worker_init_fn(worker_id: int):
    """Each worker opens its own tar handles (no parent inheritance)."""
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    ds = info.dataset
    if hasattr(ds, '_tar_handles'):
        ds._tar_handles = {}


def sav_collate_fn(batch: List[Dict]) -> Dict:
    frames = torch.stack([b['frames'] for b in batch], dim=0)         # [B, 8, 3, H, W]
    attn = torch.stack([b['attention_mask'] for b in batch], dim=0)   # [B, 8]
    masks = torch.stack([b['masks'] for b in batch], dim=0)           # [B, 8, H, W]
    valid = torch.stack([b['mask_valid'] for b in batch], dim=0)      # [B, 8]
    video_ids = [b['video_id'] for b in batch]
    return {
        'frames': frames,
        'attention_mask': attn,
        'masks': masks,
        'mask_valid': valid,
        'video_ids': video_ids,
    }
