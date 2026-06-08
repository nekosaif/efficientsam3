#!/usr/bin/env python3
"""Pre-decode SA-V videos into small per-video bundles on fast storage (NVMe).

The training bottleneck is the per-sample HDD random-seek + mp4 container open +
video decode of 8 frames. The 8 sampled frames are DETERMINISTIC per video
(np.linspace over annotated frames — depends only on the video, not the object),
so we can decode them ONCE and replay forever.

Each bundle stores exactly what __getitem__ needs to reproduce its output
byte-identically, WITHOUT re-touching the tar/video:
  - frames : the picked raw frames, JPEG-encoded at ORIGINAL resolution (resize
             happens at load time, same as the live path) — list of 8 jpeg bytes
  - masklet: the RLEs for ALL objects at the picked annotated frames (object
             selection is per-idx at load time, so we must keep every object)
  - local_picks, num_obj, frame H0/W0 — metadata to drive the shared post-decode

Bundle key = the mp4 member name (video_id), so it's independent of host path.
Format: one .npz per video under <out>/<key[:2]>/<sha1(key)>.npz (sharded dirs).
Resumable (skips existing), parallel (process pool).

Usage:
  python tools/predecode_sav.py --out /home/saif/sav_predecoded --workers 12
  python tools/predecode_sav.py --out ... --limit 2000   # a subset for benchmarking
"""
from __future__ import annotations
import argparse, hashlib, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sam3"))


def bundle_path(out_root: str, video_id: str) -> str:
    h = hashlib.sha1(video_id.encode()).hexdigest()
    return os.path.join(out_root, h[:2], h + ".npz")


def _encode_jpeg(rgb: np.ndarray, q: int) -> bytes:
    import cv2
    ok, enc = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])  # RGB->BGR
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return enc.tobytes()


def _worker(args):
    import json, cv2, tarfile  # noqa
    idx, tar_path, mp4_member, json_member, out_root, jpeg_q = args
    from stage2.data.sav_dataset import SAVVideoDataset, _decode_rle  # noqa

    p = bundle_path(out_root, mp4_member)
    if os.path.exists(p):
        return ("skip", mp4_member, 0)

    # minimal dataset instance just to reuse its tar/video helpers
    ds = _worker.ds
    try:
        mp4_bytes = ds._read_member(tar_path, mp4_member)
        json_bytes = ds._read_member(tar_path, json_member)
        T_video, handle = ds._decode_video_len(mp4_bytes)
        meta = json.loads(json_bytes)
        masklet = meta["masklet"]            # [T_a][num_obj] RLE
        num_obj = int(meta.get("masklet_num", 0))
    except Exception as e:
        return ("fail", mp4_member, 0)

    T_annot = len(masklet)
    if num_obj == 0 or T_annot == 0:
        ds._decode_close(handle)
        # write an EMPTY marker so the loader knows this is a valid-but-empty video
        _save(p, frames_jpeg=[], picked_masklet=[], local_picks=[], num_obj=0, q=jpeg_q)
        return ("empty", mp4_member, 0)

    annot_video_idx = (np.arange(T_annot) * 4).astype(np.int64)
    annot_video_idx = annot_video_idx[annot_video_idx < T_video]
    if len(annot_video_idx) == 0:
        ds._decode_close(handle)
        _save(p, frames_jpeg=[], picked_masklet=[], local_picks=[], num_obj=0, q=jpeg_q)
        return ("empty", mp4_member, 0)

    video_picks, local_picks = ds._sample_indices(annot_video_idx)
    if len(video_picks) == 0:
        ds._decode_close(handle)
        _save(p, frames_jpeg=[], picked_masklet=[], local_picks=[], num_obj=0, q=jpeg_q)
        return ("empty", mp4_member, 0)

    try:
        picked = ds._decode_frames(handle, video_picks)   # [k,H0,W0,3] uint8 RGB
    except Exception:
        ds._decode_close(handle)
        return ("fail", mp4_member, 0)
    finally:
        ds._decode_close(handle)

    frames_jpeg = [_encode_jpeg(picked[i], jpeg_q) for i in range(picked.shape[0])]
    # store the RLE for ALL objects at each picked annotated frame (object choice
    # is per-idx at load time). picked_masklet[slot] = [rle_obj0, rle_obj1, ...]
    picked_masklet = [masklet[int(li)] for li in local_picks]

    nbytes = _save(p, frames_jpeg=frames_jpeg, picked_masklet=picked_masklet,
                   local_picks=[int(x) for x in local_picks], num_obj=num_obj, q=jpeg_q)
    return ("ok", mp4_member, nbytes)


def _save(path, *, frames_jpeg, picked_masklet, local_picks, num_obj, q):
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}.npz"
    # frames as an object array of variable-length bytes; masklet as JSON (RLEs are
    # small dict/str). Atomic write via rename.
    obj = {
        "frames_jpeg": np.array(frames_jpeg, dtype=object),
        "masklet_json": json.dumps(picked_masklet),
        "local_picks": np.array(local_picks, dtype=np.int64),
        "num_obj": np.int64(num_obj),
    }
    with open(tmp, "wb") as f:
        np.savez(f, **obj)
    os.replace(tmp, path)
    return os.path.getsize(path)


def _init_worker(tar_dir, index_cache):
    from stage2.data.sav_dataset import SAVVideoDataset
    _worker.ds = SAVVideoDataset(tar_dir, img_size=1008, index_cache=index_cache,
                                 split="train", select_first_object=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/saif/sav_predecoded")
    ap.add_argument("--tar-dir", default="/mnt/exoshdd/datasets/SA-V")
    ap.add_argument("--index-cache", default="data/sav_index.pkl")
    ap.add_argument("--jpeg-q", type=int, default=95)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=-1, help="only first N videos (subset)")
    args = ap.parse_args()

    from stage2.data.sav_dataset import SAVVideoDataset
    ds = SAVVideoDataset(args.tar_dir, img_size=1008, index_cache=args.index_cache,
                         split="train", select_first_object=False)
    items = [(i, *ds.index[i], args.out, args.jpeg_q) for i in range(len(ds.index))]
    if args.limit > 0:
        items = items[:args.limit]
    print(f"[predecode] {len(items)} videos -> {args.out} (q={args.jpeg_q}, workers={args.workers})")

    import multiprocessing as mp
    t0 = time.time()
    counts = {"ok": 0, "skip": 0, "empty": 0, "fail": 0}
    total_bytes = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker,
                  initargs=(args.tar_dir, args.index_cache)) as pool:
        for i, (status, vid, nb) in enumerate(pool.imap_unordered(_worker, items, chunksize=8)):
            counts[status] = counts.get(status, 0) + 1
            total_bytes += nb
            if (i + 1) % 200 == 0:
                el = time.time() - t0
                done = i + 1
                rate = done / el
                eta = (len(items) - done) / rate / 60
                print(f"  {done}/{len(items)}  {rate:.1f} vid/s  ETA {eta:.0f}min  "
                      f"size {total_bytes/1e9:.1f}GB  {counts}", flush=True)
    el = time.time() - t0
    print(f"[predecode] done in {el/60:.1f}min  {counts}  total {total_bytes/1e9:.1f}GB")
    if counts["ok"]:
        print(f"  avg bundle: {total_bytes/max(1,counts['ok'])/1e6:.2f} MB  "
              f"-> full set proj: {total_bytes/max(1,counts['ok'])*len(ds.index)/1e9:.0f}GB")


if __name__ == "__main__":
    main()
