#!/usr/bin/env python3
"""
Milestone 0 — Mask-prompted video object tracking with the EfficientSAM3
(RepViT-M0.9) backbone, using SAM3's native tracker. NO training required.

This proves the end-to-end "select an object by a mask on frame 0, then track it
across the video" pipeline works with the *distilled* (efficient) image backbone
swapped in for SAM3's ViT-H. It is the baseline / reference that the Stage-1+
distilled student (Milestone 1) must eventually match efficiently.

What it does:
  1. Builds the SAM3 video model with backbone_type='repvit', model_name='m0.9'
     (the Stage-1 distilled encoder). The tracker / prompt-encoder / mask-decoder /
     memory modules come from the SAM3 checkpoint (frozen, reused as-is).
  2. Opens a video, selects an object on frame 0 by a MASK (or by a BOX, which the
     SAM mask-decoder turns into a mask), and propagates it through the clip.
  3. Saves per-frame mask overlays.

Why this matters for the project goal (mask-prompted tracking):
  - The Stage-2 *distilled student* (stage2/models/student.py) is object-AGNOSTIC:
    forward(frames) has no prompt input and the loader hardcodes obj_idx=0, so it
    cannot be steered to a chosen object. SAM3's tracker (used here) IS promptable.
  - The efficient piece (RepViT backbone) is already distilled; the prompt-encoder
    + mask-decoder + memory modules are comparatively cheap and reused here.

Usage:
    # Box prompt (no external mask needed — easiest first run):
    python stage2/demo_mask_tracking.py \
        --video sam3/assets/videos/bedroom.mp4 \
        --box 300,0,500,400 --max-frames 240

    # Mask prompt from a binary PNG (white = object) on frame 0:
    python stage2/demo_mask_tracking.py \
        --video sam3/assets/videos/0001 \
        --mask /path/to/frame0_mask.png

    # Point prompt:
    python stage2/demo_mask_tracking.py --video sam3/assets/videos/bedroom.mp4 \
        --points 410,260

Checkpoints:
  --sam3-ckpt   full SAM3 video checkpoint (tracker+heads+memory).
                default: /mnt/exoshdd/checkpoints/sam3/sam3.pt
  The RepViT backbone weights load from this same checkpoint via strict=False; to
  instead overlay the Stage-1 distilled backbone, pass --backbone-ckpt (optional).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

# --- path setup: make `sam3` importable (mirrors stage2/models/student.py) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for p in (_REPO, os.path.join(_REPO, "sam3")):
    if p not in sys.path:
        sys.path.insert(0, p)


def parse_args():
    p = argparse.ArgumentParser(description="Mask-prompted tracking with RepViT-M0.9 backbone")
    p.add_argument("--video", default=os.path.join(_REPO, "sam3/assets/videos/bedroom.mp4"),
                   help="Path to an .mp4 or a folder of numbered .jpg frames")
    p.add_argument("--sam3-ckpt", default="/mnt/exoshdd/checkpoints/sam3/sam3.pt",
                   help="Full SAM3 video checkpoint (tracker + heads + memory).")
    p.add_argument("--backbone-ckpt", default=None,
                   help="Optional Stage-1 distilled RepViT ckpt to overlay onto the backbone "
                        "(e.g. checkpoints/efficient_sam3_repvit_s.pt). If omitted, backbone "
                        "weights come from --sam3-ckpt.")
    p.add_argument("--backbone-type", default="repvit")
    p.add_argument("--model-name", default="m0.9")
    # --- frame-0 prompt: choose ONE of mask / box / points ---
    p.add_argument("--mask", default=None, help="Path to a binary mask image for frame 0 (white=object).")
    p.add_argument("--box", default=None, help="Frame-0 box 'xmin,ymin,xmax,ymax' in pixels.")
    p.add_argument("--points", default=None,
                   help="Frame-0 positive points 'x1,y1;x2,y2' in pixels.")
    p.add_argument("--obj-id", type=int, default=1)
    p.add_argument("--ann-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=240)
    p.add_argument("--frame-stride", type=int, default=30, help="Save every Nth frame.")
    p.add_argument("--output-dir", default=os.path.join(_REPO, "output", "demo_mask_tracking"))
    return p.parse_args()


def load_video_frames_rgb(video_path: str):
    """Return list of RGB uint8 frames (for visualization only)."""
    import cv2
    if video_path.endswith((".mp4", ".avi", ".mov")):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames
    paths = sorted(glob.glob(os.path.join(video_path, "*.jpg")),
                   key=lambda q: int(os.path.splitext(os.path.basename(q))[0])
                   if os.path.splitext(os.path.basename(q))[0].isdigit() else q)
    from PIL import Image
    return [np.array(Image.open(q).convert("RGB")) for q in paths]


_OBJ_COLORS = [(255, 80, 80), (80, 160, 255), (80, 220, 120), (240, 200, 60),
               (200, 100, 240), (60, 220, 220), (240, 140, 60), (180, 180, 180)]


def overlay_and_save(frame_rgb, mask_bool, obj_id, title, path):
    """Alpha-blend the mask over the frame and save as PNG (cv2; no matplotlib)."""
    import cv2
    img = frame_rgb.astype(np.float32).copy()
    if mask_bool is not None:
        m = np.asarray(mask_bool).squeeze().astype(bool)
        if m.any():
            color = np.array(_OBJ_COLORS[obj_id % len(_OBJ_COLORS)], np.float32)
            img[m] = 0.5 * img[m] + 0.5 * color
    bgr = cv2.cvtColor(img.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[demo] device={device}")

    if not os.path.exists(args.sam3_ckpt):
        print(f"[demo] ERROR: SAM3 checkpoint not found: {args.sam3_ckpt}")
        print("       Pass --sam3-ckpt /path/to/sam3.pt (the full video model).")
        return 1

    from sam3.model_builder import build_efficientsam3_video_model

    print(f"[demo] building video model: backbone={args.backbone_type}/{args.model_name}")
    sam3_model = build_efficientsam3_video_model(
        checkpoint_path=args.sam3_ckpt,
        backbone_type=args.backbone_type,
        model_name=args.model_name,
        load_from_HF=False,
        device=device,
        strict_state_dict_loading=False,
        apply_temporal_disambiguation=True,
        compile=False,
    )

    # Optional: overlay the Stage-1 distilled backbone weights (the truly efficient
    # encoder). strict=False — only matching backbone keys are updated.
    if args.backbone_ckpt:
        if not os.path.exists(args.backbone_ckpt):
            print(f"[demo] WARNING: --backbone-ckpt not found, skipping: {args.backbone_ckpt}")
        else:
            sd = torch.load(args.backbone_ckpt, map_location=device, weights_only=False)
            sd = sd.get("model", sd) if isinstance(sd, dict) else sd
            sd = {k.replace("student_trunk.", ""): v for k, v in sd.items()}
            msg = sam3_model.load_state_dict(sd, strict=False)
            print(f"[demo] overlaid backbone ckpt: missing={len(msg.missing_keys)} "
                  f"unexpected={len(msg.unexpected_keys)}")

    # SAM2-style tracker predictor: tracker holds init_state/add_new_mask/propagate;
    # attach the detector's backbone so the tracker can encode frames.
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    predictor.eval()

    # --- load video + init tracking state ---
    frames_rgb = load_video_frames_rgb(args.video)
    if not frames_rgb:
        print(f"[demo] ERROR: no frames at {args.video}")
        return 1
    H, W = frames_rgb[0].shape[:2]
    print(f"[demo] loaded {len(frames_rgb)} frames @ {W}x{H}")

    inference_state = predictor.init_state(video_path=args.video)

    # --- frame-0 prompt: mask | box | points (pick one; default box if none given) ---
    af, oid = args.ann_frame, args.obj_id
    with torch.inference_mode():
        if args.mask:
            from PIL import Image
            m = np.array(Image.open(args.mask).convert("L"))
            mask_bool = torch.from_numpy(m > 127).to(device)  # [h,w] bool
            print(f"[demo] add_new_mask obj={oid} frame={af} mask_sum={int(mask_bool.sum())}")
            _, out_ids, _, vres = predictor.add_new_mask(
                inference_state=inference_state, frame_idx=af, obj_id=oid, mask=mask_bool)
        elif args.points:
            pts = np.array([[float(x) for x in pair.split(",")]
                            for pair in args.points.split(";")], np.float32)
            rel = torch.tensor([[x / W, y / H] for x, y in pts], dtype=torch.float32)
            labels = torch.ones(len(pts), dtype=torch.int32)
            print(f"[demo] add_new_points obj={oid} frame={af} npts={len(pts)}")
            _, out_ids, _, vres = predictor.add_new_points(
                inference_state=inference_state, frame_idx=af, obj_id=oid,
                points=rel, labels=labels, clear_old_points=False)
        else:
            box_str = args.box or "300,0,500,400"
            b = [float(v) for v in box_str.split(",")]
            rel_box = np.array([[b[0] / W, b[1] / H, b[2] / W, b[3] / H]], np.float32)
            print(f"[demo] add_new_points_or_box (box) obj={oid} frame={af} box={b}")
            _, out_ids, _, vres = predictor.add_new_points_or_box(
                inference_state=inference_state, frame_idx=af, obj_id=oid, box=rel_box)

    os.makedirs(args.output_dir, exist_ok=True)
    m0 = (vres[0] > 0.0).squeeze().cpu().numpy()
    overlay_and_save(frames_rgb[af], m0, oid, f"frame {af} (prompt)",
                     os.path.join(args.output_dir, f"frame_{af:05d}_prompt.png"))
    print(f"[demo] frame-0 mask pixels: {int(m0.sum())}")

    # --- propagate across the video ---
    print(f"[demo] propagating up to {args.max_frames} frames...")
    video_segments = {}
    with torch.inference_mode():
        for fidx, obj_ids, _low, vres_masks, _scores in predictor.propagate_in_video(
                inference_state, start_frame_idx=0,
                max_frame_num_to_track=args.max_frames, reverse=False,
                propagate_preflight=True):
            video_segments[fidx] = {
                oid_: (vres_masks[i] > 0.0).squeeze().cpu().numpy()
                for i, oid_ in enumerate(obj_ids)
            }
    print(f"[demo] propagated {len(video_segments)} frames")

    # --- save every Nth overlay ---
    saved = 0
    for fidx in sorted(video_segments):
        if fidx % args.frame_stride != 0 or fidx >= len(frames_rgb):
            continue
        seg = video_segments[fidx]
        mask = next(iter(seg.values())) if seg else None
        overlay_and_save(frames_rgb[fidx], mask, oid, f"frame {fidx}",
                         os.path.join(args.output_dir, f"frame_{fidx:05d}.png"))
        saved += 1
    print(f"[demo] saved {saved} overlays to {args.output_dir}")
    print("[demo] DONE — mask-prompted tracking with RepViT-M0.9 backbone succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
