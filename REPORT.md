# EfficientSAM3 — Research & Training Report

**Last updated:** 2026-04-18
**Author:** Mollah Md Saif
**Target hardware:** Samsung S26 Ultra (Qualcomm Hexagon NPU)
**Status:** Stage 2 training in progress

> New agent picking up this project? Start with **[HANDOFF.md](HANDOFF.md)** for operational state (what's running, where files are, how to monitor). This file is the research write-up.

---

## 1. Project Goal

Build a mobile-deployable version of Meta's SAM3 (Segment Anything Model 3) video segmentation model, with static-shape ONNX export compatible with the Qualcomm Hexagon NPU on Samsung Galaxy S26 Ultra.

SAM3 is too large and dynamic to run on phone NPUs directly:
- **Heavy encoder:** ViT-H (~630M params)
- **Dynamic video memory:** sequential per-frame memory bank that re-queries up to 7 past frames per new frame — incompatible with static ONNX graphs required by Hexagon

**EfficientSAM3** compresses both via knowledge distillation into a fixed-shape student suitable for on-device inference.

---

## 2. Architecture Overview

| Component | Teacher (SAM3) | Student (EfficientSAM3) |
|-----------|----------------|-------------------------|
| Image encoder | ViT-H, 1008×1008 input | RepViT-M0.9 (~10M params), 1008×1008 |
| Text encoder | PerceptionEncoder | PerceptionEncoder (distilled separately in Stage 1) |
| Temporal memory | Sequential 7-frame memory bank + object pointers + transformer cross-attn | **TemporalPerceiver** — 4-layer cross-attention over 64 learned latents, processes all 8 frames in parallel |
| Output | Per-frame `pix_feat_with_mem` (memory-conditioned features) | Same shape, distilled to match |

**ONNX constraints driving design:**
- `MAX_FRAMES = 8` (static)
- `IMG_SIZE = 1008` (SAM3 RoPE constraint)
- All `MultiheadAttention` calls use `need_weights=False` (ONNX-friendly)
- No dynamic loops over past frames — single Perceiver pass

---

## 3. Research & Implementation Stages

### Stage 1 — Image & Text Encoder Distillation ✅ Complete

**Goal:** Distill SAM3 ViT-H image encoder → RepViT-M0.9. Text encoder distilled in parallel.

**Artifacts:**
- `stage1/train_image_encoder_stage1.py` — DDP + AMP training loop
- `stage1/train_text_encoder_stage1.py` — text branch
- Checkpoint: `checkpoints/efficient_sam3_repvit_s.pt`

**Outcome:** Backbone frozen for downstream stages. Single-frame segmentation IoU comparable to SAM3 ViT-H baseline.

### Stage 2 — Temporal Memory Distillation 🚧 In Progress

**Goal:** Replace teacher's dynamic memory bank with the static-shape **TemporalPerceiver**. Teacher remains frozen; Stage 1 backbone remains frozen. Only the Perceiver module is trained from scratch.

#### 3.1 Teacher memory path analysis

Studied SAM3 tracker source to identify exact distillation target:

- `sam3/sam3/model/sam3_tracker_base.py:799` — `_encode_new_memory()` produces per-frame memory features
- `sam3/sam3/model/sam3_tracker_base.py:562` — `_prepare_memory_conditioned_features()` combines current pixel features with up to 7 past memory frames via transformer cross-attention, yielding `pix_feat_with_mem` — **this is the distillation target**

Discovery: `tracker.backbone` is `None` in the composed SAM3 video model; had to call `vision_backbone` directly and manually apply `conv_s0` / `conv_s1` to reconstruct the correct feature stride.

#### 3.2 Dataset — SA-V

- Source: Meta's SA-V (Segment Anything Video) dataset, 50,583 videos across 60 tar archives, 1.1 TB at `/mnt/hdd/datasets/SA-V/`
- Chose SA-V over SA-Co Gold because Stage 2 distills the **temporal memory mechanism**, which requires video sequences with consistent per-frame masks. SA-Co Gold is for concept/text grounding (a potential future Stage 4).
- Train/val split: deterministic SHA1-hash split, `val_fraction=0.02` → 49,594 train / 989 val
- Index cache: `data/sav_index_v2.pkl`

#### 3.3 Training infrastructure

- **Distillation loss:** MSE + cosine, equal weights, on `pix_feat_with_mem` per frame
- **Optimizer:** AdamW on Perceiver params only (backbone + teacher frozen)
- **Precision:** BF16 autocast
- **Trainer features:** DDP (NCCL), auto-resume, TensorBoard, held-out validation every 5 epochs, best-checkpoint tracking, gradient clipping at 1.0
- **Schedule:** cosine LR with 5-epoch warmup, base LR 1e-4, weight decay 0.05

#### 3.4 I/O optimization sweep (performance engineering)

Initial throughput was the bottleneck — cv2 was decoding full videos to pick 8 frames.

**Tried levers:**

1. **decord random-access decoder** (pip install decord==0.6.0) — seek + decode only the 8 sampled frame indices instead of full-video decode
2. **Worker sweep** (data-only throughput, bs=2):

   | workers | data-only iters/s |
   |--------:|------------------:|
   | 14 (original) | 0.74 |
   | 24 | 2.10 |
   | **28 (chosen)** | **2.86** |
   | 32 | 1.65 (oversubscription thrash) |

3. **Pre-decoded .npy frames** — **skipped**. Would need ~1.2 TB on /mnt/hdd; only 993 GB free.

**Result — end-to-end training throughput:**
- Before: 1.9 ips (cv2 + 14 workers)
- After: **2.67 ips** (decord + 28 workers) → **1.4× end-to-end speedup** (3.6× data-only — GPU compute now the bottleneck)

**Config locked in:** `stage2/configs/sav_repvit_m0_9.yaml` — `NUM_WORKERS: 28`, `BATCH_SIZE: 2`, `PREFETCH_FACTOR: 4`.

#### 3.5 Current launch — 50 epochs on full SA-V

- Launched 2026-04-18 with `TRAIN.EPOCHS 50` (early loss descent 2.26 → 1.49 in 200 iters suggests 50 ep sufficient; 100 ep plan kept as option)
- Projected wall-clock: **~5.4 days** on single RTX PRO 6000 Blackwell 96 GB
- Output dir: `output/efficient_sam3_stage2/<run>/`
- Log: `logs/stage2_run1.log`
- Auto-resume enabled — safe to kill/restart

---

## 4. Hardware

- **GPU:** 1× NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB VRAM
- **CPU:** AMD Ryzen 9 9950X (16C / 32T)
- **RAM:** 128 GB DDR5
- **Storage:** 3.6 TB HDD at `/mnt/hdd` (datasets + teacher checkpoints), ~993 GB free

---

## 5. Roadmap

| Stage | Purpose | Status |
|------|---------|--------|
| 1 | Image + text encoder distillation | ✅ Complete |
| 2 | Temporal memory distillation (TemporalPerceiver) | 🚧 Training — 50 ep, ~5.4 days |
| 3 | ONNX export with static shapes + Hexagon deployment | ⏳ Pending |
| 4 (optional) | Concept/text-head distillation on SA-Co Gold for open-vocab PCS | ⏳ Maybe |

---

## 6. Key Files

| Path | Purpose |
|------|---------|
| `stage2/config.py` | YACS config schema |
| `stage2/configs/sav_repvit_m0_9.yaml` | Active training config |
| `stage2/train.py` | DDP training entrypoint |
| `stage2/models/perceiver.py` | TemporalPerceiver module |
| `stage2/data/sav_dataset.py` | SA-V streaming dataset (decord + cv2 fallback) |
| `stage2/loss.py` | MSE + cosine distillation loss |
| `stage2/eval/` | Validation evaluation |
| `stage2/utils/` | Checkpoint management |
| `stage2/bench_io.py` | I/O throughput benchmark script |
| `checkpoints/efficient_sam3_repvit_s.pt` | Stage 1 student checkpoint |
| `/mnt/hdd/checkpoints/sam3/sam3.pt` | Frozen SAM3 ViT-H teacher |

---

## 7. References

- SAM3 paper + codebase: `sam3/` (vendored)
- Stage 1 approach: follows RepViT distillation recipe
- Perceiver I/O reference: `arxiv.org/abs/2107.14795`
- SA-V dataset: Meta AI Research, `DATASET_SPECS.md` in `/mnt/hdd/datasets/SA-V/`
