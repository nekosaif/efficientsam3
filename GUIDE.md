# Integrating the SAM3-Distilled Student (`sam3-distill`) into Other Code

This guide explains how to reuse the **EfficientSAM3 Stage-2 distilled student** model from
another codebase. It is written to be followed literally: every path, shape, constant, and
code reference below was verified against the source (`file:line` given throughout).

> **TL;DR** — The distilled model is a **video feature extractor**. Given 8 RGB frames it
> returns per-frame **memory-conditioned feature maps** of shape **`[B, 8, 256, 72, 72]`**.
> It does **not** output segmentation masks. Load the trainable Perceiver weights from a
> Stage-2 `.pth` *on top of* a freshly-constructed model whose frozen backbone comes from the
> Stage-1 checkpoint. Inference **must** run under `torch.amp.autocast(..., cache_enabled=False)`.

---

## 1. What this model is — and what it is not

The student is built from two pieces (`stage2/models/student.py:40`):

```
StudentTemporalModel
├── vision_backbone   ← frozen RepViT-M0.9 image encoder (from Stage-1 ckpt)
├── conv_s0, conv_s1  ← frozen FPN projection convs (loaded but UNUSED in forward; see §9)
└── perceiver         ← TemporalPerceiver  (TRAINABLE — this is what Stage-2 learned)
```

Per-frame data path (`stage2/models/student.py:9-16`):

```
img [B,3,1008,1008]
  → vision_backbone.forward_image(img)
  → sam2_backbone_out["backbone_fpn"][-1]   = pix_feat [B,256,72,72]   (top FPN level)
  → flatten to tokens [B, 5184, 256]
  → perceiver(tokens, prev_latents, frame_idx=t) → mem_feat [B,256,72,72]
stack over T=8 frames → [B, 8, 256, 72, 72]
```

> The 72×72 grid is **fixed by the model**, not derived from input size: the student's
> `ImageStudentEncoder` resamples the backbone feature to `embed_size=72` via bilinear
> `F.interpolate` (`sam3/sam3/model_builder.py:831-841,956-962`). So the RepViT trunk's
> raw 32×32 output is upsampled to the canonical SAM3 72×72 feature grid before the neck.

The **TemporalPerceiver** replaces SAM3's dense sequential memory bank (up to 7 past frames +
object pointers) with **64 fixed learnable latents** that are carried across frames
(`stage2/models/perceiver.py:1-15`). It was distilled to match the SAM3 ViT-H teacher's
per-frame `pix_feat_with_mem` features via L2-normalized MSE + cosine + log-ratio-norm loss
(`stage2/loss.py`).

**Parameter counts** (verified via `smoke_test_models.py`):

| Component | Params | State |
|---|---|---|
| TemporalPerceiver | **~5.02M** (96 state_dict keys) | trainable — what the `.pth` stores |
| RepViT-M0.9 backbone + FPN convs | rest of **~388.93M** total | frozen — comes from Stage-1 ckpt |

### What it is NOT

- ❌ **Not a mask producer.** The output is 256-channel feature maps, the distillation
  *target* — not logits or masks. To get segmentation you would have to feed these features
  back into SAM3's mask decoder/tracker, **which is not implemented anywhere in this repo**
  (see §12).
- ❌ **Not variable-length.** `T` is fixed at **8** frames (`MAX_FRAMES`, see §7).
- ❌ **Not variable-resolution.** Input is fixed at **1008×1008** (RoPE constraint, see §6).

---

## 2. The checkpoint: location, selection, and format

### Which checkpoint to use

| Path | Epoch | `val_total` | Notes |
|---|---|---|---|
| `best_checkpoint/ckpt_best_ep10_val0.2920.pth` | 10 | 0.2920 | **Recommended for integration** — pinned, self-contained (ships with its own `sav_repvit_m0_9.yaml` + `README.md`) |
| `output/efficient_sam3_stage2/ep50_run4/ckpt_best.pth` | 22+ | ~0.2896 | Marginally better, but a **moving target** — overwritten as training continues |
| `output/efficient_sam3_stage2/ep50_run4/ckpt_latest.pth` | latest | — | Symlink to the newest *epoch* ckpt; not necessarily the best |

Lower `val_total` is better. Use the **curated `best_checkpoint/`** copy for a stable,
reproducible integration — it won't change under you and carries the exact config that
produced it. If you want the latest-and-greatest and don't mind it moving, copy
`ep50_run4/ckpt_best.pth` aside first.

### What's inside a `.pth` (`stage2/utils/checkpoint.py:85-98`)

A checkpoint is a Python dict (saved with `torch.save`, **not** `weights_only`):

| Key | Contents | Needed for inference? |
|---|---|---|
| `model_trainable` | 96 Perceiver keys (`perceiver.*`), ~5.02M params | ✅ **yes** |
| `backbone_bn_buffers` | 321 frozen-BN running stats from the backbone | ✅ **yes** (see §9) |
| `ema` | `{'decay', 'shadow'}`; shadow = 96 keys **without** `perceiver.` prefix | optional (smoothed weights) |
| `optimizer`, `scheduler_iter`, `scaler` | training state | ❌ no |
| `epoch`, `global_step`, `best_val`, `step_in_epoch`, `config` | metadata | ❌ no |

> ⚠️ **The backbone weights are NOT in this file.** Only the trainable Perceiver + the
> backbone's BatchNorm buffers are stored. The actual backbone convolution weights come from
> the **Stage-1 checkpoint** when you construct the model (§3). The Stage-2 `.pth` is then
> *overlaid* with `strict=False` (`checkpoint.py:204`).

---

## 3. Prerequisites and assets

| Asset | Path | Size | Needed for inference? |
|---|---|---|---|
| Stage-1 backbone ckpt | `checkpoints/efficient_sam3_repvit_s.pt` | ~1.7 GB | ✅ **required** (provides backbone weights at construction) |
| Stage-2 distilled ckpt | `best_checkpoint/ckpt_best_ep10_val0.2920.pth` | ~77 MB | ✅ **required** (Perceiver + BN buffers) |
| Training config | `best_checkpoint/sav_repvit_m0_9.yaml` | <2 KB | ✅ **required** (model dims) |
| SAM3 teacher ckpt | `/mnt/hdd/checkpoints/sam3/sam3.pt` | ~3.4 GB | ❌ **NOT needed** (training only) |
| Vendored SAM3 package | `sam3/` | — | ✅ required (backbone + builder) |

**Python deps** (the working training interpreter has these): `torch` (2.10 in the run env;
≥2.4 recommended for `torch.amp` API), `timm`, `yacs`, `opencv-python` (`cv2`), `numpy`.
A CUDA GPU is recommended; the model runs on CPU but slowly.

> The teacher (`build_teacher_model`) loads the 3.4 GB SAM3 ViT-H and is only used to
> *generate distillation targets during training*. **Do not** load it for inference — it is
> dead weight and requires the teacher checkpoint you don't need.

---

## 4. Consumption mode A — import directly from this repo

Simplest path: add this repo (and its vendored `sam3/`) to `sys.path` and reuse the existing
factory + loader. The model code already inserts `sam3/` onto `sys.path` itself
(`stage2/models/student.py:28-35`), so importing `stage2.models` is enough.

```python
import os, sys
REPO = "/home/saif/github/efficientsam3"          # <- this repo
sys.path.insert(0, REPO)

import torch
from stage2.config import get_config
from stage2.models import build_student_model, ModelEma
from stage2.utils import load_checkpoint


# get_config(args) reads YAML + applies CLI-style overrides; we feed a tiny shim object.
class _Args:
    cfg = os.path.join(REPO, "best_checkpoint/sav_repvit_m0_9.yaml")
    # STUDENT_CKPT (Stage-1 backbone) is set in the yaml; override here if your path differs:
    student_ckpt = os.path.join(REPO, "checkpoints/efficient_sam3_repvit_s.pt")
    opts = None
    batch_size = data_path = teacher_ckpt = resume = output = tag = None
    accumulation_steps = None
    use_checkpoint = disable_amp = only_cpu = eval = throughput = False
    local_rank = 0

config = get_config(_Args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build model: this loads the Stage-1 backbone (frozen) + a fresh TemporalPerceiver.
model = build_student_model(config).to(device)

# Overlay the trained Perceiver weights + backbone BN buffers from the Stage-2 ckpt.
ckpt_path = os.path.join(REPO, "best_checkpoint/ckpt_best_ep10_val0.2920.pth")
ema = ModelEma(model.perceiver, decay=config.TRAIN.EMA_DECAY, device=device)  # for EMA shadow
meta = load_checkpoint(config=config, path=ckpt_path, model=model, ema=ema, map_location=device)
print("loaded:", meta)   # {'epoch':10, 'best_val':0.292..., 'has_ema':True, ...}

model.eval()
# Optional: use the EMA-smoothed weights (typically ~0.0015–0.0023 lower val):
# ema.copy_to(model.perceiver)
```

- `build_student_model` — `stage2/models/student.py:139`. Reads `config.MODEL.PERCEIVER_*`
  and `config.MODEL.STUDENT_CKPT`.
- `load_checkpoint` — `stage2/utils/checkpoint.py:190`. Loads `model_trainable` with
  `strict=False`, restores `backbone_bn_buffers`, and (if `ema=` passed) the EMA shadow.
  Pass `optimizer=`/`scheduler=`/`scaler=` only when resuming training — omit for inference.

---

## 5. Consumption mode B — vendor a minimal file set (detached project)

To run without depending on the whole repo, copy these into your project (preserving the
relative layout so imports resolve):

```
your_project/
├── sam3/                              # the ENTIRE vendored SAM3 package (backbone + model_builder)
├── stage2/
│   ├── __init__.py
│   ├── config.py                      # MAX_FRAMES, IMG_SIZE, MEAN/STD, perceiver dims, get_config
│   ├── models/
│   │   ├── __init__.py                # exports build_student_model, ModelEma, TemporalPerceiver
│   │   ├── student.py                 # StudentTemporalModel
│   │   ├── perceiver.py               # TemporalPerceiver
│   │   ├── ema.py                     # ModelEma
│   │   └── teacher.py                 # imported by models/__init__.py — keep it even if unused
│   └── utils/
│       ├── __init__.py
│       └── checkpoint.py              # load_checkpoint
├── checkpoints/efficient_sam3_repvit_s.pt          # Stage-1 backbone (~1.7 GB)
└── best_checkpoint/
    ├── ckpt_best_ep10_val0.2920.pth                # Stage-2 distilled weights
    └── sav_repvit_m0_9.yaml                         # config
```

Notes:
- `stage2/models/__init__.py` imports `teacher.py` (`build_teacher_model`), so include
  `teacher.py` even though inference never calls it. It imports lazily from `sam3` and won't
  load the 3.4 GB teacher ckpt unless you call `build_teacher_model`.
- `student.py:28-35` inserts the repo root and `sam3/` onto `sys.path` at import time, using
  paths relative to its own location — so the layout above "just works" as long as `sam3/`
  sits beside `stage2/`.
- Then use the **exact same code as §4**, with `REPO` pointing at `your_project/`.

---

## 6. Input preprocessing (exact, verified)

Reproduce `SAVVideoDataset._resize_pad_image` (`stage2/data/sav_dataset.py:254-273`) **per
frame**:

1. **Long-side resize to 1008**, preserving aspect ratio, bilinear:
   `scale = 1008 / max(h, w)`; `new_h, new_w = round(h*scale), round(w*scale)`;
   `cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)`.
2. **Pad bottom-right** with zeros to exactly `1008 × 1008`:
   `cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, BORDER_CONSTANT, value=0)`.
3. **Normalize** (RGB, 0–255 scale): `x = (x - mean) / std` with
   - `mean = (123.675, 116.28, 103.53)`
   - `std  = (58.395, 57.12, 57.375)`
   (`stage2/config.py:24-25`).
4. **HWC → CHW**: `np.transpose(x, (2, 0, 1))`.

Stack 8 frames per clip → tensor **`[B, 8, 3, 1008, 1008]`**, dtype float32.

> ⚠️ **Gotcha:** the dataset constructor's *default* is `img_size=1024`
> (`sav_dataset.py:75`), but the real value used everywhere is **1008**
> (`config.py:26` — the SAM3 ViT-H RoPE frequencies are baked at 1008). **Always use 1008.**
> Images are RGB (the dataset decodes to RGB; if you read with OpenCV's default BGR, convert
> first).

```python
import cv2, numpy as np, torch

MEAN = np.array([123.675, 116.28, 103.53], np.float32).reshape(1, 1, 3)
STD  = np.array([58.395, 57.12, 57.375], np.float32).reshape(1, 1, 3)
SIZE = 1008

def preprocess_frame(rgb_uint8):                       # rgb_uint8: HxWx3, RGB, 0-255
    h, w = rgb_uint8.shape[:2]
    s = SIZE / max(h, w)
    nh, nw = round(h * s), round(w * s)
    img = cv2.resize(rgb_uint8, (nw, nh), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, 0, SIZE - nh, 0, SIZE - nw, cv2.BORDER_CONSTANT, value=0)
    img = (img.astype(np.float32) - MEAN) / STD
    return torch.from_numpy(np.transpose(img, (2, 0, 1)))   # [3,1008,1008]

# 8 frames -> [1, 8, 3, 1008, 1008]
frames = torch.stack([preprocess_frame(f) for f in clip_8_rgb_frames]).unsqueeze(0)
```

---

## 7. Forward signature and output semantics

`StudentTemporalModel.forward(frames, attention_mask=None)` (`stage2/models/student.py:114`):

| Arg | Shape / type | Meaning |
|---|---|---|
| `frames` | `[B, T, 3, 1008, 1008]` float | T **must be 8** |
| `attention_mask` | `[B, T]` bool, optional | **Currently unused by `forward`** — padding is only honored in the training loss. Pass `torch.ones(B, T, dtype=bool)` or `None`. |
| **returns** | `[B, T, 256, 72, 72]` | per-frame memory-conditioned features |

**`T` must equal 8** — `MAX_FRAMES=8` is a hard constant (`config.py:9`), asserted at config
load (`config.py:183`), and the Perceiver's `temporal_embed` has exactly 8 slots
(`stage2/models/perceiver.py:85`). Fewer/more frames will index incorrectly or assert. If
your clip has fewer than 8 annotated frames, pad to 8 (the training loader zero-pads and
flags padding via `attention_mask`).

**Output spatial dims = 72×72 (5184 tokens/frame).** Verified by running the real model
end-to-end against the checkpoint (output `(1, 8, 256, 72, 72)`). Mechanism, from source:
the RepViT-M0.9 trunk reduces 1008 by stride-32 → 32×32, but the student's
**`ImageStudentEncoder` resamples it to a fixed `embed_size=72`** via bilinear
`F.interpolate` (`sam3/sam3/model_builder.py:831-841`, constructed with `embed_size=72,
img_size=1008` at `:956-962`). The `Sam3DualViTDetNeck` then builds its pyramid from this
72×72 feature; `backbone_fpn[-1]` (scale 1.0, after `scalp=1` drops the lowest level) is
72×72, which `encode_frame` consumes (`student.py:111`). 72×72 is the canonical SAM3 feature
grid (also hardcoded e.g. `sam3/sam3/model/sam1_task_predictor.py:67`).

> 🛑 **`NPU_DEPLOYMENT.md` is STALE on this point** — it claims `[1, 8, 256, 63, 63]` /
> 3969 tokens / "stride-16". That is incorrect. The real, verified output is
> **`[B, 8, 256, 72, 72]`** / 5184 tokens. Trust this guide (confirmed by an actual forward
> pass with the checkpoint loaded), not that doc.

---

## 8. Mandatory inference recipe

```python
B, T = frames.shape[:2]
attn = torch.ones(B, T, dtype=torch.bool, device=device)

model.eval()
with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        feats = model(frames.to(device), attn)     # [B, 8, 256, 72, 72]
```

- **`cache_enabled=False` is REQUIRED.** Under pure BF16, `torch.amp`'s weight cache returns
  *stale* casts of the fp32 params (the cache is only invalidated by `GradScaler.unscale_()`,
  which isn't called here). Omitting it yields garbage / divergent output. This is the
  documented "autocast-cache bug" (see `HANDOFF.md` and the comment in `stage2/train.py`
  around the autocast context). The repo sets it everywhere — you must too.
- On CPU, drop autocast (or use `dtype=torch.float32`): pass `enabled=(device.type=="cuda")`.

### EMA vs live weights

The checkpoint stores an EMA shadow of the Perceiver (`stage2/models/ema.py`). The EMA
weights typically score ~0.0015–0.0023 lower val than the live weights. To use them:

```python
ema.copy_to(model.perceiver)   # overwrites perceiver params/buffers with the EMA shadow
model.eval()
```

`copy_to` (`ema.py:53`) writes in place. There's no toggle back — rebuild/reload if you want
the live weights again. For production, benchmark both on your task and pick the better.

---

## 9. Gotchas (all verified against source)

- **Keep the model in `eval()`.** `StudentTemporalModel.train()` is overridden
  (`student.py:85-99`) to force the frozen backbone + FPN convs back into eval mode so their
  BatchNorm running stats stay fixed. If BN ever switches to train mode (or you skip loading
  `backbone_bn_buffers`), the backbone feature distribution shifts and the Perceiver output
  degrades to cos-sim ≈ 0. The BN buffers in the ckpt are the SA-V-adapted stats — they must
  be loaded (the provided `load_checkpoint` does this automatically).
- **`conv_s0` / `conv_s1` are loaded and frozen but UNUSED in `forward`.** The student only
  reads `backbone_fpn[-1]` (`student.py:111`). These convs exist to mirror the teacher's key
  layout; don't expect them to affect the output.
- **`build_efficientsam3_video_model` is defined TWICE** in `sam3/sam3/model_builder.py`
  (lines 1122 and 1519); Python keeps the **second** definition. If you edit/inspect the
  builder, use the one at `:1519`.
- **No DDP unwrap needed.** Training saves the *unwrapped* trainable state
  (`_trainable_state_dict(model)` on `model.module`), so keys have **no `module.` prefix**.
  Single-GPU / CPU loading is clean.
- **`load_checkpoint` uses `weights_only=False`** (`checkpoint.py:201`) because the payload
  contains a config object. Only load checkpoints you trust.
- **Construction loads ~1.7 GB** (the Stage-1 backbone), then discards everything except the
  vision backbone + 2 convs (`student.py:60-83`). First build takes a few seconds and a
  transient memory spike.

---

## 10. Complete standalone example (`infer.py`)

A minimal, copy-pasteable script. Mirrors `stage2/smoke_test_models.py` (which is known to
run) but loads the trained checkpoint and uses real preprocessing.

```python
import os, sys
REPO = "/home/saif/github/efficientsam3"
sys.path.insert(0, REPO)

import cv2, numpy as np, torch
from stage2.config import get_config
from stage2.models import build_student_model, ModelEma
from stage2.utils import load_checkpoint

MEAN = np.array([123.675, 116.28, 103.53], np.float32).reshape(1, 1, 3)
STD  = np.array([58.395, 57.12, 57.375], np.float32).reshape(1, 1, 3)
SIZE, T = 1008, 8

def preprocess(rgb):
    h, w = rgb.shape[:2]; s = SIZE / max(h, w)
    nh, nw = round(h * s), round(w * s)
    img = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, 0, SIZE - nh, 0, SIZE - nw, cv2.BORDER_CONSTANT, value=0)
    img = (img.astype(np.float32) - MEAN) / STD
    return torch.from_numpy(np.transpose(img, (2, 0, 1)))

class _Args:
    cfg = os.path.join(REPO, "best_checkpoint/sav_repvit_m0_9.yaml")
    student_ckpt = os.path.join(REPO, "checkpoints/efficient_sam3_repvit_s.pt")
    opts = None
    batch_size = data_path = teacher_ckpt = resume = output = tag = None
    accumulation_steps = None
    use_checkpoint = disable_amp = only_cpu = eval = throughput = False
    local_rank = 0

def main():
    config = get_config(_Args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_student_model(config).to(device)
    ema = ModelEma(model.perceiver, decay=config.TRAIN.EMA_DECAY, device=device)
    meta = load_checkpoint(
        config=config,
        path=os.path.join(REPO, "best_checkpoint/ckpt_best_ep10_val0.2920.pth"),
        model=model, ema=ema, map_location=device,
    )
    print("loaded:", meta)
    model.eval()
    # ema.copy_to(model.perceiver)   # uncomment for EMA-smoothed weights

    # --- replace this with 8 real RGB frames; here we use random images ---
    clip = [np.random.randint(0, 255, (720, 1280, 3), np.uint8) for _ in range(T)]
    frames = torch.stack([preprocess(f) for f in clip]).unsqueeze(0).to(device)  # [1,8,3,1008,1008]
    attn = torch.ones(1, T, dtype=torch.bool, device=device)

    use_cuda = device.type == "cuda"
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=use_cuda, cache_enabled=False):
            feats = model(frames, attn)
    print("output:", tuple(feats.shape), feats.dtype, "finite:", torch.isfinite(feats).all().item())
    assert tuple(feats.shape) == (1, 8, 256, 72, 72)

if __name__ == "__main__":
    main()
```

Expected: `output: (1, 8, 256, 72, 72) ... finite: True`. (This exact script was run against
`ckpt_best_ep10_val0.2920.pth` and produced `(1, 8, 256, 72, 72)`, finite.)

---

## 11. Export / NPU deployment status

- **There is no working ONNX/TFLite export script in this repo.** Grep confirms only the
  static-shape *constraints* exist (`MAX_FRAMES=8`, `IMG_SIZE=1008`) plus the aspirational
  `NPU_DEPLOYMENT.md`. No `torch.onnx.export` call ships here.
- The shapes are kept static specifically to *enable* a future export: fixed `T=8`, fixed
  1008×1008 input, fixed 64 latents, and the Perceiver computes `H=W=isqrt(HW)` at runtime
  (`perceiver.py:122`; here `isqrt(5184)=72`) — for export you'd hardcode `H=W=72`.
- ⚠️ `NPU_DEPLOYMENT.md` is **inaccurate** on the core I/O shape (it says 63×63/3969 tokens;
  the truth is **72×72/5184**, see §7). Treat its export snippets as a sketch, not a recipe —
  any exporter must target `[1, 8, 256, 72, 72]` output and `[1, 8, 3, 1008, 1008]` input.
- A real export would: wrap `StudentTemporalModel.forward` (or a per-frame variant) in an
  `nn.Module` with fixed shapes, run `model.eval()`, disable flash SDPA, and `torch.onnx.export`
  with a fixed dummy `[1,8,3,1008,1008]` — then validate the ONNX output matches PyTorch
  within tolerance. None of this exists yet; it's future work.

---

## 12. Producing segmentation masks (NOT implemented)

The student outputs distilled **memory features**, not masks. SAM3 normally consumes the
equivalent `pix_feat_with_mem` in its mask decoder / tracker to produce masks. To turn these
features into segmentation you would need to:

1. Feed `feats[:, t]` (`[B,256,72,72]`) into SAM3's tracker as the memory-conditioned image
   feature, where `Sam3VideoBase` / the tracker expects it (see
   `sam3/sam3/model/sam3_tracker_base.py`, `sam3/sam3/model/sam3_video_predictor.py`,
   `sam3/sam3/model/sam3_video_base.py`).
2. Provide prompts (points/box/mask) and run the SAM mask decoder head.

**This wiring is not present in this repo** — Stage-2 trains only the memory module against
the teacher's features. Building the full predictor on top of the distilled student is a
separate, unimplemented effort. If/when added, the natural seam is where the teacher's
`pix_feat_with_mem` is produced (`stage2/models/teacher.py:_forward_with_memory`) — the
student's output is the drop-in replacement for exactly that tensor.

---

## Quick reference

| Thing | Value | Source |
|---|---|---|
| Input | `[B, 8, 3, 1008, 1008]` float, RGB, normalized | `config.py:9,26` |
| Output | `[B, 8, 256, 72, 72]` (memory features, not masks) | verified §7 |
| Frames `T` | fixed **8** | `config.py:9,183` |
| Image size | fixed **1008** | `config.py:26` |
| Norm mean | `(123.675, 116.28, 103.53)` | `config.py:24` |
| Norm std | `(58.395, 57.12, 57.375)` | `config.py:25` |
| Recommended ckpt | `best_checkpoint/ckpt_best_ep10_val0.2920.pth` | §2 |
| Backbone ckpt (required) | `checkpoints/efficient_sam3_repvit_s.pt` | §3 |
| Build fn | `stage2.models.build_student_model(config)` | `student.py:139` |
| Load fn | `stage2.utils.load_checkpoint(...)` | `checkpoint.py:190` |
| Autocast | `torch.amp.autocast('cuda', dtype=bfloat16, cache_enabled=False)` | §8 (required) |
| Trainable params | ~5.02M (96 keys) | `smoke_test_models.py` |
