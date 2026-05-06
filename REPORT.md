# EfficientSAM3: On-Device Video Segmentation via Knowledge Distillation
## Progress Report

**Date:** 2026-04-19
**Author:** Mollah Md Saif
**Affiliation:** [Your Institution / Lab]
**Target Deployment:** Samsung Galaxy S26 Ultra (Qualcomm Hexagon NPU)

---

## Executive Summary

This report describes the research and engineering progress toward **EfficientSAM3**, a mobile-deployable video segmentation model derived from Meta's Segment Anything Model 3 (SAM3). The core challenge is that SAM3 is architecturally incompatible with on-device neural processing units (NPUs), which require static computation graphs and fixed tensor shapes. This work addresses that incompatibility through a staged knowledge distillation pipeline that compresses SAM3 from approximately 827 million parameters to a student model suitable for real-time NPU inference.

Stage 1 (encoder distillation) is complete. Stage 2 (temporal memory distillation) is currently underway, with approximately 2.5% of the planned 50-epoch training run completed before a planned pause. The loss function has descended substantially from its initial value (2.33 → ~0.88), confirming that distillation is proceeding as expected.

---

## 1. Motivation and Problem Statement

Meta's SAM3 achieves state-of-the-art video object segmentation by maintaining a sequential memory bank that cross-attends over up to seven past frames per new frame. While this mechanism produces high-quality temporal consistency, it introduces two fundamental constraints that prevent direct NPU deployment:

1. **Parameter count:** The ViT-H image encoder contains approximately 630 million parameters, far exceeding the memory budget of mobile NPUs.
2. **Dynamic computation graph:** The sequential per-frame memory queries produce variable-length tensor operations that cannot be represented as the static ONNX graphs required by the Qualcomm Hexagon NPU SDK.

**EfficientSAM3** resolves both constraints via knowledge distillation: the heavy encoder is replaced by a lightweight RepViT backbone, and the dynamic memory bank is replaced by a parallel fixed-shape TemporalPerceiver module. The resulting student model is exportable to ONNX with fully static shapes.

---

## 2. Overall Approach: Staged Knowledge Distillation

The distillation is organized into three primary stages:

| Stage | Objective | Status |
|-------|-----------|--------|
| **1** | Distill ViT-H image encoder → RepViT-M0.9; distill text encoder in parallel | **Complete** |
| **2** | Distill SAM3 sequential memory bank → TemporalPerceiver (static, parallel) | **In progress** (~2.5% of 50 epochs) |
| **3** | ONNX export with static shapes; Hexagon NPU deployment and on-device evaluation | Pending |
| **4** *(optional)* | Concept/text-head distillation on SA-Co Gold for open-vocabulary segmentation | Under consideration |

In all stages, the SAM3 teacher model is fully frozen. In Stage 2, the Stage 1 backbone is additionally frozen; only the new TemporalPerceiver module (5.02 million parameters) has active gradients.

---

## 3. Architecture

### 3.1 Teacher Model (SAM3)

SAM3 processes video through the following pipeline:

- **Image encoder:** ViT-H backbone (~630M parameters), input resolution 1008×1008 (constrained by Rotary Position Embedding)
- **Memory encoder:** Per-frame `_encode_new_memory()` computes `maskmem_features [B, 256, H, W]` from pixel features and segmentation masks
- **Memory-conditioned features:** `_prepare_memory_conditioned_features()` gathers features from up to `num_maskmem=7` past frames with temporal positional encodings, passes them through transformer cross-attention, and produces `pix_feat_with_mem [B, 256, H, W]` per frame — the final distillation target

### 3.2 Student Model (EfficientSAM3)

| Component | Teacher (SAM3) | Student (EfficientSAM3) |
|-----------|----------------|-------------------------|
| Image encoder | ViT-H, ~630M params | RepViT-M0.9, ~10M params |
| Input resolution | 1008×1008 | 1008×1008 (identical) |
| Text encoder | PerceptionEncoder | PerceptionEncoder (distilled, Stage 1) |
| Temporal memory | Sequential, up to 7 past frames, dynamic graph | TemporalPerceiver — 4-layer cross-attention over 64 learned latents, all 8 frames in parallel, static graph |
| Distillation target | — | `pix_feat_with_mem` per frame |
| Total parameters | ~827M | ~388M (backbone ~383M + Perceiver 5M) |
| Trainable in Stage 2 | — | 5.02M (Perceiver only) |

**ONNX static-shape constraints:**
- `MAX_FRAMES = 8` (fixed at export time)
- `IMG_SIZE = 1008` (preserves SAM3 RoPE weight compatibility)
- All `MultiheadAttention` calls use `need_weights=False` (required for ONNX tracing)
- No dynamic loops over past frames; single parallel Perceiver pass

### 3.3 TemporalPerceiver Design

The TemporalPerceiver replaces the teacher's sequential memory bank with a module that accepts all 8 frames simultaneously:

- **Input:** Student encoder features `[B, 8, C, H, W]` + boolean attention mask `[B, 8]`
- **Architecture:** 64 learned latent queries attend to spatially-projected frame tokens via 4 cross-attention layers (8 heads, dim=256); sinusoidal temporal positional encodings distinguish frame positions
- **Output:** Memory-conditioned features `[B, 8, C, H, W]`, matching teacher's `pix_feat_with_mem` shape
- **ONNX compatibility:** Fixed tensor shapes throughout; no conditional branches on sequence length

---

## 4. Stage 1: Encoder Distillation (Complete)

### 4.1 Objective

Replace SAM3's ViT-H image encoder with RepViT-M0.9 while preserving single-frame segmentation quality. Text encoder distilled in parallel.

### 4.2 Outcome

- RepViT-M0.9 student backbone trained to match ViT-H feature maps via intermediate-layer and output-level distillation losses
- Single-frame segmentation IoU on validation set comparable to SAM3 ViT-H baseline
- Checkpoint saved to `checkpoints/efficient_sam3_repvit_s.pt`
- Backbone frozen for all subsequent stages

**Key implementation note:** In the composed SAM3 video model, `tracker.backbone` is `None`. The image encoder must be accessed via `vision_backbone` directly, with `conv_s0` and `conv_s1` applied manually to reconstruct the correct feature stride — a non-obvious implementation detail that required source analysis of `sam3_tracker_base.py`.

---

## 5. Stage 2: Temporal Memory Distillation (In Progress)

### 5.1 Objective

Train the TemporalPerceiver module to produce memory-conditioned features that match those of SAM3's sequential memory bank, enabling temporal consistency in video segmentation without dynamic graph operations.

### 5.2 Dataset: SA-V

The Meta SA-V (Segment Anything Video) dataset was selected as the training corpus for Stage 2:

- **Scale:** 50,583 video clips across 60 TAR archives; 1.1 TB total
- **Annotation:** Dense per-frame segmentation masks providing ground-truth temporal structure
- **Split:** Deterministic SHA1-hash partitioning, `val_fraction=0.02` → **49,594 training / 989 validation** videos
- **Rationale over SA-Co Gold:** SA-Co Gold targets concept and text grounding, which is relevant only to the optional Stage 4. Stage 2 requires video sequences with consistent per-frame masks to supervise temporal memory behavior — SA-V provides exactly this.

### 5.3 Distillation Methodology

The training objective is to minimize the divergence between student and teacher memory-conditioned features at every frame:

$$\mathcal{L} = \mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{cosine}}$$

Both terms are applied to `pix_feat_with_mem [B, 256, H, W]` with equal weight (1.0 each). Features are **L2-normalized along the channel dimension** before MSE computation, making the loss scale-invariant (bounded ~[0, 6]) regardless of teacher feature magnitude. This normalization was required because SA-V teacher features vary 30-50× in magnitude across videos, causing raw MSE to range from 2 to 2700 and destabilizing AdamW's second moments. Per-frame losses are averaged across all valid (non-padded) frames in the batch.

**Training configuration:**

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Base learning rate | 1×10⁻⁴ |
| LR schedule | Cosine decay with 5-epoch warmup |
| Minimum LR | 1×10⁻⁶ |
| Weight decay | 0.05 |
| Gradient clipping | 0.3 (global norm) |
| Precision | BF16 autocast |
| Batch size | 2 (per GPU) |
| Planned epochs | 50 |

### 5.4 Data Pipeline Optimization

Initial profiling revealed that the data pipeline — not GPU compute — was the primary throughput bottleneck. The default implementation used OpenCV (`cv2`) to decode full videos before subsampling 8 frames, incurring unnecessary I/O.

The following optimizations were evaluated:

| Intervention | Result |
|---|---|
| Replace cv2 with `decord` random-access decoder (seek + decode sampled indices only) | 3.6× data-only throughput improvement |
| DataLoader worker count sweep (batch size 2): | |
| — 14 workers (baseline) | 0.74 data-only iter/s |
| — 24 workers | 2.10 iter/s |
| — **28 workers (selected)** | **2.86 iter/s** |
| — 32 workers | 1.65 iter/s (CPU oversubscription) |
| Pre-decode all frames to `.npy` arrays | **Not feasible** — requires ~1.2 TB; only 993 GB free on storage |

**Final end-to-end throughput:** 2.67 samples/second (1.33 iterations/second at batch=2; up from 1.9 samples/s baseline), a **1.4× end-to-end speedup**. GPU compute is now the bottleneck; further data-side gains would not improve wall-clock time without a larger batch or additional GPU.

### 5.5 Current Training Status

| Parameter | Value |
|---|---|
| Launch date | 2026-04-24 16:34 local (resumed from ckpt_epoch_0) |
| Current status | Running — epoch 8 of 50 |
| Epochs completed | 8 of 50 |
| Loss trajectory | ~0.82 initial → steady decay; cosine component dominant early |
| Throughput | ~2.67 samples/s (~1.33 iters/s at batch=2) |
| Estimated wall-clock (full run) | ~11 days total (~5.3h/epoch); ~9.2 days remaining from ep8 |
| Auto-resume | Enabled; epoch-boundary only (SAVE_EVERY_ITERS=0) |

With L2-normalized features, initial loss is ~0.82 (MSE≈0.006, cosine≈0.81). The cosine component dominates early training, indicating the student output directions are not yet aligned with the teacher. Loss is expected to decrease as the Perceiver learns to replicate teacher memory-conditioned feature directions.

### 5.6 Compute Infrastructure

| Resource | Specification |
|---|---|
| GPU | 1× NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB VRAM |
| CPU | AMD Ryzen 9 9950X, 16 cores / 32 threads |
| RAM | 128 GB DDR5 |
| Storage | 3.6 TB HDD (`/mnt/hdd`); ~993 GB available |

---

## 6. Stage 2 Implementation: Modules Built from Scratch

All Stage 2 code was authored in this fork on top of the upstream EfficientSAM3 codebase, which contained only Stage 1 (encoder distillation). The following describes each component.

### 6.1 Training Loop (`stage2/train.py`)

DDP-aware torchrun entrypoint. Key design choices:

- **Frozen backbone:** only `student.module.perceiver` parameters have gradients; backbone and teacher are frozen at construction. `_split_param_groups()` partitions trainable params into weight-decay and no-decay groups (1-D params and biases excluded from decay).
- **BF16 AMP:** `torch.amp.autocast('cuda', dtype=bfloat16)` wraps both student and teacher forward passes. `GradScaler` is conditionally enabled (no-op for BF16 since BF16 has dynamic range comparable to FP32).
- **Gradient clipping:** global norm clip at 0.3 applied after unscaling. Reduced from 1.0 after observing output-head gradient spikes in early runs.
- **Bad-batch guard:** iterations with `loss > 10.0` are skipped and logged as bad batches; optimizer step is not taken.
- **TensorBoard logging:** per-step train loss/MSE/cosine, per-epoch val and val-EMA loss, learning rate. All written from rank-0 only.
- **Mid-epoch checkpoint:** controlled by `SAVE_EVERY_ITERS`; currently disabled (`=0`) per user preference. When enabled, writes `ckpt_running.pth` atomically via tmp+rename.
- **Val evaluation:** runs `evaluate_distill()` every `EVAL.EVERY_N_EPOCHS` epochs on the held-out SA-V val split. Both live student and EMA shadow are evaluated separately.

### 6.2 TemporalPerceiver (`stage2/models/perceiver.py`)

Replaces SAM3's sequential memory bank with a statically-shaped module:

- **Latents:** 64 learned latent queries (`init_latents [1, 64, 256]`), shared across the batch, initialized with truncated normal (std=0.02).
- **Temporal positional embedding:** per-frame additive embedding `temporal_embed [8, 1, 256]` added to latents before each frame's processing. Supports up to 8 frames (ONNX-static).
- **Cross-attention blocks (depth=4):** each block: `CrossAttn(latents ← frame_tokens)` → `SelfAttn(latents)` → `MLP`. Pre-norm (`LayerNorm` before each sub-layer). All `MultiheadAttention` calls use `need_weights=False` for ONNX traceability.
- **Output projection:** after all blocks, frame tokens attend to updated latents (`CrossAttn(tokens ← latents)`), producing `pix_feat [B, C, H, W]` — the distillation target shape.
- **Parameter count:** 5.02M total trainable. Fits in memory alongside frozen 388M student backbone and 827M teacher.

### 6.3 Loss Function (`stage2/loss.py`)

`DistillLoss` implements the L2-normalized MSE + cosine objective:

- **L2 normalization (channel dim):** both student and teacher features are normalized along `dim=2` (channel) before all loss computations. This makes loss scale-invariant regardless of teacher feature magnitude, which varies 30–50× across SA-V videos. Without normalization, raw MSE ranged from 2 to 2700 per batch, destabilizing AdamW's second moments.
- **MSE term:** `((s_norm - t_norm)² × mask).sum() / n_valid_elements`. Bounded ~[0, 4].
- **Cosine term:** `(1 − cosine_similarity(s, t, dim=channel)) × mask`, averaged over valid spatial tokens. Bounded [0, 2].
- **Attention mask:** `[B, T]` bool tensor propagated from the dataloader; padded frames (SA-V clips shorter than `MAX_FRAMES=8`) contribute zero loss.

### 6.4 EMA (`stage2/models/ema.py`)

Exponential moving average over Perceiver weights only:

- **Scope:** shadow is a deepcopy of the Perceiver (~5M params). The frozen 388M backbone is excluded — not needed in the shadow, and would inflate checkpoint size 10×.
- **FP32 shadow under BF16 training:** shadow is immediately cast to float32 after deepcopy. EMA of BF16 weights would accumulate rounding errors over 1000+ steps; FP32 shadow prevents drift.
- **Update rule:** `shadow = decay × shadow + (1 − decay) × live_param` applied in-place per parameter. Non-float buffers (integer counters, bool masks) are copied exact.
- **Decay:** 0.999, stepped every optimizer step. Shadow lags live model by ~1000 iterations.
- **Val use only:** EMA shadow is evaluated separately at each val epoch. Not used as a teacher signal — the teacher remains frozen SAM3 ViT-H throughout.
- **Checkpoint:** saved as `{'decay': float, 'shadow': state_dict}` and restored on auto-resume.

### 6.5 Optimizer and LR Schedule (`stage2/optim.py`)

- **AdamW** with param-group splitting: weight-decay applied to ≥2-D parameters only; biases and LayerNorm parameters (`ndim == 1`) are in a no-decay group.
- **`CosineWarmupScheduler`:** per-iteration stepping (not per-epoch), giving smooth LR curves. Linear warmup from `WARMUP_LR=1e-7` to `BASE_LR=1e-4` over 5 epochs (~123k iters). Cosine decay from `BASE_LR` to `MIN_LR=1e-6` over remaining 45 epochs. LR is applied by mutating `param_groups['lr']` directly.

### 6.6 Checkpoint System (`stage2/utils/checkpoint.py`)

Two-track checkpoint system designed for long multi-day runs:

- **Epoch checkpoints** (`ckpt_epoch_N.pth`): saved at each epoch boundary. `ckpt_latest.pth` symlink always points to the most recent. `ckpt_best.pth` written whenever val loss improves.
- **Running checkpoint** (`ckpt_running.pth`): optional mid-epoch save, written atomically via tmp+rename to prevent corruption on interrupted saves. `auto_resume_helper` prefers this over the epoch checkpoint when it is strictly newer (mtime comparison), so a SIGHUP mid-epoch loses at most `SAVE_EVERY_ITERS` steps rather than a full epoch.
- **Minimal size:** only trainable Perceiver weights + optimizer state + scheduler iter + EMA shadow are persisted. Frozen RepViT backbone and SAM3 teacher are not saved — they are loaded from their own checkpoints at construction time.

### 6.7 Data Pipeline (`stage2/data/sav_dataset.py`)

SA-V dataset loader with random-access decoding:

- **TAR-based streaming:** 60 TAR archives are indexed at first run and cached (`sav_index_v2.pkl`). Videos are accessed by TAR member path without extracting.
- **decord random-access:** `VideoReader.get_batch(indices)` decodes only the 8 sampled frames per video. No full-video decode. 3.6× throughput improvement over OpenCV frame-by-frame iteration.
- **Frame sampling:** every 4th frame is annotated in SA-V; the loader samples `MAX_FRAMES=8` indices evenly from annotated frames. Short clips are zero-padded; the `attention_mask` tensor tracks real vs. padded frames.
- **Deterministic train/val split:** SHA1 hash of video path, `val_fraction=0.02` → 49,594 train / 989 val. Reproducible without an index file.
- **Worker count:** 28 workers optimal on Ryzen 9 9950X (16C/32T). 32 workers oversubscribes and reduces throughput.

### 6.8 Distillation Evaluator (`stage2/eval/distill_eval.py`)

Held-out val evaluation at epoch boundaries:

- Runs both live student and EMA shadow through the full forward pass (student + teacher) over the SA-V val split.
- `torch.no_grad()` + AMP. Student switched to `eval()` mode, restored to `train()` after.
- DDP-safe: `dist.all_reduce(sums)` aggregates loss across all ranks before computing mean.
- Returns `DistillEvalResult(total, mse, cosine, n_batches)` — logged to TensorBoard and log file.

### 6.9 Training Dashboard (`dashboard/`)

FastAPI-based real-time monitoring dashboard built alongside the training infrastructure:

- **Backend (`dashboard/server.py`):** three API endpoints — `/api/snapshot` (training state + system telemetry), `/api/scalars?tag=` (TensorBoard scalar series with downsampling), `/api/scalars/all` (latest value per tag).
- **Metrics reader (`dashboard/metrics.py`):** parses training log via regex for current epoch/iter/loss/LR/ips; reads TensorBoard event files via `EventAccumulator` with mtime-based caching; computes ETA using epoch wall-time average (immune to ips unit confusion — log `ips` is samples/sec, not iters/sec). Falls back to `ips / batch_size` for ETA before the first epoch completes.
- **System telemetry (`dashboard/system.py`):** GPU memory/utilization via `nvidia-smi`; CPU/RAM via `psutil`; 1-second cache to avoid nvml overhead on every request.
- **Frontend (`dashboard/static/`):** single-page app with Plotly loss curves, canvas sparklines, progress bar, and live-refreshing metric cards.

---

## 7. Changes from Upstream Fork

This fork is based on [SimonZeng7108/efficientsam3](https://github.com/SimonZeng7108/efficientsam3). The upstream repo covers Stage 1 (static image encoder + text encoder distillation) and SAM3-LiteText. All additions below are original work added in this fork.

### 7.1 Stage 2: Temporal Memory Distillation (new, not in upstream)

The entire `stage2/` directory is new. Upstream has no video memory distillation; SAM3's temporal memory bank is replaced wholesale with a fixed-shape TemporalPerceiver suitable for ONNX export.

| File | What it adds |
|---|---|
| `stage2/train.py` | DDP training loop: BF16 AMP, gradient clip, EMA updates, auto-resume, TensorBoard logging, bad-batch guard |
| `stage2/models/perceiver.py` | TemporalPerceiver (5.02M): 64 learned latents, 4-layer cross-attention, sinusoidal temporal embeddings, `need_weights=False` throughout for ONNX compatibility |
| `stage2/models/ema.py` | FP32 EMA shadow of Perceiver only; EMA kept separate from training model; val-only evaluation |
| `stage2/models/student.py` | Wrapper that extracts multi-scale features from frozen RepViT backbone and routes them to the Perceiver |
| `stage2/models/teacher.py` | Extracts SAM3 ViT-H spatial memory features via `vision_backbone` + manual `conv_s0`/`conv_s1` application (tracker backbone is None at inference) |
| `stage2/loss.py` | L2-normalized distillation loss: channel-dim unit-norm before MSE + cosine; bounds MSE ∈ [0, 4] and cosine ∈ [0, 2]; per-frame attention mask support |
| `stage2/optim.py` | AdamW with separate param groups (decay / no-decay); `CosineWarmupScheduler` stepping per-iteration rather than per-epoch |
| `stage2/data/sav_dataset.py` | SA-V TAR streaming with decord random-access frame sampling; SHA1-based deterministic train/val split (49,594 / 989 videos); padding mask for variable-length clips |
| `stage2/eval/distill_eval.py` | DDP-aware validation: `all_reduce` for distributed loss aggregation; model/EMA eval-mode toggle; no-grad + AMP context |
| `stage2/utils/checkpoint.py` | Minimal-state checkpointing (Perceiver + optimizer + scheduler + EMA only; frozen backbone not saved); atomic tmp+rename for mid-epoch running checkpoint |
| `stage2/config.py` | CfgNode defaults for all Stage 2 hyperparameters |
| `stage2/configs/sav_repvit_m0_9.yaml` | Active run config: RepViT-M0.9 student, IMG_SIZE=1008, MAX_FRAMES=8, EPOCHS=50, EMA_DECAY=0.999 |
| `stage2/bench_io.py` | I/O throughput benchmark; used to establish workers=28 as optimal on Ryzen 9 9950X |
| `stage2/smoke_test_models.py` | Shape-validation smoke test for student/teacher/Perceiver pipeline |

**Key design decisions not present upstream:**
- **L2 normalization before loss**: SA-V teacher features vary 30–50× in channel magnitude across videos. Normalizing prevents dominant-channel collapse in cosine similarity and keeps MSE bounded.
- **Perceiver latent memory**: Fixed 64×256 latent tensor decouples temporal context size from input frame count; static shape is ONNX-exportable unlike SAM3's variable-length memory bank.
- **EMA in fp32 under bf16 training**: Accumulated EMA updates in fp32 prevent rounding drift that would occur in bf16 at decay=0.999.
- **`need_weights=False` on all MHA**: Required for ONNX tracing; PyTorch's `MultiheadAttention` with `need_weights=True` uses a non-traceable fallback path.

### 7.2 Real-time Training Dashboard (new, not in upstream)

The entire `dashboard/` directory is new. Upstream has no monitoring tooling.

| File | What it adds |
|---|---|
| `dashboard/metrics.py` | Reads TensorBoard event files (EventAccumulator) and tail-parses training log; ETA via epoch-wall-time averaging; `ips` unit corrected (samples/sec ÷ batch_size = iters/sec) |
| `dashboard/system.py` | GPU memory/utilization via `nvidia-smi`; CPU/RAM via `psutil`; 1-second cache |
| `dashboard/server.py` | FastAPI REST API: `/api/snapshot`, `/api/scalars/<tag>`, `/api/scalars/all`, `/api/system` |
| `dashboard/static/` | Single-page frontend: Plotly loss curves, canvas sparklines, live-refreshing metric cards, progress bar |
| `dashboard/launch.sh` | PID-managed launcher; writes `logs/dashboard.pid`; prevents duplicate instances |

### 7.3 Dependency Management (pyproject.toml)

`pyproject.toml` was extended with optional dependency groups:

```toml
[project.optional-dependencies]
train  = ["torch", "torchvision", "timm", "decord", "tensorboard", "psutil", "pyyaml"]
dashboard = ["fastapi", "uvicorn[standard]", "tensorboard", "psutil", "pyyaml"]
```

Upstream uses a flat `requirements.txt` with no group separation. The optional groups allow a clean `uv pip install -e ".[dashboard]"` install that does not pull in heavy training deps on inference-only machines.

### 7.4 Operational Documentation (new files)

| File | Purpose |
|---|---|
| `HANDOFF.md` | Session-continuity document: architecture decisions, gotchas, current run state |
| `RUNBOOK.md` | Operational cheat-sheet: copy-paste commands for start/stop/resume/checkpoint management |
| `REPORT.md` | This document: research context, implementation details, training progress |
| `scripts/health_check.sh` | Hourly cron health snapshot: process liveness, GPU state, log tail, disk usage |

### 7.5 Bug Fixes Applied to Upstream Code

| Fix | Commit | Description |
|---|---|---|
| Frozen-backbone key skip | `a1512fe` | `convert_stage1_weights.py` skipped SAM3 teacher keys during Stage 1 → student weight mapping |
| Geometry fine-tune | `6cf60db` | EdgeSAM-style prompt mixing for Stage 1+ geometry fine-tuning |
| MPS support | `1709977` | `get_autocast_device_type` + lazy Triton import for macOS inference |
| Weight norm functions | `2744d4c` | Fixed normalization function mapping in weight conversion scripts |

---

## 8. Planned Next Steps

### Stage 2 Completion

1. Resume training from current checkpoint; run to epoch 50 (or stop early if validation loss plateaus for 10+ consecutive epochs)
2. Evaluate final checkpoint on **DAVIS 2017** validation set; target J&F score of 75–82
3. Analyze per-epoch validation loss curves; confirm no overfitting

### Stage 3: ONNX Export and On-Device Deployment

1. Trace the full student model (RepViT backbone + TemporalPerceiver) with static shapes: `MAX_FRAMES=8`, `IMG_SIZE=1008`, `BATCH=1`
2. Validate inference correctness using `onnxruntime` on CPU
3. Convert ONNX model to Qualcomm AI Engine Direct (QNN SDK) format for Hexagon NPU
4. Deploy to Samsung Galaxy S26 Ultra; benchmark latency, peak memory, and segmentation quality relative to cloud-hosted SAM3 teacher

### Stage 4 (Optional)

Distill SAM3's concept/text grounding head using the SA-Co Gold dataset, enabling open-vocabulary prompted segmentation on-device. This stage is contingent on Stage 3 results and project scope decisions.

---

## 9. References

1. Ravi et al., "SAM 2: Segment Anything in Images and Videos," Meta AI Research, 2024
2. Wang et al., "RepViT: Revisiting Mobile CNN Training for Vision Foundation Models," 2023
3. Jaegle et al., "Perceiver IO: A General Architecture for Structured Inputs & Outputs," arXiv:2107.14795, 2021
4. Meta AI Research, SA-V Dataset, 2024
5. Qualcomm AI Engine Direct SDK Documentation, Qualcomm Technologies Inc.

---

*For operational details (how to resume training, monitor logs, checkpoint locations), see [HANDOFF.md](HANDOFF.md).*
