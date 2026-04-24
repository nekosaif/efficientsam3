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

**Final end-to-end throughput:** 2.67 iterations/second (up from 1.9 iter/s baseline), a **1.4× end-to-end speedup**. GPU compute is now the bottleneck; further data-side gains would not improve wall-clock time without a larger batch or additional GPU.

### 5.5 Current Training Status

| Parameter | Value |
|---|---|
| Launch date | 2026-04-24 16:34 local (resumed from ckpt_epoch_0) |
| Current status | Running — epoch 1 |
| Epochs completed | 0 of 50 (ckpt_epoch_0 is only saved epoch) |
| Loss trajectory | ~0.82 (normalized scale, start of ep1); expected slow decay |
| Throughput | ~2.67 iter/s |
| Estimated wall-clock (full run) | ~5.4 days on current hardware |
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

## 6. Planned Next Steps

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

## 7. References

1. Ravi et al., "SAM 2: Segment Anything in Images and Videos," Meta AI Research, 2024
2. Wang et al., "RepViT: Revisiting Mobile CNN Training for Vision Foundation Models," 2023
3. Jaegle et al., "Perceiver IO: A General Architecture for Structured Inputs & Outputs," arXiv:2107.14795, 2021
4. Meta AI Research, SA-V Dataset, 2024
5. Qualcomm AI Engine Direct SDK Documentation, Qualcomm Technologies Inc.

---

*For operational details (how to resume training, monitor logs, checkpoint locations), see [HANDOFF.md](HANDOFF.md).*
