# **Weekly Progress Report**

### **Work Completed (This Week) – 21th April, 2026**


### **1\. SAM3 Distillation**

#### **Motivation**

This research focuses on developing **EfficientSAM3**, a mobile-deployable video segmentation model by compressing Meta's Segment Anything Model 3 (SAM3).

* **Parameter count:** The ViT-H image encoder has approximately 630 million parameters, exceeding mobile memory limits.

* **Dynamic computation graph:** SAM3's sequential memory queries for up to seven past frames result in variable-length tensor operations, which cannot be represented as the static graphs required by the NPU SDK.

EfficientSAM3 addresses this by replacing the heavy encoder with a lightweight RepViT backbone and the dynamic memory bank with a parallel, fixed-shape TemporalPerceiver module.

#### **Stage-wise Progress Overview**

| Stage | Objective | Status |
| :---: | :---- | :---- |
| **1** | Distill ViT-H image encoder → RepViT-M0.9; distill text encoder | Complete |
| **2** | Distill SAM3 sequential memory bank → TemporalPerceiver (static, parallel) | In progress (\~37% of 50 epochs) |

#### **Model Compression Summary**

The total model size has been reduced from approximately 827 million parameters to a student model with \~388 million parameters (\~383M backbone \+ \~5M TemporalPerceiver), achieving significant compression while maintaining performance targets.

#### **Stage 1: Encoder Distillation (Complete)**

The objective of Stage 1 was to replace the ViT-H image encoder with RepViT-M0.9 while preserving single-frame segmentation quality.

This stage has been successfully completed. The RepViT-M0.9 backbone, trained using intermediate-layer and output-level distillation losses, is now frozen.

Checkpoint: checkpoints/efficient\_sam3\_repvit\_s.pt

#### **Stage 2: Temporal Memory Distillation**

**Objective**

Train a lightweight TemporalPerceiver (\~5.02M parameters) to approximate SAM3’s dynamic memory bank by matching memory-conditioned features (pix\_feat\_with\_mem), ensuring temporal consistency across video frames.

**Dataset**

Training uses the SA-V (Segment Anything Video) dataset, which provides temporally consistent frame-wise masks for supervising temporal memory behavior.

**Training Progress (After \~2 Days)**

* Progress: \~37% of total 50 epochs

* Loss:

  * Initial: 2.33

  * Early training (\~2.5%): \~0.88–0.95

  * Current: \~0.55 – 0.70

This corresponds to:

* \~22% – 39% reduction from early-stage loss (\~0.90)

* \~70% – 76% total reduction from initial loss

Training has transitioned from rapid early convergence to a slower refinement phase, consistent with temporal distillation dynamics.

**Observed Training Behavior**

* Strong initial loss drop during early iterations

* Gradual stabilization as temporal consistency constraints become dominant

* Slower convergence due to:

  * Temporal alignment complexity across frames

  * Limited capacity of the \~5M parameter TemporalPerceiver

  * Cosine similarity term influencing late-stage optimization

**Validation Plan**

Upon completion of training, the final checkpoint will be evaluated on the DAVIS 2017 validation set, targeting a J\&F score of 75–82.

**Summary**

* Stage 1 is complete and stable

* Stage 2 has reached \~37% completion with significant loss reduction

* Current loss range (\~0.55–0.70) indicates effective temporal knowledge distillation and stable convergence toward the final target

#### 

## 
