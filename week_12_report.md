# **Weekly Progress Report**

### **Work Completed (This Week) – 29th April, 2026**

---

### **1\. SAM3 Distillation – Stage 2 Updates**

#### **Stage-wise Progress Overview**

| Stage | Objective | Status |
| :---: | :---- | :---- |
| **1** | Distill ViT-H image encoder → RepViT-M0.9; distill text encoder | Complete |
| **2** | Distill SAM3 sequential memory bank → TemporalPerceiver (static, parallel) | In progress (~36% of 50 epochs, ep18 in progress) |

---

#### **Training Progress (This Week: Apr 21 → Apr 29)**

Training continued from epoch 0 (clean run started Apr 21) through epoch 17 completed (Apr 28), with epoch 18 in progress.

**Validation Metrics (logged every 5 epochs during training):**

| Epoch | Val Total | Val MSE | Val Cosine | EMA Total | EMA MSE | EMA Cosine |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ep4 | 0.3876 | 0.0030 | 0.3846 | 0.3669 | 0.0028 | 0.3640 |
| ep9 | **0.3461** | 0.0027 | 0.3434 | **0.3443** | 0.0027 | 0.3416 |
| ep14 | 0.3488 | 0.0027 | 0.3461 | 0.3423 | 0.0027 | 0.3396 |

* **Best checkpoint:** `ckpt_best.pth` → epoch 9 (val total = **0.3461**)
* EMA (Exponential Moving Average, decay=0.999) consistently outperforms raw checkpoint metrics
* Val loss plateaued slightly between ep9–ep14 (difference < 1%), indicating late-stage refinement

**Training Loss Progression:**

* ep1 start: \~0.82 → rapid drop within first epoch
* ep2 onwards: stabilized in range \~0.29–0.48
* ep18 (current): loss \~0.32–0.40, lr = 8.03e-05, throughput \~2.6 iterations/sec

**Throughput:** \~2.6 ips (\~5.3 hours/epoch) on single Blackwell 96GB GPU.

---

#### **Key Implementations Completed This Week**

**1\. L2-Normalized Distillation Loss**

Distillation loss uses channel-wise L2 normalization for scale invariance:

* MSE computed on normalized feature maps → invariant to magnitude drift across training stages
* Total loss = L2-MSE + cosine term (1 − cos\_sim)
* MSE range \[0, 4\], cosine range \[0, 2\]

**2\. TemporalPerceiver EMA**

Exponential Moving Average applied to TemporalPerceiver weights (decay=0.999):

* EMA shadow weights maintained alongside raw weights
* EMA evaluated separately during val passes
* EMA consistently shows 1–3% lower val loss than raw

**3\. Mid-Epoch Checkpoint Recovery**

`SAVE_EVERY_ITERS` config guards against SIGHUP / job interruption:

* Saves mid-epoch checkpoint at fixed iteration intervals
* Auto-resume wired; training can restart from mid-epoch state without full epoch loss

**4\. Trainable-Only Checkpoint Saving**

Checkpoints save only TemporalPerceiver (trainable) parameters, not full model:

* Perceiver parameters: \~5.02M
* Checkpoint size: \~77 MB (vs \~1.5 GB for full model)
* RepViT-M0.9 backbone (frozen, 107 BN layers) not included in checkpoint

**5\. StudentTemporalModel.train() Override**

Override keeps frozen backbone BatchNorm layers in eval mode during training:

* Prevents BN running stats corruption in frozen backbone during training updates
* Perceiver layers remain in train mode as expected

---

#### **Evaluation Fix: Standalone Eval BN Discrepancy**

**Issue discovered:** Standalone `--eval` mode on ep1–ep17 checkpoints produced anomalously high loss (total ≈ 1.01 vs training-time val 0.3461 for ep9).

**Root cause:** During ep1–ep17 training, backbone BN layers were in TRAIN mode (per-batch statistics). The Perceiver learned on batch-normalized features. When standalone eval calls `student.eval()`, backbone BN switches to accumulated running-stats mode, changing the feature distribution and pushing cosine similarity to ≈ 0.

**Fix implemented:** `--backbone-bn-train-mode` CLI flag added to `stage2/train.py`:

* Keeps backbone BN in TRAIN mode during eval pass for legacy checkpoints
* Uses per-batch statistics matching the training distribution exactly
* Required for all ep1–ep17 checkpoints; ep18+ checkpoints (with `.train()` override active) do not need this flag

**Usage:**
```bash
torchrun --nproc_per_node=1 stage2/train.py \
  --cfg stage2/configs/sav_repvit_m0_9.yaml \
  --data-path /mnt/hdd/datasets/SA-V/ \
  --tag ep50_run1 \
  --resume output/efficient_sam3_stage2/ep50_run1/ckpt_epoch_17.pth \
  --eval \
  --backbone-bn-train-mode
```

---

#### **Summary**

* Stage 2 training reached epoch 17 (completed) with epoch 18 in progress (\~36% of 50 epochs total)
* Best validation total loss: **0.3461** at epoch 9 (cos\_sim ≈ 65.7%)
* EMA shadow consistently tracks better than raw: ep14 EMA = **0.3423** vs raw 0.3488
* Training loss stabilized in 0.29–0.48 range — slow refinement phase confirmed
* Standalone eval BN discrepancy identified and fixed via `--backbone-bn-train-mode` flag
* Validation plan unchanged: DAVIS J\&F score target 75–82 upon Stage 2 completion (Stage 3 territory)
