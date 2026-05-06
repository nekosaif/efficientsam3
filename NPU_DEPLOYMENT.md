# EfficientSAM3 — NPU Deployment Guide for Mobile Optimization Team

**Model:** EfficientSAM3 (RepViT-M0.9 backbone + TemporalPerceiver)
**Target:** Snapdragon 8 Elite for Galaxy — Hexagon HTP (NPU)
**Contact:** Mollah Md Saif · mollahmdsaif@gmail.com

---

## 1. Model Overview

EfficientSAM3 is a distilled video segmentation model derived from Meta's SAM3 (827M params → ~5M trainable student).

### Architecture

```
Input: [1, 8, 3, 1008, 1008]  (batch=1, frames=8, RGB, 1008×1008)
         ↓
RepViT-M0.9 backbone  (~383M params, frozen)
         ↓
frame_tokens: [1, 3969, 256] per frame  (stride-16 features, 63×63)
         ↓
TemporalPerceiver  (5.02M params — the trained student)
  ├── 4× CrossAttnBlock  (latents=64, heads=8, dim=256)
  │     ├── cross-attn: latents[64] ← tokens[3969]
  │     ├── self-attn:  latents[64] ← latents[64]
  │     └── MLP (GELU, dim 256→1024→256)
  └── out-attn: tokens[3969] ← latents[64]  → pix_feat [1, 256, 63, 63] per frame
         ↓
Output: [1, 8, 256, 63, 63]  (memory-conditioned features, all 8 frames)
```

### Why these shapes are fixed

- **IMG_SIZE=1008** — locked by SAM3 teacher's Rotary Position Embeddings. Cannot be changed without retraining from scratch.
- **MAX_FRAMES=8** — fixed for static ONNX graph (NPU requirement). No dynamic sequence length.
- **Batch=1** — on-device inference only.

---

## 2. ONNX Export Instructions (ML engineer side)

### Step 1 — disable fused SDPA before export

PyTorch 2.x routes `nn.MultiheadAttention(need_weights=False)` through `scaled_dot_product_attention`, which exports as a single fused op that QNN HTP may not recognize. Force decomposition to explicit matmul+softmax:

```python
import torch

# CRITICAL: set before torch.onnx.export
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
```

### Step 2 — export with static shapes

```python
model.eval()
dummy = torch.randn(1, 8, 3, 1008, 1008)

torch.onnx.export(
    model,
    dummy,
    "efficientsam3.onnx",
    opset_version=13,
    dynamic_axes=None,           # fully static — no dynamic_axes
    input_names=["frames"],
    output_names=["pix_feat_with_mem"],
    do_constant_folding=True,
)
```

### Step 3 — simplify graph

```bash
pip install onnxsim
python -m onnxsim efficientsam3.onnx efficientsam3_sim.onnx
```

### Step 4 — verify

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("efficientsam3_sim.onnx", providers=["CPUExecutionProvider"])
dummy = np.random.randn(1, 8, 3, 1008, 1008).astype(np.float32)
out = sess.run(None, {"frames": dummy})
print(out[0].shape)  # expect (1, 8, 256, 63, 63)
```

---

## 3. Model-Specific QNN Flags (Mobile team)

### 3.1 MultiheadAttention ops

The Perceiver contains 3 types of `nn.MultiheadAttention`:

| Op | Q shape | K/V shape | Notes |
|---|---|---|---|
| `cross_attn` | [1, 64, 256] | [1, 3969, 256] | latents attend to frame tokens |
| `self_attn` | [1, 64, 256] | [1, 64, 256] | latents self-attend — small, safe |
| `out_attn` | [1, 3969, 256] | [1, 64, 256] | tokens attend to latents |

**Attention matrices are small** (64×3969 = 254K elements) — not 3969×3969. Cross-attention memory is not the bottleneck.

After SDPA decomposition (Step 2 above), these export as: `MatMul → Div → Softmax → MatMul`. QNN HTP handles this natively.

If QNN still flags attention ops, try setting `need_weights=True` in export wrapper (adds dummy output, forces legacy codepath):

```python
# export-only wrapper — does not affect trained weights
class ExportWrapper(nn.Module):
    def forward(self, x):
        # replace need_weights=False → True in all MHA calls
        ...
```

### 3.2 GELU activation

Model uses `nn.GELU()` in all MLP blocks. If HTP flags it:

- Preferred fix: `nn.GELU(approximate='tanh')` — tanh approximation, hardware-friendlier, weights unchanged
- Fallback: replace with `nn.ReLU()` — minor quality loss, maximum compatibility

This change does not require retraining; weights are compatible.

### 3.3 LayerNorm

Present in every `_CrossAttnBlock` (`norm_q1`, `norm_kv1`, `norm2`, `norm3`) and output head (`norm_out`). Total: ~10 LayerNorm ops.

QNN HTP supports LayerNorm in FP16. If using INT8 quantization, LayerNorm should run in FP16 (mixed precision). Set HTP quantization override for LayerNorm nodes to FP16.

### 3.4 Hardcoded spatial dimensions

The export wrapper should hardcode `H=W=63` (derived from 1008÷16=63) rather than computing `isqrt(HW)` at runtime:

```python
# in export wrapper — replace dynamic isqrt with constant
H, W = 63, 63  # fixed for IMG_SIZE=1008, stride=16
pix_feat = tokens.transpose(1, 2).reshape(B, C, H, W)
```

This removes a shape-inference op that can confuse QNN graph compiler.

---

## 4. Peak Memory Profile

### Feature maps

| Stage | Shape | FP16 size |
|---|---|---|
| Input (8 frames) | [1, 8, 3, 1008, 1008] | ~46 MB |
| Backbone output per frame | [1, 256, 63, 63] | ~2 MB |
| All 8 frame tokens | [1, 8, 3969, 256] | ~16 MB |
| Perceiver latents | [1, 64, 256] | ~0.03 MB |
| Output features | [1, 8, 256, 63, 63] | ~16 MB |

### Backbone peak SRAM (RepViT at 1008px)

RepViT-M0.9 intermediate activations at 1008px input:

- Early layers (stride 2, 3): ~252×252 feature maps — **largest memory pressure**
- Estimated peak backbone SRAM: **40–60 MB** (FP16)

This is the primary memory concern. IMG_SIZE cannot be reduced (RoPE constraint).

**Recommendation:** verify HTP SRAM tiling strategy handles 252×252 early feature maps. If OOM, discuss with ML engineer — would require architecture redesign and full retraining.

### Perceiver attention memory

Cross-attn matrix (per head): 64×3969×2 bytes = ~0.5 MB × 8 heads = 4 MB per block × 4 blocks = **16 MB peak** for all attention blocks. Acceptable.

---

## 5. Quantization Recommendations

### Target: INT8 weights + FP16 activations (W8A16)

Best balance of NPU performance and segmentation quality. Full INT8 activations risk visible mask quality degradation on fine edges.

### Calibration dataset

Use ~200–500 frames from SA-V validation set (`/mnt/hdd/datasets/SA-V/`). Segmentation models are sensitive to calibration — use diverse scenes with:
- Fine object boundaries
- Multiple objects per frame
- Varying illumination

### Sensitive layers (keep in FP16)

| Layer | Reason |
|---|---|
| All `LayerNorm` | Scale/bias sensitive to quantization |
| `out_attn` (final output projection) | Directly produces distillation target |
| `mlp_out` (output MLP) | Last layer before pixel features |
| `Softmax` in attention | Numerical stability |

### QAT note

Current model was trained in FP32/BF16 without quantization-aware training. If PTQ (post-training quantization) quality is insufficient (visible mask degradation), request QAT fine-tune from ML engineer — can be done in ~5–10 epochs on SA-V.

---

## 6. Recommended QNN Conversion Flow

```
efficientsam3_sim.onnx
        ↓
qnn-onnx-converter \
  --input_network efficientsam3_sim.onnx \
  --input_dim frames "1,8,3,1008,1008" \
  --output_path efficientsam3.cpp \
  --float_fallback              # fallback risky ops to FP16
        ↓
qnn-model-lib-generator \
  --model efficientsam3.cpp \
  --backend HTP
        ↓
qnn-net-run \
  --model libefficientsam3.so \
  --backend libQnnHtp.so \
  --input_list input_list.txt
```

### Flags to try if conversion fails

```bash
# if SDPA op not recognized — should be fixed by export step, but fallback:
--op_package_config sdpa_override.json

# if LayerNorm fails:
--act_bw 16                     # force FP16 activations

# if backbone memory exceeds HTP SRAM:
--set_output_tensors backbone_out  # partition: run backbone on GPU, Perceiver on HTP
```

---

## 7. Two-Model Export Strategy

Provide both versions:

| File | Config | Use |
|---|---|---|
| `efficientsam3_fp16.onnx` | FP32 export, no quantization | First conversion test, quality baseline |
| `efficientsam3_int8.onnx` | W8A16 PTQ quantized | Production NPU target |

---

## 8. Sync Checklist (before conversion starts)

Questions for ML engineer:

- [ ] SDPA decomposition confirmed in exported ONNX? (check: no `ScaledDotProductAttention` node in netron)
- [ ] `onnxsim` applied?
- [ ] ONNX Runtime CPU inference verified? (shape = `[1, 8, 256, 63, 63]`)
- [ ] GELU variant — `approximate='tanh'` or standard?

Questions for mobile engineer:

- [ ] INT8 or W8A16 target?
- [ ] HTP-only or GPU fallback allowed?
- [ ] HTP SRAM budget — can it tile 252×252 early RepViT feature maps?
- [ ] QNN SDK version? (affects supported op set)
- [ ] Any known op blacklist from previous projects?

---

## 9. Key Files

| Path | Purpose |
|---|---|
| `stage2/models/perceiver.py` | TemporalPerceiver source |
| `checkpoints/efficient_sam3_repvit_s.pt` | Stage 1 student checkpoint (backbone + Perceiver) |
| `stage2/configs/sav_repvit_m0_9.yaml` | Training config |
| `REPORT.md` | Architecture + training methodology |
| `HANDOFF.md` | ML engineer operational notes |

---

## 10. Known Non-Issues

These look risky but are actually fine for this model:

- **Large token count (3969)** — cross-attention is latents×tokens (64×3969), not tokens×tokens. Memory is bounded.
- **8-frame input** — processed in a single parallel pass through Perceiver (not sequential RNN). Static graph throughout.
- **256 channels** — standard, well within QNN op support.
- **`contiguous()` calls in forward** — stripped by ONNX export, no effect.
