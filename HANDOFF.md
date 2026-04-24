# EfficientSAM3 — Handoff for a New LLM Agent

**Purpose:** if the user shifts this project to a different agent, this document is the single entry point to pick up without re-deriving state.

**Read order:**
1. This file (HANDOFF.md) — operational state, where things are, what to do next
2. `REPORT.md` — research context, architecture, I/O sweep, methodology
3. `README.md` — user-facing project overview
4. `CLAUDE.md` / `.claude/MEMORY.md` — prior-agent memory (may or may not be present for you)

---

## 1. Current State Snapshot — 2026-04-24 16:34 +0600

- **Stage 2 (Temporal Memory Distillation) training is RUNNING.**
- Launched: 2026-04-24 16:34 local, resuming from `ckpt_epoch_0.pth`
- Current loss: ~0.82 (normalized scale; see §7 gotchas for scale change)
- Target: **50 epochs**, ~5.4 days wall-clock on the current single GPU
- Output dir: `output/efficient_sam3_stage2/ep50_run1/`
- TensorBoard: `output/efficient_sam3_stage2/ep50_run1/tb/`
- Log: `logs/stage2_run1.log`

If the process has since died, see §5 (Resume).

---

## 2. Project in One Paragraph

EfficientSAM3 distills Meta's SAM3 ViT-H video segmentation model into a mobile-deployable form for the Samsung Galaxy S26 Ultra's Qualcomm Hexagon NPU. ONNX export with static shapes requires `MAX_FRAMES=8`, `IMG_SIZE=1008`. Stage 1 (done) distilled the image encoder → RepViT-M0.9. Stage 2 (running) distills the sequential memory bank → a 4-layer TemporalPerceiver with 64 learnable latents. Stage 3 (next) = ONNX export + on-device test.

Dataset: **SA-V** (SAM2 video dataset, 50,583 videos, 1.1 TB at `/mnt/hdd/datasets/SA-V/`). 49,594 train / 989 val via deterministic SHA1-hash split, `val_fraction=0.02`.

Distillation target: teacher's `pix_feat_with_mem` per frame. Loss: MSE + cosine, equal weights. Teacher + Stage-1 backbone both frozen; only the Perceiver (5.02 M params) has gradients.

---

## 3. Filesystem Map

| Path | Purpose |
|------|---------|
| `/home/saif/github/efficientsam3/` | Project root |
| `REPORT.md` | Full research write-up |
| `HANDOFF.md` | This file |
| `stage1/` | Stage 1 code (done) |
| `stage2/` | Stage 2 code (active) |
| `stage2/configs/sav_repvit_m0_9.yaml` | **Active training config** |
| `stage2/train.py` | DDP entrypoint |
| `stage2/models/perceiver.py` | TemporalPerceiver |
| `stage2/data/sav_dataset.py` | SA-V dataset (decord + cv2 fallback) |
| `stage2/bench_io.py` | I/O throughput benchmark |
| `checkpoints/efficient_sam3_repvit_s.pt` | Stage 1 student (frozen) |
| `/mnt/hdd/checkpoints/sam3/sam3.pt` | SAM3 ViT-H teacher (frozen) |
| `/mnt/hdd/datasets/SA-V/` | 60 tar files, 1.1 TB |
| `data/sav_index_v2.pkl` | Pre-built SA-V index cache |
| `output/efficient_sam3_stage2/ep50_run1/` | **Active run output** |
| `logs/stage2_run1.log` | Training stdout/stderr |
| `logs/stage2_run1.pid` | PID of active torchrun |
| `logs/health.log` | Hourly health check (OS cron) |
| `scripts/health_check.sh` | Cron-invoked health probe |

Hardware: 1× NVIDIA RTX PRO 6000 Blackwell (96 GB), AMD Ryzen 9 9950X (16C/32T), 128 GB DDR5, 993 GB free on `/mnt/hdd`.

---

## 4. How to Check Health

### Quick live check
```bash
ps -fp $(cat /home/saif/github/efficientsam3/logs/stage2_run1.pid) 2>/dev/null | tail -1 || echo DIED
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
tail -40 /home/saif/github/efficientsam3/logs/stage2_run1.log
```

### Last hourly snapshot
```bash
tail -40 /home/saif/github/efficientsam3/logs/health.log
```

### Loss trend
```bash
grep -Eo "loss[^ ]*[ =:][0-9.]+" /home/saif/github/efficientsam3/logs/stage2_run1.log | tail -30
```

### Validation loss (every 5 epochs)
```bash
grep -Ei "val[_-]?loss" /home/saif/github/efficientsam3/logs/stage2_run1.log
```

### TensorBoard
```bash
tensorboard --logdir output/efficient_sam3_stage2/ep50_run1/tb --port 6006
# then open http://localhost:6006
```

**Expected loss:** starts ~0.82 (normalized MSE+cosine scale); slow decay over 50 epochs. Validate per-5-epoch eval shows val loss strictly decreasing (with noise).

---

## 5. How to Resume if Training Died

Training uses `AUTO_RESUME: true` — relaunching the exact same command picks up from the latest checkpoint automatically.

```bash
cd /home/saif/github/efficientsam3
# verify no existing process first
pgrep -af "stage2/train.py" || echo "nothing running"

# relaunch
nohup torchrun --nproc_per_node=1 --master_port 29510 stage2/train.py \
  --cfg stage2/configs/sav_repvit_m0_9.yaml \
  --data-path /mnt/hdd/datasets/SA-V/ \
  --tag ep50_run1 \
  --opts TRAIN.EPOCHS 50 \
         TRAIN.EMA_ENABLE True \
         TRAIN.EMA_DECAY 0.999 \
         TRAIN.SAVE_EVERY_ITERS 0 \
         TRAIN.CLIP_GRAD 0.3 \
  > logs/stage2_run1.log 2>&1 &
echo $! > logs/stage2_run1.pid
```

If the port is in use, pick a different `--master_port` (e.g. 29511).

Checkpoints live at `output/efficient_sam3_stage2/ep50_run1/ckpt_epoch_*.pth` with a `latest.pth` symlink. `AUTO_RESUME` uses `latest.pth`.

---

## 6. How to Stop Training

```bash
pkill -TERM -f "stage2/train.py"
sleep 5
pkill -KILL -f "stage2/train.py"  # if still alive
```

To stop the hourly cron:
```bash
crontab -l | grep -v health_check | crontab -
```

---

## 7. Known Gotchas

- `tracker.backbone` is `None` on the composed SAM3 video model — you must call `vision_backbone` directly and apply `conv_s0` / `conv_s1` manually. See `stage2_step3_complete.md` in `.claude/projects/*/memory/` (if present).
- ONNX export requires `need_weights=False` on every `MultiheadAttention` — already done in the Perceiver but watch out when modifying.
- SA-V video decode: decord `get_batch(indices)` only decodes the 8 sampled frames. Don't accidentally decode the whole video.
- Worker count: **28 is the sweet spot** on this CPU. `24` is ok, `32` oversubscribes and throughput drops.
- Pre-decoding SA-V to `.npy` frames was ruled out — would need ~1.2 TB, only 993 GB free.
- Stage 1 checkpoint was moved under `./checkpoints/` — do NOT use `/mnt/hdd/checkpoints/sam3/efficient_sam3_repvit_s.pt` (older, now-wrong path).
- **Loss scale changed (2026-04-24):** `stage2/loss.py` now L2-normalizes student+teacher features along channel dim before MSE. Loss is now bounded ~[0, 6] regardless of teacher feature magnitude (which varied 30-50× across SA-V videos causing optimizer corruption). Old raw-MSE runs had initial loss ~2.33; current normalized runs start ~0.82.
- **No mid-epoch checkpoints:** `SAVE_EVERY_ITERS 0` — always resumes from epoch boundary. User preference: never resume mid-epoch.
- **Grad clip = 0.3** (reduced from 1.0 to prevent output-head explosion).

---

## 8. After Stage 2 Finishes

When epoch 50 completes, or the loss + val curves plateau earlier, the plan is:

1. **DAVIS 2017 eval** (Step 4 of original plan, not yet built): compute J&F on DAVIS val. Target ≈ J&F 75–82.
2. **Stage 3 — ONNX export**: trace the student (backbone + Perceiver) with static shapes `MAX_FRAMES=8, IMG_SIZE=1008, BATCH=1`. Verify on CPU with onnxruntime, then convert for Hexagon via Qualcomm AI Engine Direct (QNN SDK).
3. **On-device test** on S26 Ultra — latency, memory, quality vs cloud teacher.

Optional **Stage 4** (SA-Co Gold, concept/text head) only if you want open-vocab PCS on-device. Not required for the core video-segmentation use case.

---

## 9. What the User Typically Asks

- "Is training still alive / what's the loss?" → §4 Quick live check
- "Did it crash?" → check log tail for `Traceback|CUDA|OOM|Killed|RuntimeError`
- "How long until done?" → `iters/s × remaining iters`; baseline 2.67 ips, 24,797 iters/epoch, 50 epochs
- "Can we speed it up?" → remaining levers: `torch.compile` on the Perceiver, larger batch if VRAM permits, mixed-precision tweaks. Pre-decode still blocked on disk.
- "Should we stop early?" → yes if val loss plateaus for 10+ epochs

---

## 10. Contact / Meta

- User: Mollah Md Saif (mollahmdsaif@gmail.com)
- Primary working dir: `/home/saif/github/efficientsam3`
- Git branch: `main`
- Caveman-mode terse text preference active on the previous agent (see `.claude/` if relevant to you)
- Search scope restriction: search inside project dir + `/mnt/hdd/datasets/` only
