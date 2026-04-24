# EfficientSAM3 Stage 2 — Runbook

Operational cheat-sheet. Copy-paste commands. Assume `cd /home/saif/github/efficientsam3`.

Also read `HANDOFF.md` (context) + `REPORT.md` (research) for background.

---

## 0. Current run (as of 2026-04-20 21:23)

| Field | Value |
|---|---|
| Run tag | `ep50_run1` |
| Train PID file | `logs/stage2_run1.pid` |
| TB PID file | `logs/tensorboard.pid` |
| Output dir | `output/efficient_sam3_stage2/ep50_run1/` |
| TB log dir | `output/efficient_sam3_stage2/ep50_run1/tb` |
| Train log | `logs/stage2_run1.log` |
| TB log | `logs/tensorboard.log` |
| Config | `stage2/configs/sav_repvit_m0_9.yaml` |
| Dataset | `/mnt/hdd/datasets/SA-V/` |
| Epochs | 50 (`TRAIN.EPOCHS 50`) |
| EMA | **ON**, decay 0.999 |
| Mid-epoch ckpt | **disabled** (SAVE_EVERY_ITERS=0) |
| Batch / GPU | 2 |
| Workers | 28 |
| Throughput | ~2.6 iters/s |
| Wall-clock | ~5.4 days for 50 ep |

---

## 1. Check training status

### Is it alive?
```bash
ps -fp $(cat logs/stage2_run1.pid) 2>/dev/null | tail -1 || echo DIED
```

### GPU usage
```bash
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
```

### Latest log lines
```bash
tail -40 logs/stage2_run1.log
```

### Loss trajectory (last 30 log points)
```bash
grep -Eo "loss=[0-9.]+" logs/stage2_run1.log | tail -30
```

### Validation loss (logged every 5 epochs)
```bash
grep -Ei "val[_-]?loss|val loss|ema" logs/stage2_run1.log
```

### Current epoch / iter
```bash
grep -Eo "ep[0-9]+ it[0-9]+/[0-9]+" logs/stage2_run1.log | tail -5
```

### Crash scan
```bash
grep -Ei "Traceback|CUDA|OOM|Killed|RuntimeError|Error" logs/stage2_run1.log | tail -20
```

### Hourly health snapshot (if cron active)
```bash
tail -40 logs/health.log
```

---

## 2. TensorBoard (port 6007, --bind_all)

### Is TB alive?
```bash
ps -fp $(cat logs/tensorboard.pid) 2>/dev/null | tail -1 || echo DIED
ss -ltn | grep 6007 || echo "NOT LISTENING"
```

### Open in browser
```
http://<host-ip>:6007/
# or locally:  http://localhost:6007/
# from same LAN: http://srbd-pc:6007/
```

### Start TB (if not running)
```bash
source /home/saif/venvs/tf/bin/activate
nohup tensorboard \
  --logdir output/efficient_sam3_stage2/ep50_run1/tb \
  --port 6007 --bind_all \
  > logs/tensorboard.log 2>&1 &
echo $! > logs/tensorboard.pid
```

### Stop TB
```bash
kill $(cat logs/tensorboard.pid) 2>/dev/null
sleep 2
pkill -f "tensorboard.*6007" 2>/dev/null
```

---

## 3. Resume training (auto-resume enabled)

Training has `AUTO_RESUME: true` → relaunching same command picks up from `output/.../latest.pth`.

```bash
cd /home/saif/github/efficientsam3

# verify nothing running first
pgrep -af "stage2/train.py" && echo "ALREADY RUNNING — do not relaunch" && exit 1

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

sleep 20 && tail -30 logs/stage2_run1.log
```

If port 29510 busy, pick another (29511, 29512…).

---

## 4. Stop training

### Graceful (saves mid-epoch ckpt if SAVE_EVERY_ITERS triggers)
```bash
pkill -TERM -f "stage2/train.py"
sleep 10
pgrep -af "stage2/train.py" && pkill -KILL -f "stage2/train.py"
```

### Verify dead
```bash
pgrep -af "stage2/train.py" || echo "STOPPED"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

### Stop hourly cron
```bash
crontab -l | grep -v health_check | crontab -
```

---

## 5. Fresh start (wipe all state → ep0)

**WARNING: destroys all checkpoints + TB logs. Confirm before running.**

```bash
cd /home/saif/github/efficientsam3

# 1. stop anything running
pkill -TERM -f "stage2/train.py" 2>/dev/null
pkill -TERM -f "tensorboard.*6007" 2>/dev/null
sleep 5
pkill -KILL -f "stage2/train.py" 2>/dev/null
pkill -KILL -f "tensorboard.*6007" 2>/dev/null

# 2. wipe outputs + logs
rm -rf output/efficient_sam3_stage2/ep50_run1
rm -f logs/stage2_run1.log logs/stage2_run1.pid
rm -f logs/tensorboard.log logs/tensorboard.pid

# 3. relaunch training (same as §3)
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

# 4. relaunch TB (same as §2)
source /home/saif/venvs/tf/bin/activate
nohup tensorboard \
  --logdir output/efficient_sam3_stage2/ep50_run1/tb \
  --port 6007 --bind_all \
  > logs/tensorboard.log 2>&1 &
echo $! > logs/tensorboard.pid

# 5. confirm
sleep 20
tail -30 logs/stage2_run1.log
ss -ltn | grep 6007
```

Expected early loss: starts ~0.82 (normalized MSE+cosine), slow decay over training.

---

## 6. Checkpoint locations

```
output/efficient_sam3_stage2/ep50_run1/
├── ckpt_epoch_0.pth        # per-epoch (only epoch-boundary saves)
├── ckpt_epoch_1.pth
├── ...
├── ckpt_latest.pth         # symlink → auto-resume target
├── best.pth                # best val loss so far
└── tb/                     # tensorboard events
```

Note: mid-epoch checkpoints disabled (`SAVE_EVERY_ITERS=0`). Resume always starts from epoch boundary.

Contents: only trainable Perceiver (5.02M params) + optimizer + scheduler + EMA shadow. Backbone + teacher **not saved** (frozen, load from disk).

### Clear mid-epoch ckpts only (keep epoch ckpts)
```bash
rm -f output/efficient_sam3_stage2/ep50_run1/ckpt_iter_*.pth
```

### Keep only latest + best
```bash
find output/efficient_sam3_stage2/ep50_run1/ -name "ckpt_epoch_*.pth" | \
  sort -V | head -n -1 | xargs -r rm -f
```

---

## 7. Key files reference

| Path | Purpose |
|---|---|
| `stage2/train.py` | DDP entrypoint |
| `stage2/config.py` | Default config (CfgNode) |
| `stage2/configs/sav_repvit_m0_9.yaml` | Run config override |
| `stage2/models/perceiver.py` | TemporalPerceiver (5.02M) |
| `stage2/models/ema.py` | ModelEma (Perceiver smoothing) |
| `stage2/data/sav_dataset.py` | SA-V decord loader |
| `stage2/utils/checkpoint.py` | Save/load incl. EMA shadow |
| `checkpoints/efficient_sam3_repvit_s.pt` | Stage 1 student (frozen) |
| `/mnt/hdd/checkpoints/sam3/sam3.pt` | SAM3 ViT-H teacher (frozen) |
| `data/sav_index_v2.pkl` | SA-V index cache (50,583 videos) |

---

## 8. Common gotchas

- **`tracker.backbone` is `None`** — use `vision_backbone` directly + apply `conv_s0`/`conv_s1` manually.
- **ONNX needs `need_weights=False`** on every `MultiheadAttention` (already done in Perceiver).
- **decord `get_batch(indices)`** — only decodes sampled 8 frames. Don't decode whole video.
- **Workers=28 is sweet spot** on Ryzen 9 9950X. 32 oversubscribes, throughput drops.
- **Pre-decode to .npy blocked** — needs ~1.2 TB, only 993 GB free on `/mnt/hdd`.
- **Stage 1 ckpt path:** `./checkpoints/efficient_sam3_repvit_s.pt` (NOT `/mnt/hdd/checkpoints/sam3/...`).
- **IMG_SIZE=1008** fixed (SAM3 RoPE). **MAX_FRAMES=8** fixed (ONNX static).
- **If GPU mem stuck after kill:** `nvidia-smi` → find orphan PID → `kill -9 <pid>`.

---

## 9. Hardware

- GPU: 1× NVIDIA RTX PRO 6000 Blackwell, 96 GB
- CPU: AMD Ryzen 9 9950X (16C/32T)
- RAM: 128 GB DDR5
- Storage: `/mnt/hdd` 3.6 TB (993 GB free)

---

## 10. Next stages (post Stage 2)

1. **Eval on DAVIS 2017 val** — target J&F 75–82
2. **Stage 3 ONNX export** — trace with `MAX_FRAMES=8, IMG_SIZE=1008, BATCH=1`; validate on CPU via onnxruntime; convert for Hexagon via Qualcomm AI Engine Direct (QNN SDK)
3. **On-device test** — Samsung Galaxy S26 Ultra — latency / memory / quality vs cloud SAM3
4. **(Optional) Stage 4** — SA-Co Gold concept/text head distill for open-vocab PCS

---

## 11. Contact

- User: Mollah Md Saif · mollahmdsaif@gmail.com
- Repo: `/home/saif/github/efficientsam3` · branch `main`
