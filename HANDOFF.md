# EfficientSAM3 — Handoff for a New LLM Agent

**Purpose:** if the user shifts this project to a different agent, this document is the single entry point to pick up without re-deriving state.

**Read order:**
1. This file (HANDOFF.md) — operational state, where things are, what to do next
2. `REPORT.md` — research context, architecture, I/O sweep, methodology
3. `RUNBOOK.md` — copy-paste shell commands
4. `README.md` — user-facing project overview
5. `CLAUDE.md` / `.claude/MEMORY.md` — prior-agent memory (may or may not be present for you)

---

## 1. Current State Snapshot — 2026-05-14 13:00 +0600

- **Stage 2 (Temporal Memory Distillation) training is RUNNING.** Run tag **`ep50_run3`** (fresh restart from epoch 0).
- Launched: 2026-05-14 12:53 local
- Output dir: `output/efficient_sam3_stage2/ep50_run3/`
- TensorBoard logs: `output/efficient_sam3_stage2/ep50_run3/tb/`
- Train log: `logs/stage2_run3.log`
- Train PID file: `logs/stage2_run3.pid`
- Target: 50 epochs (~11 days at 2.7 ips, ~5.3 h/epoch)
- Dashboard: **http://localhost:6007** (also reachable on LAN at `http://<host-ip>:6007`)

If the process has died, see §5 (Resume).

### Why ep50_run3 (and not run2)

**`ep50_run2` collapsed silently and was archived.** Investigation 2026-05-14 found that the saved checkpoints (`ckpt_epoch_0…3.pth`) contain Perceiver weights in a degenerate regime: `mlp_out.2.weight` std grew from healthy init 0.018 to **1.09** (~50× too large). Standalone `--eval` on those checkpoints reproduces `s_norm ≈ 12,000` per token vs teacher's ~15 (s/t ≈ 800×), i.e. complete output direction inversion.

The training-time log nevertheless showed `loss ≈ 0.36` because the L2-normalized MSE and cosine terms are *scale-invariant* — they cannot detect a magnitude blowup; only `norm_loss = log(s/t)²` can, and with the old `NORM_WEIGHT=0.25` it was too weak a signal to prevent the attractor.

Archived to `archive/stage2_failed_run2_collapsed/{ckpts,logs,tb}/` for forensics (309 MB).

---

## 2. Anti-Collapse Changes Active in run3

| Knob | run2 (collapsed) | run3 (active) | Why |
|---|---|---|---|
| `DISTILL.NORM_WEIGHT` | 0.25 | **2.0** | 8× stronger log-ratio regularizer on `||s||/||t||`; the only loss term that sees magnitude |
| `TRAIN.WEIGHT_DECAY` | 0.05 | **0.1** | extra decay pressure on `mlp_out`/`out_attn` weights (ndim≥2 are in the decay group) |
| `TRAIN.CLIP_GRAD` | 1.0 | **0.5** | tighter cap on Perceiver updates |
| `EVAL.EVERY_N_EPOCHS` | 5 | **1** | first held-out val at end of ep0 (~5.3 h) — early collapse detection |
| `DATA.NUM_WORKERS` | 28 | **16** | matches Ryzen 9 9950X core count; reduces dataloader RSS by ~43 % |
| `DATA.PREFETCH_FACTOR` | 4 | **2** | halves in-flight batch queue (~14 GB → ~7 GB) |
| `TRAIN.SAVE_EVERY_ITERS` | 0 | **2000** | mid-epoch atomic ckpt every ~12 min |
| OS: swap | none | **48 GiB swapfile** at `/swapfile` | OOM-killer guardrail (run2 was SIGKILLed during ep4) |

**New per-step TB scalars** (added to `stage2/train.py`):
- `train/grad_norm` — pre-clip gradient L2 norm
- `train/mlp_out_w_std` — `perceiver.mlp_out[2].weight.std()` — direct collapse signal
- `train/s_token_norm` — average per-token L2 norm of student output

`mlp_out_w_std` is the early-warning canary: in run2 it grew from 0.018 → 1.09 silently. In run3 it must stay near 0.02–0.05 throughout.

---

## 3. Project in One Paragraph

EfficientSAM3 distills Meta's SAM3 ViT-H video segmentation model into a mobile-deployable form for the Samsung Galaxy S26 Ultra's Qualcomm Hexagon NPU. ONNX export with static shapes requires `MAX_FRAMES=8`, `IMG_SIZE=1008`. Stage 1 (done) distilled the image encoder → RepViT-M0.9. Stage 2 (running) distills the sequential memory bank → a 4-layer TemporalPerceiver with 64 learnable latents. Stage 3 (next) = ONNX export + on-device test.

Dataset: **SA-V** (SAM2 video, 50,583 clips, 1.1 TB at `/mnt/hdd/datasets/SA-V/`). 49,594 train / 989 val via deterministic SHA1-hash split, `val_fraction=0.02`.

Distillation target: teacher's `pix_feat_with_mem` per frame. Loss: `MSE + cosine + 2.0 × norm_log_ratio`, all on L2-normalized features. Teacher + Stage-1 backbone both frozen; only the Perceiver (5.02 M params) has gradients.

---

## 4. Filesystem Map

| Path | Purpose |
|---|---|
| `/home/saif/github/efficientsam3/` | Project root |
| `REPORT.md` | Full research write-up |
| `RUNBOOK.md` | Shell-command cheat sheet |
| `HANDOFF.md` | This file |
| `stage1/` | Stage 1 code (done) |
| `stage2/` | Stage 2 code (active) |
| `stage2/configs/sav_repvit_m0_9.yaml` | **Active training config** |
| `stage2/train.py` | DDP entrypoint |
| `stage2/models/perceiver.py` | TemporalPerceiver |
| `stage2/data/sav_dataset.py` | SA-V dataset (decord + cv2 fallback) |
| `stage2/loss.py` | DistillLoss (MSE + cosine + log-ratio norm matching) |
| `stage2/utils/checkpoint.py` | Save/load incl. EMA shadow and backbone BN buffers |
| `checkpoints/efficient_sam3_repvit_s.pt` | Stage 1 student (frozen) |
| `/mnt/hdd/checkpoints/sam3/sam3.pt` | SAM3 ViT-H teacher (frozen) |
| `/mnt/hdd/datasets/SA-V/` | 60 tar files, 1.1 TB |
| `data/sav_index_v2.pkl` | Pre-built SA-V index cache |
| `output/efficient_sam3_stage2/ep50_run3/` | **Active run output** |
| `logs/stage2_run3.log` | Training stdout/stderr |
| `logs/stage2_run3.pid` | PID of active torchrun |
| `logs/dashboard.{log,pid}` | Dashboard process |
| `logs/health.log` | Hourly health snapshot (cron → `scripts/health_check.sh`) |
| `archive/stage2_failed_run2_collapsed/` | Old run2 (collapsed) kept for forensics |
| `dashboard/server.py` | FastAPI backend (incl. train control endpoints) |
| `dashboard/launch.sh` | Dashboard launcher (uses tf venv, port 6007) |
| `dashboard/static/index.html` | Dashboard UI (incl. Start/Stop control card) |

Hardware: 1× NVIDIA RTX PRO 6000 Blackwell (96 GB), AMD Ryzen 9 9950X (16 C/32 T), 128 GB DDR5 + **48 GB swapfile**, 562 GB free on `/mnt/hdd`.

---

## 5. How to Check Health

### From the dashboard
Open `http://localhost:6007` (or LAN IP). The status pill auto-refreshes every 5 s. Live `loss`, `mw` (mlp_out weight std — the collapse canary), `s_norm`, GPU util, RAM are all plotted.

### Quick CLI check
```bash
ps -fp $(cat /home/saif/github/efficientsam3/logs/stage2_run3.pid) 2>/dev/null | tail -1 || echo DIED
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
tail -40 /home/saif/github/efficientsam3/logs/stage2_run3.log
```

### Collapse early-warning grep
```bash
# mw (mlp_out.2.weight std) should stay below ~0.1 throughout training
grep -Eo "mw=[0-9.]+" /home/saif/github/efficientsam3/logs/stage2_run3.log | tail -20
# s_norm should hover near teacher's ~15 (allow 5-25 range)
grep -Eo "s_norm=[0-9.]+" /home/saif/github/efficientsam3/logs/stage2_run3.log | tail -20
```

### Validation loss (every 1 epoch now)
```bash
grep -Ei "\[stage2\]\[val" /home/saif/github/efficientsam3/logs/stage2_run3.log
```

### TensorBoard
```bash
source /home/saif/venvs/tf/bin/activate
tensorboard --logdir output/efficient_sam3_stage2/ep50_run3/tb --port 6008 --bind_all
# port 6008 because the dashboard already owns 6007
```

**Expected loss trajectory:** starts ~1.9 (because `total = mse + cos + 2*norm` and norm≈0.5 initially), drops to ~0.4–0.5 within ep0 as cos converges, then slowly to ~0.35 over 50 ep. **The new signal to watch is `mw` ≤ ~0.05** — if it ever climbs above 0.1, collapse has resumed.

---

## 6. How to Resume if Training Died

Training has `AUTO_RESUME: true` and `SAVE_EVERY_ITERS=2000` (mid-epoch ckpts every ~12 min). Either of these works:

### A. From the dashboard
Open `http://localhost:6007`. The Start button auto-labels as **"Continue"** when a resumable checkpoint is present. Click it.

### B. CLI
```bash
cd /home/saif/github/efficientsam3
pgrep -af "stage2/train.py" && echo "ALREADY RUNNING" && exit 1

nohup torchrun --nproc_per_node=1 --master_port 29510 stage2/train.py \
  --cfg stage2/configs/sav_repvit_m0_9.yaml \
  --data-path /mnt/hdd/datasets/SA-V/ \
  --tag ep50_run3 \
  --output output \
  > logs/stage2_run3.log 2>&1 &
echo $! > logs/stage2_run3.pid
```

All anti-collapse settings (NORM_WEIGHT, CLIP_GRAD, WEIGHT_DECAY, EVAL frequency, worker count) live in the YAML now, so no `--opts` overrides needed.

---

## 7. How to Stop Training

### From the dashboard
Click **Stop**. Two confirmation prompts (intentional, since 50 ep ≈ 11 days). The backend sends SIGTERM to the process group, then SIGKILL after 10 s if needed.

### CLI
```bash
pkill -TERM -f "stage2/train.py"
sleep 10
pgrep -af "stage2/train.py" && pkill -KILL -f "stage2/train.py"
```

---

## 8. Dashboard

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Single-page UI (Plotly + sparklines) |
| `/api/snapshot` | GET | Current train state + system telemetry |
| `/api/scalars?tag=…` | GET | TB scalar series with downsampling |
| `/api/scalars/all` | GET | All TB scalar tag names |
| `/api/train/status?tag=…` | GET | `{alive, pid, tag, has_resumable_ckpt, ...}` |
| `/api/train/start` | POST | Body `{tag, master_port?, cfg?, ...}` — launches torchrun via `subprocess.Popen(start_new_session=True)`. 409 if already running. |
| `/api/train/stop` | POST | SIGTERM the process group, SIGKILL fallback after 10 s. |

Tag whitelist: `[A-Za-z0-9_-]{1,64}`. UI requires two confirms for Stop, one for Start.

Launcher: `dashboard/launch.sh` uses `/home/saif/venvs/tf` (has fastapi, uvicorn, tensorflow) and port 6007.

---

## 9. Known Gotchas (Updated)

- **`mw` (mlp_out_w_std) is the collapse canary.** Healthy init is 0.018. In the failed run2 it grew to 1.09 (50×). New regularizer holds it flat in run3 — but always check before trusting any new checkpoint.
- **L2-normalized loss is scale-invariant** — MSE + cosine alone cannot detect magnitude drift. The `NORM_WEIGHT × log(s/t)²` term is the only magnitude sensor. Do not lower NORM_WEIGHT without restoring an unnormalized MSE.
- **`tracker.backbone` is `None`** on the composed SAM3 video model — must call `vision_backbone` directly and apply `conv_s0` / `conv_s1` manually. See `stage2/models/teacher.py:_backbone_per_frame`.
- **ONNX export requires `need_weights=False`** on every `MultiheadAttention` — already done in the Perceiver.
- **SA-V video decode**: decord `get_batch(indices)` only decodes the 8 sampled frames. Don't accidentally decode the whole video.
- **Worker count: 16 is the sweet spot** on this CPU (16 physical cores). The previous 28 caused dataloader RSS spikes that triggered the OOM killer in run2 (SIGKILL during ep4).
- **Pre-decoding SA-V to .npy frames was ruled out** — would need ~1.2 TB; only 562 GB free on `/mnt/hdd`.
- **`SAVE_EVERY_ITERS=2000`** is now on; expect `ckpt_running.pth` to appear within ~12 min of launch.
- **`StudentTemporalModel.train()` override** keeps the frozen backbone in eval mode regardless of student mode — preserves Stage-1 BN running stats. Don't remove this without a coordinated fix to the eval pathway.

---

## 10. After Stage 2 Finishes

When epoch 50 completes, or val loss plateaus for 10+ consecutive epochs:

1. **Eval on DAVIS 2017 val** — target J&F 75–82.
2. **Stage 3 — ONNX export** — trace student (RepViT + Perceiver) with static shapes `MAX_FRAMES=8, IMG_SIZE=1008, BATCH=1`. Validate via onnxruntime CPU. Convert with QNN SDK for Hexagon HTP.
3. **On-device test** on Samsung Galaxy S26 Ultra — latency / memory / quality vs cloud teacher. See `NPU_DEPLOYMENT.md`.

Optional **Stage 4** (SA-Co Gold, concept/text head) only if open-vocabulary PCS on-device is required.

---

## 11. Contact / Meta

- User: Mollah Md Saif (mollahmdsaif@gmail.com)
- Primary working dir: `/home/saif/github/efficientsam3`
- Git branch: `main`
- Search scope restriction: project dir + `/mnt/hdd/datasets/` only
