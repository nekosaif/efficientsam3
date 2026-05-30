# EfficientSAM3 Stage 2 — Conversation Memory Dump

Snapshot as of **2026-05-28**. Use this to bring a fresh Claude session (any
model) up to speed on the project's state, decisions, and outstanding work.

---

## 1. Current state in one screen

| | |
|---|---|
| Active run tag | **`ep50_run4`** |
| Output dir | `output/efficient_sam3_stage2/ep50_run4/` |
| Training process | pid 2404965 (relaunched 2026-05-28 12:55) — auto-resume from `ckpt_running.pth` |
| Dashboard | http://localhost:6007 (LAN: http://172.18.157.182:6007), pid 2735647 |
| Last logged training position | ep15 it ~14400 / 24797 (~58 % through ep15) |
| **Best val** | **0.2907 (ep13)** ← `ckpt_best.pth` |
| EMA val (latest at ep14) | 0.2890 — still descending monotonically |
| Early-stop counter | 1 / 5 (after ep14) |
| `mw` (collapse canary) | 0.036, flat all run |
| `BAD BATCH` count | 0 over ~16 epochs of training |
| Run target | 50 epochs (early stop active, will exit if 5 consecutive epochs without new best) |

---

## 2. The biggest finding of this conversation — autocast cache bug

Two prior training attempts (`ep50_run2`, `ep50_run3`) silently corrupted
themselves. The Perceiver's `mlp_out` weights grew 25–50× and any
save+fresh-load round-trip produced catastrophic loss (~12–83) on a
checkpoint whose training-time log showed loss ~0.3.

Root cause located via `tools/debug_*.py` chain: **PyTorch's
`torch.amp.autocast(..., cache_enabled=True)` caches bf16 casts of fp32
parameters and only invalidates them via `GradScaler.unscale_()`, which we
don't call in pure BF16 training**. So after every `optimizer.step()`, the
cached bf16 went stale, and the in-process forward kept using lagged
weights. When you fresh-loaded the checkpoint in a new process there was no
stale cache → forward used the *current* fp32 weights → garbage.

**Fix (one-line per call site):**
```python
with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp,
                        cache_enabled=False):
```

Applied at:
- `stage2/train.py:295`
- `stage2/eval/distill_eval.py:71, 107`
- `stage2/smoke_test_models.py:71`

Verified by `tools/debug_verify_fix.py`: trained model_A and fresh +
`load_state_dict(...)` model_B produce **bit-exact** forward output.

This is the only reason the current run4 is honest — training-time loss
values now equal what a fresh-load eval would reproduce.

---

## 3. Anti-collapse + reliability config in `sav_repvit_m0_9.yaml`

```yaml
DATA:
  BATCH_SIZE: 2
  NUM_WORKERS: 16            # was 28 — reduces dataloader RSS by ~43%
  PREFETCH_FACTOR: 2         # was 4 — halves in-flight queue (~14 → ~7 GB)

DISTILL:
  NORM_WEIGHT: 2.0           # was 0.25 — 8x stronger log-ratio regularizer

TRAIN:
  WEIGHT_DECAY: 0.1          # was 0.05 — extra decay on mlp_out / out_attn
  CLIP_GRAD: 0.5             # was 1.0 — tighter cap
  SAVE_EVERY_ITERS: 2000     # mid-epoch atomic ckpt every ~12 min
  EARLY_STOP_PATIENCE: 5     # stop after 5 epochs without new best

EVAL:
  EVERY_N_EPOCHS: 1          # was 5 — catch any collapse mode same epoch
```

Plus added: 48 GiB `/swapfile` on root NVMe (OOM-killer guardrail).

---

## 4. Code additions made this conversation

| file | added |
|---|---|
| `stage2/config.py` | `_C.TRAIN.EARLY_STOP_PATIENCE = 0` |
| `stage2/train.py` | `epochs_since_best` counter, `no new best (since_best=N, ...)` log line, `EARLY STOP` break, restore counter from ckpt |
| `stage2/utils/checkpoint.py` | persist `epochs_since_best` in both epoch + running ckpts |
| `stage2/train.py:295`, `stage2/eval/distill_eval.py:71,107`, `stage2/smoke_test_models.py:71` | `cache_enabled=False` |
| `stage2/train.py` per-iter log | added `gn=` (grad norm), `mw=` (mlp_out.2.weight std — collapse canary), `s_norm=` (avg per-token L2) |
| `stage2/train.py` TB scalars | `train/grad_norm`, `train/mlp_out_w_std`, `train/s_token_norm` |
| `dashboard/server.py` | `/api/train/start`, `/api/train/stop`, `/api/train/status` endpoints |
| `dashboard/static/index.html` | Start/Stop buttons + tag input + auto-tag-detect logic + live status polling |
| `dashboard/metrics.py` | merge all TB event files (not just newest); TB fallback for train metrics when log parser sees nothing (skip-first phase) |
| `dashboard/launch.sh` | now uses `/home/saif/venvs/tf` and port 6007 |
| `scripts/systemd/mnt-hdd-watchdog.sh` + `.service` + `INSTALL.md` | watchdog that auto-mounts /mnt/hdd and auto-relaunches training+dashboard if dead |
| `tools/debug_save_load.py`, `debug_save_load_after_train.py`, `debug_save_load_inplace.py`, `debug_per_layer.py`, `debug_first_call.py`, `debug_walk_blocks.py`, `debug_mha_isolated.py`, `debug_force_math_sdpa.py`, `debug_autocast_cache.py`, `debug_definitive.py`, `debug_verify_fix.py` | the diagnostic chain that found the autocast cache bug |

---

## 5. Full val history (run4)

| ep | val total | val cos | val norm | EMA total | counter |
|----|-----------|---------|----------|-----------|---------|
| 0  | 0.3071 | 0.2987 | 0.0031 | 0.3058 | 0 |
| 1  | 0.3006 | 0.2927 | 0.0028 | 0.2982 | 0 |
| 2  | 0.2980 | 0.2902 | 0.0028 | 0.2953 | 0 |
| 3  | 0.2961 | 0.2882 | 0.0028 | 0.2937 | 0 |
| 4  | 0.2962 | 0.2886 | 0.0027 | 0.2924 | 1 |
| 5  | 0.2933 | 0.2858 | 0.0027 | 0.2915 | 0 |
| 6  | 0.2928 | 0.2853 | 0.0026 | 0.2910 | 0 |
| 7  | 0.2924 | 0.2849 | 0.0026 | 0.2909 | 0 |
| 8  | 0.2929 | 0.2856 | 0.0026 | 0.2908 | 1 |
| 9  | 0.2928 | 0.2853 | 0.0026 | 0.2901 | 2 |
| 10 | 0.2920 | 0.2847 | 0.0026 | 0.2898 | 0 |
| 11 | 0.2921 | 0.2847 | 0.0026 | 0.2898 | 1 |
| 12 | 0.2916 | 0.2842 | 0.0026 | 0.2897 | 0 |
| **13** | **0.2907** | 0.2835 | 0.0025 | 0.2892 | 0 ← `ckpt_best.pth` |
| 14 | 0.2910 | 0.2836 | 0.0026 | 0.2890 | 1 (current state) |

Pattern: raw val noisy (~±0.0005 wobble), EMA monotonically descending
(every epoch since ep5 has produced a lower EMA). Training is genuinely
still learning.

---

## 6. Infra setup that survives crashes

1. **fstab `/mnt/hdd` entry** has `nofail` so missing disk doesn't stall boot.
2. **`mnt-hdd-watchdog.service`** (root, polls every 20 s):
   - Re-mounts `/mnt/hdd` if it disappears
   - Re-launches dashboard if dead
   - Re-launches training (`--tag ep50_run4`, auto-resume from
     `ckpt_running.pth`) if dead
   - All Python procs run as user `saif` via `runuser`
3. **48 GiB swapfile** at `/swapfile` (sysctl `vm.swappiness=10`)
4. **SAVE_EVERY_ITERS=2000** = atomic mid-epoch ckpt every ~12 min, so the
   worst-case loss to any crash is ~12 min of work

### Known gotcha

The watchdog initially failed with `nohup: failed to run command 'torchrun':
No such file or directory` because `runuser -u saif -- bash -c ...` doesn't
source `.profile` so `~/.local/bin` (where torchrun lives) isn't in PATH.

**Pending fix:** edit `scripts/systemd/mnt-hdd-watchdog.sh` so the launch
command uses absolute path, e.g.:
```bash
runuser -u "$USR" -- bash -lc "..."     # -l = login shell, sources .profile
# OR
runuser -u "$USR" -- env PATH=/home/saif/.local/bin:/usr/local/bin:/usr/bin bash -c "..."
```
Then `sudo cp` again to `/usr/local/bin/`. Until that's done, the watchdog
**re-mounts /mnt/hdd correctly but cannot relaunch training**. User is doing
manual relaunches in the meantime.

---

## 7. Recurring issues observed

| | |
|---|---|
| `/mnt/hdd` spontaneously unmounts | Happens after PC reboot / sleep cycles. `nofail` in fstab and the watchdog now handle it (~20 s recovery once watchdog fix above lands). |
| Another user (`sami`) shares the GPU | When this happens, ips drops from 2.7 → ~2.1 (a ~25 % slowdown). Doesn't break anything, just extends epoch wall-time. |
| Browser tab gets stuck with old tag | After dashboard updates, hard-refresh (Ctrl+Shift+R). The HTML now auto-fills tag from snapshot. |

---

## 8. Archived runs (forensic value, not for resume)

| dir | what + why |
|---|---|
| `archive/stage2_failed_run2_collapsed/` | ep50_run2 — collapsed silently due to autocast cache + insufficient NORM_WEIGHT. mlp_out std grew to 1.09 (50× init). 309 MB ckpts + log + tb. |
| `archive/stage2_failed_run3_autocast_cache/` | ep50_run3 — collapsed for the same autocast cache reason but more slowly thanks to NORM_WEIGHT=2. Provided the failure data that led to the autocast cache hypothesis. 386 MB. |

---

## 9. What the project is

EfficientSAM3 Stage 2 distills SAM3 ViT-H's sequential `pix_feat_with_mem`
into a 5.02 M-parameter **TemporalPerceiver** while keeping the
Stage 1 RepViT-M0.9 backbone frozen. Dataset: SA-V (50,583 videos, 1.1 TB
at `/mnt/hdd/datasets/SA-V/`), 49,594 train / 989 val via SHA1-hash split.
Loss: `MSE + cosine + 2.0 × log(s/t)²` on L2-normalized features.
Target deployment: Samsung Galaxy S26 Ultra Hexagon NPU at 1008 × 1008 × 8
frames static-shape ONNX export.

**Param accounting (queried from the actual model 2026-05-21):**

| component | params |
|---|---|
| vision encoder (RepViT-M0.9 + neck) | 30.16 M |
| TemporalPerceiver (Stage 2, only trainable) | 5.02 M |
| detector.transformer | 21.05 M |
| segmentation_head (mask decoder) | 2.30 M |
| geometry_encoder | 8.22 M |
| dot_prod_scoring | 1.18 M |
| tracker kept pieces (sam_mask_decoder + obj ptrs) | 4.44 M |
| **deployable video-tracking model (no text)** | **72.37 M** |
| (full text-prompted student incl. lang backbone) | 388.93 M |

vs SAM3 ViT-H teacher's 827 M — **6.5× compression for video-only tracking**.

---

## 10. Files you'll need to look at first if picking this up

1. `MEMORY_DUMP.md` — this file
2. `HANDOFF.md` — the older, longer handoff doc with the autocast bug already
   documented at the top
3. `REPORT.md` — research write-up
4. `RUNBOOK.md` — copy-paste shell commands for start/stop/resume
5. `stage2/train.py` — the actual training loop
6. `stage2/configs/sav_repvit_m0_9.yaml` — the live config
7. `scripts/systemd/INSTALL.md` — watchdog install + test instructions
8. `tools/debug_definitive.py` — the script that proves the autocast fix

---

## 11. Things that are next

1. **Fix the watchdog PATH issue** (see §6 "Known gotcha")
2. **Decide when to stop training**:
   - Early-stop will fire automatically after 5 epochs without a new best, OR
   - Manually if the EMA val also plateaus (currently still descending —
     0.2937 → 0.2890 over ep3 → ep14)
3. **DAVIS 2017 eval** of `ckpt_best.pth` (Stage 3 territory, REPORT.md §8)
4. **Stage 3 — ONNX export**: trace student (RepViT + Perceiver) with
   static shapes `MAX_FRAMES=8, IMG_SIZE=1008, BATCH=1`. See
   `NPU_DEPLOYMENT.md` for QNN conversion steps.

---

## 12. Quick commands

```bash
# health check
ps -fp $(cat ~/github/efficientsam3/logs/stage2_run4.pid) | tail -1
tail -5 ~/github/efficientsam3/logs/stage2_run4.log
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

# manual launch (if watchdog hasn't been fixed yet)
cd ~/github/efficientsam3
nohup torchrun --nproc_per_node=1 --master_port 29510 stage2/train.py \
  --cfg stage2/configs/sav_repvit_m0_9.yaml \
  --data-path /mnt/hdd/datasets/SA-V/ \
  --tag ep50_run4 --output output \
  >> logs/stage2_run4.log 2>&1 &
echo $! > logs/stage2_run4.pid

# pause (graceful)
kill -TERM $(cat logs/stage2_run4.pid)

# dashboard
bash dashboard/launch.sh    # → http://localhost:6007

# watchdog
systemctl status mnt-hdd-watchdog
sudo tail -f /var/log/mnt-hdd-watchdog.log
```

---

## 13. Conversation timeline (high-level)

| date | what happened |
|---|---|
| 2026-05-14 | Inherited stage2 codebase, discovered run2 had collapsed. First failed restart as run3. |
| 2026-05-14 PM | Root-caused autocast cache bug via 11 diagnostic scripts. Applied 3-line fix. |
| 2026-05-15 | Launched run4 from ep0 with the fix. Verified bit-exact save+load. |
| 2026-05-15 → 5-20 | Trained through 14 epochs across multiple crash/HDD-unmount cycles. Each resume confirmed loss continuity (proving the fix held). |
| 2026-05-19 | Added `EARLY_STOP_PATIENCE`. Persists `epochs_since_best` in checkpoints. |
| 2026-05-22 | Trained through ep15 mid-epoch. |
| 2026-05-28 | Wrote systemd `mnt-hdd-watchdog.service` for auto-mount + auto-resume. User installed but PATH bug discovered (training relaunch fails inside the service). Manually relaunched. Wrote this dump. |

---

User: **Mollah Md Saif** · mollahmdsaif@gmail.com
Branch: `main` · Repo: `/home/saif/github/efficientsam3`
