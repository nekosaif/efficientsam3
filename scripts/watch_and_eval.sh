#!/usr/bin/env bash
# scripts/watch_and_eval.sh
#
# Waits for a Stage 2 checkpoint to appear, runs a bounded distillation eval,
# and writes a PASS/FAIL health summary to the log file.
#
# Usage (run in background from repo root):
#   nohup bash scripts/watch_and_eval.sh \
#       output/efficient_sam3_stage2/ep50_run2/ckpt_epoch_1.pth \
#       logs/autoeval_ep1.log \
#       ep50_run2 \
#       > /dev/null 2>&1 &
#
# Arguments:
#   $1  CKPT_PATH  path to watch for (default: ep50_run2/ckpt_epoch_1.pth)
#   $2  LOG_FILE   where to write all output (default: logs/autoeval_ep1.log)
#   $3  TAG        training run tag (default: ep50_run2)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CKPT="${1:-output/efficient_sam3_stage2/ep50_run2/ckpt_epoch_1.pth}"
LOG="${2:-logs/autoeval_ep1.log}"
TAG="${3:-ep50_run2}"
CFG="stage2/configs/sav_repvit_m0_9.yaml"
DATA="/mnt/hdd/datasets/SA-V/"
MAX_BATCHES=100     # ~5-10 min, limits GPU memory overlap with training
MASTER_PORT=29512   # different from training's 29511

# PASS thresholds
COS_THRESHOLD=1.0   # must be < 1.0 (random baseline when cos_sim=0 is exactly 1.0)
NORM_THRESHOLD=2.0  # log(s_norm/t_norm)^2 < 2 means norms within ~4x of teacher

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== watch_and_eval START ==="
log "Watching : $CKPT"
log "Log      : $LOG"

# --- poll every 60s until checkpoint appears ---
while [ ! -f "$CKPT" ]; do
    sleep 60
done
log "Checkpoint found. Sleeping 120s for training to stabilize into next epoch..."
sleep 120

# --- run eval ---
log "Running eval (--val-max-batches $MAX_BATCHES) ..."
set +e
torchrun \
    --nproc_per_node=1 \
    --master_port "$MASTER_PORT" \
    stage2/train.py \
    --cfg "$CFG" \
    --data-path "$DATA" \
    --tag "$TAG" \
    --resume "$CKPT" \
    --eval \
    --val-max-batches "$MAX_BATCHES"
EVAL_EXIT=$?
set -e

if [ "$EVAL_EXIT" -ne 0 ]; then
    log "FAIL: eval process exited $EVAL_EXIT (crash or OOM)"
    log "  ACTION: check if GPU OOM; re-run manually with --val-max-batches 20"
    log "=== watch_and_eval END (FAIL) ==="
    exit 1
fi

# --- parse metrics from the eval result line ---
# Expected: [stage2][eval] total=X mse=Y cos=Z norm=W n=N
RESULT_LINE=$(grep '\[stage2\]\[eval\]' "$LOG" | tail -1)

if [ -z "$RESULT_LINE" ]; then
    log "FAIL: no [stage2][eval] line in log"
    log "=== watch_and_eval END (FAIL) ==="
    exit 1
fi

log "Result: $RESULT_LINE"

TOTAL=$(echo "$RESULT_LINE" | grep -oP 'total=\K[0-9.]+' || true)
COS=$(echo   "$RESULT_LINE" | grep -oP 'cos=\K[0-9.]+'   || true)
NORM=$(echo  "$RESULT_LINE" | grep -oP 'norm=\K[0-9.]+'  || true)

if [ -z "$TOTAL" ] || [ -z "$COS" ] || [ -z "$NORM" ]; then
    log "FAIL: could not parse total/cos/norm from result line"
    log "=== watch_and_eval END (FAIL) ==="
    exit 1
fi

log "--- METRICS ---"
log "  total = $TOTAL"
log "  cos   = $COS   (pass if < $COS_THRESHOLD)"
log "  norm  = $NORM  (pass if < $NORM_THRESHOLD)"

# Use python for reliable float comparison (avoids bc locale/precision issues)
if python3 - "$COS" "$NORM" "$COS_THRESHOLD" "$NORM_THRESHOLD" <<'PYEOF'
import sys
cos, norm, cos_t, norm_t = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
if cos >= cos_t:
    print(f"FAIL: cos={cos:.4f} >= {cos_t} — no better than random baseline; Perceiver may have re-collapsed")
    sys.exit(1)
if norm >= norm_t:
    print(f"FAIL: norm={norm:.4f} >= {norm_t} — magnitude collapse; mlp_out still diverging")
    sys.exit(1)
sys.exit(0)
PYEOF
then
    log "PASS: training looks healthy after 2 epochs"
    log "  total=$TOTAL  cos=$COS  norm=$NORM"
    log "  Next built-in val fires at epoch 4 (EVERY_N_EPOCHS=5)."
    log "=== watch_and_eval END (PASS) ==="
    exit 0
else
    log "  ACTION: check logs/stage2_run2.log for loss trend"
    log "  ACTION: if norm stays high after ep5, re-initialize Perceiver mlp_out"
    log "=== watch_and_eval END (FAIL) ==="
    exit 1
fi
