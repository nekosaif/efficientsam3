#!/usr/bin/env bash
# Stage 2 training health check — invoked by cron every hour.
# Appends one record to logs/health.log so you can `tail -f` it
# or have Claude read it after a session restart.

set -u
ROOT=/home/saif/github/efficientsam3
LOG=$ROOT/logs/stage2_run1.log
PIDF=$ROOT/logs/stage2_run1.pid
OUT=$ROOT/logs/health.log
TB_DIR=$ROOT/output/efficient_sam3_stage2/ep50_run1

ts=$(date '+%Y-%m-%d %H:%M:%S %z')
{
  echo "===== $ts ====="

  # Process alive?
  if [ -f "$PIDF" ]; then
    pid=$(cat "$PIDF")
    if ps -p "$pid" >/dev/null 2>&1; then
      echo "STATE: ALIVE pid=$pid"
      ps -o etime= -p "$pid" 2>/dev/null | awk '{print "ELAPSED:",$1}'
    else
      echo "STATE: DEAD pid=$pid (no process)"
    fi
  else
    echo "STATE: UNKNOWN (no pid file)"
  fi

  # GPU
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu \
               --format=csv,noheader 2>/dev/null | head -1 | awk '{print "GPU:",$0}'
  fi

  # Disk on /mnt/hdd
  df -h /mnt/hdd 2>/dev/null | awk 'NR==2{print "HDD:",$3,"used /",$2,"(",$5,"used )"}'

  # Last few iter/loss lines (best-effort regex — adjust if format changes)
  if [ -f "$LOG" ]; then
    bytes=$(stat -c%s "$LOG" 2>/dev/null || echo 0)
    echo "LOG_BYTES: $bytes"
    grep -Eo "Epoch[: ][0-9]+|iter[: ]?[0-9]+|loss[^ ]*[ =:][0-9.]+|val[_-]?loss[ =:][0-9.]+" "$LOG" 2>/dev/null | tail -8 | sed 's/^/  /'
    # Surface any errors near tail
    grep -Ei "traceback|cuda error|oom|killed|nan|^error|runtimeerror" "$LOG" 2>/dev/null | tail -3 | sed 's/^/  ERR: /'
  else
    echo "LOG: missing $LOG"
  fi

  # Checkpoint listing
  if [ -d "$TB_DIR" ]; then
    ls -la "$TB_DIR" 2>/dev/null | grep -E "ckpt|best|latest" | sed 's/^/  CKPT: /'
  fi

  echo
} >> "$OUT" 2>&1
