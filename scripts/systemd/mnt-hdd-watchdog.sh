#!/bin/bash
# Watchdog that:
#   1. Mounts /mnt/hdd whenever it disappears (boot, hot-(un)plug, sleep/wake).
#   2. After a successful mount, ensures the dashboard + Stage-2 training are
#      running. Auto-resumes from ckpt_running.pth (the latest mid-epoch save).
#
# Runs as root (systemd unit). Drops to user `saif` for launching the Python
# processes via `runuser`.
#
# Logs to /var/log/mnt-hdd-watchdog.log

set -u

MOUNT="/mnt/hdd"
LOG="/var/log/mnt-hdd-watchdog.log"
INTERVAL=20   # seconds between checks

# user-side bits
USR="saif"
REPO="/home/saif/github/efficientsam3"
TRAIN_PIDFILE="$REPO/logs/stage2_run4.pid"
DASH_PIDFILE="$REPO/logs/dashboard.pid"
TRAIN_LOG="$REPO/logs/stage2_run4.log"
DASH_PORT=6007
TAG="ep50_run4"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "$(ts) $*" >> "$LOG"; }

_alive() {
  # arg: pid file. returns 0 if pid in file is still alive.
  local f=$1
  [ -f "$f" ] || return 1
  local pid
  pid=$(cat "$f" 2>/dev/null) || return 1
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

_launch_training() {
  # Training auto-resume disabled by user request 2026-05-30.
  # HDD auto-remount + dashboard relaunch remain active.
  return 0
}

_launch_dashboard() {
  if _alive "$DASH_PIDFILE"; then
    return 0
  fi
  if ss -ltn 2>/dev/null | grep -q ":$DASH_PORT "; then
    return 0   # something is already on the port
  fi
  log "relaunching dashboard"
  runuser -u "$USR" -- bash -c "cd '$REPO' && bash dashboard/launch.sh" \
    >>"$LOG" 2>&1
}

log "watchdog start (mount=$MOUNT interval=${INTERVAL}s)"

# initial pass
if ! mountpoint -q "$MOUNT"; then
  mount "$MOUNT" 2>>"$LOG" && log "MOUNTED $MOUNT (initial)"
fi
_launch_dashboard
_launch_training

while true; do
  sleep "$INTERVAL"

  if ! mountpoint -q "$MOUNT"; then
    if mount "$MOUNT" 2>>"$LOG"; then
      log "MOUNTED $MOUNT (recovered)"
      # after a fresh mount, re-check both services
      _launch_dashboard
      _launch_training
    else
      log "MOUNT FAILED for $MOUNT (next try in ${INTERVAL}s)"
    fi
    continue
  fi

  # mount is fine — make sure the two services stayed alive
  _launch_dashboard
  _launch_training
done
