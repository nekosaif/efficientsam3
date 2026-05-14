#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=/home/saif/venvs/tf
PORT=6007
LOGFILE=logs/dashboard.log
PIDFILE=logs/dashboard.pid

# Stop existing instance (best-effort, kill anything still on the port).
if [ -f "$PIDFILE" ]; then
  OLD_PID=$(cat "$PIDFILE")
  kill "$OLD_PID" 2>/dev/null || true
  sleep 1
fi
pkill -f "uvicorn dashboard.server:app" 2>/dev/null || true
sleep 1

nohup "$VENV/bin/python" -m uvicorn dashboard.server:app \
  --host 0.0.0.0 --port "$PORT" \
  > "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"

sleep 2
echo "Dashboard PID $(cat $PIDFILE) → http://$(hostname -I | awk '{print $1}'):$PORT"
tail -5 "$LOGFILE"
