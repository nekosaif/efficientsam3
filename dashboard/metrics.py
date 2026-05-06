"""Metrics reader: TB scalars + training log parser + ETA calculator."""

from __future__ import annotations

import os
import re
import time
import subprocess
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
LOG_PATH = PROJECT_ROOT / "logs" / "stage2_run2.log"
TB_DIR = PROJECT_ROOT / "output" / "efficient_sam3_stage2" / "ep50_run2" / "tb"
CONFIG_PATH = PROJECT_ROOT / "stage2" / "configs" / "sav_repvit_m0_9.yaml"
RUN_TAG = "ep50_run2"

# Cache for TB accumulator
_tb_cache: dict[str, Any] = {"mtime": 0.0, "data": {}}

# Cache for log parse (refresh every 2s)
_log_cache: dict[str, Any] = {"ts": 0.0, "data": {}}

_config_cache: dict[str, Any] = {}


def _load_config() -> dict:
    if _config_cache:
        return _config_cache
    try:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        _config_cache["epochs"] = cfg.get("TRAIN", {}).get("EPOCHS", 100)
        _config_cache["batch_size"] = cfg.get("DATA", {}).get("BATCH_SIZE", 2)
    except Exception:
        _config_cache["epochs"] = 100
        _config_cache["batch_size"] = 2
    return _config_cache


def _active_tb_file() -> Path | None:
    if not TB_DIR.exists():
        return None
    files = sorted(TB_DIR.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def _load_tb_scalars() -> dict[str, list]:
    """Return dict tag -> list of (step, value). Cached by mtime."""
    f = _active_tb_file()
    if f is None:
        return {}
    mtime = f.stat().st_mtime
    if mtime == _tb_cache["mtime"] and _tb_cache["data"]:
        return _tb_cache["data"]

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
        ea.Reload()
        data: dict[str, list] = {}
        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            data[tag] = [(e.step, e.value) for e in events]
        _tb_cache["mtime"] = mtime
        _tb_cache["data"] = data
        return data
    except Exception:
        return _tb_cache.get("data", {})


def get_scalars(tag: str, downsample: int = 500) -> dict:
    data = _load_tb_scalars()
    points = data.get(tag, [])
    if len(points) > downsample:
        stride = len(points) // downsample
        points = points[::stride]
    steps = [p[0] for p in points]
    values = [p[1] for p in points]
    return {"tag": tag, "steps": steps, "values": values}


def get_all_scalar_tags() -> dict[str, float | None]:
    data = _load_tb_scalars()
    result = {}
    for tag, pts in data.items():
        result[tag] = pts[-1][1] if pts else None
    return result


def _parse_log() -> dict:
    now = time.time()
    if now - _log_cache["ts"] < 2.0 and _log_cache["data"]:
        return _log_cache["data"]

    result: dict[str, Any] = {
        "current_epoch": 0,
        "current_iter": 0,
        "steps_per_epoch": 24797,
        "total_epochs": None,
        "train_loss": None,
        "train_mse": None,
        "train_cos": None,
        "lr": None,
        "ips": None,
        "val_total": None,
        "val_mse": None,
        "val_cos": None,
        "val_ema_total": None,
        "val_ema_mse": None,
        "val_ema_cos": None,
        "val_at_epoch": None,
        "best_val": None,
        "epoch_times": [],
        "run_start_time": None,
    }

    if not LOG_PATH.exists():
        return result

    try:
        # Read last 100KB of log (enough for recent iters + all epoch markers)
        size = LOG_PATH.stat().st_size
        offset = max(0, size - 102400)
        with open(LOG_PATH, "rb") as f:
            f.seek(offset)
            tail = f.read().decode("utf-8", errors="replace")

        # Boot line: steps/epoch
        m = re.search(r"train ds len=\d+ steps/epoch=(\d+)", tail)
        if m:
            result["steps_per_epoch"] = int(m.group(1))

        # Boot line: total_epochs (logged by train.py at loop entry)
        m = re.search(r"entering training loop \([^)]*total_epochs=(\d+)\)", tail)
        if m:
            result["total_epochs"] = int(m.group(1))

        # Last iter line
        iter_pat = re.compile(
            r"\[stage2\] ep(\d+) it(\d+)/(\d+) loss=([\d.]+) mse=([\d.]+) cos=([\d.]+)"
            r"(?:\s+norm=([\d.]+))?\s+lr=([\d.e+\-]+) ips=([\d.]+)"
        )
        m = None
        for m in iter_pat.finditer(tail):
            pass
        if m:
            result["current_epoch"] = int(m.group(1))
            result["current_iter"] = int(m.group(2))
            result["steps_per_epoch"] = int(m.group(3))
            result["train_loss"] = float(m.group(4))
            result["train_mse"] = float(m.group(5))
            result["train_cos"] = float(m.group(6))
            result["lr"] = float(m.group(8))
            result["ips"] = float(m.group(9))

        # Val lines
        val_pat = re.compile(
            r"\[stage2\]\[val\] ep(\d+) total=([\d.]+) mse=([\d.]+) cos=([\d.]+)"
        )
        m = None
        for m in val_pat.finditer(tail):
            result["val_total"] = float(m.group(2))
            result["val_mse"] = float(m.group(3))
            result["val_cos"] = float(m.group(4))
            result["val_at_epoch"] = int(m.group(1))

        ema_pat = re.compile(
            r"\[stage2\]\[val-ema\] ep(\d+) total=([\d.]+) mse=([\d.]+) cos=([\d.]+)"
        )
        m = None
        for m in ema_pat.finditer(tail):
            result["val_ema_total"] = float(m.group(2))
            result["val_ema_mse"] = float(m.group(3))
            result["val_ema_cos"] = float(m.group(4))

        # Best val
        best_pat = re.compile(r"\[stage2\] new best val=([\d.]+)")
        for m in best_pat.finditer(tail):
            result["best_val"] = float(m.group(1))

        # Epoch wall times: scan full log (epoch-done lines may be outside last 100KB)
        ep_time_pat = re.compile(r"\[stage2\] epoch (\d+) done in ([\d.]+)s")
        with open(LOG_PATH, "r", errors="replace") as _f:
            full = _f.read()
        times = [(int(m.group(1)), float(m.group(2))) for m in ep_time_pat.finditer(full)]
        result["epoch_times"] = times[-5:]

        # Run start: first line with timestamp or first iter
        first_iter = re.search(r"\[stage2\] ep\d+ it1/", tail)
        if first_iter:
            # Approximate: use log file ctime
            result["run_start_time"] = LOG_PATH.stat().st_ctime

    except Exception:
        pass

    _log_cache["ts"] = now
    _log_cache["data"] = result
    return result


def _is_training_alive() -> tuple[bool, int | None]:
    """Check if training process is alive by scanning running processes."""
    try:
        import psutil
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmd = " ".join(p.info["cmdline"] or [])
                if "stage2/train.py" in cmd:
                    return True, p.pid
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass
    # Fallback: nvidia-smi compute processes
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            text=True, timeout=3
        )
        if out.strip():
            return True, None
    except Exception:
        pass
    return False, None


def _fmt_duration(sec: float) -> str:
    sec = int(sec)
    d = sec // 86400
    h = (sec % 86400) // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if d > 0:
        return f"{d}d {h}h {m}m"
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def get_snapshot() -> dict:
    cfg = _load_config()
    log = _parse_log()
    alive, pid = _is_training_alive()

    # Prefer log-derived epoch count (runtime --opts may override YAML)
    total_epochs = log["total_epochs"] or cfg["epochs"]
    batch_size = cfg.get("batch_size", 2)
    steps_per_epoch = log["steps_per_epoch"]
    cur_epoch = log["current_epoch"]
    cur_iter = log["current_iter"]

    # cur_epoch is 0-indexed (train.py: for epoch in range(0, total_epochs))
    # ep0 = first epoch, ep49 = last epoch for a 50-epoch run
    completed_epochs = cur_epoch  # ep{N} means epochs 0..N-1 are done
    global_step = completed_epochs * steps_per_epoch + cur_iter
    total_steps = total_epochs * steps_per_epoch
    pct_done = 100.0 * global_step / total_steps if total_steps > 0 else 0.0

    # ETA from epoch wall times (preferred — immune to ips unit confusion)
    # Note: log `ips` is SAMPLES/sec (n_window += frames.shape[0] in train.py),
    # not iterations/sec. True iter rate = ips / batch_size.
    epoch_avg_sec: float | None = None
    if log["epoch_times"]:
        times = [t for _, t in log["epoch_times"]]
        epoch_avg_sec = sum(times) / len(times)

    remaining_iters = total_steps - global_step
    eta_sec: float | None = None
    if epoch_avg_sec and steps_per_epoch > 0:
        # ep{cur_epoch} is in progress; after it: ep{cur_epoch+1}..ep{total_epochs-1}
        remaining_full_epochs = total_epochs - cur_epoch - 1
        frac_left_in_cur = (steps_per_epoch - cur_iter) / steps_per_epoch
        eta_sec = max(0.0, remaining_full_epochs * epoch_avg_sec + frac_left_in_cur * epoch_avg_sec)
    elif log["ips"] and log["ips"] > 0 and remaining_iters > 0:
        # Fallback: ips is samples/sec; divide by batch_size to get iters/sec
        iter_rate = log["ips"] / batch_size
        eta_sec = remaining_iters / iter_rate

    # Elapsed: use log file birth time (Linux statx) or oldest TB event file
    elapsed_sec: float = 0.0
    try:
        # st_birthtime available on macOS; on Linux use stat --format=%W
        birth = subprocess.check_output(
            ["stat", "--format=%W", str(LOG_PATH)], text=True, timeout=2
        ).strip()
        birth_ts = float(birth)
        if birth_ts > 0:
            elapsed_sec = time.time() - birth_ts
    except Exception:
        pass
    if elapsed_sec == 0.0 and TB_DIR.exists():
        files = sorted(TB_DIR.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
        if files:
            elapsed_sec = time.time() - files[0].stat().st_mtime

    # Augment val from TB if log tail doesn't have it (val only written at epoch boundary)
    val_total = log["val_total"]
    val_mse = log["val_mse"]
    val_cos = log["val_cos"]
    val_ema = log["val_ema_total"]
    val_epoch = log["val_at_epoch"]
    best_val = log["best_val"]
    if val_total is None:
        tb_data = _load_tb_scalars()
        def _last(tag):
            pts = tb_data.get(tag, [])
            return pts[-1][1] if pts else None
        def _last_step(tag):
            pts = tb_data.get(tag, [])
            return pts[-1][0] if pts else None
        val_total = _last("val/loss_total")
        val_mse   = _last("val/loss_mse")
        val_cos   = _last("val/loss_cosine")
        val_ema   = _last("val/ema_loss_total")
        val_epoch = _last_step("val/loss_total")
        best_val  = val_total  # TB doesn't store best separately; use latest val

    return {
        "run": {
            "tag": RUN_TAG,
            "config_epochs": total_epochs,
            "steps_per_epoch": steps_per_epoch,
            "current_epoch": cur_epoch,
            "current_iter": cur_iter,
            "global_step": global_step,
            "total_steps": total_steps,
            "pct_done": round(pct_done, 2),
        },
        "loss": {
            "train_total": log["train_loss"],
            "train_mse": log["train_mse"],
            "train_cos": log["train_cos"],
            "lr": log["lr"],
            "ips": log["ips"],
            "val_total": val_total,
            "val_mse": val_mse,
            "val_cos": val_cos,
            "val_ema_total": val_ema,
            "val_at_epoch": val_epoch,
            "best_val": best_val,
        },
        "time": {
            "elapsed_sec": round(elapsed_sec),
            "elapsed_human": _fmt_duration(elapsed_sec),
            "eta_sec": round(eta_sec) if eta_sec else None,
            "eta_human": _fmt_duration(eta_sec) if eta_sec else "estimating…",
            "remaining_iters": remaining_iters if remaining_iters >= 0 else 0,
            "epoch_avg_sec": round(epoch_avg_sec) if epoch_avg_sec else None,
            "epoch_avg_human": _fmt_duration(epoch_avg_sec) if epoch_avg_sec else None,
        },
        "process": {
            "alive": alive,
            "pid": pid,
        },
    }
