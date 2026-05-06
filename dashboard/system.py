"""System telemetry: GPU via nvidia-smi, CPU/RAM via psutil. Cached 1s."""

from __future__ import annotations

import subprocess
import time
from typing import Any

import psutil

_cache: dict[str, Any] = {"ts": 0.0, "data": {}}


def _nvidia_smi() -> dict:
    fields = "name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,fan.speed"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            text=True, timeout=3
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) < 8:
            raise ValueError("unexpected nvidia-smi output")
        return {
            "name": parts[0],
            "util_pct": _f(parts[1]),
            "mem_util_pct": _f(parts[2]),
            "mem_used_mb": _f(parts[3]),
            "mem_total_mb": _f(parts[4]),
            "temp_c": _f(parts[5]),
            "power_w": _f(parts[6]),
            "fan_pct": _f(parts[7]),
        }
    except Exception as e:
        return {"error": str(e)}


def _f(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def get_system() -> dict:
    now = time.time()
    if now - _cache["ts"] < 1.0 and _cache["data"]:
        return _cache["data"]

    gpu = _nvidia_smi()
    cpu_pct = psutil.cpu_percent(interval=None)
    load = psutil.getloadavg()
    vm = psutil.virtual_memory()

    data = {
        "gpu": gpu,
        "cpu": {
            "util_pct": cpu_pct,
            "load_1m": round(load[0], 2),
            "load_5m": round(load[1], 2),
            "load_15m": round(load[2], 2),
        },
        "ram": {
            "used_gb": round(vm.used / 1024**3, 1),
            "total_gb": round(vm.total / 1024**3, 1),
            "pct": vm.percent,
        },
    }
    _cache["ts"] = now
    _cache["data"] = data
    return data
