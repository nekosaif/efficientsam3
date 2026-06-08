"""EfficientSAM3 training dashboard — FastAPI backend."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dashboard.metrics import get_snapshot, get_scalars, get_all_scalar_tags
from dashboard.system import get_system

STATIC_DIR = Path(__file__).parent / "static"
REPO_ROOT = Path(__file__).parent.parent.resolve()
LOGS_DIR = REPO_ROOT / "logs"

# Default launch profile — matches the production runbook.
# Tag defaults to the newest existing run-dir under output/efficient_sam3_stage2/
# so "Start" on the dashboard resumes the right run instead of clobbering it
# with a different tag.
DEFAULT_CFG = "stage2/configs/sav_repvit_m0_9.yaml"
DEFAULT_DATA = "/mnt/exoshdd/datasets/SA-V/"
DEFAULT_OUTPUT = "output"
DEFAULT_PORT = 29510


def _resolve_default_tag() -> str:
    base = Path(DEFAULT_OUTPUT)
    runs_root = REPO_ROOT / base / "efficient_sam3_stage2"
    if runs_root.is_dir():
        candidates = [d for d in runs_root.iterdir() if d.is_dir()]
        if candidates:
            candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
            return candidates[0].name
    return "ep50_run4"


DEFAULT_TAG = _resolve_default_tag()

app = FastAPI(title="EfficientSAM3 Dashboard", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Read-only metric endpoints
# ---------------------------------------------------------------------------

@app.get("/api/snapshot")
def snapshot():
    data = get_snapshot()
    data["system"] = get_system()
    return JSONResponse(data)


@app.get("/api/scalars")
def scalars(tag: str = Query(...), downsample: int = Query(500)):
    return JSONResponse(get_scalars(tag, downsample))


@app.get("/api/scalars/all")
def scalars_all():
    return JSONResponse(get_all_scalar_tags())


# ---------------------------------------------------------------------------
# Training process control
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"--tag\s+(\S+)")


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _find_training_process() -> Optional[dict]:
    """Return info about a running stage2/train.py process, or None.

    Picks the Python child (not the torchrun wrapper) so the PID can be used
    for process-group kill.
    """
    try:
        out = subprocess.check_output(["pgrep", "-af", "stage2/train.py"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    best = None
    for line in out.strip().splitlines():
        # skip shells that happen to mention the path
        if "/bin/bash" in line or "pgrep" in line:
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmd = parts[1]
        if "stage2/train.py" not in cmd:
            continue
        m = _TAG_RE.search(cmd)
        tag = m.group(1) if m else None
        info = {"pid": pid, "cmd": cmd, "tag": tag, "is_torchrun_parent": "torchrun" in cmd}
        # prefer the python child (not the torchrun wrapper) for cleaner pgid kill
        if not info["is_torchrun_parent"]:
            return info
        best = best or info
    return best


def _has_resumable_ckpt(tag: str, output: str = DEFAULT_OUTPUT) -> bool:
    out_dir = REPO_ROOT / output / "efficient_sam3_stage2" / tag
    if not out_dir.is_dir():
        return False
    if (out_dir / "ckpt_latest.pth").exists() or (out_dir / "ckpt_running.pth").exists():
        return True
    return any(out_dir.glob("ckpt_epoch_*.pth"))


def _safe_tag(tag: str) -> str:
    """Whitelist tag → only ASCII letters/digits/underscore/hyphen."""
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", tag):
        raise HTTPException(400, f"Invalid tag (alnum, _, -, max 64 chars): {tag!r}")
    return tag


@app.get("/api/train/status")
def train_status(tag: str = Query(DEFAULT_TAG)):
    """Liveness + resume-availability check.

    Reports the running training process if any, plus whether the requested
    tag has on-disk checkpoints that AUTO_RESUME would pick up.
    """
    _safe_tag(tag)
    proc = _find_training_process()
    return {
        "alive": proc is not None,
        "pid": proc["pid"] if proc else None,
        "tag": proc["tag"] if proc else None,
        "cmd": proc["cmd"] if proc else None,
        "has_resumable_ckpt": _has_resumable_ckpt(tag),
        "default_tag": DEFAULT_TAG,
    }


class TrainStartReq(BaseModel):
    tag: str = Field(default=DEFAULT_TAG)
    cfg: str = Field(default=DEFAULT_CFG)
    data_path: str = Field(default=DEFAULT_DATA)
    output: str = Field(default=DEFAULT_OUTPUT)
    master_port: int = Field(default=DEFAULT_PORT, ge=1024, le=65535)


@app.post("/api/train/start")
def train_start(req: TrainStartReq):
    """Launch torchrun in a new session group so dashboard restarts don't kill it."""
    tag = _safe_tag(req.tag)

    proc = _find_training_process()
    if proc is not None:
        raise HTTPException(
            409,
            f"Training already running: pid={proc['pid']} tag={proc['tag']!r}",
        )

    cfg_path = REPO_ROOT / req.cfg
    if not cfg_path.is_file():
        raise HTTPException(400, f"cfg not found: {cfg_path}")

    LOGS_DIR.mkdir(exist_ok=True)
    # naming: logs/stage2_run3.log (strip 'ep50_' prefix if present)
    short = tag.replace("ep50_", "") if tag.startswith("ep50_") else tag
    log_path = LOGS_DIR / f"stage2_{short}.log"
    pid_path = LOGS_DIR / f"stage2_{short}.pid"

    cmd = [
        "torchrun", "--nproc_per_node=1", f"--master_port={req.master_port}",
        "stage2/train.py",
        "--cfg", req.cfg,
        "--data-path", req.data_path,
        "--tag", tag,
        "--output", req.output,
    ]

    # Open log in append mode so resumes don't clobber history.
    log_f = open(log_path, "a")
    log_f.write(f"\n=== [dashboard] launching at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log_f.flush()

    sub = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # so SIGTERM to dashboard doesn't cascade
    )
    pid_path.write_text(str(sub.pid))

    # brief settle
    time.sleep(0.5)
    alive = _process_alive(sub.pid)

    return {
        "ok": alive,
        "pid": sub.pid,
        "tag": tag,
        "cmd": " ".join(cmd),
        "log": str(log_path.relative_to(REPO_ROOT)),
        "pid_file": str(pid_path.relative_to(REPO_ROOT)),
        "resumed": _has_resumable_ckpt(tag, req.output),
    }


@app.post("/api/train/stop")
def train_stop():
    """SIGTERM the training process group; SIGKILL fallback after 10 s."""
    proc = _find_training_process()
    if proc is None:
        raise HTTPException(404, "No training process running")

    pid = proc["pid"]
    # Kill the process *group* so torchrun + python child + dataloader workers all go.
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass

    deadline = time.time() + 10.0
    while time.time() < deadline:
        if not _process_alive(pid):
            break
        time.sleep(0.5)

    killed_with_sigkill = False
    if _process_alive(pid):
        killed_with_sigkill = True
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(1.0)

    final_alive = _process_alive(pid)
    return {
        "ok": not final_alive,
        "pid": pid,
        "tag": proc["tag"],
        "killed_with_sigkill": killed_with_sigkill,
    }


# ---------------------------------------------------------------------------
# Static
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
