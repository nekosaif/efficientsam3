"""EfficientSAM3 training dashboard — FastAPI backend."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from dashboard.metrics import get_snapshot, get_scalars, get_all_scalar_tags
from dashboard.system import get_system

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="EfficientSAM3 Dashboard", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


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


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
