"""HTTP API.

Read routes are the contract ``interpretableintelligence-server`` matches. ``POST /explain`` stays
here because it has to run the model. The Rust server proxies that route.
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from interpretableintelligence import __version__
from interpretableintelligence.explain import (
    ExplainError,
    FeatureNotFound,
    RunNotFound,
    SAEUnusable,
    explain_clip,
    get_feature,
    list_runs,
    load_run,
    resolve_run_dir,
)
from interpretableintelligence.contract_view import STATIC_DIR, _resolve_pixels, load_run as load_viewer

_MODELS: dict[tuple[str, str], object] = {}
_LOCK = threading.Lock()


class ClipNotFound(ExplainError):
    """The clip or frame is not in the run."""


def create_app(runs_dir: Path, clips_dir: Path | None = None) -> FastAPI:
    """API rooted at a directory of run folders. ``run_id`` is the folder name."""
    root = Path(runs_dir)
    clips = None if clips_dir is None else Path(clips_dir)
    app = FastAPI(title="interpretableintelligence", version=__version__)

    @app.exception_handler(SAEUnusable)
    async def unusable(_request, exc: SAEUnusable):
        return JSONResponse(status_code=409, content={"detail": exc.as_dict()})

    @app.exception_handler(RunNotFound)
    async def missing_run(_request, exc: RunNotFound):
        return JSONResponse(status_code=404, content={"detail": exc.as_dict()})

    @app.exception_handler(FeatureNotFound)
    async def missing_feature(_request, exc: FeatureNotFound):
        return JSONResponse(status_code=404, content={"detail": exc.as_dict()})

    @app.exception_handler(ClipNotFound)
    async def missing_clip(_request, exc: ClipNotFound):
        return JSONResponse(status_code=404, content={"detail": exc.as_dict()})

    @app.exception_handler(ExplainError)
    async def explain_failed(_request, exc: ExplainError):
        return JSONResponse(status_code=400, content={"detail": exc.as_dict()})

    def viewer(run_dir: Path):
        key = (str(run_dir.resolve()), "" if clips is None else str(clips.resolve()))
        with _LOCK:
            cached = _MODELS.get(key)
        if cached is not None:
            return cached
        loaded = load_viewer(run_dir, clips)
        with _LOCK:
            _MODELS[key] = loaded
        return loaded

    @app.get("/runs")
    def runs() -> dict:
        return list_runs(root)

    @app.get("/runs/{run_id}")
    def run_detail(run_id: str) -> dict:
        ctx = load_run(resolve_run_dir(root, run_id))
        payload = dict(viewer(ctx.run_dir).run_payload)
        payload["model_sha"] = ctx.model_sha
        return payload

    @app.get("/features/{feature_id}")
    def feature(
        feature_id: int,
        run_id: str = Query(..., description="Run folder name under the server's runs directory."),
    ):
        return get_feature(resolve_run_dir(root, run_id), feature_id)

    @app.get("/features/{feature_id}/top_clips")
    def top_clips(
        feature_id: int,
        run_id: str = Query(..., description="Run folder name under the server's runs directory."),
    ) -> dict:
        ctx = load_run(resolve_run_dir(root, run_id))
        if feature_id not in ctx.features_by_id:
            raise FeatureNotFound(
                f"run {ctx.run_id} has no feature {feature_id}",
                run_id=ctx.run_id,
                model_sha=ctx.model_sha,
                layer=ctx.layer,
            )
        rows = viewer(ctx.run_dir).run_payload["features"]
        record = next((row for row in rows if row["id"] == feature_id), None)
        if record is None:
            raise FeatureNotFound(
                f"run {ctx.run_id} has no feature {feature_id}",
                run_id=ctx.run_id,
                model_sha=ctx.model_sha,
                layer=ctx.layer,
            )
        return {
            "run_id": ctx.run_id,
            "model_sha": ctx.model_sha,
            "layer": ctx.layer,
            "feature_id": feature_id,
            "label": record["label"],
            "badge": record["badge"],
            "top_clips": record["top_clips"],
        }

    @app.get("/clips/{clip_id}/frame/{frame}")
    def frame(
        clip_id: str,
        frame: int,
        run_id: str = Query(..., description="Run folder name under the server's runs directory."),
    ) -> dict:
        ctx = load_run(resolve_run_dir(root, run_id))
        clip = viewer(ctx.run_dir).clips.get(clip_id)
        if clip is None:
            raise ClipNotFound(
                f"run {ctx.run_id} has no clip {clip_id}",
                run_id=ctx.run_id,
                model_sha=ctx.model_sha,
                layer=ctx.layer,
            )
        n_frames = int(clip["n_frames"])
        if frame < 0 or frame >= n_frames:
            raise ClipNotFound(
                f"run {ctx.run_id} clip {clip_id} has no frame {frame}",
                run_id=ctx.run_id,
                model_sha=ctx.model_sha,
                layer=ctx.layer,
            )
        row = clip["frames"][frame]
        return {
            "run_id": ctx.run_id,
            "model_sha": ctx.model_sha,
            "layer": ctx.layer,
            "clip_id": clip_id,
            "frame": frame,
            "window": row["window"],
            "t": row["t"],
            "features": row["features"],
        }

    @app.get("/clips/{clip_id}/file")
    def clip_file(
        clip_id: str,
        run_id: str = Query(..., description="Run folder name under the server's runs directory."),
    ):
        ctx = load_run(resolve_run_dir(root, run_id))
        manifest_ids = [str(clip["clip_id"]) for clip in ctx.manifest.get("clips") or []]
        if clip_id not in manifest_ids:
            raise ClipNotFound(
                f"run {ctx.run_id} has no clip {clip_id}",
                run_id=ctx.run_id,
                model_sha=ctx.model_sha,
                layer=ctx.layer,
            )
        resolved = _resolve_pixels(ctx.manifest, ctx.run_dir, clips)
        loaded = resolved.get(clip_id)
        if loaded is None or loaded.status != "ok" or loaded.path is None:
            message = loaded.message if loaded is not None and loaded.message else "Clip file not found."
            raise ClipNotFound(
                message,
                run_id=ctx.run_id,
                model_sha=ctx.model_sha,
                layer=ctx.layer,
            )
        return FileResponse(loaded.path, media_type="application/octet-stream")

    @app.post("/explain")
    def explain(
        run_id: str = Form(..., description="Run folder name."),
        clip: UploadFile = File(..., description="A .npy clip, shaped [frames, height, width, channels]."),
    ) -> dict:
        payload = clip.file.read()
        filename = clip.filename or "upload.npy"
        clip_id = Path(filename).stem or "upload"
        handle = tempfile.NamedTemporaryFile(prefix="interpretableintelligence-clip-", suffix=".npy", delete=False)
        destination = Path(handle.name)
        try:
            handle.write(payload)
            handle.close()
            return explain_clip(resolve_run_dir(root, run_id), destination, clip_id=clip_id)
        finally:
            destination.unlink(missing_ok=True)

    if STATIC_DIR.is_dir():

        @app.get("/")
        def index():
            return FileResponse(STATIC_DIR / "index.html")

        @app.get("/index.html")
        def index_html():
            return FileResponse(STATIC_DIR / "index.html")

        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app
