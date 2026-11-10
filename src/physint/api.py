"""Local HTTP API. Routes call :mod:`physint.explain` and add no scoring of their own."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse

from physint import __version__
from physint.explain import (
    ExplainError,
    FeatureNotFound,
    RunNotFound,
    SAEUnusable,
    explain_clip,
    get_feature,
    list_runs,
    resolve_run_dir,
)


def create_app(runs_dir: Path) -> FastAPI:
    """API rooted at a directory of run folders. ``run_id`` is the folder name."""
    root = Path(runs_dir)
    app = FastAPI(title="physint", version=__version__)

    @app.exception_handler(SAEUnusable)
    async def unusable(_request, exc: SAEUnusable):
        return JSONResponse(status_code=409, content={"detail": exc.as_dict()})

    @app.exception_handler(RunNotFound)
    async def missing_run(_request, exc: RunNotFound):
        return JSONResponse(status_code=404, content={"detail": exc.as_dict()})

    @app.exception_handler(FeatureNotFound)
    async def missing_feature(_request, exc: FeatureNotFound):
        return JSONResponse(status_code=404, content={"detail": exc.as_dict()})

    @app.exception_handler(ExplainError)
    async def explain_failed(_request, exc: ExplainError):
        return JSONResponse(status_code=400, content={"detail": exc.as_dict()})

    @app.get("/runs")
    def runs() -> dict:
        return list_runs(root)

    @app.get("/features/{feature_id}")
    def feature(feature_id: int, run_id: str = Query(..., description="Run folder name under the server's runs directory.")):
        return get_feature(resolve_run_dir(root, run_id), feature_id)

    @app.post("/explain")
    def explain(
        run_id: str = Form(..., description="Run folder name."),
        clip: UploadFile = File(..., description="A .npy clip, shaped [frames, height, width, channels]."),
    ) -> dict:
        payload = clip.file.read()
        filename = clip.filename or "upload.npy"
        clip_id = Path(filename).stem or "upload"
        # The hash is of these bytes, so the file explain reads has to be exactly the upload.
        handle = tempfile.NamedTemporaryFile(prefix="physint-clip-", suffix=".npy", delete=False)
        destination = Path(handle.name)
        try:
            handle.write(payload)
            handle.close()
            return explain_clip(resolve_run_dir(root, run_id), destination, clip_id=clip_id)
        finally:
            destination.unlink(missing_ok=True)

    return app
