"""``POST /explain``, the one route that has to run the model.

Reads (``/runs``, ``/features``, ``/clips``, static assets) are served by the Rust binary
``interpretableintelligence-server`` (``server/``), which proxies ``POST /explain`` here with
``--explain-url``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from interpretableintelligence import __version__
from interpretableintelligence.explain import (
    ExplainError,
    FeatureNotFound,
    RunNotFound,
    SAEUnusable,
    explain_clip,
    resolve_run_dir,
)


def create_app(runs_dir: Path) -> FastAPI:
    """API rooted at a directory of run folders. ``run_id`` is the folder name."""
    root = Path(runs_dir)
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

    @app.exception_handler(ExplainError)
    async def explain_failed(_request, exc: ExplainError):
        return JSONResponse(status_code=400, content={"detail": exc.as_dict()})

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

    return app
