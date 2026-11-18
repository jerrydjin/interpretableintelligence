"""HTTP routing for the playground backend: API v1 (docs/PLAYGROUND-API.md) over ``PackModel``.

Routing only. Every route validates its input, calls one ``PackModel`` method and returns its
result; no model code lives here, so a gateway in another language (for example Rust, next to
``server/``) can take over routing, auth or rate limits later by speaking the same contract.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict

from interpretableintelligence.playground.compute import BadRequest, PackModel, PackNotFound, PlaygroundError

if TYPE_CHECKING:
    from interpretableintelligence.playground.builds import Builds

API_NAME = "ii-playground"
API_VERSION = 1
MAX_BODY_BYTES = 4 * 1024 * 1024


class SteerBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    image_id: str
    features: list[int]
    kind: str
    value: float = 0.0


class ModelBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    model: str
    layer: int | None = None


def create_app(
    packs: Mapping[str, PackModel] | Callable[[], Mapping[str, PackModel]],
    allow_origins: list[str] | tuple[str, ...] = ("*",),
    builds: "Builds | Callable[[], Builds | None] | None" = None,
) -> FastAPI:
    """``packs`` may be a callable, so a deployment can load the models after the app exists.

    ``builds`` (``playground.builds.Builds``, or a callable returning one) turns on bring your own
    model: ``POST /v1/models``, ``GET /v1/jobs/{id}`` and ``GET /v1/packs/{pack}/files/{name}``.
    Without it those routes answer ``builds_off``."""
    app = FastAPI(title="ii playground", version=str(API_VERSION), docs_url="/v1/docs", openapi_url="/v1/openapi.json")
    app.add_middleware(GZipMiddleware, minimum_size=2048)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(allow_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["content-type"],
        max_age=3600,
    )

    def table() -> Mapping[str, PackModel]:
        return packs() if callable(packs) else packs

    def builder() -> "Builds":
        from interpretableintelligence.playground.builds import BuildsOff

        found = builds() if callable(builds) else builds
        if found is None:
            raise BuildsOff("this server does not build models; it serves the packs it shipped with")
        return found

    def pack(pack_id: str) -> PackModel:
        found = table().get(pack_id)
        if found is None:
            raise PackNotFound(f"no pack named {pack_id!r}")
        return found

    @app.exception_handler(PlaygroundError)
    async def _playground_error(_request: Request, exc: PlaygroundError) -> JSONResponse:
        return JSONResponse({"error": {"code": exc.code, "message": exc.message}}, status_code=exc.status)

    from interpretableintelligence.playground.builds import BuildError

    @app.exception_handler(BuildError)
    async def _build_error(_request: Request, exc: BuildError) -> JSONResponse:
        return JSONResponse({"error": {"code": exc.code, "message": exc.message}}, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_request: Request, exc: RequestValidationError) -> JSONResponse:
        first = (exc.errors() or [{}])[0]
        where = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
        message = f"{where}: {first.get('msg', 'invalid')}" if where else str(first.get("msg", "invalid request"))
        return JSONResponse({"error": {"code": "bad_request", "message": message}}, status_code=400)

    @app.get("/v1/health")
    def health() -> dict:
        found = builds() if callable(builds) else builds
        out = {"api": API_NAME, "version": API_VERSION, "packs": sorted(table())}
        if found is not None:
            out["builds"] = found.info()  # present only where the page may build a model
        return out

    @app.get("/v1/packs")
    def list_packs() -> dict:
        current = table()
        summaries = getattr(current, "summaries", None)
        if callable(summaries):
            return {"packs": summaries()}
        return {"packs": [model.info() for _, model in sorted(current.items())]}

    @app.get("/v1/packs/{pack_id}")
    async def get_pack(pack_id: str) -> dict:
        # A built pack loads on first use, which can take a few seconds: off the event loop.
        return (await asyncio.to_thread(pack, pack_id)).info()

    @app.get("/v1/packs/{pack_id}/files/{name:path}")
    def pack_file(pack_id: str, name: str) -> FileResponse:
        path = builder().file(pack_id, name)
        media = "application/json" if name.endswith(".json") else "image/jpeg"
        return FileResponse(path, media_type=media, headers={"cache-control": "public, max-age=3600"})

    @app.post("/v1/models")
    async def build_model(body: ModelBody) -> JSONResponse:
        job, created = await asyncio.to_thread(builder().submit, body.model, body.layer)
        return JSONResponse({"job": job}, status_code=202 if job["state"] != "done" else 200, headers={"x-ii-new-build": str(created).lower()})

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str) -> dict:
        return {"job": await asyncio.to_thread(builder().job, job_id)}

    @app.post("/v1/packs/{pack_id}/read")
    async def read(pack_id: str, request: Request) -> dict:
        model = await asyncio.to_thread(pack, pack_id)
        content_type = request.headers.get("content-type", "").split(";")[0].strip()
        if content_type != "application/octet-stream":
            raise BadRequest("send the picture as application/octet-stream: image_size x image_size RGB, 8 bits each")
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            raise BadRequest("picture too large")
        return await asyncio.to_thread(model.read, body)

    @app.post("/v1/packs/{pack_id}/steer")
    async def steer(pack_id: str, body: SteerBody) -> dict:
        model = await asyncio.to_thread(pack, pack_id)
        return await asyncio.to_thread(model.steer, body.image_id, body.features, body.kind, body.value)

    return app
