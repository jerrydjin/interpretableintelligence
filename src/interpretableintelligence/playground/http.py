"""HTTP routing for the playground backend: API v1 (docs/PLAYGROUND-API.md) over ``PackModel``.

Routing only. Every route validates its input, calls one ``PackModel`` method and returns its
result; no model code lives here, so a gateway in another language (for example Rust, next to
``server/``) can take over routing, auth or rate limits later by speaking the same contract.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from interpretableintelligence.playground.compute import BadRequest, PackModel, PackNotFound, PlaygroundError

API_NAME = "ii-playground"
API_VERSION = 1
MAX_BODY_BYTES = 4 * 1024 * 1024


class SteerBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    image_id: str
    features: list[int]
    kind: str
    value: float = 0.0


def create_app(
    packs: Mapping[str, PackModel] | Callable[[], Mapping[str, PackModel]],
    allow_origins: list[str] | tuple[str, ...] = ("*",),
) -> FastAPI:
    """``packs`` may be a callable, so a deployment can load the models after the app exists."""
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

    def pack(pack_id: str) -> PackModel:
        found = table().get(pack_id)
        if found is None:
            raise PackNotFound(f"no pack named {pack_id!r}")
        return found

    @app.exception_handler(PlaygroundError)
    async def _playground_error(_request: Request, exc: PlaygroundError) -> JSONResponse:
        return JSONResponse({"error": {"code": exc.code, "message": exc.message}}, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_request: Request, exc: RequestValidationError) -> JSONResponse:
        first = (exc.errors() or [{}])[0]
        where = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
        message = f"{where}: {first.get('msg', 'invalid')}" if where else str(first.get("msg", "invalid request"))
        return JSONResponse({"error": {"code": "bad_request", "message": message}}, status_code=400)

    @app.get("/v1/health")
    def health() -> dict:
        return {"api": API_NAME, "version": API_VERSION, "packs": sorted(table())}

    @app.get("/v1/packs")
    def list_packs() -> dict:
        return {"packs": [model.info() for _, model in sorted(table().items())]}

    @app.get("/v1/packs/{pack_id}")
    def get_pack(pack_id: str) -> dict:
        return pack(pack_id).info()

    @app.post("/v1/packs/{pack_id}/read")
    async def read(pack_id: str, request: Request) -> dict:
        model = pack(pack_id)
        content_type = request.headers.get("content-type", "").split(";")[0].strip()
        if content_type != "application/octet-stream":
            raise BadRequest("send the picture as application/octet-stream: image_size x image_size RGB, 8 bits each")
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            raise BadRequest("picture too large")
        return await asyncio.to_thread(model.read, body)

    @app.post("/v1/packs/{pack_id}/steer")
    async def steer(pack_id: str, body: SteerBody) -> dict:
        model = pack(pack_id)
        return await asyncio.to_thread(model.steer, body.image_id, body.features, body.kind, body.value)

    return app
