"""``ii deploy-live``: put the playground backend on Modal and point the page at it."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

MODULE = "interpretableintelligence.remote.playground_app"
URL = re.compile(r"https://[A-Za-z0-9.-]+\.modal\.run\b")
CONFIG_NAME = "config.json"


class DeployError(RuntimeError):
    pass


def find_url(output: str) -> str | None:
    """The web endpoint URL ``modal deploy`` prints."""
    found = URL.findall(output)
    return found[-1] if found else None


def write_config(live_dir: Path, api: str | None) -> Path:
    """``demo-site/live/config.json``: where the page sends read and steer. ``null`` runs in the browser."""
    path = Path(live_dir) / CONFIG_NAME
    if api is not None and not re.fullmatch(r"https?://[^\s\"']+", api):
        raise DeployError(f"{api!r} is not an http(s) URL")
    path.write_text(json.dumps({"api": api.rstrip("/") if api else None}, indent=2) + "\n", encoding="utf-8")
    return path


def deploy(
    packs_dir: Path,
    gpu: str,
    idle_s: int,
    max_containers: int,
    origins: str,
    log: Callable[[str], None] = print,
) -> str:
    try:
        import modal  # noqa: F401
    except ImportError as exc:
        raise DeployError("ii deploy-live needs the Modal client: uv sync --extra remote, then modal setup") from exc
    modal_cli = shutil.which("modal")
    command = [modal_cli, "deploy", "-m", MODULE] if modal_cli else [sys.executable, "-m", "modal", "deploy", "-m", MODULE]
    env = dict(os.environ)
    env.update(
        II_PLAYGROUND_PACKS=str(Path(packs_dir).resolve()),
        II_PLAYGROUND_GPU=gpu,
        II_PLAYGROUND_IDLE=str(idle_s),
        II_PLAYGROUND_MAX_CONTAINERS=str(max_containers),
        II_PLAYGROUND_ORIGINS=origins,
    )
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        log(line.rstrip("\n"))
    if process.wait() != 0:
        raise DeployError("modal deploy failed (output above). If it asks for a token, run modal setup once.")
    url = find_url("".join(lines))
    if url is None:
        raise DeployError("deployed, but no *.modal.run URL in the output; pass it with ii deploy-live --api <url>")
    return url
