"""Find the Rust ``interpretableintelligence-server`` binary that serves the viewer and read API.

In order:

1. ``II_SERVER``: a path to a binary, used as is.
2. A repository checkout with ``cargo``: ``cargo build -p interpretableintelligence-server``
   (a no-op when it is up to date), so edits to ``server/`` are always what runs.
3. The prebuilt binary for this platform from the GitHub release named in
   ``server_release.json``, downloaded once into the user cache and checked against the
   SHA-256 recorded there. A download that does not match is deleted and refused.

This module reads no run folder and loads no model; it only locates a program.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

NAME = "interpretableintelligence-server"
REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_FILE = Path(__file__).resolve().parent / "server_release.json"
DOWNLOAD = "https://github.com/{repo}/releases/download/{tag}/{asset}"


class ServerNotFound(RuntimeError):
    """No binary could be found, built or downloaded."""


def platform_asset() -> str | None:
    """Release asset name for this machine, or ``None`` when no prebuilt binary exists for it."""
    system = sys.platform
    machine = platform.machine().lower()
    arch = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "arm64"}.get(machine)
    if system.startswith("linux") and arch in {"x86_64", "aarch64"}:
        return f"{NAME}-linux-{arch}"
    if system == "darwin" and arch in {"arm64", "aarch64"}:
        return f"{NAME}-macos-arm64"
    return None


def release() -> dict:
    """``{"repo", "tag", "sha256": {asset: hex}}``. ``tag`` is ``None`` until a release is cut."""
    payload = json.loads(RELEASE_FILE.read_text(encoding="utf-8"))
    return {"repo": payload.get("repo"), "tag": payload.get("tag"), "sha256": dict(payload.get("sha256") or {})}


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "interpretableintelligence" / "server"


def find_server(*, build: bool = True) -> Path:
    """Path to a runnable binary. Raises :class:`ServerNotFound` with what to do next."""
    override = os.environ.get("II_SERVER")
    if override:
        path = Path(override)
        if not path.is_file():
            raise ServerNotFound(f"II_SERVER points at {path}, which is not a file")
        return path
    checkout = REPO_ROOT / "server" / "Cargo.toml"
    if build and checkout.is_file() and shutil.which("cargo"):
        return _cargo_build()
    return _prebuilt()


def _cargo_build() -> Path:
    command = ["cargo", "build", "-p", NAME]
    done = subprocess.run(command, cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if done.returncode != 0:
        tail = "\n".join(done.stderr.strip().splitlines()[-15:])
        raise ServerNotFound(f"`{' '.join(command)}` failed:\n{tail}")
    built = REPO_ROOT / "target" / "debug" / NAME
    if not built.is_file():
        raise ServerNotFound(f"cargo finished but {built} is missing")
    return built


def _prebuilt() -> Path:
    pinned = release()
    asset = platform_asset()
    hint = "Install Rust (https://rustup.rs) and run from a checkout, or set II_SERVER to a built binary."
    if pinned["tag"] is None or not pinned["repo"]:
        raise ServerNotFound(f"no prebuilt viewer binary has been released yet. {hint}")
    if asset is None or asset not in pinned["sha256"]:
        raise ServerNotFound(f"no prebuilt viewer binary for {sys.platform} {platform.machine()}. {hint}")
    expected = pinned["sha256"][asset]
    target = cache_dir() / pinned["tag"] / asset
    if target.is_file() and _sha256(target) == expected:
        return target
    url = DOWNLOAD.format(repo=pinned["repo"], tag=pinned["tag"], asset=asset)
    print(f"downloading the viewer binary ({pinned['tag']}) from {url}", file=sys.stderr, flush=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=target.parent, prefix=".download-", delete=False)
    partial = Path(handle.name)
    try:
        with handle, urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 (fixed https URL)
            shutil.copyfileobj(response, handle)
        got = _sha256(partial)
        if got != expected:
            raise ServerNotFound(f"downloaded {asset} has sha256 {got}, expected {expected}; refusing to run it")
        partial.chmod(0o755)
        partial.replace(target)
    except OSError as exc:
        raise ServerNotFound(f"could not download {url}: {exc}. {hint}") from exc
    finally:
        partial.unlink(missing_ok=True)
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
