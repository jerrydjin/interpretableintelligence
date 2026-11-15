"""Where a remote run executes.

A backend stores clips and run folders under one root and runs ``worker.run_events`` next to
them. ``ModalBackend`` is the real one: a Modal Volume plus a GPU function. ``LocalBackend``
runs the same worker in this process against a directory; the tests use it, and it is the
reference for what any future backend has to do.
"""

from __future__ import annotations

import io
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from interpretableintelligence.remote.worker import run_events

CLIPS_MARKER_SUFFIX = ".complete"


class RemoteSetupError(RuntimeError):
    """The remote backend is not installed or not logged in. The message says how to fix it."""


def clip_files(clips_dir: Path) -> list[Path]:
    """Files a remote run needs from ``clips_dir``: everything except hidden files."""
    clips_dir = Path(clips_dir)
    found = []
    for dirpath, dirnames, filenames in os.walk(clips_dir):
        dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
        for name in sorted(filenames):
            if not name.startswith("."):
                found.append(Path(dirpath) / name)
    return found


class LocalBackend:
    """Run the remote worker in this process, with ``root`` standing in for the Volume."""

    name = "local"

    def __init__(self, root: Path, gpu: str = "L4", timeout_s: int = 3600) -> None:
        self.root = Path(root)
        self.gpu = gpu
        self.timeout_s = timeout_s
        self.uploads = 0

    def check(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def has_clips(self, key: str) -> bool:
        return (self.root / "clips" / f"{key}{CLIPS_MARKER_SUFFIX}").is_file()

    def upload_clips(self, clips_dir: Path, key: str) -> None:
        dest = self.root / "clips" / key
        for path in clip_files(clips_dir):
            target = dest / path.relative_to(clips_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        (self.root / "clips" / f"{key}{CLIPS_MARKER_SUFFIX}").write_text("ok\n", encoding="utf-8")
        self.uploads += 1

    @contextmanager
    def session(self) -> Iterator[_LocalSession]:
        yield _LocalSession(self.root)


class _LocalSession:
    def __init__(self, root: Path) -> None:
        self.root = root

    def events(self, request: dict) -> Iterator[dict]:
        yield from run_events(request, self.root)

    def fetch(self, path: str, dest: Path) -> None:
        shutil.copyfile(self.root / path, dest)


class ModalBackend:
    """A Modal GPU function and a Modal Volume, on the account ``modal setup`` logged in."""

    name = "modal"

    def __init__(self, gpu: str, timeout_s: int, groot: bool = False) -> None:
        self.gpu = gpu
        self.timeout_s = int(timeout_s)
        self.groot = bool(groot)

    def check(self) -> None:
        try:
            import modal  # noqa: F401, PLC0415
        except ImportError as exc:
            raise RemoteSetupError(
                "--remote runs on Modal and needs its client: uv sync --extra remote "
                "(or pip install 'interpretableintelligence[remote]'), then modal setup"
            ) from exc
        from modal.config import config  # noqa: PLC0415

        if not (config.get("token_id") and config.get("token_secret")):
            raise RemoteSetupError(
                "Modal is not logged in on this machine: run `modal setup` once "
                "(or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET)"
            )

    def _volume(self):
        from interpretableintelligence.remote import modal_app  # noqa: PLC0415

        return modal_app.volume

    def has_clips(self, key: str) -> bool:
        from modal.exception import NotFoundError  # noqa: PLC0415

        try:
            for _chunk in self._volume().read_file(f"clips/{key}{CLIPS_MARKER_SUFFIX}"):
                pass
        except (NotFoundError, FileNotFoundError):
            return False
        return True

    def upload_clips(self, clips_dir: Path, key: str) -> None:
        volume = self._volume()
        with volume.batch_upload(force=True) as batch:
            for path in clip_files(clips_dir):
                batch.put_file(str(path), f"/clips/{key}/{path.relative_to(clips_dir).as_posix()}")
        # The marker goes in last, in its own commit, so a half-finished upload is never reused.
        with volume.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(b"ok\n"), f"/clips/{key}{CLIPS_MARKER_SUFFIX}")

    @contextmanager
    def session(self) -> Iterator[_ModalSession]:
        import modal  # noqa: PLC0415

        from interpretableintelligence.remote import modal_app  # noqa: PLC0415

        app = modal_app.groot_app if self.groot else modal_app.app
        function = modal_app.run_pipeline_groot if self.groot else modal_app.run_pipeline
        options: dict = {"gpu": self.gpu, "timeout": self.timeout_s}
        token = os.environ.get("HF_TOKEN")
        if token:
            # Gated Hugging Face models need it. It goes to your own Modal account only.
            options["secrets"] = [modal.Secret.from_dict({"HF_TOKEN": token})]
        with modal.enable_output(), app.run():
            yield _ModalSession(function.with_options(**options), modal_app.volume)


class _ModalSession:
    def __init__(self, function, volume) -> None:
        self.function = function
        self.volume = volume

    def events(self, request: dict) -> Iterator[dict]:
        yield from self.function.remote_gen(request)

    def fetch(self, path: str, dest: Path) -> None:
        with Path(dest).open("wb") as handle:
            for chunk in self.volume.read_file(path):
                handle.write(chunk)
