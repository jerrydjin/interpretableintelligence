"""``ii import``: turn rollout logs captured outside ``ii`` into a run folder.

A format is a reader that returns an :class:`~interpretableintelligence.importers.common.ImportedCapture`.
The shared writer in ``common`` turns that into the run contract with the same
features stage ``ii run`` uses. Anything it cannot compute without the model in
the loop is written as ``null`` and given a reason in ``manifest.import.nulls``.
``SCHEMA.md`` ("Imported runs") is the contract.

Register a new format in :data:`FORMATS`.
"""

from __future__ import annotations

import hashlib
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from interpretableintelligence.importers.common import ImportedCapture, write_run

# name -> (reader module, reader function, one-line description)
FORMATS: dict[str, tuple[str, str, str]] = {
    "groot-v9": (
        "interpretableintelligence.importers.groot_v9",
        "read_groot_v9",
        "GR00T v9 rollout capture (rollcap-v0 layout: acts/, sae/, split.json, episodes.jsonl)",
    ),
}


def format_names() -> list[str]:
    return sorted(FORMATS)


def reader_for(name: str) -> Callable[..., ImportedCapture]:
    key = name.strip().lower()
    if key not in FORMATS:
        raise ValueError(f"unknown import format {name!r}; known formats: {', '.join(format_names())}")
    module_name, function, _description = FORMATS[key]
    import importlib

    return getattr(importlib.import_module(module_name), function)


def import_run(
    format_name: str,
    source: Path,
    out: Path,
    *,
    log: Callable[[str], None] = print,
    **options,
) -> dict:
    """Import ``source`` (a directory, or a .tar / .tar.xz / .tar.gz of one) into the run folder ``out``.

    Returns the written manifest. ``out`` must not exist or must be empty.
    """
    source = Path(source)
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"refusing to write into non-empty {out}")
    reader = reader_for(format_name)
    if source.is_dir():
        capture = reader(source, log=log, **options)
        return write_run(capture, out, log=log)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not tarfile.is_tarfile(source):
        raise ValueError(f"{source} is neither a directory nor a tar archive")
    out.parent.mkdir(parents=True, exist_ok=True)
    archive = {"name": source.name, "sha256": _file_sha256(source), "bytes": int(source.stat().st_size)}
    log(f"archive {source.name}  sha256={archive['sha256']}")
    with tempfile.TemporaryDirectory(prefix=".ii-import-", dir=out.parent) as scratch:
        root = extract_archive(source, Path(scratch), log=log)
        capture = reader(root, log=log, **options)
        capture.source["archive"] = archive
        return write_run(capture, out, log=log)


def extract_archive(archive: Path, dest: Path, *, log: Callable[[str], None] = print) -> Path:
    """Extract regular files and directories only, then return the capture root.

    Absolute paths, ``..`` components, links and device files are refused. The
    root is ``dest`` itself, or its single top-level directory.
    """
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"unsafe path in archive: {member.name!r}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"archive member {member.name!r} is not a regular file or directory")
        log(f"extracting {len(members)} entries")
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, members=members, filter="data")
        else:  # pragma: no cover - Python < 3.11.4; members were checked above
            tar.extractall(dest, members=members)
    entries = [path for path in dest.iterdir()]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return dest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "FORMATS",
    "ImportedCapture",
    "extract_archive",
    "format_names",
    "import_run",
    "reader_for",
    "write_run",
]
