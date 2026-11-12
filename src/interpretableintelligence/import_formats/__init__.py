"""Import external capture layouts into the ``ii`` run contract."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from interpretableintelligence.import_formats.groot_v9 import import_groot_v9

IMPORTERS: dict[str, Callable[..., None]] = {
    "groot-v9": import_groot_v9,
}


def import_run(format_name: str, src: Path, out: Path, **options) -> None:
    """Dispatch to a registered import format."""
    key = format_name.strip().lower()
    try:
        importer = IMPORTERS[key]
    except KeyError as exc:
        known = ", ".join(sorted(IMPORTERS))
        raise ValueError(f"unknown import format {format_name!r}; known: {known}") from exc
    importer(src=Path(src), out=Path(out), **options)
