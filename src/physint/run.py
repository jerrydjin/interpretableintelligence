"""Backwards-compatible entry for the resumable pipeline.

``execute_run`` writes a finished scoreboard and returns. The CLI serves the
viewer unless ``--no-serve`` is set; call :func:`physint.pipeline.execute_pipeline`
directly when a script needs the report or the server.
"""

from __future__ import annotations

from pathlib import Path

from physint.pipeline import execute_pipeline
from physint.thresholds import DEFAULT_HELDOUT_FRAC, DEFAULT_LAYER, DEFAULT_SEED


def execute_run(
    adapter_name: str,
    clips_dir: Path,
    out_dir: Path,
    layer: int = DEFAULT_LAYER,
    seed: int = DEFAULT_SEED,
    heldout_frac: float = DEFAULT_HELDOUT_FRAC,
    tiny: bool = False,
    log=print,
) -> dict:
    """Run capture, SAE, features, and score. Does not start the viewer."""
    result = execute_pipeline(
        adapter_name=adapter_name,
        clips_dir=clips_dir,
        out_dir=out_dir,
        layer=layer,
        seed=seed,
        heldout_frac=heldout_frac,
        tiny=tiny,
        score=True,
        serve=False,
        open_browser=False,
        log=log,
    )
    scoreboard = result["scoreboard"]
    if scoreboard is None:
        raise RuntimeError("scoring did not produce a scoreboard")
    return scoreboard
