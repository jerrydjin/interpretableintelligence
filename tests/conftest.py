"""Shared fixtures. The fake acceptance run is built once."""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import pytest
from typer.testing import CliRunner

from physint.cli import app

runner = CliRunner()


class Demo(NamedTuple):
    run: Path
    clips: Path


@pytest.fixture(scope="session")
def demo(tmp_path_factory) -> Demo:
    os.environ.setdefault("PHYSINT_DEVICE", "cpu")
    root = tmp_path_factory.mktemp("physint")
    clips = root / "clips"
    out = root / "runs" / "demo"
    generated = runner.invoke(app, ["generate-clips", "--out", str(clips), "--n", "30", "--seed", "0"])
    assert generated.exit_code == 0, generated.output
    ran = runner.invoke(
        app,
        [
            "run",
            "fake",
            "--clips",
            str(clips),
            "--layer",
            "16",
            "--seed",
            "0",
            "--heldout-frac",
            "0.2",
            "--out",
            str(out),
            "--no-serve",
            "--no-open",
        ],
    )
    assert ran.exit_code == 0, ran.output
    checked = runner.invoke(app, ["validate", str(out)])
    assert checked.exit_code == 0, checked.output + (ran.output or "")
    return Demo(out, clips)


@pytest.fixture(scope="session")
def demo_run(demo: Demo) -> Path:
    return demo.run
