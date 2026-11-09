"""One reference run shared by the scoreboard and explain tests."""

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from physint.cli import app

runner = CliRunner()


@pytest.fixture(scope="session")
def demo_run(tmp_path_factory) -> Path:
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
        ],
    )
    assert ran.exit_code == 0, ran.output
    checked = runner.invoke(app, ["validate", str(out)])
    assert checked.exit_code == 0, checked.output
    return out
