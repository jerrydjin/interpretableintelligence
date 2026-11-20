"""Shared fixtures. The fake acceptance run is built once."""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import pytest
from typer.testing import CliRunner

from interpretableintelligence.cli import app

runner = CliRunner()


class Demo(NamedTuple):
    run: Path
    clips: Path
    # Model calls during scoring, counted while the run was built: ``attribution`` is one forward and one
    # backward per window; ``random_controls`` is every edited forward the random-feature controls made.
    passes: dict[str, int] | None = None


@pytest.fixture(scope="session", autouse=True)
def _activation_cache(tmp_path_factory):
    """Keep the shared activation cache inside the test session, never in the user's home."""
    os.environ["II_ACTS_CACHE"] = str(tmp_path_factory.mktemp("acts-cache"))
    yield


@pytest.fixture(scope="session", autouse=True)
def _no_label_provider():
    """A developer's API key must never turn tests into paid vision-model calls."""
    os.environ["II_LABEL_PROVIDER"] = "none"
    yield


@pytest.fixture(scope="session")
def demo(tmp_path_factory, _activation_cache) -> Demo:
    os.environ.setdefault("II_DEVICE", "cpu")
    root = tmp_path_factory.mktemp("interpretableintelligence")
    clips = root / "clips"
    out = root / "runs" / "demo"
    generated = runner.invoke(app, ["generate-clips", "--out", str(clips), "--n", "30", "--seed", "0"])
    assert generated.exit_code == 0, generated.output
    with pytest.MonkeyPatch.context() as patch:
        passes = _count_scoring_passes(patch)
        ran = _run_demo(clips, out)
    assert ran.exit_code == 0, ran.output
    checked = runner.invoke(app, ["validate", str(out)])
    assert checked.exit_code == 0, checked.output + (ran.output or "")
    return Demo(out, clips, passes)


def _count_scoring_passes(patch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count the fake adapter's attribution passes and the random controls' forwards, in place of wall-clock time."""
    from interpretableintelligence import pipeline
    from interpretableintelligence.adapters.fake import FakeAdapter

    passes = {"attribution": 0, "random_controls": 0}
    in_controls = [False]
    objective, predict, percentile = FakeAdapter.attribution_objective, FakeAdapter.predict_masked, pipeline._random_percentile

    def counted_objective(self, *args, **kwargs):
        passes["attribution"] += 1
        return objective(self, *args, **kwargs)

    def counted_predict(self, *args, **kwargs):
        if in_controls[0]:
            passes["random_controls"] += 1
        return predict(self, *args, **kwargs)

    def counted_percentile(*args, **kwargs):
        in_controls[0] = True
        try:
            return percentile(*args, **kwargs)
        finally:
            in_controls[0] = False

    patch.setattr(FakeAdapter, "attribution_objective", counted_objective)
    patch.setattr(FakeAdapter, "predict_masked", counted_predict)
    patch.setattr(pipeline, "_random_percentile", counted_percentile)
    return passes


def _run_demo(clips: Path, out: Path):
    return runner.invoke(
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


@pytest.fixture(scope="session")
def demo_run(demo: Demo) -> Path:
    return demo.run
