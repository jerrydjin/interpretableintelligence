"""GR00T rollouts CLI wiring, schema export, and v9 comparison."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from interpretableintelligence.cli import app
from interpretableintelligence.validate import _check_rollouts

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "groot-1028-verify-slice.json"
V9_SNAPSHOT = Path("/cursor/stores/bot/data-export/ablate-1028/ablate_snapshot.final.tar")

runner = CliRunner()


def _minimal_run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "groot-run"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "model": "nvidia/gr00t17-lerobot-libero_10-640",
                "layer": 5,
                "tool_version": "test",
            }
        )
    )
    return run


def test_groot_verify_slice_writes_schema(tmp_path: Path):
    run = _minimal_run_dir(tmp_path)
    result = runner.invoke(
        app,
        ["rollouts", str(run), "--policy", "groot-mock", "--verify-slice"],
    )
    assert result.exit_code == 0, result.output
    assert (run / "rollouts" / "episodes.parquet").is_file()
    summary = json.loads((run / "rollouts" / "summary.json").read_text())
    assert summary["policy"] == "groot-mock"
    assert summary["seed"] == 40_000
    assert summary["task_id"] == "1028"
    assert summary["candidate_feature_ids"] == [720]
    assert summary["control_feature_ids"] == [7774, 2343]
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest.get("schema_version") == 1
    errors: list[str] = []
    _check_rollouts(run, 1, errors)
    assert errors == []


def test_import_v9_and_compare(tmp_path: Path):
    if not V9_SNAPSHOT.is_file():
        pytest.skip("bot store v9 snapshot not mounted")
    extract = tmp_path / "v9"
    extract.mkdir()
    subprocess.check_call(
        [
            "tar",
            "-xf",
            str(V9_SNAPSHOT),
            "-C",
            str(extract),
            "physint/groot/groot-1028-v9-ablate",
        ]
    )
    v9_root = extract / "physint/groot/groot-1028-v9-ablate"
    run = _minimal_run_dir(tmp_path / "run")
    result = runner.invoke(
        app,
        [
            "rollouts",
            str(run),
            "--policy",
            "groot-1028",
            "--verify-slice",
            "--v9-out",
            str(v9_root),
        ],
    )
    assert result.exit_code == 0
    cmp = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compare_groot_v9_rollouts.py"),
            str(run),
            "--v9-dir",
            str(v9_root),
            "--policy",
            "groot-1028",
        ],
        capture_output=True,
        text=True,
    )
    assert cmp.returncode == 0, cmp.stderr + cmp.stdout


def test_groot_policy_registered():
    from interpretableintelligence.policies import get_policy

    env = get_policy("groot-mock")
    assert env.layer == 5
    assert env.n_features == 8192