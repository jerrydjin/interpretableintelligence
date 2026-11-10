"""Citation rules: the answer may contain only real features and only ablation effects."""

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from interpretableintelligence.adapters.fake import render_clip, render_cues
from interpretableintelligence.api import create_app
from interpretableintelligence.cli import app
from interpretableintelligence.explain import SAEUnusable, cites_feature, explain_clip


def _client(runs_dir: Path) -> TestClient:
    return TestClient(create_app(runs_dir))

runner = CliRunner()


def _scoreboard(run_dir: Path) -> dict:
    return json.loads((run_dir / "scoreboard.json").read_text())


def _save_clip(path: Path, frames: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, frames)
    return path


def assert_citation_contract(payload: dict, scoreboard: dict) -> None:
    """Every cited row is a real scoreboard feature with an ablation effect.

    ``unverified`` may hold estimate-only rows and features whose scoreboard
    record is not real. A real ablation never sits only in that list.
    """
    assert isinstance(payload["run_id"], str) and payload["run_id"]
    assert payload["model_sha"] == scoreboard["run"]["model_sha"]
    assert isinstance(payload["model_sha"], str) and len(payload["model_sha"]) == 64
    assert payload["layer"] == scoreboard["run"]["layer"]
    by_id = {feature["id"]: feature for feature in scoreboard["features"]}
    previous = math.inf
    assert "answer" in payload and "unverified" in payload
    for group in payload["answer"]:
        assert group["features"], "answer group has no features"
        effects = [feature["effect"] for feature in group["features"]]
        assert effects == sorted(effects, reverse=True)
        assert effects[0] <= previous
        previous = effects[0]
        for feature in group["features"]:
            assert feature["effect_kind"] == "ablation"
            assert feature["real"] is True
            record = by_id[feature["feature_id"]]
            assert record["real"] is True
            assert feature["label"] == record["label"]
            assert set(feature) == {"feature_id", "label", "effect", "effect_kind", "real"}
    for row in payload["unverified"]:
        record = by_id[row["feature_id"]]
        assert row["label"] == record["label"]
        assert row["real"] is record["real"]
        assert row["effect_kind"] in {"ablation", "estimate"}
        assert not (record["real"] is True and row["effect_kind"] == "ablation")
        if row["effect_kind"] == "estimate":
            assert row["reason"] == "estimate only"
        else:
            assert record["real"] is not True
            assert row["reason"] == "scoreboard real is false"


def _flat(payload: dict) -> list[dict]:
    return [feature for group in payload["answer"] for feature in group["features"]]


@pytest.mark.parametrize(
    ("real", "kind", "cited"),
    [
        (True, "ablation", True),
        (True, "estimate", False),
        (False, "ablation", False),
        (False, "estimate", False),
        (None, "ablation", False),
    ],
)
def test_cites_feature_matrix(real, kind, cited):
    assert cites_feature(real=real, effect_kind=kind) is cited


def test_citation_checker_rejects_a_non_real_or_an_estimate():
    scoreboard = {
        "run": {"model_sha": "a" * 64, "layer": 16},
        "features": [
            {"id": 0, "label": "stripe", "real": False},
            {"id": 1, "label": "pulse", "real": True},
        ],
    }
    base = {"run_id": "demo", "model_sha": "a" * 64, "layer": 16, "unverified": []}
    non_real = {
        **base,
        "answer": [
            {
                "window": 0,
                "t": 0,
                "features": [
                    {
                        "feature_id": 0,
                        "label": "stripe",
                        "effect": 1.0,
                        "effect_kind": "ablation",
                        "real": True,
                    }
                ],
            }
        ],
    }
    estimate = {
        **base,
        "answer": [
            {
                "window": 0,
                "t": 0,
                "features": [
                    {
                        "feature_id": 1,
                        "label": "pulse",
                        "effect": 1.0,
                        "effect_kind": "estimate",
                        "real": True,
                    }
                ],
            }
        ],
    }
    leaked = {
        **base,
        "answer": [],
        "unverified": [
            {
                "feature_id": 1,
                "label": "pulse",
                "window": 0,
                "t": 0,
                "effect": 1.0,
                "effect_kind": "ablation",
                "real": True,
                "reason": "not cited",
            }
        ],
    }
    with pytest.raises(AssertionError):
        assert_citation_contract(non_real, scoreboard)
    with pytest.raises(AssertionError):
        assert_citation_contract(estimate, scoreboard)
    with pytest.raises(AssertionError):
        assert_citation_contract(leaked, scoreboard)
    good = {
        **base,
        "answer": [
            {
                "window": 0,
                "t": 1,
                "features": [
                    {
                        "feature_id": 1,
                        "label": "pulse",
                        "effect": 0.4,
                        "effect_kind": "ablation",
                        "real": True,
                    }
                ],
            }
        ],
        "unverified": [
            {
                "feature_id": 0,
                "label": "stripe",
                "window": 0,
                "t": 0,
                "effect": 0.2,
                "effect_kind": "ablation",
                "real": False,
                "reason": "scoreboard real is false",
            },
            {
                "feature_id": 1,
                "label": "pulse",
                "window": 0,
                "t": 2,
                "effect": 0.01,
                "effect_kind": "estimate",
                "real": True,
                "reason": "estimate only",
            },
        ],
    }
    assert_citation_contract(good, scoreboard)


def test_render_cues_paints_only_the_named_channel():
    full = render_clip(40, 0)
    stripe = render_cues(40, 0, ["stripe"])
    flicker = render_cues(41, 0, ["flicker"])
    assert np.allclose(full[..., 0], stripe[..., 0])
    assert np.all(stripe[..., 1:6] == 0)
    assert np.allclose(render_clip(41, 0)[..., 3], flicker[..., 3])
    assert np.all(flicker[..., :3] == 0)
    assert np.all(flicker[..., 4:6] == 0)


def test_citation_rules_hold_for_every_explain_response(demo_run: Path, tmp_path: Path):
    scoreboard = _scoreboard(demo_run)
    clips = {
        "stripe": _save_clip(tmp_path / "probe-stripe.npy", render_cues(40, 0, ["stripe"])),
        "flicker": _save_clip(tmp_path / "probe-flicker.npy", render_cues(41, 0, ["flicker"])),
        "both": _save_clip(tmp_path / "probe-both.npy", render_cues(42, 0, ["stripe", "flicker"])),
    }
    payloads: list[dict] = []
    with _client(demo_run.parent) as client:
        for name, path in clips.items():
            payloads.append(explain_clip(demo_run, path))
            with path.open("rb") as handle:
                response = client.post(
                    "/explain",
                    data={"run_id": demo_run.name},
                    files={"clip": (path.name, handle, "application/octet-stream")},
                )
            assert response.status_code == 200, response.text
            payloads.append(response.json())
            cli = runner.invoke(app, ["explain", "--run", str(demo_run), "--clip", str(path), "--json"])
            assert cli.exit_code == 0, cli.output
            payloads.append(json.loads(cli.output))
            assert name in path.name

    assert len(payloads) == 9
    for payload in payloads:
        assert_citation_contract(payload, scoreboard)

    stripe_payloads = payloads[0:3]
    flicker_payloads = payloads[3:6]
    both_payloads = payloads[6:9]
    for payload in stripe_payloads:
        cited = _flat(payload)
        assert cited, payload
        assert cited[0]["label"] == "stripe"
        assert cited[0]["effect"] > 0
        assert {feature["label"] for feature in cited} == {"stripe"}
        assert payload["answer"][0]["features"][0]["label"] == "stripe"
    for payload in flicker_payloads:
        assert payload["answer"] == []
        assert payload["unverified"]
        assert {row["label"] for row in payload["unverified"]} == {"flicker"}
        assert all(row["real"] is False for row in payload["unverified"])
        assert all(row["effect_kind"] == "ablation" for row in payload["unverified"])
    for payload in both_payloads:
        cited = _flat(payload)
        assert cited[0]["label"] == "stripe"
        assert "flicker" not in {feature["label"] for feature in cited}
        assert any(row["label"] == "flicker" and row["real"] is False for row in payload["unverified"])


def test_cli_text_ranks_the_planted_feature_and_hides_the_decoy(demo_run: Path, tmp_path: Path):
    stripe = _save_clip(tmp_path / "probe-stripe.npy", render_cues(40, 0, ["stripe"]))
    flicker = _save_clip(tmp_path / "probe-flicker.npy", render_cues(41, 0, ["flicker"]))
    planted = runner.invoke(app, ["explain", "--run", str(demo_run), "--clip", str(stripe)])
    assert planted.exit_code == 0, planted.output
    assert f"run_id: {demo_run.name}" in planted.output
    assert "model_sha:" in planted.output
    assert "layer: 16" in planted.output
    first = next(line for line in planted.output.splitlines() if line.startswith("  1."))
    assert "stripe" in first
    assert "ablation" in first
    decoy = runner.invoke(app, ["explain", "--run", str(demo_run), "--clip", str(flicker)])
    assert decoy.exit_code == 0, decoy.output
    assert "(no cited features)" in decoy.output
    assert "flicker" in decoy.output
    assert "real=false" in decoy.output
    assert "  1." not in decoy.output


def test_unusable_sae_is_refused(demo_run: Path, tmp_path: Path):
    copied = tmp_path / "demo"
    shutil.copytree(demo_run, copied)
    scoreboard_path = copied / "scoreboard.json"
    scoreboard = json.loads(scoreboard_path.read_text())
    scoreboard["sae"]["usable"] = False
    scoreboard["sae"]["loss_recovered"] = 0.1
    scoreboard_path.write_text(json.dumps(scoreboard, indent=2) + "\n")
    clip = _save_clip(tmp_path / "probe-stripe.npy", render_cues(40, 0, ["stripe"]))
    with pytest.raises(SAEUnusable) as caught:
        explain_clip(copied, clip)
    message = str(caught.value)
    assert "not usable" in message
    assert "No features were cited" in message
    assert caught.value.run_id == "demo"
    assert caught.value.model_sha == scoreboard["run"]["model_sha"]
    assert caught.value.layer == 16
    assert "answer" not in caught.value.as_dict()

    cli = runner.invoke(app, ["explain", "--run", str(copied), "--clip", str(clip)])
    text = cli.output + (cli.stderr or "")
    assert cli.exit_code == 1
    assert "not usable" in text
    assert "run_id: demo" in text
    assert f"model_sha: {scoreboard['run']['model_sha']}" in text
    assert "layer: 16" in text

    runs = tmp_path / "runs"
    runs.mkdir()
    api_copy = runs / "demo"
    shutil.copytree(copied, api_copy)
    with _client(runs) as client, clip.open("rb") as handle:
        response = client.post(
            "/explain",
            data={"run_id": "demo"},
            files={"clip": ("probe-stripe.npy", handle, "application/octet-stream")},
        )
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "not usable" in detail["error"]
    assert detail["run_id"] == "demo"
    assert detail["model_sha"] == scoreboard["run"]["model_sha"]
    assert detail["layer"] == 16
    assert "answer" not in detail


def test_clearing_real_moves_the_planted_feature_to_unverified(demo_run: Path, tmp_path: Path):
    copied = tmp_path / "demo"
    shutil.copytree(demo_run, copied)
    scoreboard_path = copied / "scoreboard.json"
    scoreboard = json.loads(scoreboard_path.read_text())
    for feature in scoreboard["features"]:
        feature["real"] = False
    scoreboard_path.write_text(json.dumps(scoreboard, indent=2) + "\n")
    clip = _save_clip(tmp_path / "probe-stripe.npy", render_cues(40, 0, ["stripe"]))
    payload = explain_clip(copied, clip)
    assert_citation_contract(payload, scoreboard)
    assert payload["answer"] == []
    assert any(row["label"] == "stripe" and row["effect_kind"] == "ablation" for row in payload["unverified"])


def test_api_lists_runs_and_reads_a_feature(demo_run: Path):
    scoreboard = _scoreboard(demo_run)
    with _client(demo_run.parent) as client:
        listed = client.get("/runs")
        assert listed.status_code == 200
        runs = listed.json()["runs"]
        assert len(runs) == 1
        summary = runs[0]
        assert summary["run_id"] == demo_run.name
        assert summary["model_sha"] == scoreboard["run"]["model_sha"]
        assert summary["layer"] == scoreboard["run"]["layer"]
        assert summary["usable"] is True

        real = next(feature for feature in scoreboard["features"] if feature["real"])
        got = client.get(f"/features/{real['id']}", params={"run_id": demo_run.name})
        assert got.status_code == 200, got.text
        body = got.json()
        assert body["run_id"] == demo_run.name
        assert body["model_sha"] == scoreboard["run"]["model_sha"]
        assert body["layer"] == scoreboard["run"]["layer"]
        assert body["feature"]["id"] == real["id"]
        assert body["feature"]["label"] == real["label"]
        assert body["feature"]["real"] is True
        assert body["feature"]["ablation_effect_kind"] == "ablation"
        assert body["feature"]["ablation_effect"] == real["ablation_effect"]

        missing = client.get("/features/999", params={"run_id": demo_run.name})
        assert missing.status_code == 404
        detail = missing.json()["detail"]
        assert detail["run_id"] == demo_run.name
        assert detail["model_sha"] == scoreboard["run"]["model_sha"]
        assert detail["layer"] == scoreboard["run"]["layer"]

        escaped = client.get("/features/0", params={"run_id": "../demo"})
        assert escaped.status_code == 400
        assert "run_id" in escaped.json()["detail"]
        assert "model_sha" in escaped.json()["detail"]
        assert "layer" in escaped.json()["detail"]
