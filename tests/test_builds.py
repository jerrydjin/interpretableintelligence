"""Bring your own model: checks, limits, jobs and a real build through API v1 (playground.builds)."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from interpretableintelligence.playground import builds as b
from interpretableintelligence.playground.http import create_app
from interpretableintelligence.playground.registry import PackRegistry

VIT = {"model_type": "vit", "image_size": 224, "patch_size": 16, "num_hidden_layers": 12}


def _hub(params=86_000_000, config=None, gated=False, private=False, revision="0123456789abcdef"):
    return lambda _model: b.HubModel(revision=revision, params=params, gated=gated, private=private, config=config or VIT)


LIMITS = b.Limits()


def test_check_model_picks_the_middle_block_and_a_stable_pack_id():
    plan = b.check_model("google/vit-base-patch16-224", None, LIMITS, _hub())
    assert plan.layer == 6 and plan.n_layers == 12
    assert plan.pack_id == "google--vit-base-patch16-224-l6-01234567r1"
    assert b.check_model("google/vit-base-patch16-224", None, LIMITS, _hub()).pack_id == plan.pack_id
    assert b.check_model("google/vit-base-patch16-224", 3, LIMITS, _hub()).pack_id != plan.pack_id


@pytest.mark.parametrize(
    ("model", "layer", "hub", "error"),
    [
        ("not an id", None, _hub(), b.BuildError),
        ("../../etc/passwd", None, _hub(), b.BuildError),
        ("a/b", 0, _hub(), b.BuildError),
        ("a/b", 12, _hub(), b.BuildError),  # nothing would run after the edit
        ("a/b", True, _hub(), b.BuildError),
        ("a/b", None, _hub(params=1_100_000_000), b.ModelTooLarge),
        ("a/b", None, _hub(gated=True), b.ModelNotSupported),
        ("a/b", None, _hub(private=True), b.ModelNotSupported),
        ("a/b", None, _hub(config={"model_type": "resnet", "num_channels": 3, "depths": [2, 2]}), b.ModelNotSupported),
        ("a/b", None, _hub(config={"model_type": "clip", "vision_config": VIT, "text_config": {}}), b.ModelNotSupported),
    ],
)
def test_check_model_refuses_before_downloading(model, layer, hub, error):
    with pytest.raises(error):
        b.check_model(model, layer, LIMITS, hub)


def test_local_directories_only_when_allowed(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(VIT))
    with pytest.raises(b.BuildError):
        b.check_model(str(tmp_path), None, LIMITS, _hub())


def test_progress_follows_the_pipeline_log():
    assert b.progress_from_log("run capture", "loading") == ("capture", 0.08)
    stage, frac = b.progress_from_log("capture: 20/40 clips, about 1 min left", "capture")
    assert stage == "capture" and 0.08 < frac < 0.55
    stage, frac = b.progress_from_log("step 500/1000 mse=0.1 l0=32.00", "train")
    assert stage == "train" and frac == pytest.approx(0.70)
    assert b.progress_from_log("something else", "train") == ("train", None)


class _Recorder:
    def __init__(self):
        self.plans = []

    def __call__(self, plan):
        self.plans.append(plan)


def _builds(tmp_path, limits=LIMITS, hub=None):
    runner = _Recorder()
    return b.Builds(b.JobStore(tmp_path), runner, limits, hub or _hub()), runner


def test_submit_reuses_a_running_or_finished_build(tmp_path):
    builds, runner = _builds(tmp_path)
    job, new = builds.submit("google/vit-base-patch16-224")
    assert new and job["state"] == "queued" and len(runner.plans) == 1
    again, new = builds.submit("google/vit-base-patch16-224")
    assert not new and again["id"] == job["id"] and len(runner.plans) == 1
    # Once the pack exists the job reads as done, and nothing builds again.
    pack = tmp_path / "packs" / job["pack"]
    pack.mkdir(parents=True)
    (pack / "pack.json").write_text("{}")
    done, new = builds.submit("google/vit-base-patch16-224")
    assert not new and done["state"] == "done" and done["progress"] == 1.0 and len(runner.plans) == 1


def test_limits_cap_concurrent_and_daily_builds(tmp_path):
    hubs = {"a/one": "1" * 16, "a/two": "2" * 16, "a/three": "3" * 16}
    lookup = lambda model: b.HubModel(revision=hubs[model], params=1, gated=False, private=False, config=VIT)  # noqa: E731
    builds, runner = _builds(tmp_path, b.Limits(max_running=1, builds_per_day=2), lookup)
    first, _ = builds.submit("a/one")
    with pytest.raises(b.BuildLimit):
        builds.submit("a/two")  # one already running
    job = builds.store.load(first["pack"])
    job.state = "failed"
    builds.store.save(job)
    builds.submit("a/two")
    job = builds.store.load(b.check_model("a/two", None, LIMITS, lookup).pack_id)
    job.state = "failed"
    builds.store.save(job)
    with pytest.raises(b.BuildLimit, match="today"):
        builds.submit("a/three")
    assert len(runner.plans) == 2


def test_a_silent_build_times_out(tmp_path, monkeypatch):
    builds, _ = _builds(tmp_path, b.Limits(timeout_s=10))
    job, _ = builds.submit("google/vit-base-patch16-224")
    later = time.time() + 1000
    monkeypatch.setattr(b, "_now", lambda: later)
    assert builds.job(job["id"])["state"] == "failed"


def test_files_are_limited_to_what_the_page_needs(tmp_path):
    builds, _ = _builds(tmp_path)
    pack = tmp_path / "packs" / "p"
    (pack / "samples").mkdir(parents=True)
    (pack / "pack.json").write_text("{}")
    (pack / "sae.safetensors").write_bytes(b"x")
    assert builds.file("p", "pack.json").name == "pack.json"
    for name in ("sae.safetensors", "../jobs/p.json", "samples/../../x.jpg"):
        with pytest.raises(b.BuildError):
            builds.file("p", name)


def test_api_without_builds_says_so(tmp_path):
    client = TestClient(create_app({}))
    assert "builds" not in client.get("/v1/health").json()
    response = client.post("/v1/models", json={"model": "a/b"})
    assert response.status_code == 503 and response.json()["error"]["code"] == "builds_off"


def test_api_reports_model_errors(tmp_path):
    builds, _ = _builds(tmp_path, hub=_hub(params=10**10))
    client = TestClient(create_app({}, builds=builds))
    response = client.post("/v1/models", json={"model": "a/b"})
    assert response.status_code == 400 and response.json()["error"]["code"] == "model_too_large"
    assert client.post("/v1/models", json={"model": 3}).json()["error"]["code"] == "bad_request"
    assert client.get("/v1/jobs/nope").json()["error"]["code"] == "job_not_found"


def test_build_a_tiny_model_end_to_end(tiny_run, tmp_path):  # noqa: F811
    """POST /v1/models on a local ViT, poll the job, then read and steer the new pack."""
    root = tmp_path / "builds"
    store = b.JobStore(root)
    builds = b.Builds(store, b.thread_runner(store, tiny_run["clips"]), b.Limits(allow_local=True))
    table = PackRegistry({}, built_dir=root / "packs", device=torch.device("cpu"))
    client = TestClient(create_app(table, builds=builds))

    response = client.post("/v1/models", json={"model": str(tiny_run["model"]), "layer": 1})
    assert response.status_code == 202, response.text
    job = response.json()["job"]
    deadline = time.time() + 300
    while job["state"] in ("queued", "running") and time.time() < deadline:
        time.sleep(0.5)
        job = client.get(f"/v1/jobs/{job['id']}").json()["job"]
    assert job["state"] == "done", job
    pack_id = job["pack"]

    assert pack_id in client.get("/v1/health").json()["packs"]
    pack = client.get(f"/v1/packs/{pack_id}/files/pack.json").json()
    assert pack["runs_in"] == ["backend"] and "front" not in pack["files"]
    assert pack["built"]["recipe"] == b.RECIPE and pack["n_real"] == 0
    features = client.get(f"/v1/packs/{pack_id}/files/features.json").json()
    assert len(features) == pack["sae"]["n_features"] and not any(f.get("real") for f in features)
    sample = pack["samples"][0]["file"]
    assert client.get(f"/v1/packs/{pack_id}/files/{sample}").content[:2] == b"\xff\xd8"
    assert not (root / "work" / pack_id / "acts").exists()  # activations are dropped once packed

    rgb = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8).tobytes()
    read = client.post(f"/v1/packs/{pack_id}/read", content=rgb, headers={"content-type": "application/octet-stream"}).json()
    assert len(read["patches"]["offsets"]) == 17
    fired = read["patches"]["ids"][0]
    steer = client.post(f"/v1/packs/{pack_id}/steer", json={"image_id": read["image_id"], "features": [fired], "kind": "ablate"}).json()
    assert len(steer["shift"]) == 16 and steer["max"] >= 0

    # Asking again reuses the pack.
    again = client.post("/v1/models", json={"model": str(tiny_run["model"]), "layer": 1})
    assert again.status_code == 200 and again.json()["job"]["state"] == "done"
    assert again.headers["x-ii-new-build"] == "false"


from test_live import tiny_run  # noqa: E402,F401  (fixture)
