"""The steering scorecard's real-GR00T path on CPU: ``ii record-libero`` (with a stand-in env and policy),
``ii run groot --sae-from`` (a frozen SAE, ii or v9 layout), then ``ii steer-scorecard`` on the recorded clips."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file
from typer.testing import CliRunner

from interpretableintelligence.adapters import get_adapter
from interpretableintelligence.adapters.groot.colab import build_steer_notebook
from interpretableintelligence.adapters.groot.record_libero import (
    INDEX_FILE,
    RECORD_FILE,
    Step,
    episode_id,
    hwc_uint8,
    record,
)
from interpretableintelligence.cli import app
from interpretableintelligence.clips import list_clip_meta, load_clip
from interpretableintelligence.sae import load_sae
from interpretableintelligence.sae_source import check_source, load_external_sae

TASK = "put both moka pots on the stove"
SIZE = 16


class FakeRuntime:
    """Deterministic stand-in for LeRobotLibero: frames and state depend on (seed, step)."""

    task_name = "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_view_0_0_100_0_0_initstate_1"
    checkpoint = "fake"

    def __init__(self, steps: int = 9, parity_pass: bool = True) -> None:
        self.steps, self.parity_pass = steps, parity_pass
        self.seed = self.t = 0
        self.resets: list[tuple[int, int]] = []

    def n_init_states(self) -> int:
        return 5

    def reset(self, seed, init_state_id):
        self.seed, self.t = seed, 0
        self.resets.append((seed, init_state_id))
        return TASK, 520

    def observe(self) -> Step:
        rng = np.random.default_rng([self.seed, self.t])
        image = rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)
        wrist = (255 - image).astype(np.uint8)
        state = np.array([self.seed % 7, self.t, 0.5, -1, 0, 0, 0, 1], dtype=np.float32)
        return Step(image=image, wrist=wrist, state=state)

    def act(self):
        return np.zeros(7, np.float32)

    def step(self, action):
        self.t += 1
        success = self.seed % 2 == 0 and self.t == self.steps
        return success, self.t >= self.steps

    def parity(self, step, task):
        return {"pass": self.parity_pass, "backbone_features": {"rel": 0.0 if self.parity_pass else 0.5}}


def _record(out: Path, episodes: int = 20, **kw) -> dict:
    return record(FakeRuntime(**kw), out, episodes=episodes, log=lambda _m: None)


def test_record_writes_clips_the_groot_adapter_reads(tmp_path):
    out = tmp_path / "clips"
    report = _record(out)
    assert report["parity"]["pass"] and len(report["episodes"]) == 20
    assert report["n_success"] == 10 and report["episodes"][3]["seed"] == 60003 and report["episodes"][7]["init_state_id"] == 2
    index = json.loads((out / INDEX_FILE).read_text())
    assert index["fps"] == 20.0 and [c["clip_id"] for c in index["clips"]] == [episode_id(i) for i in range(20)]
    meta = list_clip_meta(out)  # the wrist frames in wrist/ are not clips
    assert len(meta) == 20 and meta[0]["n_frames"] == 9

    adapter = get_adapter("groot:tiny")(layer=5)
    adapter.load()
    clip = load_clip(meta[1])
    windows = adapter.windows(clip)
    assert len(windows) == 3  # 9 frames, chunk 4
    obs = adapter.observation(clip, windows[1])
    runtime = FakeRuntime()
    runtime.reset(60001, 1)
    for _ in range(4):
        runtime.step(None)
    want = runtime.observe()  # what the policy saw before step 4, the input of the second chunk
    assert obs.task == TASK
    assert np.array_equal(obs.image, want.image) and np.array_equal(obs.wrist, want.wrist)
    assert np.array_equal(obs.state, want.state)
    sidecar = json.loads((out / "ep-00001.groot.json").read_text())
    assert sidecar["libero"]["seed"] == 60001 and sidecar["libero"]["task_id"] == 1028


def test_record_resumes_and_refuses_other_settings(tmp_path):
    out = tmp_path / "clips"
    _record(out, episodes=2)
    runtime = FakeRuntime()
    record(runtime, out, episodes=3, log=lambda _m: None)
    assert runtime.resets == [(60002, 2)]  # only the new episode
    with pytest.raises(RuntimeError, match="other settings"):
        record(FakeRuntime(), out, episodes=3, seed0=1, log=lambda _m: None)


def test_record_stops_on_failed_parity_and_wrong_task(tmp_path):
    with pytest.raises(RuntimeError, match="parity"):
        _record(tmp_path / "a", episodes=1, parity_pass=False)
    assert not (tmp_path / "a" / "ep-00000.npy").exists()
    assert json.loads((tmp_path / "a" / RECORD_FILE).read_text())["parity"]["pass"] is False
    runtime = FakeRuntime()
    runtime.task_name = "LIVING_ROOM_SCENE2_something_else"
    with pytest.raises(RuntimeError, match="expected"):
        record(runtime, tmp_path / "b", episodes=1, log=lambda _m: None)


def test_hwc_uint8_round_trips_prepared_images():
    image = np.random.default_rng(0).integers(0, 256, (6, 5, 3), dtype=np.uint8)
    chw = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    assert np.array_equal(hwc_uint8(chw), image)
    assert np.array_equal(hwc_uint8(image[None]), image)


def _v9_sae(directory: Path, d: int, m: int, seed: int = 0) -> dict:
    """A GR00T v9 BatchTopK SAE folder (the capture's layout) with random weights."""
    rng = np.random.default_rng(seed)
    w_dec = rng.normal(size=(m, d)).astype(np.float32)
    w_dec /= np.linalg.norm(w_dec, axis=1, keepdims=True)
    weights = {
        "W_enc": rng.normal(size=(d, m)).astype(np.float32),
        "W_dec": w_dec,
        "b_enc": (0.1 * rng.normal(size=m)).astype(np.float32),
        "b_dec": (0.1 * rng.normal(size=d)).astype(np.float32),
        "norm_mu": rng.normal(size=d).astype(np.float32),
        "norm_scale": np.array([2.5], np.float32),
        "threshold": np.array([0.4], np.float32),
    }
    sae = directory / "sae"
    sae.mkdir(parents=True)
    save_file(weights, str(sae / "weights.safetensors"))
    (sae / "config.json").write_text(json.dumps({"arch": "batchtopk", "d_in": d, "d_sae": m, "k": 4, "inference_threshold": 0.4}))
    return weights


def test_v9_sae_folds_to_the_same_active_set(tmp_path):
    weights = _v9_sae(tmp_path / "capture", d=32, m=48)
    external = load_external_sae(tmp_path / "capture", 32)
    assert external.record["format"] == "groot-v9" and external.record["n_features"] == 48
    x = np.random.default_rng(1).normal(size=(64, 32)).astype(np.float32) * 3
    pre = np.maximum(((x - weights["norm_mu"]) / 2.5 - weights["b_dec"]) @ weights["W_enc"] + weights["b_enc"], 0)
    source = pre * (pre > 0.4)
    ours = external.model.encode(torch.from_numpy(x)).numpy()
    assert np.array_equal(ours > 0, source > 0)
    np.testing.assert_allclose(ours, 2.5 * source, rtol=1e-4, atol=1e-4)
    errors: list[str] = []
    check_source(external.record, errors)
    assert errors == []
    check_source({**external.record, "format": "other", "weights_sha256": "x"}, errors)
    assert len(errors) == 2
    with pytest.raises(ValueError, match="d_model"):
        load_external_sae(tmp_path / "capture", 16)


def test_sae_from_then_steer_scorecard_on_recorded_clips(tmp_path):
    from interpretableintelligence.adapters.vjepa2 import generate_tiny_clips

    runner = CliRunner()
    recorded = tmp_path / "recorded"
    _record(recorded)

    # A frozen SAE trained elsewhere: a groot:tiny run on other clips.
    other = tmp_path / "other"
    generate_tiny_clips(other, 20)
    source = tmp_path / "runs" / "source"
    made = runner.invoke(app, ["run", "groot", "--tiny", "--clips", str(other), "--out", str(source), "--no-score", "--no-serve"])
    assert made.exit_code == 0, made.output

    out = tmp_path / "runs" / "steer"
    refused = runner.invoke(app, ["run", "groot", "--tiny", "--clips", str(recorded), "--out", str(out), "--sae-from", str(source), "--no-serve"])
    assert refused.exit_code == 1 and "--no-score" in refused.output

    args = ["run", "groot", "--tiny", "--clips", str(recorded), "--out", str(out), "--sae-from", str(source), "--heldout-frac", "0.5", "--no-score", "--no-serve"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "frozen" in result.output and "run sae" in result.output
    a, _ = load_sae(source / "sae", torch.device("cpu"))
    b, _ = load_sae(out / "sae", torch.device("cpu"))
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["sae_source"]["format"] == "ii" and manifest["sae_source"]["path"] == str((source / "sae").resolve())
    again = runner.invoke(app, args)
    assert again.exit_code == 0 and "skip sae" in again.output

    scored = runner.invoke(
        app, ["steer-scorecard", str(out), "--clips", str(recorded), "--features", "0,1", "--controls", "19", "--bootstrap", "50"]
    )
    assert scored.exit_code == 0, scored.output
    card = json.loads((out / "steering" / "scorecard.json").read_text())
    assert card["target"] == "action_chunk" and card["feature_ids"] == [0, 1] and card["n_episodes"] == 10
    assert card["rollout_finding"] is False and not (out / "scoreboard.json").exists()

    # Training its own SAE again drops the provenance.
    retrained = runner.invoke(app, ["run", "groot", "--tiny", "--clips", str(recorded), "--out", str(out), "--heldout-frac", "0.5", "--no-score", "--no-serve"])
    assert retrained.exit_code == 0, retrained.output
    assert "sae_source" not in json.loads((out / "manifest.json").read_text())


def test_steer_notebook_cells_parse():
    nb = build_steer_notebook.build("claude/some-branch")
    code = ["".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"]
    assert len(code) == 3 and "claude/some-branch" in code[0]
    assert "6262,7353,7775" in code[2] and "scripts/groot_steer_scorecard.sh" in code[2]
    for text in code:
        python = "\n".join("pass" if re.match(r"\s*[!%]", line) else line for line in text.splitlines())
        ast.parse(python)
    script = Path(__file__).resolve().parents[1] / "scripts" / "groot_steer_scorecard.sh"
    assert build_steer_notebook.ASSET_SHA256 in script.read_text()


def test_groot_adapter_keeps_few_backbone_contexts(tmp_path):
    from interpretableintelligence.adapters.groot.adapter import CONTEXT_CACHE

    out = tmp_path / "clips"
    _record(out, episodes=4, steps=13)
    adapter = get_adapter("groot:tiny")(layer=5)
    adapter.load()
    seen = 0
    for path in sorted(out.glob("ep-*.npy")):
        clip = load_clip({"clip_id": path.stem, "path": path, "fps": 20.0, "sha256": path.stem})
        for window in adapter.windows(clip):
            adapter.predict_masked(clip, window)
            seen += 1
    contexts = [key for key in adapter._cache if key[0] == "context"]
    assert seen > CONTEXT_CACHE and len(contexts) == CONTEXT_CACHE
    assert contexts[-1][1:] == ("ep-00003", 3)


def test_trim_to_common_length_keeps_whole_chunks(tmp_path):
    from interpretableintelligence.adapters.groot.record_libero import trim_to_common_length

    src = tmp_path / "clips"
    record(FakeRuntime(steps=11), src, episodes=1, log=lambda _m: None)
    record(FakeRuntime(steps=11), src, episodes=2, log=lambda _m: None)
    sidecar = json.loads((src / "ep-00001.groot.json").read_text())
    steps = trim_to_common_length(src, tmp_path / "trim", chunk=4, log=lambda _m: None)
    assert steps == 8
    meta = list_clip_meta(tmp_path / "trim")
    assert [m["n_frames"] for m in meta] == [8, 8]
    trimmed = json.loads((tmp_path / "trim" / "ep-00001.groot.json").read_text())
    assert trimmed["state"] == sidecar["state"][:8] and trimmed["libero"]["trimmed_to"] == 8
    assert np.array_equal(np.load(tmp_path / "trim" / "wrist" / "ep-00001.npy"), np.load(src / "wrist" / "ep-00001.npy")[:8])
