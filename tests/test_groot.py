"""GR00T adapter on the random-weights ``groot:tiny`` stand-in. No network, no LeRobot.

The real checkpoint path (``LeRobotGroot``) needs ``lerobot[groot]`` and a
12.6 GB download, so it is exercised by hand, not in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from interpretableintelligence.adapters import get_adapter
from interpretableintelligence.adapters.base import Clip
from interpretableintelligence.adapters.groot.adapter import (
    DEFAULT_CHECKPOINT,
    DEFAULT_LAYER,
    TinyGroot,
    make_spec,
    parse_spec,
)
from interpretableintelligence.cli import app
from interpretableintelligence.steer import Steerer
from interpretableintelligence.validate import validate_run
from interpretableintelligence.view import load_run as load_view_model

CPU = torch.device("cpu")
H = TinyGroot.action_horizon


def _adapter(random_init: bool = False, layer: int = DEFAULT_LAYER):
    adapter = get_adapter(make_spec(tiny=True))(layer=layer, device=CPU)
    adapter.load(random_init=random_init)
    return adapter


def _clip(tmp_path: Path, seed: int = 0, n_frames: int = 10, sidecar: dict | None = None) -> Clip:
    frames = np.random.default_rng(seed).random((n_frames, 24, 24, 3), dtype=np.float32)
    path = tmp_path / f"c{seed}.npy"
    np.save(path, frames)
    if sidecar is not None:
        (tmp_path / f"c{seed}.groot.json").write_text(json.dumps(sidecar), encoding="utf-8")
    return Clip(clip_id=f"c{seed}", path=path, frames=frames, fps=8.0, sha256="ab" * 32)


def test_spec_names_the_checkpoint():
    assert make_spec() == f"groot:{DEFAULT_CHECKPOINT}"
    assert parse_spec(make_spec("org/ckpt")) == "org/ckpt"
    assert make_spec(tiny=True) == "groot:tiny"
    with pytest.raises(ValueError):
        parse_spec("groot:")
    with pytest.raises(ValueError, match="--tiny"):
        make_spec("org/ckpt", tiny=True)
    with pytest.raises(ValueError, match="--tiny"):
        get_adapter("groot:org/ckpt")(layer=5, device=CPU, tiny=True)


def test_grid_is_state_then_action_tokens_and_windows_follow_chunks(tmp_path):
    adapter = _adapter()
    grid = adapter.grid()
    assert (grid.t, grid.h, grid.w, grid.tubelet) == (1, 1, 1 + H, TinyGroot.chunk)
    clip = _clip(tmp_path, n_frames=10)
    windows = adapter.windows(clip)
    assert [w.start_frame for w in windows] == [0, 4, 8]
    mask = adapter.mask(windows[0])
    assert not bool(mask[0, 0, 0]) and bool(mask[0, 0, 1:].all())

    with pytest.warns(UserWarning, match="off-distribution"):
        with adapter.hook(DEFAULT_LAYER, mode="capture") as handle:
            acts = adapter.encode(clip, windows[0])
    assert tuple(acts.shape) == (1, 1, 1 + H, TinyGroot.d_model)
    assert torch.equal(handle.captured[0], acts)

    pred, target = adapter.predict_masked(clip, windows[0])
    assert tuple(target.shape) == (H, TinyGroot.action_dim)  # one row per masked action token
    assert int(mask.sum()) == target.shape[0]
    assert torch.equal(pred, target)
    same, _ = adapter.predict_masked(clip, windows[0], edit=lambda x: x)
    assert torch.allclose(same, target, atol=1e-6)  # same noise, no edit, same chunk
    with adapter.hook(DEFAULT_LAYER, mode="zero"):
        zeroed, _ = adapter.predict_masked(clip, windows[0])
    assert float(((zeroed - target) ** 2).mean()) > 1e-4
    assert adapter.visible_mean_baseline(clip, windows[0]).shape == target.shape


def test_edits_hit_the_final_denoise_step_and_batch_like_single_calls(tmp_path, monkeypatch):
    import interpretableintelligence.adapters.groot.adapter as groot

    monkeypatch.setattr(groot, "MANY_BATCH", 2)  # forces several batches
    adapter = _adapter()
    clip = _clip(tmp_path, seed=1)
    window = adapter.windows(clip)[1]
    adapter.predict_masked(clip, window)  # caches the clean chunk
    calls = []
    handle = adapter.backend.blocks[DEFAULT_LAYER].register_forward_hook(lambda *_: calls.append(1))
    try:
        adapter.predict_masked(clip, window, edit=lambda x: x * 0.5)
    finally:
        handle.remove()
    assert len(calls) == TinyGroot.n_denoise  # the block runs once per denoise step; only the last is edited

    edits = [(lambda x, k=k: x * (1.0 - 0.2 * k)) for k in range(5)]
    with torch.no_grad():
        batched = adapter.predict_masked_many(clip, window, edits)
        looped = [adapter.predict_masked(clip, window, edit=edit)[0] for edit in edits]
    for one, many in zip(looped, batched, strict=True):
        assert torch.allclose(one, many, atol=1e-5)


def test_attribution_carries_gradient_and_random_init_changes_the_sha(tmp_path):
    adapter = _adapter()
    clip = _clip(tmp_path, seed=2)
    window = adapter.windows(clip)[0]
    scale = torch.ones((), requires_grad=True)
    loss = adapter.attribution_objective(clip, window, "prediction_loss", edit=lambda x: x * scale)
    loss.backward()
    assert scale.grad is not None
    control = _adapter(random_init=True)
    assert control.random_init and control.model_sha() != adapter.model_sha()
    with pytest.raises(ValueError, match="layer 8"):
        _adapter(layer=8)


def test_sidecar_supplies_wrist_state_and_task(tmp_path):
    adapter = _adapter()
    wrist = np.full((10, 24, 24, 3), 200, dtype=np.uint8)
    np.save(tmp_path / "c3.wrist.npy", wrist)
    state = [[float(i)] * 8 for i in range(10)]
    clip = _clip(tmp_path, seed=3, sidecar={"task": "put both moka pots on the stove", "state": state, "wrist": "c3.wrist.npy"})
    window = adapter.windows(clip)[1]
    obs = adapter.observation(clip, window)
    assert obs.task == "put both moka pots on the stove"
    assert obs.state.tolist() == [4.0] * 8  # the window's first frame
    assert obs.wrist.dtype == np.uint8 and int(obs.wrist[0, 0, 0]) == 200
    bare = _clip(tmp_path, seed=4)
    with pytest.warns(UserWarning):
        assert not adapter.observation(bare, adapter.windows(bare)[0]).state.any()


def test_ii_run_groot_tiny_end_to_end_then_steer(tmp_path):
    from interpretableintelligence.adapters.vjepa2 import generate_tiny_clips

    clips = tmp_path / "clips"
    generate_tiny_clips(clips, 20)
    out = tmp_path / "runs" / "groot"
    result = CliRunner().invoke(app, ["run", "groot", "--tiny", "--clips", str(clips), "--out", str(out), "--prefilter", "2", "--no-serve"])
    assert result.exit_code == 0, result.output
    assert validate_run(out) == []
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["adapter"] == "groot:tiny"
    assert manifest["layer"] == DEFAULT_LAYER
    assert manifest["grid"]["w"] == 1 + H

    steerer = Steerer(out, clips)
    clip_id = manifest["clips"][0]["clip_id"]
    same = steerer.steer({"clip_id": clip_id, "features": [0], "kind": "scale", "value": 1.0})
    moved = steerer.steer({"clip_id": clip_id, "features": [0], "kind": "clamp", "value": 5.0})
    assert same["summary"]["mean_shift"] < 1e-5
    assert moved["summary"]["mean_shift"] > 1e-4
    assert moved["shift"][0][0][0] is None  # the state token is not a prediction
    assert all(value is not None for value in moved["shift"][0][0][1:])

    viewer = load_view_model(out, clips)
    assert viewer._build_clip(clip_id)["layout"] == "actions"
    assert viewer.run_payload["steering"] is None

    # Steering scorecard on the planned action chunk. The random stand-in has no verified
    # features, so the default refuses; named features are measured and stay unverified.
    scoreboard_before = (out / "scoreboard.json").read_bytes()
    refused = CliRunner().invoke(app, ["steer-scorecard", str(out), "--clips", str(clips)])
    assert refused.exit_code == 1 and "--features" in refused.output
    scored = CliRunner().invoke(
        app, ["steer-scorecard", str(out), "--clips", str(clips), "--features", "0,1", "--controls", "19", "--bootstrap", "100"]
    )
    assert scored.exit_code == 0, scored.output
    assert "planned actions" in scored.output
    card = json.loads((out / "steering" / "scorecard.json").read_text(encoding="utf-8"))
    assert card["target"] == "action_chunk" and card["adapter"] == "groot:tiny"
    assert card["selection"] == "requested" and card["rollout_finding"] is False
    assert all(row["real"] is False for row in card["rows"])
    assert (out / "scoreboard.json").read_bytes() == scoreboard_before
    assert validate_run(out) == []


def test_cli_rejects_blocks_for_groot(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    result = CliRunner().invoke(app, ["run", "groot", "--blocks", "x", "--clips", str(clips), "--out", str(tmp_path / "r"), "--no-serve"])
    assert result.exit_code == 1 and "--blocks" in result.output
