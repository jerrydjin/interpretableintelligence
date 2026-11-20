"""``ii action-effects`` (V-JEPA 2-AC, hypothesis H2) and ``ii get-clips --with-states``, offline."""

from __future__ import annotations

import json

import av
import numpy as np
import pytest
from typer.testing import CliRunner

from interpretableintelligence import sample_clips
from interpretableintelligence.adapters.vjepa2ac.model import (
    euler_xyz_to_matrix,
    matrix_to_euler_xyz,
    poses_to_diff,
    resample_indices,
    tiny_world_model,
)
from interpretableintelligence.cli import app
from interpretableintelligence.worldmodel import action_effects as ae
from interpretableintelligence.worldmodel.regions import transition_regions, write_png


# ----------------------------------------------------------------------------- pose math


def test_euler_round_trip_and_known_matrix():
    rng = np.random.default_rng(0)
    angles = np.stack([rng.uniform(-3, 3, 50), rng.uniform(-1.4, 1.4, 50), rng.uniform(-3, 3, 50)], axis=1)
    back = matrix_to_euler_xyz(euler_xyz_to_matrix(angles))
    np.testing.assert_allclose(euler_xyz_to_matrix(back), euler_xyz_to_matrix(angles), atol=1e-9)
    # A pure yaw of 90 degrees maps x to y.
    m = euler_xyz_to_matrix(np.array([0.0, 0.0, np.pi / 2]))
    np.testing.assert_allclose(m @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-12)
    # Extrinsic x then z: R = Rz @ Rx.
    rx = euler_xyz_to_matrix(np.array([0.3, 0, 0]))
    rz = euler_xyz_to_matrix(np.array([0, 0, 0.7]))
    np.testing.assert_allclose(euler_xyz_to_matrix(np.array([0.3, 0, 0.7])), rz @ rx, atol=1e-12)


def test_poses_to_diff_matches_meta_definition():
    s = np.array(
        [[0.4, 0.0, 0.5, -2.9, -0.2, 0.1, 0.0], [0.42, 0.01, 0.49, -2.9, -0.2, 0.2, 0.3]],
        dtype=np.float32,
    )
    a = poses_to_diff(s)
    assert a.shape == (1, 7)
    np.testing.assert_allclose(a[0, :3], [0.02, 0.01, -0.01], atol=1e-6)
    np.testing.assert_allclose(a[0, 6], 0.3, atol=1e-6)
    # The angle part is the Euler angles of R1 @ R0^T: applying it to R0 gives R1.
    r0, r1 = euler_xyz_to_matrix(s[:, 3:6].astype(np.float64))
    np.testing.assert_allclose(euler_xyz_to_matrix(a[0, 3:6].astype(np.float64)) @ r0, r1, atol=1e-5)
    assert np.allclose(poses_to_diff(np.repeat(s[:1], 3, axis=0)), 0)


def test_resample_to_training_rate():
    assert resample_indices(60, 15.0, 4.0)[:5] == [0, 4, 8, 11, 15]
    assert resample_indices(1, 15.0) == [0]


# ----------------------------------------------------------------------------- regions


def _scene(t: int, size: int = 64, wide: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Grey frame; the top 16 rows are 'arm'; a white block slides along the bottom row of patches.

    ``wide`` swaps the block for a striped band across the whole bottom row, shifting each step, so
    at least 3 patches two rows away from the arm change.
    """
    f = np.full((size, size, 3), 90, dtype=np.uint8)
    f[:16] = 30 + 60 * (t % 2)
    if wide:
        cols = (np.arange(size) + 3 * t) % 8 < 4
        f[50:62, cols] = 250
    else:
        x = 4 + 6 * (t % 6)
        f[50:62, x : x + 10] = 250
    arm = np.zeros((size, size), dtype=bool)
    arm[:16] = True
    return f, arm


def test_regions_label_arm_moved_and_still():
    frames, arms = zip(*(_scene(t) for t in range(3)))
    regs = transition_regions(np.stack(frames), np.stack(arms), crop=64, patch=16)
    assert len(regs) == 2
    r = regs[0]
    assert r.arm.reshape(4, 4)[0].all() and not r.arm.reshape(4, 4)[1:].any()
    assert r.moved.reshape(4, 4)[3].any()  # the block
    assert not r.moved.reshape(4, 4)[:3].any()
    assert not (r.still & (r.arm | r.moved)).any()
    assert not r.still.reshape(4, 4)[1].any()  # next to the arm


def test_write_png_header(tmp_path):
    write_png(tmp_path / "x.png", np.zeros((4, 5, 3), dtype=np.uint8))
    assert (tmp_path / "x.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# ----------------------------------------------------------------------------- verdict


def _rec(ep: int, *, arm: float, moved: float, matched: float, n: int = 4) -> dict:
    """A transition whose real action removes the given share of each region's change."""
    r = {"clip_id": f"c{ep}", "episode": ep, "t": 0, "gripper": 0.0, "action_xyz_m": 0.01}
    for name, share in (("arm", arm), ("moved", moved), ("matched", matched), ("still", matched), ("all", arm / 2)):
        r[f"n_{name}"] = n if name != "still" else 3 * n
        r[f"copy_{name}"] = 1.0
        r[f"err_zero_{name}"] = 1.0
        r[f"err_real_{name}"] = 1.0 - share
        r[f"err_shuffled_{name}"] = 1.0
        r[f"err_negated_{name}"] = 1.0
        r[f"effect_{name}"] = share
    return r


@pytest.mark.parametrize(
    ("arm", "moved", "matched", "noise", "expected"),
    [
        (0.4, 0.0, 0.0, 0.005, "arm_simulator"),
        (0.4, 0.3, 0.0, 0.005, "object_aware"),
        (0.0, 0.0, 0.0, 0.005, "no_arm_effect"),
        # A real object effect just under a quarter of the arm's, too noisy to rule out a quarter.
        (0.4, 0.09, 0.0, 0.05, "mixed"),
    ],
)
def test_verdict_rules(arm, moved, matched, noise, expected):
    rng = np.random.default_rng(1)
    recs = [
        _rec(ep, arm=arm + rng.normal(0, noise), moved=moved + rng.normal(0, noise), matched=matched + rng.normal(0, noise))
        for ep in range(20)
        for _ in range(3)
    ]
    out = ae.summarize(recs, n_boot=300)
    assert out["moving"]["verdict"] == expected
    assert out["moving"]["n_episodes"] == 20


def test_too_few_episodes_is_insufficient():
    out = ae.summarize([_rec(0, arm=0.4, moved=0.0, matched=0.0)], n_boot=50)
    assert out["moving"]["verdict"] == "insufficient_data"


# ----------------------------------------------------------------------------- end to end


def _write_clip(path, frames, fps=15):
    with av.open(str(path), "w") as dst:
        stream = dst.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
        stream.options = {"crf": "10"}
        for f in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                dst.mux(packet)
        for packet in stream.encode():
            dst.mux(packet)


def _clips_with_states(tmp_path, n: int = 4, frames_per_clip: int = 30):
    out = tmp_path / "clips"
    out.mkdir()
    entries = []
    for i in range(n):
        frames = np.stack([_scene(t // 4 + i, wide=True)[0] for t in range(frames_per_clip)])
        states = np.zeros((frames_per_clip, 7), dtype=np.float32)
        states[:, 0] = 0.4 + 0.004 * np.arange(frames_per_clip)
        states[:, 6] = (np.arange(frames_per_clip) > frames_per_clip // 2).astype(np.float32)
        _write_clip(out / f"droid-{i:03d}.mp4", frames)
        np.save(out / f"droid-{i:03d}.states.npy", states)
        entries.append({"clip_id": f"droid-{i:03d}", "file": f"droid-{i:03d}.mp4", "states": f"droid-{i:03d}.states.npy", "episode": i})
    (out / "clips.json").write_text(json.dumps({"clips": entries}))
    return out


def test_tiny_world_model_moves_only_its_action_tokens():
    import torch

    model = tiny_world_model(seed=0)
    side = model.grid[0]
    frames = np.stack([_scene(t)[0] for t in range(2)])
    z = model.encode_frames(frames)
    assert z.shape == (2, side * side, 16)
    s = torch.zeros(2, 7)
    a = torch.zeros(2, 7)
    a[:, 0] = 0.05
    moved = model.predict_next(z, s, a) - model.predict_next(z, s, torch.zeros_like(a))
    per_token = moved.abs().mean(-1)
    assert (per_token[:, :side] > 0).all()
    assert torch.allclose(per_token[:, side:], torch.zeros(2, side * side - side), atol=1e-6)


def test_cli_action_effects_tiny(tmp_path):
    clips = _clips_with_states(tmp_path)
    out = tmp_path / "ae"
    result = CliRunner().invoke(app, ["action-effects", "--clips", str(clips), "--out", str(out), "--tiny", "--bootstrap", "50"])
    assert result.exit_code == 0, result.output
    data = json.loads((out / "action_effects.json").read_text())
    assert data["schema"] == ae.SCHEMA and data["model"] == "tiny"
    assert data["n_episodes"] == 4 and data["n_transitions"] > 0
    assert data["moving"]["verdict"] in {"arm_simulator", "object_aware", "mixed", "no_arm_effect", "insufficient_data"}
    assert (out / "transitions.parquet").is_file()
    # The tiny model's actions never touch the moving block, so moved patches get no action effect.
    assert data["moving"]["n_transitions"] > 0
    assert abs(data["moving"]["stats"]["effect_moved"]["value"]) < 1e-6
    assert data["moving"]["stats"]["effect_arm"]["value"] > 0
    assert "verdict:" in result.output


def test_cli_action_effects_needs_states(tmp_path):
    clips = _clips_with_states(tmp_path, n=2)
    index = json.loads((clips / "clips.json").read_text())
    for e in index["clips"]:
        del e["states"]
    (clips / "clips.json").write_text(json.dumps(index))
    result = CliRunner().invoke(app, ["action-effects", "--clips", str(clips), "--out", str(tmp_path / "o"), "--tiny"])
    assert result.exit_code == 1
    assert "--with-states" in result.output


# ----------------------------------------------------------------------------- get-clips --with-states


def test_get_clips_with_states(monkeypatch, tmp_path):
    import huggingface_hub

    source = tmp_path / "source.mp4"
    _write_clip(source, np.zeros((15 * 6, 48, 64, 3), dtype=np.uint8))
    episodes = [{"episode": i, "task": f"task {i}", "file": (0, 0), "start": 3.0 * i, "end": 3.0 * (i + 1)} for i in range(2)]
    times = np.arange(45) / 15.0
    table = {i: (times, np.stack([np.full(7, 100 * i + k, dtype=np.float32) for k in range(45)])) for i in range(2)}
    monkeypatch.setattr(sample_clips, "_episodes", lambda: episodes)
    monkeypatch.setattr(sample_clips, "_episode_states", lambda: table)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(source))

    out = tmp_path / "c"
    entries = sample_clips.fetch_droid_clips(out, n=2, seconds=1.0, say=lambda _: None, with_states=True)
    for e in entries:
        states = np.load(out / e["states"])
        assert states.shape == (e["frames"], 7)
        # 1 s from the middle of a 3 s episode starts 1 s in: row 15 of that episode.
        assert states[0, 0] == 100 * e["episode"] + 15
        assert np.all(np.diff(states[:, 0]) == 1)
    assert sample_clips.clips_ready(out, 2, with_states=True)
    assert "states" in json.loads((out / "clips.json").read_text())


# ----------------------------------------------------------------------------- plan-probe (H5)


def test_cem_finds_the_cost_minimum_on_a_known_landscape():
    import torch

    from interpretableintelligence.worldmodel import plan_probe as pp

    class Bowl:
        """Cost is the distance of the action's xyz from a target: CEM must land near it."""

        def __init__(self, target):
            self.target = torch.tensor(target)

        def energy(self, z, state, goal, actions, per_token=False):
            return (actions[:, :3] - self.target).abs().sum(-1)

    near = pp.cem(Bowl([0.03, -0.02, 0.01]), None, None, None, maxnorm=0.075, grip_max=0.75, samples=200, steps=10,
                  generator=torch.Generator().manual_seed(0))
    np.testing.assert_allclose(near.action[:3], [0.03, -0.02, 0.01], atol=0.01)
    assert np.all(near.action[3:6] == 0)
    # A target outside the range: the in-range search stops at the edge, the wide one reaches it.
    far = [0.3, 0.0, 0.0]
    clipped = pp.cem(Bowl(far), None, None, None, maxnorm=0.075, grip_max=0.75, generator=torch.Generator().manual_seed(0))
    wide = pp.cem(Bowl(far), None, None, None, maxnorm=0.5, grip_max=3.0, steps=12, generator=torch.Generator().manual_seed(0))
    assert clipped.action[0] <= 0.075 + 1e-6 and abs(wide.action[0] - 0.3) < 0.03
    assert wide.cost < clipped.cost


def test_plan_probe_verdicts():
    from interpretableintelligence.worldmodel import plan_probe as pp

    def recs(fooled_rate, n_ep=20, per=10, seed=0):
        rng = np.random.default_rng(seed)
        out = []
        for ep in range(n_ep):
            for t in range(per):
                f = rng.random() < fooled_rate
                out.append({"episode": ep, "fooled": f, "e_wide": 1.0 if f else 2.0, "e_in": 1.5, "e_real": 1.6, "e_zero": 2.0,
                            "wide_in_range": not f, "wide_impossible": f, "ray_costs": [1.0, 1.1, 1.2, 1.3], "in_cos_real": 0.5, "wide_step_m": 0.2, "real_step_m": 0.02,
                            "offmanifold_wide": 1.0, "offmanifold_in": 0.5, "gain_changed_tokens": 1.0, "gain_other_tokens": 3.0})
        return out

    assert pp.summarize(recs(0.4), n_boot=200)["verdict"] == "exploitable"
    assert pp.summarize(recs(0.0), n_boot=200)["verdict"] == "robust"
    s = pp.summarize(recs(0.4), n_boot=50)["stats"]
    assert abs(s["gain_share_on_changed_tokens"]["value"] - 0.25) < 1e-9


def test_cli_plan_probe_tiny(tmp_path):
    clips = _clips_with_states(tmp_path, n=3)
    out = tmp_path / "pp"
    args = ["plan-probe", "--clips", str(clips), "--out", str(out), "--tiny", "--stride", "2",
            "--samples", "20", "--steps", "2", "--bootstrap", "20"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    data = json.loads((out / "plan_probe.json").read_text())
    assert data["schema"] == "plan-probe/v1" and data["n_transitions"] > 0
    assert data["verdict"] in {"exploitable", "robust", "mixed", "insufficient_data"}
    assert (out / "transitions.parquet").is_file()


# ----------------------------------------------------------------------------- gripper-probe (H1)


def test_find_gripper_events():
    from interpretableintelligence.worldmodel.gripper_probe import find_events

    s = np.zeros((12, 7), dtype=np.float32)
    s[:, 0] = 0.4
    s[4:, 6] = 1.0  # closes between 3 and 4, arm still
    s[9:, 6] = 0.0  # opens between 8 and 9 ...
    s[9:, 0] = 0.5  # ... while the arm jumps 10 cm: not an event
    assert find_events(s) == [(3, 4)]


def test_cli_gripper_probe_tiny(tmp_path):
    clips = _clips_with_states(tmp_path, n=3)
    out = tmp_path / "gp"
    result = CliRunner().invoke(app, ["gripper-probe", "--clips", str(clips), "--out", str(out), "--tiny",
                                      "--samples", "20", "--steps", "2", "--bootstrap", "20"])
    assert result.exit_code == 0, result.output
    data = json.loads((out / "gripper_probe.json").read_text())
    assert data["schema"] == "gripper-probe/v1"
    assert data["n_events"] == 3  # each test clip closes its gripper once, mid-clip
    assert data["verdict"] in {"reads_gripper", "gripper_blind", "mixed", "insufficient_data"}


# ----------------------------------------------------------------------------- gripper-trace


def test_trace_event_finds_where_the_tiny_model_mixes_the_command():
    import torch

    from interpretableintelligence.worldmodel.gripper_trace import trace_event

    model = tiny_world_model(seed=0)
    z = model.encode_frames(np.stack([_scene(0)[0]]))[0]
    t = trace_event(model, z, np.zeros(7, dtype=np.float32), np.zeros(3, dtype=np.float32), 1.0)
    # The tiny predictor reads the action token in block 1 (layer 2 = its output). Before that the
    # command lives only on the action token: patching it recovers the whole gap, patching the frame
    # nothing. After it, the reverse.
    rec_a, rec_f = t["recovered_action"], t["recovered_frame"]
    assert len(rec_a) == len(model.predictor.predictor_blocks) + 1
    np.testing.assert_allclose(rec_a[:2], 1.0, atol=1e-5)
    np.testing.assert_allclose(rec_f[:2], 0.0, atol=1e-5)
    np.testing.assert_allclose(rec_a[2:], 0.0, atol=1e-5)
    np.testing.assert_allclose(rec_f[2:], 1.0, atol=1e-5)
    d = t["drift"]
    assert d["grip_frame"][0] == 0 and d["grip_frame"][1] == 0 and d["grip_frame"][2] > 0
    assert d["grip_action"][0] > 0
    assert torch.is_tensor(t["pred_right"])


def test_encoder_readout_reads_a_visible_gripper(tmp_path):
    from interpretableintelligence.worldmodel import gripper_trace as gt

    model = tiny_world_model(seed=0)
    trs = []
    rng = np.random.default_rng(0)
    for ep in range(8):
        closed = (np.arange(20) % 7) > 3
        frames = np.stack([_scene(t, wide=True)[0] for t in range(20)])
        frames[closed, :16, :16] = 255  # the "gripper" shows in one corner when closed
        frames = np.clip(frames.astype(int) + rng.integers(-5, 6, frames.shape), 0, 255).astype(np.uint8)
        states = np.zeros((20, 7), dtype=np.float32)
        states[:, 6] = closed
        trs.append(ae.Trajectory(clip_id=f"c{ep}", episode=ep, task="", frames=frames, states=states))
    latents = [model.encode_frames(tr.frames) for tr in trs]
    out, probes = gt.encoder_readout(latents, trs, seed=0, n_boot=50)
    assert out["encoder"]["value"] > 0.9 and out["pixels"]["value"] > 0.9
    assert probes[0] is not None and probes[1] is not None


def test_gripper_trace_verdicts():
    from interpretableintelligence.worldmodel.gripper_trace import verdicts

    enc = {"encoder": {"value": 0.95, "lo": 0.9, "hi": 0.97}}
    assert verdicts(enc, {"value": 0.05, "lo": -0.1, "hi": 0.2}) == {"encoder": "encodes_gripper", "command": "dropped"}
    assert verdicts(enc, {"value": 0.8, "lo": 0.6, "hi": 1.0})["command"] == "carried"
    assert verdicts(enc, {"value": 0.4, "lo": 0.2, "hi": 0.6})["command"] == "partial"
    weak = {"encoder": {"value": 0.7, "lo": 0.6, "hi": 0.8}}
    assert verdicts(weak, {"value": float("nan"), "lo": float("nan"), "hi": float("nan")}) == {"encoder": "weak", "command": "insufficient_data"}


def test_cli_gripper_trace_tiny(tmp_path):
    clips = _clips_with_states(tmp_path, n=4)
    out = tmp_path / "gt"
    result = CliRunner().invoke(app, ["gripper-trace", "--clips", str(clips), "--out", str(out), "--tiny", "--bootstrap", "20"])
    assert result.exit_code == 0, result.output
    data = json.loads((out / "gripper_trace.json").read_text())
    assert data["schema"] == "gripper-trace/v1"
    assert data["n_events"] == 4
    assert len(data["layers"]["recovered_action"]) == 5
    assert data["verdict"]["command"] in {"dropped", "carried", "partial", "insufficient_data"}
