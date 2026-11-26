"""``ii freeze-check``: planted models give the answer the rule promises, adapters keep the interface."""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
from typer.testing import CliRunner

from interpretableintelligence.cli import app
from interpretableintelligence.worldmodel.action_effects import Trajectory
from interpretableintelligence.worldmodel.freeze_check import measure, reset_weights, run_freeze_check, summarize, verdict

from test_action_effects import _clips_with_states, _scene


def _trajectories(n: int = 4, steps: int = 10) -> list[Trajectory]:
    out = []
    for i in range(n):
        frames = np.stack([_scene(t + i, size=64, wide=True)[0] for t in range(steps)])
        states = np.zeros((steps, 7), dtype=np.float32)
        states[:, 0] = 0.4 + 0.02 * np.arange(steps)  # 2 cm per step: every step moves
        out.append(Trajectory(clip_id=f"c{i}", episode=i, task="", frames=frames, states=states))
    return out


class _Planted:
    """Encoder = mean colour per 16-pixel patch. ``blend`` 0 predicts the current frame, 1 the real next one."""

    crop, patch = 64, 16
    info: dict = {}

    def __init__(self, trajectories, blend: float, name: str):
        self.blend, self.name = blend, name
        self.next: dict[bytes, torch.Tensor] = {}
        for tr in trajectories:
            z = self.encode_frames(tr.frames)
            for t in range(len(z) - 1):
                self.next[z[t].numpy().tobytes()] = z[t + 1]

    def encode_frames(self, frames):
        x = torch.from_numpy(np.ascontiguousarray(frames)).float()
        n = x.shape[0]
        return x.view(n, 4, 16, 4, 16, 3).mean(dim=(2, 4)).reshape(n, 16, 3)

    def predict_next(self, z, states, actions):
        out = []
        for k in range(len(z)):
            nxt = self.next.get(z[k].numpy().tobytes(), z[k])
            moved = actions[k].abs().sum() > 0
            out.append(z[k] + (self.blend if moved else 0.0) * (nxt - z[k]))
        return torch.stack(out)


def test_frozen_model_is_yes_and_a_good_model_is_no():
    trs = _trajectories()
    frozen = summarize(measure(_Planted(trs, 0.0, "frozen"), trs, say=lambda *_: None), n_boot=200)
    good = summarize(measure(_Planted(trs, 1.0, "good"), trs, say=lambda *_: None), n_boot=200)
    sliver = summarize(measure(_Planted(trs, 0.1, "sliver"), trs, say=lambda *_: None), n_boot=200)
    assert sliver["frozen_wins"]["value"] == 1.0
    assert abs(sliver["change_predicted"]["value"] - 0.1) < 1e-5
    assert abs(sliver["change_from_action"]["value"] - 0.1) < 1e-5
    assert good["frozen_wins"]["value"] == 0.0 and abs(good["change_predicted"]["value"] - 1.0) < 1e-5
    assert frozen["change_predicted"]["value"] == 0.0 and frozen["frozen_wins"]["value"] == 1.0
    assert verdict(sliver["frozen_wins"], good["frozen_wins"]) == "yes"
    assert verdict(sliver["frozen_wins"], sliver["frozen_wins"]) == "unclear"  # no better than its control
    assert verdict(sliver["frozen_wins"], None) == "unclear"  # no control, no yes
    assert verdict(good["frozen_wins"], sliver["frozen_wins"]) == "no"


def test_verdict_rule_edges():
    def ci(v, lo, hi):
        return {"value": v, "lo": lo, "hi": hi}

    assert verdict(ci(0.9, 0.8, 0.95), ci(0.5, 0.45, 0.55)) == "yes"
    assert verdict(ci(0.9, 0.8, 0.95), ci(0.6, 0.55, 0.65)) == "unclear"  # 0.15 above the control: not enough
    assert verdict(ci(0.5, 0.4, 0.59), None) == "no"
    assert verdict(ci(0.5, float("nan"), float("nan")), None) == "insufficient_data"


def test_reset_weights_is_seeded_and_touches_every_parameter():
    torch.manual_seed(1)
    a = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    a.register_parameter("pos", nn.Parameter(torch.ones(3)))
    b = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    b.register_parameter("pos", nn.Parameter(torch.ones(3)))
    with torch.no_grad():
        for p in list(a.parameters()) + list(b.parameters()):
            p.fill_(5.0)
    reset_weights(a, seed=3)
    reset_weights(b, seed=3)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
        assert not torch.all(pa == 5.0)


def test_run_writes_json_card_and_heatmap(tmp_path):
    trs = _trajectories()
    res = run_freeze_check(_Planted(trs, 0.1, "sliver"), trs, tmp_path, control=_Planted(trs, 1.0, "good"), n_boot=100, say=lambda *_: None)
    assert res["verdict"] == "yes"
    data = json.loads((tmp_path / "freeze_check.json").read_text())
    assert data["schema"] == "freeze-check/v1" and data["control_name"] == "good"
    assert "Yes: it expects the world to stop." in (tmp_path / "index.html").read_text()
    assert (tmp_path / "heatmaps" / "moving_frozen.png").is_file()


def test_cli_freeze_check_tiny(tmp_path):
    clips = _clips_with_states(tmp_path, n=3)
    out = tmp_path / "fc"
    result = CliRunner().invoke(app, ["freeze-check", "--clips", str(clips), "--out", str(out), "--model", "tiny", "--bootstrap", "20"])
    assert result.exit_code == 0, result.output
    data = json.loads((out / "freeze_check.json").read_text())
    assert data["model_name"] == "tiny" and data["control_name"] == "tiny-untrained"
    assert data["verdict"] in {"yes", "no", "unclear", "insufficient_data"}
    assert data["model"]["n_steps"] > 0
    assert "verdict:" in result.output


def test_cli_freeze_check_rejects_unknown_model(tmp_path):
    clips = _clips_with_states(tmp_path, n=1)
    result = CliRunner().invoke(app, ["freeze-check", "--clips", str(clips), "--out", str(tmp_path / "x"), "--model", "nope"])
    assert result.exit_code == 1 and "unknown --model" in result.output


# ----------------------------------------------------------------------------- jepa-wms adapter


class _FakeVideoWM(nn.Module):
    """Shapes of jepa-wms' VideoWM: grid 4 x 4, width 8, a linear 'predictor' reading the action."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 8)
        self.predictor = nn.Linear(8, 8)
        self.action_encoder = nn.Linear(7, 8)
        self.proprio_encoder = None
        self.heads = None
        self.img_size, self.grid_size, self.enc_type, self.use_proprio, self.proprio_tokens = 64, 4, "dino", False, 0


class _FakeEncPredWM(nn.Module):  # like EncPredWM, an nn.Module holding the VideoWM
    def __init__(self):
        super().__init__()
        self.model = _FakeVideoWM()

    def encode(self, x):  # [B, 1, C, H, W] 0-255
        b = x.shape[0]
        pooled = x[:, 0].view(b, 3, 4, 16, 4, 16).mean(dim=(3, 5)).permute(0, 2, 3, 1)  # [B, 4, 4, 3]
        return self.model.encoder(pooled / 255.0).view(b, 1, 1, 4, 4, 8)

    def unroll(self, z_ctxt, act_suffix=None):
        a = self.model.action_encoder(act_suffix[0])  # [B, 8]
        nxt = self.model.predictor(z_ctxt[:, -1]) + a[:, None, None, None, :]
        return torch.stack([z_ctxt[:, -1], nxt])  # [2, B, 1, 4, 4, 8]


def test_jepawm_adapter_shapes_and_untrained_copy():
    from interpretableintelligence.adapters.jepawm import JepaWorldModel, untrained_copy

    model = JepaWorldModel(wm=_FakeEncPredWM(), crop=64, patch=16, device=torch.device("cpu"), name="fake-wm")
    frames = np.stack([_scene(t)[0] for t in range(3)])
    z = model.encode_frames(frames)
    assert z.shape == (3, 16, 8)
    y = model.predict_next(z, torch.zeros(3, 7), torch.ones(3, 7) * 0.01)
    assert y.shape == (3, 16, 8)
    before = model.wm.model.predictor.weight.clone()
    ctl = untrained_copy(model, seed=0)
    assert torch.equal(model.wm.model.predictor.weight, before)  # the trained model is left as it was
    assert torch.equal(model.predict_next(z, torch.zeros(3, 7), torch.ones(3, 7) * 0.01), y)
    assert ctl.wm.model.encoder is model.wm.model.encoder  # encoder shared, not reset
    assert torch.equal(ctl.encode_frames(frames), z)
    assert not torch.equal(ctl.wm.model.predictor.weight, model.wm.model.predictor.weight)
    assert not torch.allclose(ctl.predict_next(z, torch.zeros(3, 7), torch.ones(3, 7) * 0.01), y)
    assert ctl.name == "fake-wm-untrained"
