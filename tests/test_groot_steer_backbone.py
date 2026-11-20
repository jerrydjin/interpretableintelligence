"""Backbone steering runner for the task 1028 pot-order test (adapters/groot/v9/steer_backbone.py), without a GPU."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from interpretableintelligence.adapters.groot.v9 import steer_backbone as sb


class FakeHead(torch.nn.Module):
    def _encode_features(self, x):
        return SimpleNamespace(backbone_features=x.clone(), state_features=None)


def test_steerer_adds_vector_only_to_edited_rows():
    head = FakeHead()
    steer = sb.BackboneSteerer().install_on(head)
    x = torch.ones(2, 3, 4)
    edit = sb.SteerEdit("a", torch.tensor([1.0, 0.0, 0.0, 2.0]))
    steer.begin([None, edit])
    out = head._encode_features(x).backbone_features
    stats = steer.end()
    assert torch.equal(out[0], x[0])
    assert torch.allclose(out[1], x[1] + torch.tensor([1.0, 0.0, 0.0, 2.0]))
    assert stats[0] is None and stats[1]["rel_delta"] == pytest.approx(5 ** 0.5 / 2.0)
    steer.remove()
    assert "_encode_features" not in head.__dict__


def test_steerer_disarmed_and_baseline_rows_are_untouched():
    head = FakeHead()
    steer = sb.BackboneSteerer().install_on(head)
    x = torch.randn(1, 3, 4)
    assert torch.equal(head._encode_features(x).backbone_features, x)  # not armed
    steer.begin([None])
    assert torch.equal(head._encode_features(x).backbone_features, x)
    steer.end()


def test_capture_keeps_every_other_call_and_leaves_features_unchanged():
    head = FakeHead()
    steer = sb.BackboneSteerer().install_on(head)
    edit = sb.CaptureEdit("baseline", 7, every=2)
    for t in range(5):
        x = torch.full((2, 3, 4), float(t))
        steer.begin([edit, None])
        out = head._encode_features(x).backbone_features
        steer.end()
        assert torch.equal(out, x)
    tokens = steer.take("baseline", 7)
    assert tokens.shape == (3, 3, 4) and tokens.dtype == np.float16
    assert [float(tokens[i, 0, 0]) for i in range(3)] == [0.0, 2.0, 4.0]
    assert steer.take("baseline", 7) is None


def test_steerer_refuses_a_predict_that_never_reached_the_backbone():
    steer = sb.BackboneSteerer().install_on(FakeHead())
    steer.begin([sb.SteerEdit("a", torch.zeros(4))])
    with pytest.raises(RuntimeError, match="expected 1"):
        steer.end()


def _sim(lift0, lift1, T=50):
    pos = np.zeros((T, 2, 3))
    pos[:, :, 2] = 1.0
    if lift0 is not None:
        pos[lift0:, 0, 2] += 0.05
    if lift1 is not None:
        pos[lift1:, 1, 2] += 0.05
    return {"obj_pos": pos, "obj_names": np.array(sb.POT_NAMES)}


def test_pot_order():
    assert sb.pot_order(_sim(30, 10)) == {"pot0_lift_t": 30, "pot1_lift_t": 10, "pot0_first": False, "any_lift": True}
    assert sb.pot_order(_sim(5, 10))["pot0_first"] is True
    assert sb.pot_order(_sim(5, None))["pot0_first"] is True
    assert sb.pot_order(_sim(None, None)) == {"pot0_lift_t": None, "pot1_lift_t": None, "pot0_first": False, "any_lift": False}


def test_summarize_pairs_seeds_and_compares_random_arms(tmp_path):
    plan = {"schema": sb.SCHEMA, "outcome": "pot0_first",
            "arms": [{"id": "baseline", "kind": "baseline", "n": 4}, {"id": "dom", "kind": "direction", "n": 4},
                     {"id": "rand1", "kind": "random", "n": 4}],
            "sprt": {"p0": 0.05, "p1_rise": 0.35, "p1_fall": 0.01, "alpha": 0.05, "beta": 0.05, "cap": 4}}
    outcomes = {"baseline": [0, 0, 0, 0], "dom": [1, 1, 1, 0], "rand1": [0, 1, 0, 0]}
    for arm, ys in outcomes.items():
        (tmp_path / arm).mkdir()
        lines = [json.dumps({"ep": i, "pot0_first": bool(y), "success": True, "any_lift": True}) for i, y in enumerate(ys)]
        (tmp_path / arm / "episodes.jsonl").write_text("\n".join(lines) + "\n")
    s = sb.summarize(plan, tmp_path)
    rows = {t["arm"]: t for t in s["table"]}
    assert rows["dom"]["pot0_first"] == 3 and rows["dom"]["vs_baseline"]["n_cond_only_success"] == 3
    assert rows["dom"]["beats_every_random_arm"] is True
    assert "sprt" in rows["dom"] and s["finding"] is False


def test_arm_vectors(tmp_path):
    v = np.array([3.0, 4.0], np.float32)
    np.save(tmp_path / "d.npy", v)
    plan = {"arms": [{"id": "baseline", "kind": "baseline"},
                     {"id": "d2", "kind": "direction", "vector": str(tmp_path / "d.npy"), "scale": 2.0},
                     {"id": "r", "kind": "random", "random_seed": 7, "norm_of": str(tmp_path / "d.npy"), "scale": 2.0}]}
    out = sb.arm_vectors(plan)
    assert out["baseline"] is None
    assert torch.allclose(out["d2"], torch.tensor([6.0, 8.0]))
    assert float(out["r"].norm()) == pytest.approx(10.0, rel=1e-5)
