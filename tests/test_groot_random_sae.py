"""Research rule 4 on GR00T: untrained SAEs, one picker for every dictionary, and rise-test scoring.

A small synthetic capture stands in for ``groot-1028-v9-n100`` (same layout, a few hundred tokens), so the real
picker (``picker/pick_candidates.py``) runs on it in seconds. Feature 5 of its trained SAE is planted to fire
before failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from typer.testing import CliRunner

from interpretableintelligence.adapters.groot import random_sae
from interpretableintelligence.adapters.groot.v9 import ablate_v9
from interpretableintelligence.cli import app
from interpretableintelligence.rollout_groot import execute_groot_pick
from interpretableintelligence.validate import _check_rollouts

D, M, K = 12, 24, 3
PLANTED = 5
TOKENS = (0, 1, 2, 3)
N_EPISODES = 32

runner = CliRunner()


def _write_capture(root: Path) -> Path:
    """A capture run folder in the groot-v9 picker layout, with feature PLANTED rising before every failure."""
    rng = np.random.default_rng(0)
    capture = root / "capture"
    (capture / "acts").mkdir(parents=True)
    (capture / "sae").mkdir()
    directions, _ = np.linalg.qr(rng.standard_normal((D, D)))
    w_dec = np.concatenate([directions.T, rng.standard_normal((M - D, D))], 0)
    w_dec /= np.linalg.norm(w_dec, axis=1, keepdims=True)
    trained = {
        "W_enc": torch.tensor(w_dec.T.copy(), dtype=torch.float32),
        "W_dec": torch.tensor(w_dec, dtype=torch.float32),
        "b_enc": torch.zeros(M),
        "b_dec": torch.zeros(D),
        "norm_mu": torch.zeros(D),
        "norm_scale": torch.ones(1),
        "threshold": torch.tensor(0.5),
    }
    save_file(trained, str(capture / "sae" / "weights.safetensors"))
    (capture / "sae" / "config.json").write_text(json.dumps({"arch": "batchtopk", "k": K, "d_in": D, "d_sae": M}))
    episodes = []
    for episode in range(N_EPISODES):
        success = episode % 2 == 0
        n_steps = 160
        fail_t = None if success else 128
        steps = list(range(0, n_steps, 16))
        rows, step_col, tok_col = [], [], []
        for step in steps:
            for tok in TOKENS:
                code = np.zeros(M)
                live = rng.choice(np.arange(D), size=K, replace=False)
                code[live] = rng.uniform(1.0, 2.0, size=K)
                code[PLANTED] = 0.0
                near_failure = fail_t is not None and fail_t - 48 <= step < fail_t
                if near_failure or rng.uniform() < 0.15:
                    code[PLANTED] = rng.uniform(2.0, 3.0) if near_failure else rng.uniform(1.0, 2.0)
                rows.append(code @ w_dec + 0.01 * rng.standard_normal(D))
                step_col.append(step)
                tok_col.append(tok)
        save_file(
            {
                "acts": torch.tensor(np.array(rows), dtype=torch.float16),
                "episode_idx": torch.full((len(rows),), episode, dtype=torch.int32),
                "step": torch.tensor(step_col, dtype=torch.int32),
                "tok": torch.tensor(tok_col, dtype=torch.int16),
            },
            str(capture / "acts" / f"shard-{episode:05d}.safetensors"),
        )
        episodes.append(
            {
                "episode_idx": episode,
                "episode_id": f"ep-{episode:05d}",
                "task_id": "1028",
                "seed": 20_000 + episode,
                "success": success,
                "fail_t": fail_t,
                "n_steps": n_steps,
            }
        )
    manifest = {
        "model": "nvidia/gr00t17-lerobot-libero_10-640",
        "layer": 5,
        "d_model": D,
        "task_index": 1027,
        "task_id": "1028",
        "task_name": "synthetic",
        "n_init_states": 50,
        "token_indices": list(TOKENS),
        "episodes": episodes,
    }
    (capture / "manifest.json").write_text(json.dumps(manifest))
    held = [episode for episode in range(N_EPISODES) if episode % 4 == 3]
    (capture / "split.json").write_text(
        json.dumps({"heldout": held, "train": [episode for episode in range(N_EPISODES) if episode not in held]})
    )
    return capture


@pytest.fixture(scope="module")
def capture(tmp_path_factory) -> Path:
    return _write_capture(tmp_path_factory.mktemp("groot-capture"))


def _run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "groot-run"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"model": "nvidia/gr00t17-lerobot-libero_10-640", "layer": 5, "tool_version": "test"}))
    return run


def _chunks(capture: Path):
    from interpretableintelligence.rollout_groot_pick import capture_info, train_chunks

    return train_chunks(capture, capture_info(capture)["train"])


def test_untrained_groot_sae_keeps_shape_normalisation_and_k(capture: Path):
    path = capture / "sae" / "weights.safetensors"
    weights, record = random_sae.build_untrained_sae(path, 7, lambda: _chunks(capture), k=K)
    trained = random_sae.load_tensors(path)
    for key in random_sae.SAE_KEYS:
        assert weights[key].shape == trained[key].shape or key in ("norm_scale", "threshold"), key
    assert torch.equal(weights["norm_mu"], trained["norm_mu"])
    assert torch.allclose(weights["W_dec"].norm(dim=1), torch.ones(M))
    assert torch.equal(weights["W_enc"], weights["W_dec"].T)
    assert not torch.allclose(weights["W_dec"], trained["W_dec"])
    rows = torch.cat(list(_chunks(capture)))
    z = random_sae.pre_activations(weights, rows)
    l0 = float((z >= weights["threshold"]).sum(1).float().mean())
    assert l0 == pytest.approx(K, abs=1e-6)
    assert record["calibration_rows"] == rows.shape[0]
    again, _ = random_sae.build_untrained_sae(path, 7, lambda: _chunks(capture), k=K)
    other, _ = random_sae.build_untrained_sae(path, 8, lambda: _chunks(capture), k=K)
    assert all(torch.equal(weights[key], again[key]) for key in random_sae.SAE_KEYS)
    assert not torch.equal(weights["W_dec"], other["W_dec"])


def _subset(seed: int, latents) -> ablate_v9.SAESubset:
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(D, M, generator=g)
    w = w / w.norm(dim=0, keepdim=True)
    full = ablate_v9.SAESubset(list(range(M)), w, torch.zeros(M), w.T.contiguous(), torch.zeros(D), torch.zeros(D), 2.0, 0.05, {})
    pos = full.pos(latents)
    return ablate_v9.SAESubset(list(latents), full.W_enc[:, pos], full.b_enc[pos], full.W_dec[pos], full.b_dec, full.norm_mu, 2.0, 0.05, {})


def test_latent_ablator_edits_each_row_through_its_dictionary():
    trained, untrained = _subset(0, [3, 4]), _subset(1, [9])
    edits = [
        ablate_v9.Edit("cand", (0,), (0,), "trained"),
        ablate_v9.Edit("rsae", (0,), (0,), "random:1"),
        None,
        ablate_v9.Edit("rsae-recon", (), (0,), "random:1"),
    ]
    ablator = ablate_v9.LatentAblator({"trained": trained, "random:1": untrained})
    ablator.begin(edits)
    x = torch.randn(4, 5, D, generator=torch.Generator().manual_seed(2)) * 3
    out, _stats = ablator.edit_tensor(x)
    for row, sae in ((0, trained), (1, untrained)):
        f = sae.encode(x[row].float())[:, :1]
        assert f.abs().sum() > 0
        expect = x[row] - (f @ sae.W_dec[:1]) * sae.norm_scale
        assert torch.allclose(out[row], expect, atol=1e-5)
    assert torch.equal(out[2], x[2])
    assert torch.equal(out[3], x[3])  # an empty edit of an untrained SAE keeps its error: bit-identical


def test_make_edit_needs_the_named_dictionary():
    cond = {"id": "rsae", "latents": [9], "dictionary": "random:1"}
    with pytest.raises(ValueError, match="only the trained SAE"):
        ablate_v9.make_edit("rsae", cond, _subset(0, [3, 9]))
    edit = ablate_v9.make_edit("rsae", cond, {"trained": _subset(0, [3]), "random:1": _subset(1, [9])})
    assert edit.dictionary == "random:1" and edit.sel == (0,)


def test_groot_mock_pick_runs_rule_4_and_validates(tmp_path: Path, capture: Path):
    run = _run_dir(tmp_path)
    args = ["rollouts", str(run), "--policy", "groot-mock", "--pick", "2", "--random-sae", "2", "--controls", "2",
            "--episodes", "8", "--capture", str(capture)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    summary = json.loads((run / "rollouts" / "summary.json").read_text())
    plan = json.loads((run / "rollouts" / "ablate_plan.json").read_text())
    assert summary["direction"] == "rise"
    assert summary["candidate_source"] == "pick" and summary["picker"] == "groot_failure_contrast"
    assert PLANTED in summary["candidate_feature_ids"]
    assert len(summary["random_sae_seeds"]) == 2
    assert [len(picks) for picks in summary["random_sae_feature_ids"]] == [2, 2]
    roles = [feature["role"] for feature in summary["features"]]
    assert roles.count("random_sae") == 4
    candidates = [feature for feature in summary["features"] if feature["role"] == "candidate"]
    assert all(feature["n_random_sae"] == 4 and feature["beats_random_sae"] is False for feature in candidates)
    assert summary["finding_feature_ids"] == []  # groot-mock's default outcome has no effect anywhere
    names = set(plan["sae"]["dictionaries"])
    assert names == {"trained", *(f"random:{seed}" for seed in summary["random_sae_seeds"])}
    for name, entry in plan["sae"]["dictionaries"].items():
        assert ablate_v9.sha256_file(run / "rollouts" / entry["subset_file"]) == entry["subset_sha256"]
        if name != "trained":
            assert entry["k"] == K and entry["threshold"] > 0
    # the hook can load every dictionary the plan names
    paths = {name: run / "rollouts" / entry["subset_file"] for name, entry in plan["sae"]["dictionaries"].items()}
    saes = ablate_v9.load_dictionaries(plan, paths.pop("trained"), paths)
    assert set(saes) == names
    errors: list[str] = []
    _check_rollouts(run, 1, errors)
    assert errors == []
    again = runner.invoke(app, args)  # rerun reuses the plan and the picks
    assert again.exit_code == 0, again.output
    assert "reusing" in again.output
    changed = list(args)
    changed[changed.index("--episodes") + 1] = "12"
    refused = runner.invoke(app, changed)  # a plan built with other arguments is never overwritten
    assert refused.exit_code != 0


def _planted(rise_random: bool):
    def simulate(cond: dict, ep: int, seed: int) -> bool:
        base = ep % 3 == 0  # baseline success 1/3, the same seeds on every arm
        if cond["kind"] == "candidate" and cond["latents"] == [PLANTED]:
            return True
        if rise_random and cond["kind"] == "random_sae":
            return True
        return base

    return simulate


def test_planted_rise_is_a_finding_only_above_every_random_sae_arm(tmp_path: Path, capture: Path):
    kwargs = dict(policy="groot-mock", capture=capture, pick=2, random_sae=2, controls=2, episodes=24, log=lambda *_: None)
    summary = execute_groot_pick(_run_dir(tmp_path / "a"), simulate=_planted(False), **kwargs)
    planted = next(f for f in summary["features"] if f["role"] == "candidate" and f["feature_id"] == PLANTED)
    assert planted["delta"] == pytest.approx(16 / 24)  # rise: intervention - baseline
    assert planted["causal"] and planted["beats_random_sae"] is True and planted["sprt_verdict"] == "rise"
    assert planted["finding"] is True
    assert summary["finding_feature_ids"] == [PLANTED]

    blocked = execute_groot_pick(_run_dir(tmp_path / "b"), simulate=_planted(True), **kwargs)
    planted = next(f for f in blocked["features"] if f["role"] == "candidate" and f["feature_id"] == PLANTED)
    assert planted["causal"]  # beats the trained SAE's controls ...
    assert planted["beats_random_sae"] is False and planted["finding"] is False  # ... but ties the untrained SAEs
    assert blocked["finding_feature_ids"] == []


def test_groot_random_sae_without_pick_is_refused(tmp_path: Path):
    result = runner.invoke(app, ["rollouts", str(_run_dir(tmp_path)), "--policy", "groot-mock", "--verify-slice", "--random-sae", "2"])
    assert result.exit_code != 0
    result = runner.invoke(app, ["rollouts", str(_run_dir(tmp_path / "x")), "--policy", "groot-mock", "--pick", "2"])
    assert result.exit_code != 0


def test_rise_needs_the_sobel_wald_call_too(tmp_path: Path, capture: Path):
    """Rule 1: causal and above every untrained SAE is still not a finding until the plan's SPRT calls a rise."""

    def simulate(cond: dict, ep: int, seed: int) -> bool:
        if cond["kind"] == "candidate" and cond["latents"] == [PLANTED]:
            return ep % 3 != 1  # 16/24 against a 1/3 baseline: McNemar 8 to 0, but the LLR hovers near 0
        return ep % 3 == 0

    run = _run_dir(tmp_path)
    kwargs = dict(policy="groot-mock", capture=capture, pick=2, random_sae=2, controls=2, episodes=24, log=lambda *_: None)
    summary = execute_groot_pick(run, simulate=simulate, **kwargs)
    plan = json.loads((run / "rollouts" / "ablate_plan.json").read_text())
    assert plan["sprt"]["p0"] == 0.5 and plan["sprt"]["p1_rise"] == 0.8  # the capture's success rate, +0.30
    assert summary["rise_sprt"] == {key: plan["sprt"][key] for key in summary["rise_sprt"]}
    assert all(cond["sprt"] is False for cond in plan["conditions"])  # still a fixed budget
    planted = next(f for f in summary["features"] if f["role"] == "candidate" and f["feature_id"] == PLANTED)
    assert planted["causal"] and planted["beats_random_sae"] is True
    assert planted["sprt_verdict"] == "cap_undecided"
    assert planted["finding"] is False and summary["finding_feature_ids"] == []
    errors: list[str] = []
    _check_rollouts(run, 1, errors)  # ii validate rebuilds the same verdicts from the summary's rise_sprt
    assert errors == []


def test_plan_without_a_rise_design_has_no_findings():
    """A --pick plan written before it carried the rule 1 design can no longer produce a finding."""
    from interpretableintelligence.rollout_groot import plan_rise_sprt

    assert plan_rise_sprt({"sprt": {"p0": None, "cap": 24}}) is None
    assert plan_rise_sprt({}) is None


def test_named_features_positive_control_scores_a_fall_and_validates(tmp_path: Path, capture: Path):
    """--features with --capture: a fixed-budget plan for hand-named features, here scored as a fall (positive control)."""

    def simulate(cond: dict, ep: int, seed: int) -> bool:
        if cond["kind"] == "candidate":
            return False  # the strong edit breaks every episode
        return ep % 3 != 0  # baseline and controls succeed 2/3, on the same seeds

    run = _run_dir(tmp_path)
    from interpretableintelligence.rollout_groot import execute_groot_named

    summary = execute_groot_named(run, policy="groot-mock", capture=capture, candidates=[PLANTED], controls=[1, 2, 3],
                                  episodes=24, direction="fall", simulate=simulate, log=lambda *_: None)
    plan = json.loads((run / "rollouts" / "ablate_plan.json").read_text())
    assert plan["direction"] == "fall" and plan["pick"]["picker"] == "named"
    assert [c["kind"] for c in plan["conditions"]] == ["baseline", "recon_only", "candidate", "control", "control", "control"]
    assert summary["direction"] == "fall" and summary["candidate_source"] == "features" and summary["rise_sprt"] is None
    planted = next(f for f in summary["features"] if f["role"] == "candidate")
    assert planted["delta"] == pytest.approx(16 / 24)  # fall: baseline - intervention
    assert planted["causal"] is True and planted["finding"] is False  # a fall is never a finding
    errors: list[str] = []
    _check_rollouts(run, 1, errors)
    assert errors == []


def test_named_features_cli_refuses_overlap_and_rebuild(tmp_path: Path, capture: Path):
    run = _run_dir(tmp_path)
    base = ["rollouts", str(run), "--policy", "groot-mock", "--capture", str(capture), "--episodes", "8"]
    overlap = runner.invoke(app, [*base, "--features", str(PLANTED), "--control-features", f"1,{PLANTED}"])
    assert overlap.exit_code != 0 and "both candidates and controls" in overlap.output
    ok = runner.invoke(app, [*base, "--features", str(PLANTED), "--control-features", "1,2"])
    assert ok.exit_code == 0, ok.output
    changed = runner.invoke(app, [*base, "--features", str(PLANTED), "--control-features", "1,3"])
    assert changed.exit_code != 0  # a plan built with other arguments is never overwritten


def test_named_selection_with_random_sae_arms_can_be_a_finding(tmp_path: Path, capture: Path):
    """--selection: the untrained SAEs' picks from the same selection become rule 4's arms, so a rise can be a finding."""
    import shutil

    random_dir = tmp_path / "random"
    for item in (11, 12):
        shutil.copytree(capture / "sae", random_dir / str(item) / "sae")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"picker": "contrast_then_action_screen", "random_sae_dir": str(random_dir),
                                     "random_sae": {"11": [7], "12": [8]}, "note": "test"}))

    def simulate(cond: dict, ep: int, seed: int) -> bool:
        return True if cond["kind"] == "candidate" else ep % 2 == 0

    run = _run_dir(tmp_path)
    from interpretableintelligence import rollout_groot

    original = rollout_groot.null_mock_outcome
    rollout_groot.null_mock_outcome = simulate
    try:
        result = runner.invoke(app, ["rollouts", str(run), "--policy", "groot-mock", "--capture", str(capture), "--episodes", "24",
                                     "--features", str(PLANTED), "--control-features", "1,2", "--selection", str(selection)])
    finally:
        rollout_groot.null_mock_outcome = original
    assert result.exit_code == 0, result.output
    summary = json.loads((run / "rollouts" / "summary.json").read_text())
    assert summary["candidate_source"] == "pick" and summary["picker"] == "contrast_then_action_screen"
    assert summary["random_sae_seeds"] == [11, 12] and summary["random_sae_feature_ids"] == [[7], [8]]
    planted = next(f for f in summary["features"] if f["role"] == "candidate")
    assert planted["causal"] is True and planted["sprt_verdict"] == "rise" and planted["finding"] is True
    assert summary["finding_feature_ids"] == [PLANTED]
    errors: list[str] = []
    _check_rollouts(run, 1, errors)
    assert errors == []
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"random_sae_dir": str(random_dir), "random_sae": {"11": [7]}}))
    refused = runner.invoke(app, ["rollouts", str(_run_dir(tmp_path / "b")), "--policy", "groot-mock", "--capture", str(capture),
                                  "--episodes", "8", "--features", str(PLANTED), "--selection", str(bad)])
    assert refused.exit_code != 0  # random-SAE arms without the selection's name
