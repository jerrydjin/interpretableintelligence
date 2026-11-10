"""Causal rollouts on the fake reacher.

The planted feature must be flagged and ten random features must not, inside
a stated rollout budget. Reruns skip episodes that are already on disk.
A null policy that plants nothing must not flag features above alpha.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from interpretableintelligence.cli import app
from interpretableintelligence.policies import get_policy
from interpretableintelligence.policies.base import Intervention
from interpretableintelligence.policies.fake import PLANTED_FEATURE_ID, TASK_ID, FakePolicyEnv
from interpretableintelligence.policies.robots import RobotPolicyStub
from interpretableintelligence.artifacts import read_rollout_episodes, write_rollout_episodes
from interpretableintelligence.rollout import declared_stop
from interpretableintelligence.rollout_stats import (
    control_percentile,
    discordant_counts,
    feature_is_causal,
    mcnemar_exact_pvalue,
    paired_success_delta_ci,
    sprt_log_lr,
    sprt_separated,
    sprt_threshold,
    wilson_interval,
)
from interpretableintelligence.thresholds import SPRT_FAMILY_POWER, SPRT_LAMBDA

# Wilson early stop measured 184 on seed 0 and reused alpha at every look.
# The two-sided SPRT measured 264. The efficacy SPRT needs 14 discordant
# pairs; on seed 0 the last interim look (28) has 12, so this budget is full.
WILSON_ROLLOUTS_PER_VALIDATED_FEATURE = 184
TWO_SIDED_SPRT_ROLLOUTS_PER_VALIDATED_FEATURE = 264
EFFICACY_ROLLOUTS_PER_VALIDATED_FEATURE = 384
DONE_EPISODES = 32
DONE_CONTROLS = 10
NULL_SIMS = 2000
NULL_ALPHA = 0.05
# Andrej's power sim: 50% base, 20-point drop, about 68 episodes and ~744 total.
POWER_TARGET_EPISODES = 68.0
POWER_TARGET_TOTAL = 744.0

runner = CliRunner()


def _successes(ablate: int | None, n: int, seed0: int = 0) -> np.ndarray:
    env = FakePolicyEnv()
    out = np.zeros(n, dtype=bool)
    intervention = None
    if ablate is not None:
        intervention = Intervention((ablate,), env.layer, "ablate", 0.0)
    for offset in range(n):
        obs = env.reset(task_id=TASK_ID, seed=seed0 + offset)
        done = False
        info: dict = {}
        with env.intervene(intervention):
            while not done:
                obs, _reward, done, info = env.step(env.act(obs))
        out[offset] = bool(info["success"])
    return out


def test_planted_feature_moves_success_and_random_features_do_not():
    n = 2000
    baseline = _successes(None, n)
    planted = _successes(PLANTED_FEATURE_ID, n)
    assert 0.75 <= float(baseline.mean()) <= 0.85
    assert 0.25 <= float(planted.mean()) <= 0.36
    assert float(baseline.mean() - planted.mean()) > 0.4
    for feature_id in (0, 1, 2, 4, 15):
        other = _successes(feature_id, n)
        assert np.array_equal(other, baseline)

    env = FakePolicyEnv()
    obs = env.reset(task_id=TASK_ID, seed=0)
    informed = env.act(obs)
    with env.intervene(Intervention((PLANTED_FEATURE_ID,), env.layer, "ablate", 0.0)):
        ablated = env.act(obs)
    restored = env.act(obs)
    assert not np.allclose(informed, ablated)
    assert np.allclose(informed, restored)

    scale_zero = _successes_kind("scale", 0.0, n=200)
    clamp_zero = _successes_kind("clamp", 0.0, n=200)
    assert np.array_equal(scale_zero, _successes(PLANTED_FEATURE_ID, 200))
    assert np.array_equal(clamp_zero, _successes(PLANTED_FEATURE_ID, 200))


def _successes_kind(kind: str, value: float, n: int) -> np.ndarray:
    env = FakePolicyEnv()
    intervention = Intervention((PLANTED_FEATURE_ID,), env.layer, kind, value)
    out = np.zeros(n, dtype=bool)
    for offset in range(n):
        obs = env.reset(task_id=TASK_ID, seed=offset)
        done = False
        info: dict = {}
        with env.intervene(intervention):
            while not done:
                obs, _reward, done, info = env.step(env.act(obs))
        out[offset] = bool(info["success"])
    return out


def test_mcnemar_exact_and_causal_rule():
    assert mcnemar_exact_pvalue(8, 0) == pytest.approx(2 / 256)
    assert mcnemar_exact_pvalue(0, 0) == 1.0
    assert mcnemar_exact_pvalue(3, 3) == 1.0
    assert feature_is_causal(delta=0.5, p_value=0.01, control_percentile=100.0, alpha=0.05) is True
    assert feature_is_causal(delta=0.5, p_value=0.01, control_percentile=None, alpha=0.05) is False
    assert feature_is_causal(delta=0.0, p_value=1.0, control_percentile=100.0, alpha=0.05) is False


def test_control_percentile_splits_ties():
    assert control_percentile(0.0, [0.0] * 10) == 50.0
    assert control_percentile(0.5, [0.0] * 10) == 100.0
    assert control_percentile(-0.2, [0.0] * 10) == 0.0
    # Two strictly below, one tie, one strictly above.
    assert control_percentile(0.2, [0.0, 0.0, 0.2, 0.4]) == pytest.approx(62.5)
    assert control_percentile(0.1, []) is None
    assert control_percentile(0.0, [1e-12, -1e-12]) == 50.0


def test_sprt_stops_only_for_efficacy():
    alpha = 0.05
    n_controls = DONE_CONTROLS
    beta = 1.0 - SPRT_FAMILY_POWER ** (1.0 / n_controls)
    threshold = sprt_threshold(alpha, n_controls)
    assert beta == pytest.approx(0.022, abs=0.001)
    assert threshold == pytest.approx(math.log((1.0 - beta) / (alpha / n_controls)))
    # 13 pure efficacy pairs sit just under the boundary; 14 cross it.
    efficacy = [True] * 14
    ablated = [False] * 14
    assert sprt_log_lr(efficacy[:13], ablated[:13], SPRT_LAMBDA) < threshold
    assert sprt_log_lr(efficacy, ablated, SPRT_LAMBDA) >= threshold
    assert sprt_separated(efficacy, [ablated], alpha, n_controls=n_controls) is True
    # The opposite direction is not efficacy, however long it runs.
    assert sprt_separated([False] * 200, [[True] * 200], alpha, n_controls=n_controls) is False
    assert sprt_separated([True] * 40, [[True] * 40], alpha, n_controls=n_controls) is False
    assert sprt_separated([], [[]], alpha, n_controls=n_controls) is False


def test_robot_policies_are_stubs():
    with pytest.raises(NotImplementedError, match="robots"):
        get_policy("libero")
    with pytest.raises(NotImplementedError, match="pi0.5"):
        RobotPolicyStub("pi0.5").reset(task_id="reach", seed=0)
    with pytest.raises(ValueError, match="fake"):
        get_policy("vjepa2")


def _invoke(out: Path, workers: int, episodes: int = DONE_EPISODES, controls: int = DONE_CONTROLS, min_episodes: int = 8):
    return runner.invoke(
        app,
        [
            "rollouts",
            str(out),
            "--policy",
            "fake",
            "--episodes",
            str(episodes),
            "--features",
            str(PLANTED_FEATURE_ID),
            "--controls",
            str(controls),
            "--workers",
            str(workers),
            "--min-episodes",
            str(min_episodes),
            "--batch-size",
            "4",
            "--alpha",
            "0.05",
            "--seed",
            "0",
        ],
    )


def test_done_check_planted_is_causal_and_rerun_is_idempotent(tmp_path: Path):
    out = tmp_path / "reach"
    first = _invoke(out, workers=2)
    assert first.exit_code == 0, first.output

    summary = json.loads((out / "rollouts" / "summary.json").read_text())
    by_id = {feature["feature_id"]: feature for feature in summary["features"]}
    planted = by_id[PLANTED_FEATURE_ID]
    controls = [feature for feature in summary["features"] if feature["role"] == "control"]
    assert planted["role"] == "candidate"
    assert planted["causal"] is True
    assert planted["delta"] >= 0.15
    assert planted["p_value"] <= 0.05
    assert planted["control_percentile"] == 100.0
    assert planted["stopped_early"] is False
    assert planted["episodes_run"] == DONE_EPISODES
    assert len(controls) == DONE_CONTROLS
    assert all(feature["causal"] is False for feature in controls)
    assert all(feature["stopped_early"] is False for feature in controls)
    assert all(feature["reason"] == "fixed_budget" for feature in controls)
    assert all(feature["episodes_run"] == DONE_EPISODES for feature in controls)
    assert all(feature["delta"] == 0.0 for feature in controls)
    assert all(feature["control_percentile"] == 50.0 for feature in controls)

    control_conditions = [item for item in summary["conditions"] if item["role"] == "control"]
    assert len(control_conditions) == DONE_CONTROLS
    threshold = sprt_threshold(0.05, DONE_CONTROLS)
    for condition in control_conditions:
        assert condition["reason"] == "fixed_budget"
        assert condition["stopped_early"] is False
        assert condition["sprt_threshold"] == pytest.approx(threshold)
    assert summary["gap_bound"] == "sprt_discordant"
    assert summary["sprt_lambda"] == SPRT_LAMBDA

    total = summary["total_episodes"]
    naive = (1 + DONE_CONTROLS) * DONE_EPISODES
    assert total == EFFICACY_ROLLOUTS_PER_VALIDATED_FEATURE, (
        f"got {total}; Wilson measured {WILSON_ROLLOUTS_PER_VALIDATED_FEATURE}, "
        f"two-sided SPRT measured {TWO_SIDED_SPRT_ROLLOUTS_PER_VALIDATED_FEATURE}"
    )
    assert total == summary["fixed_budget_episodes"]
    assert summary["episodes_saved"] == 0
    assert summary["validated_feature_ids"] == [PLANTED_FEATURE_ID]
    assert summary["rollouts_per_validated_feature"] == pytest.approx(total)
    assert f"feature_arms_x_N={naive}" in first.output
    assert "rollouts_per_validated_feature=" in first.output
    assert f"total_episodes={total}" in first.output
    assert f"feature_arms_x_N={naive}" in first.output

    rows = read_rollout_episodes(out / "rollouts" / "episodes.parquet")
    text = (out / "rollouts" / "summary.json").read_text()
    second = _invoke(out, workers=2)
    assert second.exit_code == 0, second.output
    assert (out / "rollouts" / "summary.json").read_text() == text
    assert read_rollout_episodes(out / "rollouts" / "episodes.parquet") == rows
    assert second.output == first.output

    # A hole in the parquet is filled with the same successes and the same stop.
    kept = [row for row in rows if row["seed"] % 5 != 0]
    assert len(kept) < len(rows)
    write_rollout_episodes(out / "rollouts" / "episodes.parquet", kept)
    resumed = _invoke(out, workers=2)
    assert resumed.exit_code == 0, resumed.output
    refilled = read_rollout_episodes(out / "rollouts" / "episodes.parquet")
    assert [(row["episode_id"], row["success"], row["steps"]) for row in refilled] == [
        (row["episode_id"], row["success"], row["steps"]) for row in rows
    ]
    kept_ids = {row["episode_id"] for row in kept}
    by_old = {row["episode_id"]: row for row in rows}
    for row in refilled:
        if row["episode_id"] in kept_ids:
            assert row["wall_time_s"] == by_old[row["episode_id"]]["wall_time_s"]
    assert (out / "rollouts" / "summary.json").read_text() == text


def test_parallel_workers_match_serial_outcomes(tmp_path: Path):
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    first = _invoke(serial, workers=1, episodes=16, controls=4)
    second = _invoke(parallel, workers=2, episodes=16, controls=4)
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    left = json.loads((serial / "rollouts" / "summary.json").read_text())
    right = json.loads((parallel / "rollouts" / "summary.json").read_text())
    left["workers"] = right["workers"] = 0
    assert left == right


def test_min_episodes_can_disable_early_stopping(tmp_path: Path):
    out = tmp_path / "full"
    result = _invoke(out, workers=1, episodes=8, controls=2, min_episodes=8)
    assert result.exit_code == 0, result.output
    summary = json.loads((out / "rollouts" / "summary.json").read_text())
    assert summary["total_episodes"] == summary["fixed_budget_episodes"] == 4 * 8
    assert all(condition["stopped_early"] is False for condition in summary["conditions"])
    assert all(condition["reason"] == "fixed_budget" for condition in summary["conditions"])


def test_rollouts_rejects_a_bad_request(tmp_path: Path):
    missing = runner.invoke(app, ["rollouts", str(tmp_path / "x"), "--policy", "fake", "--features", ""])
    assert missing.exit_code != 0
    unknown = runner.invoke(
        app,
        ["rollouts", str(tmp_path / "x"), "--policy", "libero", "--features", "0", "--episodes", "2", "--controls", "0"],
    )
    assert unknown.exit_code != 0
    assert "stub" in (unknown.output + (unknown.stderr or "")).lower() or "stub" in unknown.output
    too_soon = runner.invoke(
        app,
        [
            "rollouts",
            str(tmp_path / "x"),
            "--policy",
            "fake",
            "--features",
            "3",
            "--episodes",
            "4",
            "--min-episodes",
            "8",
            "--controls",
            "1",
        ],
    )
    assert too_soon.exit_code != 0


def _null_successes(ablate: int | None, n: int, seed0: int = 0) -> np.ndarray:
    env = FakePolicyEnv(plant=False)
    out = np.zeros(n, dtype=bool)
    intervention = None
    if ablate is not None:
        intervention = Intervention((ablate,), env.layer, "ablate", 0.0)
    for offset in range(n):
        obs = env.reset(task_id=TASK_ID, seed=seed0 + offset)
        done = False
        info: dict = {}
        with env.intervene(intervention):
            while not done:
                obs, _reward, done, info = env.step(env.act(obs))
        out[offset] = bool(info["success"])
    return out


def test_null_policy_is_exchangeable_and_not_identical():
    env = get_policy("fake-null")
    assert env.name == "fake-null"
    assert env.plant is False
    n = 400
    baseline = _null_successes(None, n)
    planted = _null_successes(PLANTED_FEATURE_ID, n)
    other = _null_successes(0, n)
    for name, flags in ("baseline", baseline), ("planted", planted), ("other", other):
        rate = float(flags.mean())
        assert 0.40 <= rate <= 0.60, f"{name} success rate {rate:.3f} is not near 1/2"
    assert not np.array_equal(baseline, planted)
    assert not np.array_equal(baseline, other)
    # The planted policy is unchanged: feature 3 still moves success, feature 0 does not.
    assert not np.array_equal(_successes(None, 50), _successes(PLANTED_FEATURE_ID, 50))
    assert np.array_equal(_successes(None, 50), _successes(0, 50))


def _stop_length(control: list[bool], candidate: list[bool], *, episodes: int, min_episodes: int, batch_size: int) -> int:
    length, _reason = declared_stop(
        control,
        [candidate],
        episodes=episodes,
        min_episodes=min_episodes,
        batch_size=batch_size,
        alpha=NULL_ALPHA,
        n_controls=DONE_CONTROLS,
    )
    return length


def _flagged(baseline: list[bool], candidate: list[bool], controls: list[list[bool]]) -> bool:
    """Causal gate after efficacy stopping, matching ``summarize``."""
    stopped = [
        control[: _stop_length(control, candidate, episodes=len(candidate), min_episodes=8, batch_size=4)]
        for control in controls
    ]
    delta, _low, _high = paired_success_delta_ci(baseline, candidate, NULL_ALPHA)
    n10, n01 = discordant_counts(baseline, candidate)
    p_value = mcnemar_exact_pvalue(n10, n01)
    control_deltas = []
    for flags in stopped:
        control_delta, _lo, _hi = paired_success_delta_ci(baseline[: len(flags)], flags, NULL_ALPHA)
        control_deltas.append(control_delta)
    return feature_is_causal(
        delta=delta,
        p_value=p_value,
        control_percentile=control_percentile(delta, control_deltas),
        alpha=NULL_ALPHA,
    )


def test_power_sim_twenty_point_drop_stops_near_68():
    """50% base success, switching the feature off drops 20 points.

    Paired seeds share one uniform draw: the control succeeds when it is
    below 0.50 and the candidate when it is below 0.30, so every discordant
    pair is efficacy. Ten controls see that same base outcome. The candidate
    is run for the same length. Andrej's target is about 68 episodes per
    condition and about 744 total.
    """
    rng = np.random.default_rng(0)
    cap = 500
    reps = 2000
    stops = np.empty(reps)
    for i in range(reps):
        draw = rng.random(cap)
        candidate = (draw < 0.30).tolist()
        control = (draw < 0.50).tolist()
        stops[i] = _stop_length(control, candidate, episodes=cap, min_episodes=1, batch_size=1)
    mean_episodes = float(stops.mean())
    # One candidate plus ten identical controls.
    total = (1 + DONE_CONTROLS) * mean_episodes
    message = (
        f"mean_episodes_per_condition={mean_episodes:.2f} "
        f"(target {POWER_TARGET_EPISODES:.0f}) total={total:.1f} "
        f"(target {POWER_TARGET_TOTAL:.0f}) capped={(stops == cap).mean():.3f}"
    )
    print(message)
    # Measured on this boundary: about 71 episodes and about 778 total,
    # next to Andrej's 68 and 744.
    assert mean_episodes == pytest.approx(70.8, abs=2.0), message
    assert total == pytest.approx(11 * 70.8, abs=22.0), message
    assert float((stops == cap).mean()) < 0.01


def test_null_false_flag_rate_stays_within_alpha():
    """Nothing planted. Exact arms never flag; noisy arms stay within alpha.

    Two thousand sims of one candidate and ten controls. The Wilson upper
    bound on each false-flag rate has to sit at or below alpha.
    """
    rng = np.random.default_rng(0)
    n = DONE_EPISODES
    exact_flags = 0
    for _ in range(NULL_SIMS):
        bits = (rng.random(n) < 0.5).tolist()
        controls = [list(bits) for _control in range(DONE_CONTROLS)]
        exact_flags += _flagged(bits, list(bits), controls)
    exact_low, exact_high = wilson_interval(exact_flags, NULL_SIMS, NULL_ALPHA)
    noisy_flags = 0
    for _ in range(NULL_SIMS):
        baseline = (rng.random(n) < 0.5).tolist()
        candidate = (rng.random(n) < 0.5).tolist()
        controls = [(rng.random(n) < 0.5).tolist() for _control in range(DONE_CONTROLS)]
        noisy_flags += _flagged(baseline, candidate, controls)
    noisy_low, noisy_high = wilson_interval(noisy_flags, NULL_SIMS, NULL_ALPHA)
    exact_message = (
        f"exact null {exact_flags}/{NULL_SIMS}={exact_flags / NULL_SIMS:.4f} "
        f"wilson95=[{exact_low:.4f},{exact_high:.4f}]"
    )
    noisy_message = (
        f"noisy controls {noisy_flags}/{NULL_SIMS}={noisy_flags / NULL_SIMS:.4f} "
        f"wilson95=[{noisy_low:.4f},{noisy_high:.4f}] alpha={NULL_ALPHA}"
    )
    print(exact_message)
    print(noisy_message)
    assert NULL_SIMS >= 2000
    assert exact_flags == 0, exact_message
    assert exact_high <= NULL_ALPHA
    assert noisy_high <= NULL_ALPHA, noisy_message
