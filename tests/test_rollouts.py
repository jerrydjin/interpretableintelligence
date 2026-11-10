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

from physint.cli import app
from physint.policies import get_policy
from physint.policies.base import Intervention
from physint.policies.fake import PLANTED_FEATURE_ID, TASK_ID, FakePolicyEnv
from physint.policies.robots import RobotPolicyStub
from physint.artifacts import read_rollout_episodes, write_rollout_episodes
from physint.rollout import execute_rollouts
from physint.rollout_stats import (
    control_percentile,
    feature_is_causal,
    mcnemar_exact_pvalue,
    sprt_log_lr,
    sprt_separated,
    sprt_threshold,
    wilson_interval,
)
from physint.thresholds import SPRT_LAMBDA

# The Wilson stop reused alpha at every look and measured 184 on seed 0.
# That figure is not a valid early-stop result. The SPRT (lambda 0.6) stops
# the ten controls at 20 episodes: 32 baseline + 32 candidate + 10 * 20 = 264.
WILSON_ROLLOUTS_PER_VALIDATED_FEATURE = 184
SPRT_ROLLOUTS_PER_VALIDATED_FEATURE = 264
# Budget for one validated feature. The fixed design without stopping is 12*N
# and the feature-arm budget 11*N is 352, so 320 is still under that.
STATED_MAX_ROLLOUTS_PER_VALIDATED_FEATURE = 320
DONE_EPISODES = 32
DONE_CONTROLS = 10
NULL_FEATURE_ARM_TESTS = 200
NULL_ALPHA = 0.05

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


def test_sprt_crosses_only_after_enough_one_sided_pairs():
    alpha = 0.05
    threshold = sprt_threshold(alpha)
    assert threshold == pytest.approx(math.log(2.0 / alpha))
    control = [True] * 8
    candidate = [False] * 8
    assert sprt_log_lr(control[:7], candidate[:7], SPRT_LAMBDA) < threshold
    assert sprt_log_lr(control, candidate, SPRT_LAMBDA) >= threshold
    assert sprt_separated(control, [candidate], alpha, SPRT_LAMBDA) is True
    assert sprt_separated([True] * 20, [[True] * 20], alpha, SPRT_LAMBDA) is False
    mixed_control = [True, False] * 10
    mixed_candidate = [False, True] * 10
    assert sprt_separated(mixed_control, [mixed_candidate], alpha, SPRT_LAMBDA) is False
    assert sprt_separated([], [[]], alpha, SPRT_LAMBDA) is False


def test_sprt_type_i_error_stays_at_alpha():
    """Exchangeable pairs, looked at on the same grid the runner uses.

    The Wilson upper bound on the stop rate has to sit at or below alpha.
    """
    rng = np.random.default_rng(0)
    alpha = 0.05
    n = DONE_EPISODES
    ends = [8, 12, 16, 20, 24, 28]
    reps = 2000
    hits = 0
    for _ in range(reps):
        control = (rng.random(n) < 0.5).tolist()
        candidate = (rng.random(n) < 0.5).tolist()
        for end in ends:
            if sprt_separated(control[:end], [candidate[:end]], alpha, SPRT_LAMBDA):
                hits += 1
                break
    _low, high = wilson_interval(hits, reps, alpha)
    assert high <= alpha, f"sprt stops {hits}/{reps}, wilson upper {high:.4f}"


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
    assert all(feature["stopped_early"] is True for feature in controls)
    assert all(feature["reason"] == "sprt_separated" for feature in controls)
    assert all(feature["episodes_run"] == 20 for feature in controls)
    assert all(feature["delta"] == 0.0 for feature in controls)
    assert all(feature["control_percentile"] == 50.0 for feature in controls)

    control_conditions = [item for item in summary["conditions"] if item["role"] == "control"]
    assert len(control_conditions) == DONE_CONTROLS
    for condition in control_conditions:
        assert condition["reason"] == "sprt_separated"
        assert condition["sprt_log_lr"] >= condition["sprt_threshold"]
        assert condition["sprt_threshold"] == pytest.approx(math.log(2.0 / 0.05))
    assert summary["gap_bound"] == "sprt_discordant"
    assert summary["sprt_lambda"] == SPRT_LAMBDA

    total = summary["total_episodes"]
    naive = (1 + DONE_CONTROLS) * DONE_EPISODES
    # 32 baseline + 32 candidate + 10 controls x 20 episodes. 11*N is 352.
    assert total == SPRT_ROLLOUTS_PER_VALIDATED_FEATURE, (
        f"got {total}; the Wilson stop on this seed measured {WILSON_ROLLOUTS_PER_VALIDATED_FEATURE}"
    )
    assert total == summary["fixed_budget_episodes"] - summary["episodes_saved"]
    assert total < naive
    assert total <= STATED_MAX_ROLLOUTS_PER_VALIDATED_FEATURE
    assert summary["validated_feature_ids"] == [PLANTED_FEATURE_ID]
    assert summary["rollouts_per_validated_feature"] == pytest.approx(total)
    assert summary["rollouts_per_validated_feature"] <= STATED_MAX_ROLLOUTS_PER_VALIDATED_FEATURE
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


def _null_feature_arm(root: Path, index: int) -> bool:
    """One candidate-arm test with the planted effect off. Returns whether it was flagged."""
    summary = execute_rollouts(
        root / f"null-{index}",
        policy="fake-null",
        features=[0],
        episodes=DONE_EPISODES,
        controls=DONE_CONTROLS,
        workers=1,
        seed=index * DONE_EPISODES,
        alpha=NULL_ALPHA,
        min_episodes=8,
        batch_size=4,
    )
    candidates = [feature for feature in summary["features"] if feature["role"] == "candidate"]
    if len(candidates) != 1 or summary["policy"] != "fake-null":
        raise RuntimeError(f"null arm {index} did not test one fake-null candidate")
    return bool(candidates[0]["causal"])


def test_null_false_flag_rate_stays_within_alpha(tmp_path: Path):
    """Plant nothing. The Wilson upper bound on the causal-flag rate is at most alpha.

    Two hundred candidate arms, each with the same N, K, and alpha as the
    planted done-check. Episode seeds do not overlap across arms.
    """
    flags = [_null_feature_arm(tmp_path, index) for index in range(NULL_FEATURE_ARM_TESTS)]
    n_tests = len(flags)
    n_flag = sum(bool(item) for item in flags)
    assert n_tests >= 200
    low, high = wilson_interval(n_flag, n_tests, NULL_ALPHA)
    rate = n_flag / n_tests
    message = (
        f"false-flag rate {n_flag}/{n_tests}={rate:.4f} "
        f"wilson95=[{low:.4f},{high:.4f}] alpha={NULL_ALPHA}"
    )
    print(message)
    assert high <= NULL_ALPHA, message
