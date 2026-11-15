"""Random-weights SAE control (AGENTS.md research rule 4) on the fake reacher.

The planted feature must beat every untrained dictionary sent through the same
picker. The null policy must produce no finding. A candidate tied with a
random-SAE arm is not a finding. Old episode files without the ``dictionary``
column still read and validate.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from typer.testing import CliRunner

from interpretableintelligence.artifacts import ROLLOUT_COLUMNS_V1_0, read_rollout_episodes
from interpretableintelligence.cli import app
from interpretableintelligence.policies import get_policy
from interpretableintelligence.policies.base import (
    Intervention,
    dictionary_seed,
    edit_through_dictionary,
    random_dictionary,
)
from interpretableintelligence.policies.fake import PLANTED_FEATURE_ID, TASK_ID
from interpretableintelligence.rollout import RolloutConfig, _spec, arms_for, execute_rollouts, summarize
from interpretableintelligence.rollout_pick import discovery_seeds, pick_features, random_sae_seeds
from interpretableintelligence.sae import SAEConfig, train_sae, untrained_sae
from interpretableintelligence.validate import validate_run

runner = CliRunner()
EPISODES = 32
RANDOM_SAE = 5


def _rollouts(out: Path, policy: str = "fake", *extra: str):
    return runner.invoke(
        app,
        [
            "rollouts", str(out), "--policy", policy, "--episodes", str(EPISODES), "--pick", "1",
            "--random-sae", str(RANDOM_SAE), "--controls", "4", "--seed", "0", *extra,
        ],
    )


def test_untrained_sae_is_exactly_where_training_starts():
    torch.manual_seed(123)
    rows = torch.randn(64, 6)
    config = SAEConfig(d_model=6, n_features=10, steps=0, seed=4)
    started = train_sae(rows, config, torch.device("cpu"))
    fresh = untrained_sae(config, rows.mean(dim=0), torch.device("cpu"))
    for name, tensor in started.state_dict().items():
        assert torch.equal(tensor, fresh.state_dict()[name]), name
    other = untrained_sae(SAEConfig(d_model=6, n_features=10, seed=5), rows.mean(dim=0), torch.device("cpu"))
    assert not torch.equal(other.W_dec, fresh.W_dec)


def test_dictionary_edit_keeps_the_sae_error():
    sae = untrained_sae(SAEConfig(d_model=8, n_features=8, seed=1), torch.zeros(8), torch.device("cpu"))
    x = np.random.default_rng(0).normal(size=8)
    identity = Intervention((2,), 0, "scale", 1.0, random_dictionary(1))
    assert np.array_equal(edit_through_dictionary(x, sae, identity), x)
    z = sae.encode(torch.as_tensor(x, dtype=torch.float32)).detach().double().numpy()
    feature = int(np.argmax(z))
    ablated = edit_through_dictionary(x, sae, Intervention((feature,), 0, "ablate", 0.0, random_dictionary(1)))
    expected = x - z[feature] * sae.W_dec.detach().double().numpy()[:, feature]
    assert np.allclose(ablated, expected)


def test_dictionary_names_are_validated():
    assert dictionary_seed("trained") is None
    assert dictionary_seed(random_dictionary(7)) == 7
    for bad in ("random:", "random:-1", "random:x", "untrained", ""):
        with pytest.raises(ValueError):
            Intervention((0,), 0, "ablate", 0.0, bad)
    with pytest.raises(ValueError):
        random_dictionary(-1)


def test_picker_finds_the_planted_feature_and_is_deterministic():
    env = get_policy("fake")
    args = {"kind": "ablate", "value": 0.0, "layer": 0, "task_id": TASK_ID, "seed": 0}
    assert pick_features(env, "trained", 1, **args) == [PLANTED_FEATURE_ID]
    seeds = random_sae_seeds(0, 3)
    assert random_sae_seeds(0, 5)[:3] == seeds  # more dictionaries extend the same list
    assert len(set(seeds)) == 3
    first = [pick_features(env, random_dictionary(seed), 2, **args) for seed in seeds]
    again = [pick_features(get_policy("fake"), random_dictionary(seed), 2, **args) for seed in seeds]
    assert first == again
    # Discovery never reuses a rollout seed.
    assert min(discovery_seeds(0)) > 10_000


def test_planted_feature_beats_every_random_sae_and_is_a_finding(demo_run: Path, tmp_path: Path):
    out = tmp_path / "demo"
    shutil.copytree(demo_run, out)
    result = _rollouts(out)
    assert result.exit_code == 0, result.output
    summary = json.loads((out / "rollouts" / "summary.json").read_text())
    assert summary["candidate_source"] == "pick"
    assert summary["picker"] == "action_divergence"
    assert summary["candidate_feature_ids"] == [PLANTED_FEATURE_ID]
    assert summary["random_sae_seeds"] == random_sae_seeds(0, RANDOM_SAE)
    assert [len(picks) for picks in summary["random_sae_feature_ids"]] == [1] * RANDOM_SAE

    candidate = next(item for item in summary["features"] if item["role"] == "candidate")
    randoms = [item for item in summary["features"] if item["role"] == "random_sae"]
    assert len(randoms) == RANDOM_SAE
    assert {item["dictionary"] for item in randoms} == {random_dictionary(seed) for seed in summary["random_sae_seeds"]}
    assert candidate["causal"] is True
    assert candidate["n_random_sae"] == RANDOM_SAE
    assert candidate["random_sae_percentile"] == 100.0
    assert candidate["beats_random_sae"] is True
    assert candidate["finding"] is True
    assert all(item["delta"] < candidate["delta"] for item in randoms)
    assert all(item["finding"] is False and item["beats_random_sae"] is None for item in randoms)
    assert summary["finding_feature_ids"] == [PLANTED_FEATURE_ID]
    assert "finding=true" in result.output

    rows = read_rollout_episodes(out / "rollouts" / "episodes.parquet")
    assert {row["dictionary"] for row in rows} == {None, "trained", *(item["dictionary"] for item in randoms)}
    assert validate_run(out) == []
    rerun = _rollouts(out)
    assert rerun.exit_code == 0 and rerun.output == result.output


def test_null_policy_has_no_finding(tmp_path: Path):
    result = _rollouts(tmp_path / "null", "fake-null")
    assert result.exit_code == 0, result.output
    summary = json.loads((tmp_path / "null" / "rollouts" / "summary.json").read_text())
    assert summary["finding_feature_ids"] == []
    assert all(item["finding"] is False for item in summary["features"])


def test_hand_picked_features_get_no_random_sae_comparison(tmp_path: Path):
    refused = runner.invoke(
        app, ["rollouts", str(tmp_path / "x"), "--policy", "fake", "--features", "3", "--random-sae", "2"]
    )
    assert refused.exit_code != 0
    assert "--pick" in refused.output
    both = runner.invoke(app, ["rollouts", str(tmp_path / "x"), "--policy", "fake", "--features", "3", "--pick", "1"])
    assert both.exit_code != 0
    plain = runner.invoke(
        app, ["rollouts", str(tmp_path / "y"), "--policy", "fake", "--features", "3", "--episodes", "8", "--controls", "2"]
    )
    assert plain.exit_code == 0, plain.output
    summary = json.loads((tmp_path / "y" / "rollouts" / "summary.json").read_text())
    candidate = next(item for item in summary["features"] if item["role"] == "candidate")
    assert candidate["causal"] in (True, False)
    assert candidate["random_sae_percentile"] is None and candidate["beats_random_sae"] is None
    assert candidate["finding"] is False
    assert summary["finding_feature_ids"] == []
    assert "no finding is possible without the random-weights SAE control" in plain.output


def _synthetic_rows(config: RolloutConfig, success: dict[str, list[bool]]) -> list[dict]:
    rows = []
    for arm in arms_for(config):
        for offset in range(config.episodes_per_arm):
            spec = _spec(config, arm, offset, max_steps=1)
            rows.append(
                {
                    "episode_id": spec["episode_id"], "task_id": spec["task_id"], "seed": spec["seed"],
                    "condition": spec["condition"], "feature_ids": spec["feature_ids"], "layer": spec["layer"],
                    "kind": spec["kind"], "value": spec["value"], "success": success[arm.name][offset],
                    "steps": 1, "wall_time_s": 0.0, "dictionary": spec["dictionary"],
                }
            )
    return rows


def test_a_candidate_tied_with_a_random_sae_arm_is_not_a_finding():
    n = 40
    config = RolloutConfig(
        policy="fake", task_id=TASK_ID, layer=0, kind="ablate", value=0.0, seed=0, episodes_per_arm=n,
        batch_size=4, min_episodes=n, alpha=0.05, workers=1, candidate_feature_ids=(3,), control_feature_ids=(5, 6),
        candidate_source="pick", picker="action_divergence", pick_discovery_episodes=16,
        random_sae_seeds=(11, 12), random_sae_feature_ids=((4,), (9,)),
    )
    names = [arm.name for arm in arms_for(config)]
    base = [True] * n
    hurt = [index % 2 == 0 for index in range(n)]  # delta 0.5, McNemar p tiny
    success = {name: list(base) for name in names}
    success["candidate:3"] = hurt
    success["random_sae:random:12:9"] = list(hurt)  # one untrained dictionary does exactly as well
    summary = summarize(_synthetic_rows(config, success), config)
    candidate = next(item for item in summary["features"] if item["role"] == "candidate")
    assert candidate["causal"] is True  # rules 1-3 alone would call it
    assert candidate["random_sae_percentile"] == 75.0
    assert candidate["beats_random_sae"] is False
    assert candidate["finding"] is False
    assert summary["validated_feature_ids"] == [3] and summary["finding_feature_ids"] == []
    # The same random arm, a little weaker: now the candidate beats every one.
    success["random_sae:random:12:9"] = [index % 4 != 0 for index in range(n)]
    beaten = summarize(_synthetic_rows(config, success), config)
    assert beaten["finding_feature_ids"] == [3]


def test_episode_files_from_before_the_dictionary_column_still_read(tmp_path: Path):
    out = tmp_path / "old"
    summary = execute_rollouts(out, policy="fake", episodes=8, features=[3], controls=2, min_episodes=8)
    path = out / "rollouts" / "episodes.parquet"
    rows = read_rollout_episodes(path)
    old = pq.read_table(path).select(list(ROLLOUT_COLUMNS_V1_0))
    pq.write_table(old, path)
    assert tuple(pq.read_table(path).column_names) == ROLLOUT_COLUMNS_V1_0
    assert read_rollout_episodes(path) == rows
    again = execute_rollouts(out, policy="fake", episodes=8, features=[3], controls=2, min_episodes=8)
    assert again == summary
    assert pq.read_table(path).column_names[-1] == "dictionary"


def test_random_sae_needs_a_policy_that_can_edit_one(tmp_path: Path, monkeypatch):
    import interpretableintelligence.policies.fake as fake

    monkeypatch.setattr(fake.FakePolicyEnv, "random_dictionaries", False)
    with pytest.raises(ValueError, match="cannot edit an untrained SAE"):
        execute_rollouts(tmp_path / "x", policy="fake", episodes=8, pick=1, random_sae=1, controls=0)
