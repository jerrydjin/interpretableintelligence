"""Baseline vs intervention rollouts, with early stopping on controls.

Episodes are paired on seed: baseline, each candidate feature, each
matched random-feature control, and each random-weights SAE arm see the same
tasks. After each batch, a
control stops only for efficacy: a one-sided SPRT on discordant pairs
(control succeeds, candidate fails) crosses the family-wise boundary.
A control that never crosses runs to the episode cap. Baseline and
candidates always run the full ``--episodes`` budget. Reruns skip episode
ids already in ``rollouts/episodes.parquet``.

Random-weights SAE control (AGENTS.md research rule 4): with ``--pick N`` the
candidates are the top ``N`` of the policy's picker on the trained SAE, and
``--random-sae K`` sends ``K`` untrained SAEs of the same shape through the same
picker. Each of their picks becomes an arm with the same edit. These arms join
the controls' early-stopping family. A candidate is a ``finding`` only when it is
``causal`` and its delta is above every random-SAE arm's.

``direction`` says which way an effect counts. ``fall`` (the default): the edit
should lower success, and ``delta = baseline - intervention``. ``rise`` (GR00T,
whose candidates fire before failure, so removing them should help):
``delta = intervention - baseline``. Every gate and the early-stopping test read
the oriented outcome, so the same thresholds apply both ways.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from interpretableintelligence.artifacts import read_json, read_rollout_episodes, write_json, write_rollout_episodes
from interpretableintelligence.policies import get_policy
from interpretableintelligence.policies.base import TRAINED_DICTIONARY, Intervention, dictionary_seed, random_dictionary
from interpretableintelligence.rollout_stats import (
    binding_sprt,
    control_percentile,
    discordant_counts,
    feature_is_causal,
    mcnemar_exact_pvalue,
    paired_success_delta_ci,
    sprt_separated,
)
from interpretableintelligence.thresholds import (
    DEFAULT_ROLLOUT_ALPHA,
    DEFAULT_ROLLOUT_BATCH,
    DEFAULT_ROLLOUT_CONTROLS,
    DEFAULT_ROLLOUT_EPISODES,
    DEFAULT_ROLLOUT_MIN_EPISODES,
    DEFAULT_RANDOM_SAE,
    DEFAULT_ROLLOUT_WORKERS,
    PICK_DISCOVERY_EPISODES,
    SPRT_LAMBDA,
)

# Arms that are compared against, not tested: they share the early-stopping family.
CONTROL_ROLES = ("control", "random_sae")
CANDIDATE_SOURCES = ("features", "pick")
DIRECTIONS = ("fall", "rise")


def oriented(flags, direction: str) -> list[bool]:
    """Outcomes the effect is measured on: success for ``fall``, failure for ``rise``.

    ``paired_success_delta_ci(oriented(base), oriented(edit))`` is ``base - edit`` for ``fall`` and
    ``edit - base`` for ``rise``, and the discordant-pair SPRT's efficacy pair (control "succeeds",
    candidate does not) becomes control fails, candidate succeeds.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    return [bool(item) for item in flags] if direction == "fall" else [not bool(item) for item in flags]

SUMMARY_KEYS = {
    "schema_version",
    "policy",
    "task_id",
    "layer",
    "kind",
    "value",
    "seed",
    "episodes_per_arm",
    "batch_size",
    "min_episodes",
    "alpha",
    "workers",
    "gap_bound",
    "sprt_lambda",
    "test",
    "ci_method",
    "candidate_feature_ids",
    "control_feature_ids",
    "candidate_source",
    "direction",
    "picker",
    "pick_discovery_episodes",
    "random_sae_seeds",
    "random_sae_feature_ids",
    "rise_sprt",
    "fixed_budget_episodes",
    "total_episodes",
    "episodes_saved",
    "validated_feature_ids",
    "finding_feature_ids",
    "rollouts_per_validated_feature",
    "conditions",
    "features",
}
RISE_SPRT_KEYS = {"p0", "p1_rise", "p1_fall", "alpha", "beta", "cap", "stop_rule"}
CONDITION_KEYS = {
    "role",
    "condition",
    "dictionary",
    "feature_ids",
    "layer",
    "kind",
    "value",
    "episodes_run",
    "episodes_budget",
    "stopped_early",
    "reason",
    "success_rate",
    "gap_vs_candidate",
    "sprt_log_lr",
    "sprt_threshold",
}
FEATURE_KEYS = {
    "feature_id",
    "role",
    "dictionary",
    "baseline_success",
    "intervention_success",
    "delta",
    "ci_low",
    "ci_high",
    "p_value",
    "n_pairs",
    "n_baseline_only",
    "n_intervention_only",
    "control_percentile",
    "n_controls",
    "causal",
    "random_sae_percentile",
    "n_random_sae",
    "beats_random_sae",
    "sprt_verdict",
    "finding",
    "episodes_run",
    "stopped_early",
    "reason",
}


@dataclass(frozen=True)
class RolloutConfig:
    policy: str
    task_id: str
    layer: int
    kind: str
    value: float
    seed: int
    episodes_per_arm: int
    batch_size: int
    min_episodes: int
    alpha: float
    workers: int
    candidate_feature_ids: tuple[int, ...]
    control_feature_ids: tuple[int, ...]
    # "features": ids passed by hand. "pick": the policy's picker on the trained SAE.
    candidate_source: str = "features"
    picker: str | None = None
    pick_discovery_episodes: int | None = None
    # Untrained SAEs, and each one's picks (same picker, as many as there are candidates).
    random_sae_seeds: tuple[int, ...] = ()
    random_sae_feature_ids: tuple[tuple[int, ...], ...] = ()
    # "fall": the edit should lower success (fake policies). "rise": it should raise it (GR00T rise test).
    direction: str = "fall"
    # Research rule 1 on GR00T --pick plans: the Sobel-Wald SPRT (rollout_stats.sequential_decision)
    # read on each arm's outcomes in episode order. {p0, p1_rise, p1_fall, alpha, beta, cap, stop_rule}, or None.
    rise_sprt: dict | None = None

    def as_dict(self) -> dict:
        return {
            "policy": self.policy,
            "task_id": self.task_id,
            "layer": self.layer,
            "kind": self.kind,
            "value": self.value,
            "seed": self.seed,
            "episodes_per_arm": self.episodes_per_arm,
            "batch_size": self.batch_size,
            "min_episodes": self.min_episodes,
            "alpha": self.alpha,
            "workers": self.workers,
            "candidate_feature_ids": list(self.candidate_feature_ids),
            "control_feature_ids": list(self.control_feature_ids),
            "candidate_source": self.candidate_source,
            "direction": self.direction,
            "picker": self.picker,
            "pick_discovery_episodes": self.pick_discovery_episodes,
            "random_sae_seeds": list(self.random_sae_seeds),
            "random_sae_feature_ids": [list(item) for item in self.random_sae_feature_ids],
            "rise_sprt": None if self.rise_sprt is None else dict(self.rise_sprt),
        }


@dataclass(frozen=True)
class Arm:
    role: str
    condition: str
    feature_ids: tuple[int, ...]
    dictionary: str = TRAINED_DICTIONARY

    @property
    def name(self) -> str:
        if self.role == "baseline":
            return "baseline"
        ids = "+".join(str(item) for item in self.feature_ids)
        if self.dictionary != TRAINED_DICTIONARY:
            return f"{self.role}:{self.dictionary}:{ids}"
        return f"{self.role}:{ids}"


def parse_feature_ids(text: str) -> list[int]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise ValueError("--features is required, for example --features 3")
    ids: list[int] = []
    for part in parts:
        if part.startswith("-") or not part.isdigit():
            raise ValueError(f"feature id must be a non-negative integer, got {part!r}")
        ids.append(int(part))
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate feature id in {ids}")
    return ids


def choose_control_features(n_features: int, exclude: set[int], k: int, seed: int) -> list[int]:
    """First ``k`` of a seeded permutation of the features that are not candidates.

    The draw does not depend on ``k``, so a larger control count extends the
    same list.
    """
    if k < 0:
        raise ValueError(f"controls must be >= 0, got {k}")
    pool = [index for index in range(n_features) if index not in exclude]
    if k > len(pool):
        raise ValueError(f"need {k} controls but only {len(pool)} features are left after excluding {sorted(exclude)}")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xC0FFEE]))
    order = rng.permutation(pool)
    return sorted(int(item) for item in order[:k])


def batch_ends(episodes: int, min_episodes: int, batch_size: int) -> list[int]:
    """Seed-prefix lengths at which early stopping is evaluated.

    The first check is at ``min_episodes``. Later checks add ``batch_size``
    until the arm budget is covered.
    """
    if episodes < 1:
        raise ValueError(f"episodes must be >= 1, got {episodes}")
    if min_episodes < 1 or min_episodes > episodes:
        raise ValueError(f"min_episodes must be in [1, episodes], got {min_episodes} for {episodes}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    ends: list[int] = []
    cursor = 0
    first = True
    while cursor < episodes:
        step = min_episodes if first else batch_size
        first = False
        cursor = min(episodes, cursor + step)
        ends.append(cursor)
    return ends


def make_episode_id(
    *,
    policy: str,
    condition: str,
    task_id: str,
    seed: int,
    feature_ids: tuple[int, ...] | list[int],
    layer: int | None,
    kind: str | None,
    value: float | None,
    dictionary: str | None = None,
) -> str:
    """Episode identity. A trained-SAE (or baseline) id is unchanged from before random dictionaries."""
    feats = "-" if not feature_ids else "+".join(str(int(item)) for item in feature_ids)
    layer_s = "-" if layer is None else str(int(layer))
    kind_s = "-" if kind is None else str(kind)
    value_s = "-" if value is None else format(float(value), ".8g")
    episode_id = f"p={policy}|c={condition}|t={task_id}|s={int(seed)}|f={feats}|L={layer_s}|k={kind_s}|v={value_s}"
    if dictionary is not None and dictionary != TRAINED_DICTIONARY:
        episode_id += f"|d={dictionary}"
    return episode_id


def _check_config(config: RolloutConfig) -> None:
    if config.kind not in {"ablate", "scale", "clamp"}:
        raise ValueError(f"kind must be ablate|scale|clamp, got {config.kind!r}")
    if not 0.0 < config.alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {config.alpha}")
    if config.episodes_per_arm < 1:
        raise ValueError("episodes must be >= 1")
    if config.min_episodes < 1 or config.min_episodes > config.episodes_per_arm:
        raise ValueError(
            f"min_episodes must be in [1, episodes], got {config.min_episodes} for {config.episodes_per_arm}"
        )
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if config.workers < 1:
        raise ValueError("workers must be >= 1")
    if isinstance(config.layer, bool) or not isinstance(config.layer, int) or config.layer < 0:
        raise ValueError(f"layer must be a non-negative integer, got {config.layer!r}")
    if not config.candidate_feature_ids:
        raise ValueError("at least one candidate feature is required")
    if len(set(config.candidate_feature_ids)) != len(config.candidate_feature_ids):
        raise ValueError("duplicate candidate feature id")
    if len(set(config.control_feature_ids)) != len(config.control_feature_ids):
        raise ValueError("duplicate control feature id")
    overlap = set(config.candidate_feature_ids) & set(config.control_feature_ids)
    if overlap:
        raise ValueError(f"controls overlap candidates: {sorted(overlap)}")
    if config.direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {config.direction!r}")
    if config.rise_sprt is not None:
        if config.direction != "rise":
            raise ValueError("rise_sprt is the rule 1 rise call: it needs direction rise")
        if set(config.rise_sprt) != RISE_SPRT_KEYS:
            raise ValueError(f"rise_sprt keys must be {sorted(RISE_SPRT_KEYS)}, got {sorted(config.rise_sprt)}")
    if config.candidate_source not in CANDIDATE_SOURCES:
        raise ValueError(f"candidate_source must be one of {CANDIDATE_SOURCES}, got {config.candidate_source!r}")
    if (config.candidate_source == "pick") != (config.picker is not None):
        raise ValueError("picker is recorded exactly when candidates were picked")
    if (config.picker == "action_divergence") != (config.pick_discovery_episodes is not None):
        raise ValueError("pick_discovery_episodes is recorded exactly when the default picker ran")
    if config.random_sae_seeds and config.candidate_source != "pick":
        raise ValueError(
            "the random-weights SAE control compares picker against picker: pass --pick instead of --features"
        )
    if len(set(config.random_sae_seeds)) != len(config.random_sae_seeds):
        raise ValueError("duplicate random-sae seed")
    if len(config.random_sae_feature_ids) != len(config.random_sae_seeds):
        raise ValueError("random_sae_feature_ids needs one pick list per random-sae seed")
    for seed, picks in zip(config.random_sae_seeds, config.random_sae_feature_ids, strict=True):
        random_dictionary(int(seed))  # validates the seed
        if len(picks) != len(config.candidate_feature_ids) or len(set(picks)) != len(picks):
            raise ValueError(
                f"random:{seed} needs {len(config.candidate_feature_ids)} distinct picks, one per candidate, got {list(picks)}"
            )


def arms_for(config: RolloutConfig) -> list[Arm]:
    arms = [Arm("baseline", "baseline", ())]
    for feature_id in config.candidate_feature_ids:
        arms.append(Arm("candidate", "intervention", (int(feature_id),)))
    for feature_id in config.control_feature_ids:
        arms.append(Arm("control", "intervention", (int(feature_id),)))
    for seed, picks in zip(config.random_sae_seeds, config.random_sae_feature_ids, strict=True):
        for feature_id in picks:
            arms.append(Arm("random_sae", "intervention", (int(feature_id),), random_dictionary(int(seed))))
    return arms


def _episode_seed(config: RolloutConfig, offset: int) -> int:
    return int(config.seed) + int(offset)


def _arm_episode_id(config: RolloutConfig, arm: Arm, offset: int) -> str:
    if arm.condition == "baseline":
        return make_episode_id(
            policy=config.policy,
            condition="baseline",
            task_id=config.task_id,
            seed=_episode_seed(config, offset),
            feature_ids=(),
            layer=None,
            kind=None,
            value=None,
        )
    return make_episode_id(
        policy=config.policy,
        condition="intervention",
        task_id=config.task_id,
        seed=_episode_seed(config, offset),
        feature_ids=arm.feature_ids,
        layer=config.layer,
        kind=config.kind,
        value=config.value,
        dictionary=arm.dictionary,
    )


def _spec(config: RolloutConfig, arm: Arm, offset: int, max_steps: int) -> dict:
    seed = _episode_seed(config, offset)
    if arm.condition == "baseline":
        feature_ids: list[int] = []
        layer = None
        kind = None
        value = None
        dictionary = None
    else:
        feature_ids = list(arm.feature_ids)
        layer = config.layer
        kind = config.kind
        value = config.value
        dictionary = arm.dictionary
    return {
        "policy": config.policy,
        "task_id": config.task_id,
        "seed": seed,
        "condition": arm.condition,
        "feature_ids": feature_ids,
        "layer": layer,
        "kind": kind,
        "value": value,
        "dictionary": dictionary,
        "max_steps": max_steps,
        "episode_id": _arm_episode_id(config, arm, offset),
    }


def run_episode_spec(spec: dict) -> dict:
    """Run one episode. Top-level so process workers can import it."""
    import time

    env = get_policy(spec["policy"])
    intervention = None
    if spec["condition"] == "intervention":
        intervention = Intervention(
            feature_ids=tuple(spec["feature_ids"]),
            layer=int(spec["layer"]),
            kind=str(spec["kind"]),
            value=float(spec["value"]),
            dictionary=str(spec["dictionary"]),
        )
    started = time.perf_counter()
    observation = env.reset(task_id=spec["task_id"], seed=int(spec["seed"]))
    steps = 0
    info: dict = {}
    done = False
    with env.intervene(intervention):
        while not done:
            action = env.act(observation)
            observation, _reward, done, info = env.step(action)
            steps += 1
            if steps > int(spec["max_steps"]):
                done = True
                info = {**info, "success": False}
                break
    wall = time.perf_counter() - started
    return {
        "episode_id": spec["episode_id"],
        "task_id": spec["task_id"],
        "seed": int(spec["seed"]),
        "condition": spec["condition"],
        "feature_ids": list(spec["feature_ids"]),
        "layer": spec["layer"],
        "kind": spec["kind"],
        "value": None if spec["value"] is None else float(spec["value"]),
        "success": bool(info.get("success", False)),
        "steps": int(steps),
        "wall_time_s": float(wall),
        "dictionary": spec["dictionary"],
    }


class _Pool:
    def __init__(self, workers: int) -> None:
        self.workers = workers
        self._pool: ProcessPoolExecutor | None = None

    def run(self, jobs: list[dict]) -> list[dict]:
        if not jobs:
            return []
        if self.workers == 1:
            return [run_episode_spec(job) for job in jobs]
        if self._pool is None:
            context = multiprocessing.get_context("spawn")
            self._pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=context)
        chunksize = max(1, len(jobs) // (self.workers * 4))
        return list(self._pool.map(run_episode_spec, jobs, chunksize=chunksize))

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None


def _flags(config: RolloutConfig, arm: Arm, n: int, by_id: dict[str, dict]) -> list[bool]:
    flags: list[bool] = []
    for offset in range(n):
        episode_id = _arm_episode_id(config, arm, offset)
        try:
            row = by_id[episode_id]
        except KeyError as exc:
            raise RuntimeError(f"missing episode {episode_id}") from exc
        flags.append(bool(row["success"]))
    return flags


def declared_stop(
    control_flags: list[bool],
    candidate_flags: list[list[bool]],
    *,
    episodes: int,
    min_episodes: int,
    batch_size: int,
    alpha: float,
    n_controls: int,
) -> tuple[int, str]:
    """How many control episodes the schedule should have run, and why."""
    for end in batch_ends(episodes, min_episodes, batch_size):
        if end > len(control_flags):
            break
        if end < episodes and sprt_separated(
            control_flags[:end],
            [flags[:end] for flags in candidate_flags],
            alpha,
            SPRT_LAMBDA,
            n_controls=n_controls,
        ):
            return end, "sprt_separated"
    return episodes, "fixed_budget"


def _index_rows(rows: list[dict]) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for row in rows:
        episode_id = row["episode_id"]
        if episode_id in by_id:
            raise ValueError(f"duplicate episode_id {episode_id}")
        by_id[episode_id] = row
    return by_id


def _rows_for_arm(config: RolloutConfig, arm: Arm, n: int, by_id: dict[str, dict]) -> list[dict]:
    selected = []
    for offset in range(n):
        episode_id = _arm_episode_id(config, arm, offset)
        try:
            selected.append(by_id[episode_id])
        except KeyError as exc:
            raise ValueError(f"rollouts are missing {episode_id}") from exc
    seeds = [int(row["seed"]) for row in selected]
    expected = [_episode_seed(config, offset) for offset in range(n)]
    if seeds != expected:
        raise ValueError(f"{arm.name} seeds are not the paired prefix {expected[0]}..{expected[-1]}")
    return selected


def summarize(rows: list[dict], config: RolloutConfig) -> dict:
    """Build the rollout summary from episode rows. Raises if the schedule is wrong."""
    _check_config(config)
    by_id = _index_rows(rows)
    arms = arms_for(config)
    baseline = arms[0]
    candidates = [arm for arm in arms if arm.role == "candidate"]
    controls = [arm for arm in arms if arm.role in CONTROL_ROLES]
    n = config.episodes_per_arm

    baseline_rows = _rows_for_arm(config, baseline, n, by_id)
    baseline_flags = [bool(row["success"]) for row in baseline_rows]
    candidate_flags = []
    for arm in candidates:
        arm_rows = _rows_for_arm(config, arm, n, by_id)
        candidate_flags.append([bool(row["success"]) for row in arm_rows])

    control_plan: list[tuple[Arm, int, str, list[bool]]] = []
    for arm in controls:
        # The prefix that should exist is determined by the successes that were run.
        # Recover it by testing every batch end that is fully present.
        present = 0
        while present < n and _arm_episode_id(config, arm, present) in by_id:
            present += 1
        if present == 0:
            raise ValueError(f"{arm.name} has no episodes")
        present_flags = _flags(config, arm, present, by_id)
        length, reason = declared_stop(
            oriented(present_flags, config.direction),
            [oriented(flags, config.direction) for flags in candidate_flags],
            episodes=n,
            min_episodes=config.min_episodes,
            batch_size=config.batch_size,
            alpha=config.alpha,
            n_controls=len(controls),
        )
        if length != present:
            raise ValueError(
                f"{arm.name} has {present} episodes but the early-stop schedule requires {length} ({reason})"
            )
        control_plan.append((arm, length, reason, present_flags))

    expected_ids = {_arm_episode_id(config, baseline, offset) for offset in range(n)}
    for arm in candidates:
        for offset in range(n):
            expected_ids.add(_arm_episode_id(config, arm, offset))
    for arm, length, _reason, _flags_ in control_plan:
        for offset in range(length):
            expected_ids.add(_arm_episode_id(config, arm, offset))
    extra = set(by_id) - expected_ids
    missing = expected_ids - set(by_id)
    if extra or missing:
        raise ValueError(f"episode set does not match the schedule, missing={len(missing)} extra={len(extra)}")

    def score_arm(arm: Arm, flags: list[bool], episodes_run: int, reason: str) -> dict:
        paired_base = baseline_flags[: len(flags)]
        delta, ci_low, ci_high = paired_success_delta_ci(
            oriented(paired_base, config.direction), oriented(flags, config.direction), config.alpha
        )
        n10, n01 = discordant_counts(paired_base, flags)
        return {
            "feature_id": arm.feature_ids[0],
            "role": arm.role,
            "dictionary": arm.dictionary,
            "baseline_success": float(sum(paired_base) / len(paired_base)),
            "intervention_success": float(sum(flags) / len(flags)),
            "delta": delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": mcnemar_exact_pvalue(n10, n01),
            "n_pairs": len(flags),
            "n_baseline_only": n10,
            "n_intervention_only": n01,
            "episodes_run": episodes_run,
            "stopped_early": bool(episodes_run < n),
            "reason": reason,
        }

    raw_features: list[dict] = []
    for arm, flags in zip(candidates, candidate_flags, strict=True):
        raw_features.append(score_arm(arm, flags, n, "fixed_budget"))
    for arm, length, reason, flags in control_plan:
        raw_features.append(score_arm(arm, flags, length, reason))
    # Matched random features of the trained SAE, and the random-weights SAE arms.
    control_deltas = [float(item["delta"]) for item in raw_features if item["role"] == "control"]
    random_deltas = [float(item["delta"]) for item in raw_features if item["role"] == "random_sae"]

    features: list[dict] = []
    control_index = 0
    for scored in raw_features:
        if scored["role"] == "control":
            # Compared with the other controls: drop this control's own delta.
            others = [delta for item, delta in enumerate(control_deltas) if item != control_index]
            control_index += 1
        else:
            others = control_deltas
        percentile = control_percentile(float(scored["delta"]), others)
        causal = feature_is_causal(
            delta=float(scored["delta"]),
            p_value=float(scored["p_value"]),
            control_percentile=percentile,
            alpha=config.alpha,
        )
        random_percentile = None
        beats_random = None
        if scored["role"] == "candidate" and random_deltas:
            random_percentile = control_percentile(float(scored["delta"]), random_deltas)
            # Rule 4: above every random-SAE arm, not merely most of them. Ties do not beat.
            beats_random = bool(random_percentile is not None and random_percentile >= 100.0)
        features.append(
            {
                **scored,
                "control_percentile": percentile,
                "n_controls": len(others),
                "causal": causal,
                "random_sae_percentile": random_percentile,
                "n_random_sae": len(random_deltas) if scored["role"] == "candidate" else 0,
                "beats_random_sae": beats_random,
                "sprt_verdict": None,
                "finding": bool(scored["role"] == "candidate" and causal and beats_random is True),
            }
        )

    def condition_record(arm: Arm, episodes_run: int, reason: str, flags: list[bool]) -> dict:
        gap = None
        log_lr = None
        threshold = None
        if arm.role in CONTROL_ROLES:
            gap, log_lr, threshold = binding_sprt(
                oriented(flags, config.direction),
                [oriented(item, config.direction) for item in candidate_flags],
                config.alpha,
                SPRT_LAMBDA,
                n_controls=len(controls),
            )
        layer = None if arm.role == "baseline" else config.layer
        kind = None if arm.role == "baseline" else config.kind
        value = None if arm.role == "baseline" else config.value
        return {
            "role": arm.role,
            "condition": arm.condition,
            "dictionary": None if arm.role == "baseline" else arm.dictionary,
            "feature_ids": list(arm.feature_ids),
            "layer": layer,
            "kind": kind,
            "value": value,
            "episodes_run": episodes_run,
            "episodes_budget": n,
            "stopped_early": bool(episodes_run < n and reason == "sprt_separated"),
            "reason": reason,
            "success_rate": float(sum(flags) / len(flags)),
            "gap_vs_candidate": gap,
            "sprt_log_lr": log_lr,
            "sprt_threshold": threshold,
        }

    conditions = [
        condition_record(baseline, n, "fixed_budget", baseline_flags),
    ]
    for arm, flags in zip(candidates, candidate_flags, strict=True):
        conditions.append(condition_record(arm, n, "fixed_budget", flags))
    for arm, length, reason, flags in control_plan:
        conditions.append(condition_record(arm, length, reason, flags))

    n_arms = 1 + len(candidates) + len(controls)
    fixed_budget = n_arms * n
    total = len(rows)
    validated = [int(feature["feature_id"]) for feature in features if feature["causal"] and feature["role"] != "random_sae"]
    findings = [int(feature["feature_id"]) for feature in features if feature["finding"]]
    per = None if not validated else float(total) / float(len(validated))
    summary = {
        "schema_version": 1,
        **config.as_dict(),
        "gap_bound": "sprt_discordant",
        "sprt_lambda": SPRT_LAMBDA,
        "test": "mcnemar_exact",
        "ci_method": "paired_wald",
        "fixed_budget_episodes": fixed_budget,
        "total_episodes": total,
        "episodes_saved": fixed_budget - total,
        "validated_feature_ids": validated,
        "finding_feature_ids": findings,
        "rollouts_per_validated_feature": per,
        "conditions": conditions,
        "features": features,
    }
    if set(summary) != SUMMARY_KEYS:
        raise RuntimeError(f"summary keys drifted: {set(summary) ^ SUMMARY_KEYS}")
    return summary


def format_report(summary: dict) -> str:
    episodes = int(summary["episodes_per_arm"])
    n_random = sum(len(picks) for picks in summary["random_sae_feature_ids"])
    n_feature_arms = len(summary["candidate_feature_ids"]) + len(summary["control_feature_ids"]) + n_random
    lines = [
        (
            f"policy={summary['policy']} task={summary['task_id']} kind={summary['kind']} "
            f"layer={summary['layer']} episodes={episodes} controls={len(summary['control_feature_ids'])} "
            f"alpha={summary['alpha']} min_episodes={summary['min_episodes']} "
            f"sprt_lambda={summary['sprt_lambda']} candidates={summary['candidate_source']} "
            f"random_sae={len(summary['random_sae_seeds'])}"
        )
    ]
    for feature in summary["features"]:
        causal = "true" if feature["causal"] else "false"
        stopped = "true" if feature["stopped_early"] else "false"
        percentile = feature["control_percentile"]
        percentile_s = "none" if percentile is None else f"{float(percentile):.1f}"
        role = feature["role"] if feature["role"] != "random_sae" else f"random_sae[{feature['dictionary']}]"
        extra = ""
        if feature["role"] == "candidate":
            random_pct = feature["random_sae_percentile"]
            random_s = "none" if random_pct is None else f"{float(random_pct):.1f}"
            extra = f" random_sae_pct={random_s} finding={'true' if feature['finding'] else 'false'}"
        lines.append(
            f"feature {feature['feature_id']} {role} causal={causal}{extra} "
            f"delta={float(feature['delta']):.3f} p={float(feature['p_value']):.4g} "
            f"ci=[{float(feature['ci_low']):.3f},{float(feature['ci_high']):.3f}] "
            f"pct={percentile_s} baseline={float(feature['baseline_success']):.3f} "
            f"intervention={float(feature['intervention_success']):.3f} "
            f"episodes={feature['episodes_run']} stopped_early={stopped} reason={feature['reason']}"
        )
    per = summary["rollouts_per_validated_feature"]
    per_s = "none" if per is None else f"{float(per):.1f}"
    lines.append(
        f"total_episodes={summary['total_episodes']} fixed_budget={summary['fixed_budget_episodes']} "
        f"episodes_saved={summary['episodes_saved']} feature_arms_x_N={n_feature_arms * episodes} "
        f"validated={summary['validated_feature_ids']} findings={summary['finding_feature_ids']} "
        f"rollouts_per_validated_feature={per_s}"
    )
    if not summary["random_sae_seeds"]:
        lines.append("no finding is possible without the random-weights SAE control: rerun with --pick N --random-sae K")
    return "\n".join(lines)


def config_from_summary(summary: dict) -> RolloutConfig:
    return RolloutConfig(
        policy=str(summary["policy"]),
        task_id=str(summary["task_id"]),
        layer=int(summary["layer"]),
        kind=str(summary["kind"]),
        value=float(summary["value"]),
        seed=int(summary["seed"]),
        episodes_per_arm=int(summary["episodes_per_arm"]),
        batch_size=int(summary["batch_size"]),
        min_episodes=int(summary["min_episodes"]),
        alpha=float(summary["alpha"]),
        workers=int(summary["workers"]),
        candidate_feature_ids=tuple(int(item) for item in summary["candidate_feature_ids"]),
        control_feature_ids=tuple(int(item) for item in summary["control_feature_ids"]),
        candidate_source=str(summary["candidate_source"]),
        picker=None if summary["picker"] is None else str(summary["picker"]),
        pick_discovery_episodes=None if summary["pick_discovery_episodes"] is None else int(summary["pick_discovery_episodes"]),
        random_sae_seeds=tuple(int(item) for item in summary["random_sae_seeds"]),
        random_sae_feature_ids=tuple(tuple(int(item) for item in picks) for picks in summary["random_sae_feature_ids"]),
        direction=str(summary["direction"]),
        rise_sprt=None if summary["rise_sprt"] is None else dict(summary["rise_sprt"]),
    )


def execute_rollouts(
    run_dir: Path,
    *,
    policy: str = "fake",
    episodes: int = DEFAULT_ROLLOUT_EPISODES,
    features: list[int] | None = None,
    controls: int = DEFAULT_ROLLOUT_CONTROLS,
    workers: int = DEFAULT_ROLLOUT_WORKERS,
    seed: int = 0,
    kind: str = "ablate",
    value: float = 0.0,
    layer: int | None = None,
    task_id: str = "reach",
    alpha: float = DEFAULT_ROLLOUT_ALPHA,
    min_episodes: int = DEFAULT_ROLLOUT_MIN_EPISODES,
    batch_size: int = DEFAULT_ROLLOUT_BATCH,
    pick: int | None = None,
    random_sae: int | None = None,
    pick_episodes: int = PICK_DISCOVERY_EPISODES,
) -> dict:
    """Run the paired design into ``run_dir/rollouts`` and return the summary.

    Pass ``features`` (ids chosen by hand) or ``pick`` (the top ``pick`` of the
    policy's picker on the trained SAE), not both. ``random_sae`` untrained SAEs
    go through the same picker and add one arm per pick; it needs ``pick``.
    ``None`` means ``DEFAULT_RANDOM_SAE`` with ``pick`` and 0 without.
    """
    from interpretableintelligence.rollout_pick import pick_features, picker_name, random_sae_seeds

    if features and pick is not None:
        raise ValueError("pass --features or --pick, not both")
    if not features and pick is None:
        raise ValueError("at least one feature id is required: pass --features or --pick")
    if random_sae is None:
        random_sae = DEFAULT_RANDOM_SAE if pick is not None else 0
    if random_sae < 0:
        raise ValueError(f"--random-sae must be >= 0, got {random_sae}")
    if random_sae and pick is None:
        raise ValueError(
            "the random-weights SAE control compares picker against picker: pass --pick instead of --features"
        )
    env = get_policy(policy)
    if layer is None:
        layer = int(env.layer)
    elif int(layer) != int(env.layer):
        raise ValueError(f"policy {policy!r} is instrumented at layer {env.layer}, got --layer {layer}")
    if random_sae and not getattr(env, "random_dictionaries", False):
        raise ValueError(f"policy {policy!r} cannot edit an untrained SAE, so --random-sae is unavailable for it")
    if pick_episodes < 1:
        raise ValueError(f"--pick-episodes must be >= 1, got {pick_episodes}")
    pick_args = {"kind": kind, "value": float(value), "layer": int(layer), "task_id": task_id, "seed": int(seed), "episodes": int(pick_episodes)}
    if pick is not None:
        features = pick_features(env, TRAINED_DICTIONARY, int(pick), **pick_args)
    assert features is not None
    random_seeds = random_sae_seeds(int(seed), int(random_sae))
    random_picks = [tuple(pick_features(env, random_dictionary(item), len(features), **pick_args)) for item in random_seeds]
    for feature_id in features:
        if feature_id < 0 or feature_id >= int(env.n_features):
            raise ValueError(f"feature id {feature_id} is outside 0..{int(env.n_features) - 1} for policy {policy!r}")
    control_ids = choose_control_features(int(env.n_features), set(features), int(controls), int(seed))
    config = RolloutConfig(
        policy=policy,
        task_id=task_id,
        layer=int(layer),
        kind=kind,
        value=float(value),
        seed=int(seed),
        episodes_per_arm=int(episodes),
        batch_size=int(batch_size),
        min_episodes=int(min_episodes),
        alpha=float(alpha),
        workers=int(workers),
        candidate_feature_ids=tuple(int(item) for item in features),
        control_feature_ids=tuple(control_ids),
        candidate_source="features" if pick is None else "pick",
        picker=None if pick is None else picker_name(env),
        pick_discovery_episodes=None if pick is None or picker_name(env) == "policy" else int(pick_episodes),
        random_sae_seeds=tuple(random_seeds),
        random_sae_feature_ids=tuple(random_picks),
    )
    _check_config(config)

    run_dir = Path(run_dir)
    rollout_dir = run_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = rollout_dir / "episodes.parquet"
    by_id: dict[str, dict] = {}
    if episodes_path.is_file():
        by_id = _index_rows(read_rollout_episodes(episodes_path))

    arms = arms_for(config)
    baseline = arms[0]
    candidates = [arm for arm in arms if arm.role == "candidate"]
    control_arms = [arm for arm in arms if arm.role in CONTROL_ROLES]
    max_steps = int(getattr(env, "horizon", 10_000))
    control_stop = {arm.name: config.episodes_per_arm for arm in control_arms}
    pool = _Pool(config.workers)
    try:
        previous = 0
        for end in batch_ends(config.episodes_per_arm, config.min_episodes, config.batch_size):
            jobs: list[dict] = []
            active = [baseline, *candidates]
            active.extend(arm for arm in control_arms if control_stop[arm.name] > previous)
            for arm in active:
                limit = end if arm.role not in CONTROL_ROLES else min(end, control_stop[arm.name])
                for offset in range(previous, limit):
                    spec = _spec(config, arm, offset, max_steps)
                    if spec["episode_id"] not in by_id:
                        jobs.append(spec)
            for row in pool.run(jobs):
                by_id[row["episode_id"]] = row
            if end < config.episodes_per_arm:
                cand_flags = [_flags(config, arm, end, by_id) for arm in candidates]
                for arm in control_arms:
                    if control_stop[arm.name] <= previous:
                        continue
                    control_flags = _flags(config, arm, end, by_id)
                    if sprt_separated(
                        oriented(control_flags, config.direction),
                        [oriented(item, config.direction) for item in cand_flags],
                        config.alpha,
                        SPRT_LAMBDA,
                        n_controls=len(control_arms),
                    ):
                        control_stop[arm.name] = end
            previous = end
    finally:
        pool.close()

    kept: list[dict] = []
    for offset in range(config.episodes_per_arm):
        kept.append(by_id[_arm_episode_id(config, baseline, offset)])
        for arm in candidates:
            kept.append(by_id[_arm_episode_id(config, arm, offset)])
    for arm in control_arms:
        for offset in range(control_stop[arm.name]):
            kept.append(by_id[_arm_episode_id(config, arm, offset)])
    kept.sort(key=lambda row: row["episode_id"])
    summary = summarize(kept, config)
    write_rollout_episodes(episodes_path, kept)
    _write_summary(rollout_dir / "summary.json", summary)
    _bump_manifest(run_dir)
    return summary


def _write_summary(path: Path, summary: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_json(tmp, summary)
    tmp.replace(path)


def _bump_manifest(run_dir: Path) -> None:
    path = run_dir / "manifest.json"
    if not path.is_file():
        return
    manifest = read_json(path)
    if manifest.get("schema_version") == 1:
        return
    if "schema_version" in manifest and manifest.get("schema_version") not in (0, 1):
        raise ValueError(f"manifest.schema_version must be 0 or 1, got {manifest.get('schema_version')!r}")
    updated: dict = {}
    inserted = False
    for key, val in manifest.items():
        if key == "schema_version":
            continue
        updated[key] = val
        if key == "tool_version":
            updated["schema_version"] = 1
            inserted = True
    if not inserted:
        updated["schema_version"] = 1
    write_json(path, updated)
