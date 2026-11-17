"""GR00T LIBERO-plus rollouts via the v9 ablation plan (task 1028).

``ii rollouts`` uses this module for policies ``groot-1028`` (live v9 runner)
and ``groot-mock`` (deterministic replay from a frozen verification slice for CI).

Episodes use v9 seeds ``40000 + episode_index`` and ``init_state_id = index % 50``.
Rows are written to ``rollouts/episodes.parquet`` and ``rollouts/summary.json``
(SCHEM.md version 1). Recon-only episodes are stored as ``kind=scale``, ``value=1``
on latent 720 so ``episode_id`` differs from a true ablation of 720.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from interpretableintelligence.artifacts import write_rollout_episodes
from interpretableintelligence.policies.groot import is_groot_policy
from interpretableintelligence.policies.base import TRAINED_DICTIONARY, random_dictionary
from interpretableintelligence.rollout import (
    SUMMARY_KEYS,
    RolloutConfig,
    _check_config,
    _index_rows,
    batch_ends,
    make_episode_id,
    oriented,
)
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
    DEFAULT_ROLLOUT_MIN_EPISODES,
    DEFAULT_ROLLOUT_WORKERS,
    SPRT_LAMBDA,
)

GROOT_TASK_ID = "1028"
GROOT_LAYER = 5
GROOT_SEED_BASE = 40_000
RECON_KIND = "scale"
RECON_VALUE = 1.0
RECON_MARKER_FEATURE = 10

DEFAULT_PLAN = (
    Path(__file__).resolve().parent / "adapters" / "groot" / "v9" / "ablate" / "ablate_plan.json"
)
VERIFY_FIXTURE = (
  Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "groot-1028-verify-slice.json"
)

VERIFY_SLICE_CONDITIONS: tuple[str, ...] = (
    "baseline",
    "recon-only",
    "A-cand-720",
    "A-ctrl01-7774",
    "A-ctrl02-2343",
)
VERIFY_SLICE_EPISODES = 8

COND_TO_POLICY = {
    "baseline": ("baseline", (), None, None, None),
    "recon-only": ("intervention", (RECON_MARKER_FEATURE,), GROOT_LAYER, RECON_KIND, RECON_VALUE),
    "A-cand-720": ("intervention", (720,), GROOT_LAYER, "ablate", 0.0),
    "B-cand-7081+5852": ("intervention", (7081, 5852), GROOT_LAYER, "ablate", 0.0),
}


@dataclass(frozen=True)
class GrootRolloutOptions:
    plan_path: Path
    only: tuple[str, ...] | None
    include_recon: bool
    verify_slice: bool


def default_plan_path(run_dir: Path) -> Path:
    sibling = run_dir / "ablate_plan.json"
    if sibling.is_file():
        return sibling
    if DEFAULT_PLAN.is_file():
        return DEFAULT_PLAN
    raise FileNotFoundError(
        f"no ablate plan at {sibling} or {DEFAULT_PLAN}; pass --plan"
    )


def load_ablate_plan(path: Path) -> dict:
    plan = json.loads(Path(path).read_text())
    if plan.get("schema") != "groot-ablate-plan-v1":
        raise ValueError(f"plan schema must be groot-ablate-plan-v1, got {plan.get('schema')!r}")
    return plan


def verify_slice_options() -> GrootRolloutOptions:
    return GrootRolloutOptions(
        plan_path=DEFAULT_PLAN,
        only=VERIFY_SLICE_CONDITIONS,
        include_recon=True,
        verify_slice=True,
    )


def options_from_plan(
    plan: dict,
    *,
    only: list[str] | None,
    verify_slice: bool,
) -> GrootRolloutOptions:
    if verify_slice:
        return verify_slice_options()
    include_recon = False
    selected = only
    if selected:
        include_recon = "recon-only" in selected
    return GrootRolloutOptions(
        plan_path=DEFAULT_PLAN,
        only=tuple(selected) if selected else None,
        include_recon=include_recon,
        verify_slice=False,
    )


def cond_by_id(plan: dict) -> dict[str, dict]:
    return {str(c["id"]): c for c in plan["conditions"]}


def selected_conditions(plan: dict, options: GrootRolloutOptions) -> list[dict]:
    conds = plan["conditions"]
    if options.only:
        allowed = set(options.only)
        conds = [c for c in conds if c["id"] in allowed]
        missing = allowed - {c["id"] for c in conds}
        if missing:
            raise ValueError(f"plan is missing conditions {sorted(missing)}")
    return conds


def episode_limit(cond: dict, *, verify_slice: bool) -> int:
    if verify_slice and cond["id"] in VERIFY_SLICE_CONDITIONS:
        return VERIFY_SLICE_EPISODES
    return int(cond["n_max"])


def v9_episode_seed(ep_index: int, plan: dict) -> int:
    base = int(plan["seeds"]["base"])
    return base + int(ep_index)


def row_from_v9_episode(
    *,
    policy: str,
    cond: dict,
    episode: dict,
    plan: dict,
) -> dict:
    cid = cond["id"]
    dictionary = cond.get("dictionary") or TRAINED_DICTIONARY
    if cid not in COND_TO_POLICY:
        latents = cond.get("latents")
        if latents is None:
            spec = ("baseline", (), None, None, None)
        elif latents == []:
            spec = ("intervention", (RECON_MARKER_FEATURE,), GROOT_LAYER, RECON_KIND, RECON_VALUE)
        else:
            spec = ("intervention", tuple(int(x) for x in latents), GROOT_LAYER, "ablate", 0.0)
    else:
        spec = COND_TO_POLICY[cid]
    condition, feature_ids, layer, kind, value = spec
    if condition == "baseline":
        dictionary = None
    ep = int(episode["ep"])
    seed = v9_episode_seed(ep, plan)
    task_id = str((plan.get("task") or {}).get("id", GROOT_TASK_ID))
    episode_id = make_episode_id(
        policy=policy,
        condition=condition,
        task_id=task_id,
        seed=seed,
        feature_ids=feature_ids,
        layer=layer,
        kind=kind,
        value=value,
        dictionary=dictionary,
    )
    steps = int(episode.get("n_steps") or episode.get("steps") or 1)
    if steps < 1:
        steps = 1
    wall = float(episode.get("seconds") or episode.get("wall_time_s") or 0.0)
    return {
        "episode_id": episode_id,
        "task_id": task_id,
        "seed": seed,
        "condition": condition,
        "feature_ids": list(feature_ids),
        "layer": layer,
        "kind": kind,
        "value": value,
        "success": bool(episode["success"]),
        "steps": steps,
        "wall_time_s": wall,
        "dictionary": dictionary,
        "_cond_id": cid,
        "_ep_index": ep,
        "_actions_sha256": episode.get("actions_sha256"),
    }


def public_row(row: dict) -> dict:
    return {k: v for k, v in row.items() if not str(k).startswith("_")}


def _recon_arm() -> tuple[str, str, tuple[int, ...], str, float]:
    return (
        "recon",
        "intervention",
        (RECON_MARKER_FEATURE,),
        RECON_KIND,
        RECON_VALUE,
    )


GrootArm = tuple  # (name, role, condition, feature_ids, dictionary)


def groot_arms(config: RolloutConfig, *, include_recon: bool) -> list[GrootArm]:
    """Return (name, role, condition, feature_ids, dictionary) arms for schedule checks."""
    arms: list[GrootArm] = []
    arms.append(("baseline", "baseline", "baseline", (), None))
    if include_recon:
        arms.append(("recon", "recon", "intervention", (RECON_MARKER_FEATURE,), TRAINED_DICTIONARY))
    for feature_id in config.candidate_feature_ids:
        arms.append((f"candidate:{feature_id}", "candidate", "intervention", (int(feature_id),), TRAINED_DICTIONARY))
    for feature_id in config.control_feature_ids:
        arms.append((f"control:{feature_id}", "control", "intervention", (int(feature_id),), TRAINED_DICTIONARY))
    for seed, picks in zip(config.random_sae_seeds, config.random_sae_feature_ids, strict=True):
        name = random_dictionary(int(seed))
        for feature_id in picks:
            arms.append((f"random_sae:{name}:{feature_id}", "random_sae", "intervention", (int(feature_id),), name))
    return arms


def _groot_arm_episode_id(config: RolloutConfig, arm: GrootArm, offset: int) -> str:
    _name, role, condition, feature_ids, dictionary = arm
    seed = v9_episode_seed(offset, {"seeds": {"base": config.seed}})
    if role == "baseline":
        return make_episode_id(
            policy=config.policy,
            condition="baseline",
            task_id=config.task_id,
            seed=seed,
            feature_ids=(),
            layer=None,
            kind=None,
            value=None,
        )
    layer = config.layer
    kind = config.kind
    value = config.value
    if role == "recon":
        kind, value = RECON_KIND, RECON_VALUE
    return make_episode_id(
        policy=config.policy,
        condition=condition,
        task_id=config.task_id,
        seed=seed,
        feature_ids=feature_ids,
        layer=layer,
        kind=kind,
        value=value,
        dictionary=dictionary,
    )


def config_for_selection(
    policy: str,
    plan: dict,
    selected: list[dict],
    *,
    include_recon: bool,
    verify_slice: bool,
    workers: int,
    alpha: float,
    min_episodes: int,
    batch_size: int,
) -> RolloutConfig:
    candidates = [c for c in selected if c.get("kind") == "candidate"]
    controls = [c for c in selected if c.get("kind") == "control"]
    if not candidates:
        raise ValueError("groot rollouts need at least one candidate condition in the selection")
    candidate_ids: list[int] = []
    for cand in candidates:
        for latent in cand["latents"]:
            latent_id = int(latent)
            if latent_id not in candidate_ids:
                candidate_ids.append(latent_id)
    control_ids: list[int] = []
    for ctrl in controls:
        if len(ctrl["latents"]) != 1:
            raise ValueError(
                f"ii rollouts schema maps one control feature per arm; {ctrl['id']!r} has {ctrl['latents']!r}"
            )
        control_ids.append(int(ctrl["latents"][0]))
    episodes = max(episode_limit(c, verify_slice=verify_slice) for c in selected if c["id"] != "recon-only")
    if include_recon or any(c["id"] == "recon-only" for c in selected):
        recon = next(c for c in plan["conditions"] if c["id"] == "recon-only")
        episodes = max(episodes, episode_limit(recon, verify_slice=verify_slice))
    seed_base = int(plan["seeds"]["base"])
    return RolloutConfig(
        policy=policy,
        task_id=GROOT_TASK_ID,
        layer=GROOT_LAYER,
        kind="ablate",
        value=0.0,
        seed=seed_base,
        episodes_per_arm=episodes,
        batch_size=batch_size,
        min_episodes=min_episodes,
        alpha=alpha,
        workers=workers,
        candidate_feature_ids=tuple(candidate_ids),
        control_feature_ids=tuple(control_ids),
        direction="rise",
    )


def load_fixture_episodes(path: Path) -> dict[str, list[dict]]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"fixture must be a JSON object, got {type(data).__name__}")
    out: dict[str, list[dict]] = {}
    for cond, rows in data.items():
        out[str(cond)] = list(rows)
    return out


def mock_run_conditions(
    *,
    policy: str,
    plan: dict,
    selected: list[dict],
    fixture_path: Path,
) -> list[dict]:
    fixture = load_fixture_episodes(fixture_path)
    rows: list[dict] = []
    for cond in selected:
        cid = cond["id"]
        limit = episode_limit(cond, verify_slice=cid in VERIFY_SLICE_CONDITIONS)
        episodes = fixture.get(cid)
        if episodes is None:
            raise FileNotFoundError(f"fixture {fixture_path} missing condition {cid!r}")
        if len(episodes) < limit:
            raise ValueError(f"fixture {cid} has {len(episodes)} rows, need {limit}")
        for episode in episodes[:limit]:
            rows.append(row_from_v9_episode(policy=policy, cond=cond, episode=episode, plan=plan))
    return rows


def import_v9_jsonl(
    *,
    policy: str,
    plan: dict,
    v9_out: Path,
    selected: list[dict],
    verify_slice: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    cond_index = cond_by_id(plan)
    for cond in selected:
        cid = cond["id"]
        ep_path = v9_out / cond["dir"] / "episodes.jsonl"
        if not ep_path.is_file():
            raise FileNotFoundError(f"missing v9 log {ep_path}")
        limit = episode_limit(cond, verify_slice=verify_slice)
        by_ep: dict[int, dict] = {}
        for line in ep_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") != "episode":
                continue
            by_ep[int(row["ep"])] = row
        for ep in range(limit):
            if ep not in by_ep:
                raise ValueError(f"{ep_path} missing episode {ep}")
            rows.append(
                row_from_v9_episode(
                    policy=policy,
                    cond=cond_index[cid],
                    episode=by_ep[ep],
                    plan=plan,
                )
            )
    return rows


def summarize_groot(rows: list[dict], config: RolloutConfig, *, include_recon: bool) -> dict:
    """Summary for a GR00T rollout table. The recon-only arm is the hook sanity check, stored as a control row.

    Deltas, intervals and the early-stopping test use ``config.direction`` (``rise`` for the GR00T plans). Random-SAE
    arms (rule 4) join the control family, and a candidate is a ``finding`` only when it is ``causal`` and strictly
    above every random-SAE arm.
    """
    direction = config.direction
    public = [public_row(row) for row in rows]
    by_id = _index_rows(public)
    arms = groot_arms(config, include_recon=include_recon)
    n = config.episodes_per_arm
    recon_n = min(n, VERIFY_SLICE_EPISODES)

    def flags_for(arm: GrootArm, count: int) -> list[bool]:
        out: list[bool] = []
        for offset in range(count):
            episode_id = _groot_arm_episode_id(config, arm, offset)
            if episode_id not in by_id:
                raise ValueError(f"rollouts are missing {episode_id}")
            out.append(bool(by_id[episode_id]["success"]))
        return out

    baseline_arm = next(arm for arm in arms if arm[1] == "baseline")
    baseline_flags = flags_for(baseline_arm, n)

    candidate_arms = [arm for arm in arms if arm[1] == "candidate"]
    family_arms = [arm for arm in arms if arm[1] in ("control", "random_sae")]
    candidate_flags = [flags_for(arm, n) for arm in candidate_arms]
    oriented_candidates = [oriented(flags, direction) for flags in candidate_flags]

    control_plan: list[tuple[GrootArm, int, str, list[bool]]] = []
    for arm in family_arms:
        present = 0
        while present < n and _groot_arm_episode_id(config, arm, present) in by_id:
            present += 1
        if present == 0:
            raise ValueError(f"{arm[0]} has no episodes")
        present_flags = flags_for(arm, present)
        length, reason = declared_stop_groot(
            oriented(present_flags, direction),
            oriented_candidates,
            episodes=n,
            min_episodes=config.min_episodes,
            batch_size=config.batch_size,
            alpha=config.alpha,
            n_controls=len(family_arms),
        )
        if length != present:
            raise ValueError(f"{arm[0]} has {present} episodes but schedule requires {length} ({reason})")
        control_plan.append((arm, length, reason, present_flags))

    expected_ids = {_groot_arm_episode_id(config, baseline_arm, offset) for offset in range(n)}
    if include_recon:
        recon_arm = next(arm for arm in arms if arm[1] == "recon")
        expected_ids |= {_groot_arm_episode_id(config, recon_arm, offset) for offset in range(recon_n)}
    for arm in candidate_arms:
        for offset in range(n):
            expected_ids.add(_groot_arm_episode_id(config, arm, offset))
    for arm, length, _reason, _flags in control_plan:
        for offset in range(length):
            expected_ids.add(_groot_arm_episode_id(config, arm, offset))
    extra = set(by_id) - expected_ids
    missing = expected_ids - set(by_id)
    if extra or missing:
        raise ValueError(f"episode set mismatch, missing={len(missing)} extra={len(extra)}")

    def score_arm(arm: GrootArm, flags: list[bool], episodes_run: int, reason: str) -> dict:
        role = arm[1]
        paired_base = baseline_flags[: len(flags)]
        delta, ci_low, ci_high = paired_success_delta_ci(oriented(paired_base, direction), oriented(flags, direction), config.alpha)
        n10, n01 = discordant_counts(paired_base, flags)
        return {
            "feature_id": int(arm[3][0]),
            "role": "control" if role == "recon" else role,
            "dictionary": arm[4],
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
            "_recon": role == "recon",
            "_flags": list(flags),
        }

    raw_features: list[dict] = []
    for arm, flags in zip(candidate_arms, candidate_flags, strict=True):
        raw_features.append(score_arm(arm, flags, n, "fixed_budget"))
    if include_recon:
        recon_arm = next(arm for arm in arms if arm[1] == "recon")
        raw_features.append(score_arm(recon_arm, flags_for(recon_arm, recon_n), recon_n, "fixed_budget"))
    for arm, length, reason, flags in control_plan:
        raw_features.append(score_arm(arm, flags, length, reason))
    # Feature controls of the trained SAE (the recon-only check is not one), and the random-weights SAE arms.
    control_deltas = [float(item["delta"]) for item in raw_features if item["role"] == "control" and not item["_recon"]]
    random_deltas = [float(item["delta"]) for item in raw_features if item["role"] == "random_sae"]

    features: list[dict] = []
    control_index = 0
    sprt_verdicts: dict[int, str] = {}
    if config.rise_sprt is not None:
        from interpretableintelligence.adapters.groot.v9.ablate_v9 import sequential_decision

        # Rule 1: the Sobel-Wald three-way SPRT on the arm's successes in episode (seed) order. A fixed-budget run
        # never stops early, so the call is where the plan's SPRT would have stopped on this prefix (or cap_undecided).
        for index, item in enumerate(raw_features):
            if not item["_recon"] and item["role"] != "baseline":
                sprt_verdicts[index] = sequential_decision(item["_flags"], config.rise_sprt)["verdict"]
    for arm_key, scored in enumerate(raw_features):
        scored.pop("_flags")
        recon = scored.pop("_recon")
        if scored["role"] == "control" and not recon:
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
            beats_random = bool(random_percentile is not None and random_percentile >= 100.0)
        verdict = sprt_verdicts.get(arm_key)
        features.append(
            {
                **scored,
                "control_percentile": percentile,
                "n_controls": len(others),
                "causal": causal,
                "random_sae_percentile": random_percentile,
                "n_random_sae": len(random_deltas) if scored["role"] == "candidate" else 0,
                "beats_random_sae": beats_random,
                "sprt_verdict": verdict,
                # Rules 1, 2+3 (causal: McNemar p and above every matched control) and 4 (above every untrained SAE).
                "finding": bool(scored["role"] == "candidate" and causal and beats_random is True and verdict == "rise"),
            }
        )

    def condition_record(arm: GrootArm, episodes_run: int, reason: str, flags: list[bool]) -> dict:
        role = "control" if arm[1] == "recon" else arm[1]
        gap = log_lr = threshold = None
        if arm[1] in ("control", "random_sae"):
            gap, log_lr, threshold = binding_sprt(
                oriented(flags, direction),
                oriented_candidates,
                config.alpha,
                SPRT_LAMBDA,
                n_controls=len(family_arms),
            )
        layer = None if arm[1] == "baseline" else config.layer
        kind = None if arm[1] == "baseline" else config.kind
        value = None if arm[1] == "baseline" else config.value
        if arm[1] == "recon":
            kind, value = RECON_KIND, RECON_VALUE
        return {
            "role": role,
            "condition": arm[2],
            "dictionary": arm[4],
            "feature_ids": list(arm[3]),
            "layer": layer,
            "kind": kind,
            "value": value,
            "episodes_run": episodes_run,
            "episodes_budget": n,
            "stopped_early": bool(episodes_run < n and reason == "sprt_separated"),
            "reason": reason,
            "success_rate": float(sum(flags) / len(flags)) if flags else 0.0,
            "gap_vs_candidate": gap,
            "sprt_log_lr": log_lr,
            "sprt_threshold": threshold,
        }

    conditions = [condition_record(baseline_arm, n, "fixed_budget", baseline_flags)]
    if include_recon:
        recon_arm = next(arm for arm in arms if arm[1] == "recon")
        conditions.append(condition_record(recon_arm, recon_n, "fixed_budget", flags_for(recon_arm, recon_n)))
    for arm, flags in zip(candidate_arms, candidate_flags, strict=True):
        conditions.append(condition_record(arm, n, "fixed_budget", flags))
    for arm, length, reason, flags in control_plan:
        conditions.append(condition_record(arm, length, reason, flags))

    n_arms = len(conditions)
    fixed_budget = n_arms * n
    total = len(public)
    validated = [
        int(feature["feature_id"]) for feature in features if feature["causal"] and feature["role"] in ("candidate", "control")
    ]
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


def declared_stop_groot(
    control_flags: list[bool],
    candidate_flags: list[list[bool]],
    *,
    episodes: int,
    min_episodes: int,
    batch_size: int,
    alpha: float,
    n_controls: int,
) -> tuple[int, str]:
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


def summarize_rollout_rows(rows: list[dict], config: RolloutConfig) -> dict:
    if is_groot_policy(config.policy):
        include_recon = any(
            row.get("kind") == RECON_KIND and float(row.get("value") or 0) == RECON_VALUE
            for row in rows
            if row.get("condition") == "intervention" and row.get("feature_ids") == [RECON_MARKER_FEATURE]
        )
        return summarize_groot(rows, config, include_recon=include_recon)
    from interpretableintelligence.rollout import summarize

    return summarize(rows, config)


def run_live_v9(
    run_dir: Path,
    plan: dict,
    *,
    plan_path: Path,
    sae_path: Path | None,
    only: tuple[str, ...] | None,
    dictionaries: dict[str, Path] | None = None,
) -> Path:
    try:
        import interpretableintelligence.adapters.groot.v9.ablate_v9 as ablate_v9
        import interpretableintelligence.adapters.groot.v9.physint_inproc_v9 as v9
        import interpretableintelligence.adapters.groot.v9.physint_inproc_v8 as v8
    except ImportError as exc:
        raise NotImplementedError(
            "groot-1028 live rollouts need the Colab v9 stack (lerobot, LIBERO-plus, GPU). "
            "See docs/groot-rollouts-verify.md"
        ) from exc
    v9_out = run_dir / "rollouts" / "v9-ablate"
    sae = sae_path or Path(plan["sae"].get("subset_file", "sae_subset.safetensors"))
    if not sae.is_file():
        raise FileNotFoundError(f"SAE subset not found: {sae}")
    v9_out.mkdir(parents=True, exist_ok=True)
    ctx = v9.Ctx(run=v9_out, run_id=run_dir.name, layer=int(plan["hook"]["layer"]))  # run_ablation writes under ctx.run
    v8.setup(ctx, device="cuda")
    ablate_v9.run_ablation(ctx, plan, sae_path=sae, only=list(only) if only else None, dictionaries=dictionaries)
    return v9_out


def execute_groot_rollouts(
    run_dir: Path,
    *,
    policy: str,
    plan_path: Path | None = None,
    verify_slice: bool = False,
    only: list[str] | None = None,
    workers: int = DEFAULT_ROLLOUT_WORKERS,
    alpha: float = DEFAULT_ROLLOUT_ALPHA,
    min_episodes: int = DEFAULT_ROLLOUT_MIN_EPISODES,
    batch_size: int = DEFAULT_ROLLOUT_BATCH,
    sae_path: Path | None = None,
    v9_out: Path | None = None,
    fixture_path: Path | None = None,
) -> dict:
    if not is_groot_policy(policy):
        raise ValueError(f"not a groot rollout policy: {policy!r}")
    plan_file = Path(plan_path) if plan_path else default_plan_path(run_dir)
    plan = load_ablate_plan(plan_file)
    options = options_from_plan(plan, only=only, verify_slice=verify_slice)
    selected = selected_conditions(plan, options)
    include_recon = options.include_recon or any(c["id"] == "recon-only" for c in selected)
    config = config_for_selection(
        policy,
        plan,
        selected,
        include_recon=include_recon,
        verify_slice=options.verify_slice,
        workers=workers,
        alpha=alpha,
        min_episodes=min_episodes,
        batch_size=batch_size,
    )
    _check_config(config)

    if policy == "groot-mock":
        fixture = Path(fixture_path) if fixture_path else VERIFY_FIXTURE
        built = mock_run_conditions(policy=policy, plan=plan, selected=selected, fixture_path=fixture)
    elif policy == "groot-1028":
        out = v9_out or run_dir / "rollouts" / "v9-ablate"
        if out.is_dir() and any(out.rglob("episodes.jsonl")):
            built = import_v9_jsonl(
                policy=policy,
                plan=plan,
                v9_out=out,
                selected=selected,
                verify_slice=options.verify_slice,
            )
        else:
            out = run_live_v9(run_dir, plan, plan_path=plan_file, sae_path=sae_path, only=options.only)
            built = import_v9_jsonl(
                policy=policy,
                plan=plan,
                v9_out=out,
                selected=selected,
                verify_slice=options.verify_slice,
            )
    else:
        raise ValueError(f"unknown groot policy {policy!r}")

    public = [public_row(row) for row in built]
    summary = summarize_groot(public, config, include_recon=include_recon)

    rollout_dir = Path(run_dir) / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    write_rollout_episodes(rollout_dir / "episodes.parquet", public)
    from interpretableintelligence.rollout import _bump_manifest, _write_summary

    _write_summary(rollout_dir / "summary.json", summary)
    _bump_manifest(Path(run_dir))
    return summary


def plan_rise_sprt(plan: dict) -> dict | None:
    """The plan's rule 1 SPRT design, or None for a plan built before it carried one (then nothing is a finding)."""
    from interpretableintelligence.rollout import RISE_SPRT_KEYS

    block = plan.get("sprt") or {}
    if block.get("p0") is None:
        return None
    return {key: block[key] for key in sorted(RISE_SPRT_KEYS)}


def null_mock_outcome(cond: dict, ep: int, seed: int) -> bool:
    """``groot-mock`` with ``--pick``: success depends on the seed only, so every arm matches baseline (no effect)."""
    del cond, ep
    return int(hashlib.sha256(f"groot-mock:{int(seed)}".encode()).hexdigest()[:8], 16) / 16**8 < 0.56


def mock_plan_rows(policy: str, plan: dict, simulate) -> list[dict]:
    """One row per (condition, episode index) of ``plan``, success from ``simulate(cond, ep, seed)``."""
    rows: list[dict] = []
    for cond in plan["conditions"]:
        for ep in range(int(cond["n_max"])):
            seed = v9_episode_seed(ep, plan)
            episode = {"ep": ep, "success": bool(simulate(cond, ep, seed)), "n_steps": 1, "seconds": 0.0}
            rows.append(row_from_v9_episode(policy=policy, cond=cond, episode=episode, plan=plan))
    return rows


def execute_groot_pick(
    run_dir: Path,
    *,
    policy: str,
    capture: Path,
    pick: int,
    random_sae: int,
    controls: int,
    episodes: int,
    seed: int = GROOT_SEED_BASE,
    alpha: float = DEFAULT_ROLLOUT_ALPHA,
    workers: int = DEFAULT_ROLLOUT_WORKERS,
    sae_path: Path | None = None,
    v9_out: Path | None = None,
    simulate=None,
    log=print,
) -> dict:
    """``ii rollouts --policy groot-* --pick N --random-sae K --capture CAPTURE``: research rule 4 on GR00T.

    Picks candidates and each untrained SAE's features with the same picker (``rollout_groot_pick``), writes the plan,
    runs it (``groot-1028``: the v9 runner, or its logs from ``v9_out``; ``groot-mock``: ``simulate``, by default no
    effect anywhere), and scores it as a rise test with the random-SAE arms in the control family. Fixed budget: every
    arm runs ``episodes`` paired episodes.
    """
    from interpretableintelligence.rollout_groot_pick import PICKER, build_pick_plan, dictionary_paths
    from interpretableintelligence.rollout_pick import random_sae_seeds

    if not is_groot_policy(policy):
        raise ValueError(f"not a groot rollout policy: {policy!r}")
    if random_sae < 0:
        raise ValueError(f"--random-sae must be >= 0, got {random_sae}")
    run_dir = Path(run_dir)
    seeds = random_sae_seeds(int(seed), int(random_sae))
    plan = build_pick_plan(
        run_dir, Path(capture), n=int(pick), random_seeds=seeds, controls=int(controls), episodes=int(episodes), seed=int(seed), log=log
    )
    info = plan["pick"]
    config = RolloutConfig(
        policy=policy,
        task_id=str(plan["task"]["id"]),
        layer=int(plan["hook"]["layer"]),
        kind="ablate",
        value=0.0,
        seed=int(plan["seeds"]["base"]),
        episodes_per_arm=int(episodes),
        batch_size=int(episodes),
        min_episodes=int(episodes),
        alpha=float(alpha),
        workers=int(workers),
        candidate_feature_ids=tuple(int(item) for item in info["candidates"]),
        control_feature_ids=tuple(int(item) for item in info["controls"]),
        candidate_source="pick",
        picker=PICKER,
        pick_discovery_episodes=None,
        random_sae_seeds=tuple(int(item) for item in info["random_sae_seeds"]),
        random_sae_feature_ids=tuple(tuple(int(x) for x in picks) for picks in info["random_sae_feature_ids"]),
        direction="rise",
        rise_sprt=plan_rise_sprt(plan),
    )
    _check_config(config)
    rollout_dir = run_dir / "rollouts"
    if policy == "groot-mock":
        built = mock_plan_rows(policy, plan, simulate or null_mock_outcome)
    else:
        out = Path(v9_out) if v9_out else rollout_dir / "v9-ablate"
        conds = list(plan["conditions"])
        if not (out.is_dir() and any(out.rglob("episodes.jsonl"))):
            paths = dictionary_paths(run_dir, plan)
            trained = Path(sae_path) if sae_path else paths.pop("trained")
            paths.pop("trained", None)
            out = run_live_v9(run_dir, plan, plan_path=rollout_dir / "ablate_plan.json", sae_path=trained, only=None, dictionaries=paths)
        built = import_v9_jsonl(policy=policy, plan=plan, v9_out=out, selected=conds)
    public = [public_row(row) for row in built]
    summary = summarize_groot(public, config, include_recon=True)
    rollout_dir.mkdir(parents=True, exist_ok=True)
    write_rollout_episodes(rollout_dir / "episodes.parquet", public)
    from interpretableintelligence.rollout import _bump_manifest, _write_summary

    _write_summary(rollout_dir / "summary.json", summary)
    _bump_manifest(run_dir)
    return summary


def format_groot_report(summary: dict, *, plan_sha: str | None = None) -> str:
    from interpretableintelligence.rollout import format_report

    lines = [format_report(summary)]
    if plan_sha:
        lines.append(f"ablate_plan_sha256={plan_sha}")
    return "\n".join(lines)