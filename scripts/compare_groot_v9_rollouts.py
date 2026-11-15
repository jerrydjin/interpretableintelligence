#!/usr/bin/env python3
"""Compare ``ii rollouts`` GR00T output against v9 per-condition summary.json files.

Success flags are matched episode-by-episode on ``ep`` index (seed 40000+i).
When v9 ``sim/ep-*.npz`` files exist, action traces are compared via sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from interpretableintelligence.artifacts import read_rollout_episodes
from interpretableintelligence.rollout_groot import (
    COND_TO_POLICY,
    GROOT_SEED_BASE,
    RECON_KIND,
    RECON_MARKER_FEATURE,
    RECON_VALUE,
    cond_by_id,
    load_ablate_plan,
    row_from_v9_episode,
)


def _episode_rows_from_parquet(parquet_rows: list[dict], plan: dict) -> dict[str, list[dict]]:
    by_cond: dict[str, list[dict]] = {}
    for row in parquet_rows:
        seed = int(row["seed"])
        ep = seed - GROOT_SEED_BASE
        cond = _infer_cond(row, plan)
        by_cond.setdefault(cond, []).append({"ep": ep, **row})
    for cond in by_cond:
        by_cond[cond].sort(key=lambda r: r["ep"])
    return by_cond


def _infer_cond(row: dict, plan: dict) -> str:
    if row["condition"] == "baseline":
        return "baseline"
    feats = tuple(int(x) for x in row["feature_ids"])
    kind = row.get("kind")
    value = row.get("value")
    if kind == RECON_KIND and value == RECON_VALUE and feats == (RECON_MARKER_FEATURE,):
        return "recon-only"
    if feats == (720,):
        return "A-cand-720"
    if feats == (7081, 5852):
        return "B-cand-7081+5852"
    for cond in plan["conditions"]:
        latents = cond.get("latents")
        if latents is None:
            continue
        if tuple(int(x) for x in latents) == feats:
            return str(cond["id"])
    return f"multi-{feats}"


def load_v9_episodes(v9_dir: Path, cond: dict) -> list[dict]:
    path = v9_dir / cond["dir"] / "episodes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type") == "episode":
            out.append(row)
    out.sort(key=lambda r: int(r["ep"]))
    return out


def actions_sha(npz_path: Path) -> str | None:
    if not npz_path.is_file():
        return None
    data = np.load(npz_path)
    if "actions" not in data:
        return None
    acts = np.ascontiguousarray(np.asarray(data["actions"], np.float32))
    return hashlib.sha256(acts.tobytes()).hexdigest()


def compare_condition(
    *,
    cond: dict,
    parquet_eps: list[dict],
    v9_eps: list[dict],
    v9_dir: Path,
    plan: dict,
    policy: str,
) -> list[str]:
    mismatches: list[str] = []
    v9_by_ep = {int(r["ep"]): r for r in v9_eps}
    for prow in parquet_eps:
        ep = int(prow["ep"])
        if ep not in v9_by_ep:
            mismatches.append(f"{cond['id']} ep={ep}: missing in v9 log")
            continue
        v9_row = v9_by_ep[ep]
        if bool(prow["success"]) != bool(v9_row["success"]):
            mismatches.append(
                f"{cond['id']} ep={ep} seed={GROOT_SEED_BASE + ep}: "
                f"success parquet={prow['success']} v9={v9_row['success']}"
            )
        expected = row_from_v9_episode(policy=policy, cond=cond, episode=v9_row, plan=plan)
        ref_sha = v9_row.get("actions_sha256")
        sim = v9_dir / cond["dir"] / "sim" / f"ep-{ep:05d}.npz"
        disk_sha = actions_sha(sim)
        if ref_sha and disk_sha and ref_sha != disk_sha:
            mismatches.append(f"{cond['id']} ep={ep}: v9 npz sha {disk_sha} != log {ref_sha}")
    summary_path = v9_dir / cond["dir"] / "summary.json"
    if summary_path.is_file() and parquet_eps:
        summary = json.loads(summary_path.read_text())
        n = int(summary.get("n") or 0)
        if n == len(parquet_eps):
            pq_succ = sum(bool(r["success"]) for r in parquet_eps)
            if int(summary.get("successes", -1)) != pq_succ:
                mismatches.append(
                    f"{cond['id']}: summary successes={summary.get('successes')} parquet={pq_succ}"
                )
    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run folder with rollouts/episodes.parquet")
    parser.add_argument(
        "--v9-dir",
        type=Path,
        required=True,
        help="v9 ablation output root (contains baseline/, arm-A/, ...)",
    )
    parser.add_argument("--plan", type=Path, default=None, help="ablate_plan.json (default: bundled plan)")
    parser.add_argument("--policy", default="groot-1028", help="policy name used in parquet episode ids")
    parser.add_argument(
        "--conditions",
        default="",
        help="Comma-separated condition ids (default: all with parquet rows)",
    )
    args = parser.parse_args(argv)
    plan_path = args.plan
    if plan_path is None:
        plan_path = (
            Path(__file__).resolve().parent.parent
            / "src/interpretableintelligence/adapters/groot/v9/ablate/ablate_plan.json"
        )
    plan = load_ablate_plan(plan_path)
    cond_index = cond_by_id(plan)
    parquet = read_rollout_episodes(args.run_dir / "rollouts" / "episodes.parquet")
    by_cond = _episode_rows_from_parquet(parquet, plan)
    if args.conditions.strip():
        wanted = [part.strip() for part in args.conditions.split(",") if part.strip()]
    else:
        wanted = sorted(by_cond)
    all_mismatches: list[str] = []
    for cid in wanted:
        if cid not in cond_index:
            print(f"skip unknown condition {cid}", file=sys.stderr)
            continue
        cond = cond_index[cid]
        v9_eps = load_v9_episodes(args.v9_dir, cond)
        pq = by_cond.get(cid, [])
        all_mismatches.extend(
            compare_condition(
                cond=cond,
                parquet_eps=pq,
                v9_eps=v9_eps,
                v9_dir=args.v9_dir,
                plan=plan,
                policy=args.policy,
            )
        )
    if all_mismatches:
        print("MISMATCH", file=sys.stderr)
        for line in all_mismatches:
            print(line, file=sys.stderr)
        return 1
    print(f"OK: {len(wanted)} condition(s) match v9 references under {args.v9_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())