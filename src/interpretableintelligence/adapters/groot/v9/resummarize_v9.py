"""Recompute the ablation summary's statistics from records that already exist. No rollouts, no GPU, no model.

Two sources, same statistics (ablate_v9: Wilson, exact sequential interval and p-values under the plan's SPRT, per-arm
control comparison, Holm across candidates):

  --logs DIR    an ablation output folder (plan.json + <arm>/<cond>/episodes.jsonl, i.e. an unpacked ablate_snapshot.tar).
                Rebuilt with ablate_v9.AblationState from the per-episode rows, exactly as the runner does.
  --table FILE  a vendored top-level summary.json (demo-data/groot/groot-1028-ablate-summary.json) plus --plan.
                The SPRT's log-likelihood ratio after n episodes depends only on (n, successes), so the sequential
                columns, control counts and Holm values computed from the table equal what --logs would give. Every row
                is checked first: the plan sha must match, Wilson must reproduce, and each SPRT row's (n, successes)
                must be a stopping point of the plan with the logged verdict. Paired deltas / McNemar need episode-level
                outcomes and are carried over unchanged.

Writes nothing if anything bears on a rise finding (an SPRT rise verdict on a candidate, a Holm-adjusted sequential rise
p < alpha, a permutation rise p <= alpha, a candidate beating every control): it prints the crossings and exits 2.
It never sets `real` (asserted false) and never changes a verdict.

    python -m interpretableintelligence.adapters.groot.v9.resummarize_v9 --table demo-data/groot/groot-1028-ablate-summary.json \\
        --plan src/interpretableintelligence/adapters/groot/v9/ablate/ablate_plan.json --write
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import ablate_v9 as A

KEY_ORDER = ("plan_sha256", "version", "unvalidated", "real", "updated", "stats_updated", "stats_source", "schedule_order",
             "episodes_logged", "duplicates_ignored", "table", "arms", "pool", "session_wall_s", "all_finished")


class StatsMismatch(ValueError):
    pass


def from_table(summary: dict, plan: dict) -> dict:
    """Recompute every derived column of a top-level summary from its (n, successes) rows; see module docstring."""
    sha = A.plan_sha(plan)
    if summary.get("plan_sha256") != sha:
        raise StatsMismatch(f"summary plan_sha256 {summary.get('plan_sha256')!r} != plan {sha}")
    if summary.get("real") is not False:
        raise StatsMismatch("summary real is not false")
    conds = {c["id"]: c for c in plan["conditions"]}
    sprt = plan["sprt"]
    stops = {(n, s): v for n, s, v, _ in A.sprt_stop_points(sprt)}
    if [r["cond"] for r in summary["table"]] != plan["schedule"]["order"]:
        raise StatsMismatch("table rows are not the plan's conditions in schedule order")
    table = []
    for r in summary["table"]:
        c = conds[r["cond"]]
        n, k = int(r["n"]), int(r["successes"])
        if A.wilson(k, n) != r["wilson95"]:
            raise StatsMismatch(f"{r['cond']}: wilson95 {r['wilson95']} != recomputed {A.wilson(k, n)}")
        seq = None
        if c["sprt"]:
            if stops.get((n, k)) != r["verdict"]:
                raise StatsMismatch(f"{r['cond']}: (n={n}, s={k}) verdict {r['verdict']!r} but the plan's SPRT stops there "
                                    f"with {stops.get((n, k))!r}")
            seq = A.sequential_inference(n, k, sprt)
        elif r["verdict"] is not None:
            raise StatsMismatch(f"{r['cond']}: fixed-n condition has a verdict")
        keep = {key: r[key] for key in ("cond", "n", "successes", "rate", "wilson95", "verdict", "delta_paired", "n_shared")}
        table.append(keep | A.table_sequential_columns(seq))
    out = dict(summary)
    out["table"] = table
    out["arms"] = A.arms_block(table, conds, sprt)
    return out


def from_logs(out_dir: Path) -> dict:
    """Rebuild from per-episode rows with the runner's own state (reads plan.json in the folder)."""
    plan = json.loads((out_dir / "plan.json").read_text())
    old = json.loads((out_dir / "summary.json").read_text()) if (out_dir / "summary.json").is_file() else {}
    st = A.AblationState(plan, out_dir, log=lambda _m: None)
    keep = {k: old[k] for k in ("pool", "session_wall_s") if k in old}
    top = st.write_summaries(extra=keep | {"all_finished": all(st.finished(c) for c in st.conds)})
    return top


def finalize(top: dict, source: str, alpha: float = 0.05) -> tuple[dict, dict]:
    if top.get("real") is not False:
        raise StatsMismatch("real must stay false")
    top = dict(top)
    top["stats_updated"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    top["stats_source"] = source
    ordered = {k: top[k] for k in KEY_ORDER if k in top} | {k: v for k, v in top.items() if k not in KEY_ORDER}
    return ordered, A.threshold_crossings(ordered, alpha)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--logs", type=Path, help="ablation output folder with plan.json and episodes.jsonl files")
    src.add_argument("--table", type=Path, help="vendored top-level summary.json")
    ap.add_argument("--plan", type=Path, help="plan for --table (ablate/ablate_plan.json)")
    ap.add_argument("--out", type=Path, help="where to write (default: the --table file, or <logs>/summary.json)")
    ap.add_argument("--write", action="store_true", help="write the result (default: print the crossings only)")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args(argv)
    if a.table:
        if not a.plan:
            ap.error("--table needs --plan")
        top = from_table(json.loads(a.table.read_text()), json.loads(a.plan.read_text()))
        source = ("vendored table (n, successes, verdict per condition; sufficient for the SPRT statistics), checked "
                  "against the plan; per-episode logs not in the repo")
        dest = a.out or a.table
    else:
        top = from_logs(a.logs)
        source = f"per-episode logs ({a.logs.name}/*/episodes.jsonl)"
        dest = a.out or a.logs / "summary.json"
    top, cross = finalize(top, source, a.alpha)
    print(json.dumps(cross, indent=1))
    if cross["rise"]:
        print("STOP: something bears on a rise finding; nothing written. Report it, do not change a verdict.", file=sys.stderr)
        return 2
    if a.write:
        Path(dest).write_text(json.dumps(top, indent=2) + "\n")
        print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
