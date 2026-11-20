"""Closed-loop steering of GR00T's backbone features on LIBERO-plus (task 1028 pot-order test, Nov 19 2026).

The action head cross-attends to ``backbone_features`` (the VLM output after the head's own projection,
``action_head._encode_features``). A steering arm adds one fixed vector to every backbone token at every policy call
of an episode. The outcome is the pot order, not task success: in task 1028 the policy lifts moka_pot_2 (pot 1) first
in 96 of 96 capture episodes, so ``pot0_first`` (moka_pot_1 lifted before moka_pot_2) has a near-zero baseline and a
rise is easy to see.

This is a supervised-direction test, not an SAE feature test: research rule 4 (beat untrained SAEs) cannot apply,
so nothing here is ever a finding. It answers whether the backbone holds a lever for a named behaviour.

Plan (JSON, ``schema: groot-steer-backbone-v1``)::

    {"schema": ..., "task": {"index", "n_init_states", "name"}, "seeds": {"base": 80000},
     "arms": [{"id": "baseline", "kind": "baseline", "n": 30},
              {"id": "dom-x3", "kind": "direction", "vector": "/vol/.../dom.npy", "scale": 3.0, "n": 30},
              {"id": "rand1-x6", "kind": "random", "random_seed": 1, "norm_of": "/vol/.../dom.npy", "scale": 6.0,
               "n": 30}, ...],
     "outcome": "pot0_first", "sprt": {"p0", "p1_rise", "p1_fall", "alpha", "beta", "cap"}}

Episodes run interleaved by index (episode i of every arm before episode i + 1), so a run stopped by its time cap
stays balanced across arms. Resumable: rows already in ``<out>/<arm>/episodes.jsonl`` are kept.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = "groot-steer-backbone-v1"
Z_LIFT = 0.020  # pot_events: lift = pot z more than 2 cm above its step-0 z
POT_NAMES = ("moka_pot_1", "moka_pot_2")  # obj_names order; pot 0 = moka_pot_1


@dataclass(frozen=True)
class SteerEdit:
    """One arm's vector, added to every backbone token. Short repr: Job.meta is printed in timing lines."""

    arm: str
    vector: object  # torch.Tensor [D]

    def __repr__(self) -> str:
        return f"SteerEdit({self.arm})"


@dataclass(frozen=True)
class CaptureEdit:
    """Leaves the features unchanged and keeps every ``every``-th policy call's backbone tokens (float16, on CPU) for
    one episode, so closed-loop episodes double as SAE training data (plan field ``capture``)."""

    arm: str
    ep: int
    every: int = 2

    def __repr__(self) -> str:
        return f"CaptureEdit({self.arm}, {self.ep})"


class BackboneSteerer:
    """The ``editor`` protocol of ``physint_inproc_v9.run_pool``, acting on ``action_head._encode_features``.

    Rows whose edit is None are untouched. While armed with any edit, the wrapped method must run exactly once per
    predict call, or ``end()`` raises: a silent no-op would make every arm look like the baseline."""

    def __init__(self) -> None:
        self.head = None
        self.rows = None
        self.calls = 0
        self.stats = None
        self.buffers: dict[tuple[str, int], list] = {}
        self.seen: dict[tuple[str, int], int] = {}

    def take(self, arm: str, ep: int):
        """The captured tokens of one episode, [calls, tokens, D] float16 numpy, or None; forgets them."""
        self.seen.pop((arm, ep), None)
        parts = self.buffers.pop((arm, ep), None)
        return np.stack([p.numpy() for p in parts]) if parts else None

    def install(self, policy, layer: int):
        del layer
        self.install_on(policy._groot_model.action_head)
        return self

    def install_on(self, head):
        self.remove()
        original = head._encode_features

        def wrapped(*args, **kwargs):
            feats = original(*args, **kwargs)
            return self._edit(feats)

        head._encode_features = wrapped
        self.head = head
        return self

    def remove(self):
        if self.head is not None and "_encode_features" in self.head.__dict__:
            del self.head.__dict__["_encode_features"]
        self.head = None

    def begin(self, rows: list):
        self.rows = list(rows)
        self.calls = 0
        self.stats = [None] * len(rows)

    def disarm(self):
        self.rows = None

    def end(self) -> list:
        if self.rows is None:
            raise RuntimeError("BackboneSteerer.end() without begin()")
        if any(r is not None for r in self.rows) and self.calls != 1:
            raise RuntimeError(f"_encode_features ran {self.calls}x in one predict call, expected 1")
        out, self.rows = self.stats, None
        return out

    def _edit(self, feats):
        if self.rows is None:
            return feats
        self.calls += 1
        if all(r is None for r in self.rows):
            return feats
        bf = feats.backbone_features
        if bf.shape[0] != len(self.rows):
            raise RuntimeError(f"backbone_features has {bf.shape[0]} rows, the predict call has {len(self.rows)} edits")
        delta = bf.new_zeros(bf.shape[0], 1, bf.shape[-1])
        for b, edit in enumerate(self.rows):
            if isinstance(edit, CaptureEdit):
                key = (edit.arm, int(edit.ep))
                n = self.seen.get(key, 0)
                self.seen[key] = n + 1
                if n % max(1, int(edit.every)) == 0:
                    self.buffers.setdefault(key, []).append(bf[b].detach().to("cpu", copy=True).half())
                self.stats[b] = {"rel_delta": 0.0}
            elif edit is not None:
                delta[b, 0] = edit.vector.to(device=bf.device, dtype=bf.dtype)
                token_norm = float(bf[b].float().norm(dim=-1).mean())
                self.stats[b] = {"rel_delta": float(edit.vector.float().norm()) / max(token_norm, 1e-12)}
        feats.backbone_features = bf + delta
        return feats

    def accumulate(self, acc, s):
        if s is None:
            return acc
        if acc is None:
            return {"n_calls": 1, "rel_delta_sum": s["rel_delta"]}
        return {"n_calls": acc["n_calls"] + 1, "rel_delta_sum": acc["rel_delta_sum"] + s["rel_delta"]}

    @staticmethod
    def summarize(acc) -> dict | None:
        if acc is None:
            return None
        return {"n_calls": acc["n_calls"], "mean_rel_delta": round(acc["rel_delta_sum"] / max(1, acc["n_calls"]), 6)}


def pot_order(sim: dict) -> dict:
    """First lift step of each pot (z more than ``Z_LIFT`` above its step-0 z) and whether pot 0 went first."""
    pos = np.asarray(sim["obj_pos"], np.float64)
    names = [str(n) for n in np.asarray(sim.get("obj_names", POT_NAMES)).reshape(-1)]
    if names[:2] != list(POT_NAMES):
        raise ValueError(f"expected objects {POT_NAMES}, got {names}")
    rel = pos[:, :2, 2] - pos[:1, :2, 2]
    lift = [int(np.argmax(rel[:, k] > Z_LIFT)) if bool((rel[:, k] > Z_LIFT).any()) else None for k in range(2)]
    pot0_first = lift[0] is not None and (lift[1] is None or lift[0] < lift[1])
    return {"pot0_lift_t": lift[0], "pot1_lift_t": lift[1], "pot0_first": bool(pot0_first),
            "any_lift": lift[0] is not None or lift[1] is not None}


def arm_vectors(plan: dict):
    """{arm id: torch vector or None}, from each arm's ``vector`` file (times ``scale``) or a seeded random direction at
    the norm of ``norm_of`` (times ``scale``)."""
    import torch  # noqa: PLC0415

    out = {}
    for arm in plan["arms"]:
        scale = float(arm.get("scale", 1.0))
        if arm.get("vector"):
            v = np.load(arm["vector"]).astype(np.float32).reshape(-1) * scale
        elif arm.get("random_seed") is not None:
            ref = np.load(arm["norm_of"]).astype(np.float32).reshape(-1)
            r = np.random.default_rng(int(arm["random_seed"])).standard_normal(ref.shape[0]).astype(np.float32)
            v = r / np.linalg.norm(r) * np.linalg.norm(ref) * scale
        else:
            out[arm["id"]] = None
            continue
        out[arm["id"]] = torch.from_numpy(v)
    return out


def plan_sha(plan: dict) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


def read_rows(out: Path, arm: str) -> dict[int, dict]:
    path = Path(out) / arm / "episodes.jsonl"
    rows: dict[int, dict] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[int(row["ep"])] = row
    return rows


def summarize(plan: dict, out: Path) -> dict:
    """Per arm: pot0_first and success counts, the paired McNemar test against the baseline on shared seeds, and the
    Sobel-Wald SPRT on pot0_first in episode order. Every non-baseline arm is compared with every random arm."""
    from interpretableintelligence.adapters.groot.v9.ablate_v9 import paired_delta, sequential_decision, wilson  # noqa: PLC0415

    outcome = plan.get("outcome", "pot0_first")
    rows = {arm["id"]: read_rows(out, arm["id"]) for arm in plan["arms"]}
    base = {i: bool(r[outcome]) for i, r in rows["baseline"].items()} if "baseline" in rows else {}
    base_success = {i: bool(r["success"]) for i, r in rows.get("baseline", {}).items()}
    table = []
    for arm in plan["arms"]:
        r = rows[arm["id"]]
        eps = sorted(r)
        y = [bool(r[i][outcome]) for i in eps]
        s = [bool(r[i]["success"]) for i in eps]
        item = {"arm": arm["id"], "kind": arm["kind"],
                "n": len(eps), outcome: int(sum(y)), "rate": round(sum(y) / len(y), 4) if y else None,
                "wilson95": wilson(int(sum(y)), len(y)), "successes": int(sum(s)),
                "success_rate": round(sum(s) / len(s), 4) if s else None,
                "any_lift": int(sum(bool(r[i]["any_lift"]) for i in eps))}
        if arm["id"] != "baseline":
            item["vs_baseline"] = paired_delta({i: bool(r[i][outcome]) for i in eps}, base)
            item["success_vs_baseline"] = paired_delta({i: bool(r[i]["success"]) for i in eps}, base_success)
            if plan.get("sprt"):
                item["sprt"] = sequential_decision(y, plan["sprt"])
        table.append(item)
    randoms = [t for t in table if t["kind"] == "random"]
    for t in table:
        if t["kind"] == "direction" and randoms:
            t["beats_every_random_arm"] = all((t["rate"] or 0) > (c["rate"] or 0) for c in randoms)
    return {"schema": SCHEMA, "plan_sha256": plan_sha(plan), "outcome": outcome, "table": table,
            "finding": False, "note": "supervised backbone direction, no SAE: research rule 4 cannot apply; never a finding"}


def run_steer(ctx, plan: dict, *, out: Path | None = None) -> dict:
    """Run ``plan`` with the live policy in ``ctx`` (physint_inproc_v9.Ctx after v8.setup). Resumable."""
    import physint_inproc_v9 as v9  # noqa: PLC0415

    if plan.get("schema") != SCHEMA:
        raise ValueError(f"plan schema {plan.get('schema')!r} != {SCHEMA}")
    kinds = {a["id"]: a.get("kind") for a in plan["arms"]}
    if kinds.get("baseline") != "baseline" or any(k not in ("baseline", "direction", "random") for k in kinds.values()):
        raise ValueError(f"plan needs a 'baseline' arm and kinds in baseline/direction/random, got {kinds}")
    out = Path(out or ctx.run)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
    vectors = arm_vectors(plan)
    edits = {arm: (SteerEdit(arm, v) if v is not None else None) for arm, v in vectors.items()}
    capture = plan.get("capture") or {}
    capture_arms = set(capture.get("arms", []))
    if capture_arms - {a["id"] for a in plan["arms"]} or any(edits.get(a) is not None for a in capture_arms):
        raise ValueError("plan.capture.arms must name arms without a vector")
    steerer = BackboneSteerer()
    task = plan["task"]
    key = ("plus", int(task["index"]))
    n_init = int(task["n_init_states"])
    base_seed = int(plan["seeds"]["base"])
    arms = [a["id"] for a in plan["arms"]]
    caps = {a["id"]: int(a["n"]) for a in plan["arms"]}
    done = {arm: set(read_rows(out, arm)) for arm in arms}
    queue = [(arm, i) for i in range(max(caps.values())) for arm in arms if i < caps[arm] and i not in done[arm]]
    queue.reverse()
    t0 = time.time()

    def next_job():
        if not queue:
            return None
        arm, i = queue.pop()
        edit = CaptureEdit(arm, i, int(capture.get("every", 2))) if arm in capture_arms else edits[arm]
        return v9.Job(key=key, seed=base_seed + i, init_state_id=i % n_init, meta={"arm": arm, "ep": i, "edit": edit})

    def on_done(job, ep):
        arm, i = job.meta["arm"], int(job.meta["ep"])
        order = pot_order(ep.sim)
        row = {"ep": i, "seed": job.seed, "init_state_id": job.init_state_id, "arm": arm, "success": bool(ep.success),
               **order, "n_steps": int(ep.n_steps), "seconds": ep.seconds, "task_name": ep.task_name,
               "edit_stats": getattr(ep, "edit_stats", None), "t": time.strftime("%Y-%m-%d %H:%M:%S")}
        d = out / arm
        (d / "sim").mkdir(parents=True, exist_ok=True)
        with (d / "episodes.jsonl").open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        np.savez_compressed(d / "sim" / f"ep-{i:05d}.npz", actions=np.asarray(ep.actions, np.float32),
                            **{k: v for k, v in ep.sim.items()})
        if arm in capture_arms:
            tokens = steerer.take(arm, i)
            if tokens is not None:
                (d / "tokens").mkdir(exist_ok=True)
                np.save(d / "tokens" / f"ep-{i:05d}.npy", tokens)
        print(f"EPISODE steer arm={arm} ep={i} seed={job.seed} success={ep.success} pot0_first={order['pot0_first']} "
              f"lifts={order['pot0_lift_t']},{order['pot1_lift_t']} n_steps={ep.n_steps}", flush=True)
        summary = summarize(plan, out)
        summary["session_wall_s"] = round(time.time() - t0, 1)
        (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    def on_error(job, err):
        ctx.log(f"[steer] ERROR {job.meta} {err[:500]}")
        steerer.take(job.meta["arm"], int(job.meta["ep"]))

    def keep_going():
        if ctx.hours_left() < 0.25:
            ctx.log("[steer] stop: soft cap")
            return False
        return True

    ctx.log(f"[steer] start plan_sha={plan_sha(plan)[:12]} arms={arms} queued={len(queue)}")
    v9.run_pool(ctx, [], next_job=next_job, editor=steerer, on_done=on_done, on_error=on_error,
                capture=False, keep_going=keep_going, label="steer")
    summary = summarize(plan, out)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary
