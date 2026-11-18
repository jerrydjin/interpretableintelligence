"""Carry one GR00T SAE feature into another dictionary trained on the same activations (``ii map-feature``).

Feature ids mean nothing across dictionaries: feature 6262 of the v9 capture SAE and feature 6262 of a retrained
SAE are unrelated. This finds the new dictionary's counterpart of one old feature by what it does on the same tokens,
before any rollout, so the choice cannot be tuned on outcomes.

On the capture's held-out tokens (``split.json`` ``heldout``, ``tok >= 0``) both SAEs encode every row:

* ``corr``: Pearson correlation of each new feature's activation with the old feature's activation, over all
  held-out tokens (zeros included, so firing pattern and size both count).
* ``cos``: cosine between the two decoder directions in raw activation units (``W_dec[i] * norm_scale``, per channel
  when the new SAE was trained with ``--norm channel``).

Decision rules, in order (pre-registered in the task 1028 design for feature 6262):

1. ``one``: the best ``corr`` is at least ``MATCH_CORR`` (0.7), that feature's ``cos`` is at least ``MATCH_COS``
   (0.7), and the runner-up's ``corr`` is below ``RUNNER_UP_CORR`` (0.5).
2. ``split``: a group of at most ``MAX_GROUP`` (3) new features, grown greedily from the ``POOL`` best by ``corr``,
   whose best non-negative weighted sum (least squares with an intercept) reaches multiple correlation
   ``GROUP_R`` (0.8) with the old activation, every weight positive.
3. ``none``: neither. Nothing is carried over.

With ``direction=True`` the GR00T picker (``picker/pick_candidates.py``, the same arguments ``ii rollouts --pick``
uses) also runs on the capture through each SAE, and the report gives each mapped feature's failure contrast (Cohen's
d of window-mean activation before failure vs matched times in successes, its permutation p and q). A rise test
(removing the feature raises success) only makes sense for features that fire more before failure, so the direction
check passes only when every mapped feature was tested by the picker and has ``d > 0``.

Offline and read-only: nothing here changes a capture, a run folder or a scoreboard, and a match is not a finding.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch

MAP_VERSION = 1
MATCH_CORR = 0.7
MATCH_COS = 0.7
RUNNER_UP_CORR = 0.5
GROUP_R = 0.8
MAX_GROUP = 3
POOL = 20
TOP_REPORT = 10
CHUNK_ROWS = 4096
PICKER_SEED = 0
REPORT = "mapping.json"



def _rollout_io():
    """The picker's SAE reader (one value or a per-channel ``norm_scale``, optional ``pos_mean``)."""
    from interpretableintelligence.rollout_groot_pick import PICKER_PATH

    picker_dir = PICKER_PATH.parent
    name = "ii_groot_rollout_io"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, picker_dir / "rollout_io.py")
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"picker reader not found in {picker_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_sae(sae_dir: Path):
    """A capture-layout SAE folder (``weights.safetensors`` + ``config.json``) as the picker reads it."""
    sae_dir = Path(sae_dir)
    if not (sae_dir / "weights.safetensors").is_file():
        raise FileNotFoundError(f"{sae_dir}: no weights.safetensors")
    return _rollout_io().PolicySAE(sae_dir)


def heldout_rows(capture: Path, log: Callable[[str], None] = print):
    """The capture's held-out token rows (raw float16 activations, token positions)."""
    from interpretableintelligence.adapters.groot.retrain_sae import load_capture

    _train, held = load_capture(Path(capture), torch.device("cpu"), log=log)
    return held


def _unit(rows: torch.Tensor) -> torch.Tensor:
    return rows / rows.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def correlations(old, new, feature: int, acts: torch.Tensor, tok: torch.Tensor, chunk: int = CHUNK_ROWS) -> dict:
    """Streaming Pearson correlation of every new feature with old ``feature``, plus firing rates."""
    m = int(new.M)
    n = 0
    s_a = s_aa = 0.0
    s_f = torch.zeros(m, dtype=torch.float64)
    s_ff = torch.zeros(m, dtype=torch.float64)
    s_fa = torch.zeros(m, dtype=torch.float64)
    fires = torch.zeros(m, dtype=torch.float64)
    old_fires = 0.0
    with torch.no_grad():
        for start in range(0, int(acts.shape[0]), chunk):
            x = acts[start : start + chunk].float()
            t = tok[start : start + chunk]
            a = old.encode(x, t, cols=[int(feature)])[:, 0].double()
            f = new.encode(x, t).double()
            n += int(x.shape[0])
            s_a += float(a.sum())
            s_aa += float((a * a).sum())
            old_fires += float((a > 0).sum())
            s_f += f.sum(0)
            s_ff += (f * f).sum(0)
            s_fa += (f * a[:, None]).sum(0)
            fires += (f > 0).double().sum(0)
    mean_a = s_a / n
    var_a = s_aa / n - mean_a**2
    mean_f = s_f / n
    var_f = s_ff / n - mean_f**2
    cov = s_fa / n - mean_f * mean_a
    denom = torch.sqrt(torch.clamp(var_f, min=0.0) * max(var_a, 0.0))
    corr = torch.where(denom > 1e-30, cov / denom.clamp(min=1e-30), torch.zeros_like(cov))
    if var_a <= 0:
        raise ValueError(f"old feature {feature} never varies on the held-out tokens, so nothing can be matched to it")
    return {"n": n, "corr": corr.numpy(), "rate": (fires / n).numpy(), "old_rate": old_fires / n}


def decoder_cosines(old, new, feature: int) -> np.ndarray:
    """|cos| would hide an inverted direction: this is the signed cosine of raw-unit decoder directions."""
    ref = _unit(old.raw_dir([int(feature)]).float())[0]
    dirs = _unit(new.raw_dir(list(range(int(new.M)))).float())
    return (dirs @ ref).numpy()


def _columns(old, new, feature: int, cols: list[int], acts: torch.Tensor, tok: torch.Tensor, chunk: int = CHUNK_ROWS):
    a_parts, f_parts = [], []
    with torch.no_grad():
        for start in range(0, int(acts.shape[0]), chunk):
            x = acts[start : start + chunk].float()
            t = tok[start : start + chunk]
            a_parts.append(old.encode(x, t, cols=[int(feature)])[:, 0].double())
            f_parts.append(new.encode(x, t, cols=cols).double())
    return torch.cat(a_parts).numpy(), torch.cat(f_parts).numpy()


def _fit(a: np.ndarray, columns: np.ndarray) -> tuple[float, np.ndarray]:
    """Multiple correlation of ``a`` with a least-squares weighted sum of ``columns`` (intercept included)."""
    design = np.column_stack([columns, np.ones(len(a))])
    coef, *_ = np.linalg.lstsq(design, a, rcond=None)
    fitted = design @ coef
    if np.std(fitted) <= 0 or np.std(a) <= 0:
        return 0.0, coef[:-1]
    return float(np.corrcoef(fitted, a)[0, 1]), coef[:-1]


def greedy_group(a: np.ndarray, columns: np.ndarray, ids: list[int], max_group: int = MAX_GROUP) -> list[dict]:
    """Forward selection: each step adds the feature that most raises the multiple correlation."""
    chosen: list[int] = []
    path: list[dict] = []
    for _ in range(min(int(max_group), len(ids))):
        best = None
        for j in range(len(ids)):
            if j in chosen:
                continue
            r, w = _fit(a, columns[:, chosen + [j]])
            if best is None or r > best[0]:
                best = (r, w, j)
        if best is None:
            break
        chosen.append(best[2])
        path.append({"features": [int(ids[j]) for j in chosen], "r": round(best[0], 6),
                     "weights": [round(float(v), 6) for v in best[1]]})
    return path


def decide(corr: np.ndarray, cos: np.ndarray, group_path: list[dict]) -> dict:
    """Apply the decision rules (module docstring). ``group_path`` is :func:`greedy_group`'s output."""
    order = np.argsort(-corr, kind="stable")
    best, runner = int(order[0]), int(order[1]) if len(order) > 1 else None
    runner_corr = float(corr[runner]) if runner is not None else 0.0
    if corr[best] >= MATCH_CORR and cos[best] >= MATCH_COS and runner_corr < RUNNER_UP_CORR:
        return {"outcome": "one", "features": [best],
                "reason": f"feature {best}: corr {corr[best]:.3f} >= {MATCH_CORR}, cos {cos[best]:.3f} >= {MATCH_COS}, "
                          f"runner-up corr {runner_corr:.3f} < {RUNNER_UP_CORR}"}
    for step in group_path:
        if len(step["features"]) > 1 and step["r"] >= GROUP_R and all(w > 0 for w in step["weights"]):
            return {"outcome": "split", "features": sorted(step["features"]), "weights": step["weights"],
                    "reason": f"{len(step['features'])} features reach multiple correlation {step['r']:.3f} >= {GROUP_R} "
                              "with positive weights"}
    return {"outcome": "none", "features": [],
            "reason": f"best single corr {corr[best]:.3f} (cos {cos[best]:.3f}, runner-up {runner_corr:.3f}); best group "
                      f"r {max((s['r'] for s in group_path), default=0.0):.3f}"}


def picker_contrast(capture: Path, sae_dir: Path, out: Path, features: list[int], log: Callable[[str], None] = print) -> dict:
    """Run the GR00T picker on ``capture`` read through ``sae_dir``; each requested feature's contrast row (or why
    it was not tested)."""
    from interpretableintelligence.rollout_groot_pick import run_picker

    log(f"picker on {capture} through {sae_dir}")
    run = run_picker(Path(capture), Path(sae_dir), Path(out), seed=PICKER_SEED, top=20)
    rows = {}
    for line in (Path(out) / "contrast.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[int(row["feature_id"])] = row
    result = {}
    for fid in features:
        row = rows.get(int(fid))
        if row is None:
            result[str(int(fid))] = {"tested": False, "note": "excluded by the picker (dead, dense, time- or "
                                     "token-index, or too rare in the windows); see its log"}
        else:
            result[str(int(fid))] = {"tested": True, **{k: row[k] for k in ("d", "d_fire", "p", "q", "firing_rate",
                                                                             "mean_fail", "mean_succ")}}
    return {"features": result, "argv": run["argv"], "n_tested": len(rows)}


def map_feature(
    capture: Path,
    from_sae: Path,
    to_sae: Path,
    feature: int,
    out: Path,
    *,
    direction: bool = True,
    log: Callable[[str], None] = print,
) -> dict:
    """Write ``<out>/mapping.json`` and return it (module docstring)."""
    out = Path(out)
    report_path = out / REPORT
    if report_path.exists():
        raise FileExistsError(f"{report_path} exists; use a new --out")
    old, new = load_sae(from_sae), load_sae(to_sae)
    if old.D != new.D:
        raise ValueError(f"the SAEs read different widths ({old.D} vs {new.D}); they cannot share activations")
    if not 0 <= int(feature) < int(old.M):
        raise ValueError(f"feature {feature} is not in the old SAE (0..{old.M - 1})")
    held = heldout_rows(capture, log=log)
    stats = correlations(old, new, int(feature), held.acts, held.tok)
    corr, rate = stats["corr"], stats["rate"]
    cos = decoder_cosines(old, new, int(feature))
    pool = [int(i) for i in np.argsort(-corr, kind="stable")[:POOL]]
    a, columns = _columns(old, new, int(feature), pool, held.acts, held.tok)
    path = greedy_group(a, columns, pool)
    decision = decide(corr, cos, path)
    log(f"mapping: {decision['outcome']} {decision['features']} ({decision['reason']})")
    top = [{"id": i, "corr": round(float(corr[i]), 6), "cos": round(float(cos[i]), 6), "rate": round(float(rate[i]), 6)}
           for i in pool[:TOP_REPORT]]
    report = {
        "version": MAP_VERSION,
        "capture": str(Path(capture).resolve()),
        "from_sae": str(Path(from_sae).resolve()),
        "to_sae": str(Path(to_sae).resolve()),
        "feature": int(feature),
        "tokens": {"split": "heldout", "n": int(stats["n"]), "rule": "split.json heldout episodes, tok >= 0"},
        "old_rate": round(float(stats["old_rate"]), 6),
        "rules": {"match_corr": MATCH_CORR, "match_cos": MATCH_COS, "runner_up_corr": RUNNER_UP_CORR, "group_r": GROUP_R,
                  "max_group": MAX_GROUP, "pool": POOL},
        "top": top,
        "group_path": path,
        "decision": decision,
        "direction": None,
        "note": "Offline. A match is not a finding and changes no scoreboard.",
    }
    if direction and decision["features"]:
        new_side = picker_contrast(capture, to_sae, out / "picker-to", decision["features"], log=log)
        old_side = picker_contrast(capture, from_sae, out / "picker-from", [int(feature)], log=log)
        rows = new_side["features"].values()
        passed = all(row["tested"] and float(row["d"]) > 0 for row in rows)
        report["direction"] = {
            "rule": "every mapped feature tested by the picker with d > 0 (more active before failure)",
            "passed": bool(passed),
            "to": new_side,
            "from": old_side,
        }
        log(f"direction: {'pass' if passed else 'fail'} {json.dumps(new_side['features'])}")
    out.mkdir(parents=True, exist_ok=True)
    tmp = report_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=1) + "\n")
    tmp.replace(report_path)
    return report
