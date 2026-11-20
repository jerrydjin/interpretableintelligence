"""GR00T ``ii rollouts --pick``: one picker for the trained SAE and every untrained one (research rule 4).

The picker is ``picker/pick_candidates.py`` (failure contrast on a capture run, then an offline prune), the one
``rollout_pick`` names as GR00T's. It runs on the capture once with the trained SAE and once per random-weights SAE
(``adapters/groot/random_sae.py``), with the same arguments and seed, and each run's ranked list is cut the same way:

1. the picker's ``candidate_ids`` (features that passed its gates), in its order, then
2. every other tested feature in ``contrast.jsonl`` order (permutation p, then larger |Cohen's d|),

and the first ``n`` are the picks. Taking the ranked list, not only gated candidates, means an untrained SAE always
yields ``n`` arms even when nothing in it passes the picker's own gates, so the control is never skipped.

The candidates' matched controls are the picker's firing-rate matched random features of the trained SAE
(``offline.jsonl``), taken round robin across candidates. Everything lands in ``<run>/rollouts/``: the plan
(``ablate_plan.json``, ``groot-ablate-plan-v1`` with a ``dictionary`` per condition), one SAE subset file per
dictionary under ``sae/``, and each picker run under ``pick/``. A rerun with the same arguments reuses the plan.

The design is a fixed budget: every condition runs ``episodes`` paired episodes, no SPRT stops (``sprt: false``).
The plan's ``sprt`` block still carries a Sobel-Wald design (``rise_sprt``): ``p0`` is the capture's success rate and
``p1 = p0 +/- RISE_SPRT_EFFECT``, the 1028 design's spacing (0.56 vs 0.86 / 0.26). ``summarize_groot`` reads it on
each arm's outcomes in episode order for research rule 1, so ``finding`` needs the rise call too.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

PICKER = "groot_failure_contrast"
PICKER_PATH = Path(__file__).resolve().parent.parent.parent / "picker" / "pick_candidates.py"
# pick_groot_pot0 (the picker behind the 1028 candidates) windowed 48 env steps = 3 policy calls before the anchor.
PICKER_ARGS: tuple[str, ...] = ("--W", "48", "--match", "time", "--quiet")
PICK_DIR = "pick"
SAE_DIR = "sae"
PLAN_FILE = "ablate_plan.json"
RECON_EPISODES = 8
PLAN_LAYER = 5
RISE_SPRT_EFFECT = 0.30
RISE_SPRT_ALPHA = 0.05
RISE_SPRT_BETA = 0.05


def rise_sprt(p0: float, cap: int) -> dict:
    """The Sobel-Wald design rule 1 is read with: H0 p = p0 against p0 +/- RISE_SPRT_EFFECT (clipped to (0, 1))."""
    if not 0.0 < p0 < 1.0:
        raise ValueError(f"the capture's success rate must be strictly between 0 and 1 for an SPRT, got {p0}")
    return {"p0": round(float(p0), 6), "p1_rise": round(min(p0 + RISE_SPRT_EFFECT, 0.99), 6),
            "p1_fall": round(max(p0 - RISE_SPRT_EFFECT, 0.01), 6), "alpha": RISE_SPRT_ALPHA, "beta": RISE_SPRT_BETA,
            "cap": int(cap), "stop_rule": "h1_or_both_h0"}


def min_rise_episodes(sprt: dict) -> int | None:
    """Fewest episodes in which the design can call a rise (every one a success), or None if it cannot by ``cap``."""
    from interpretableintelligence.rollout_stats import sequential_decision

    for n in range(1, int(sprt["cap"]) + 1):
        if sequential_decision([True] * n, sprt)["verdict"] == "rise":
            return n
    return None


def capture_success_rate(manifest: dict) -> float:
    episodes = manifest.get("episodes") or []
    flags = [bool(item["success"]) for item in episodes if item.get("success") is not None]
    if not flags:
        raise ValueError("the capture manifest lists no episode outcomes: the rule 1 SPRT needs its success rate as p0")
    return sum(flags) / len(flags)


def _load_picker():
    spec = importlib.util.spec_from_file_location("ii_groot_pick_candidates", PICKER_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"GR00T picker not found at {PICKER_PATH}")
    picker_dir = str(PICKER_PATH.parent)
    if picker_dir not in sys.path:
        sys.path.insert(0, picker_dir)  # it imports rollout_io from its own folder
    import torch

    grad = torch.is_grad_enabled()
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # the picker turns autograd off at import
    finally:
        torch.set_grad_enabled(grad)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def capture_info(capture: Path) -> dict:
    """What the plan needs from a capture run folder (``groot-1028-v9-n100`` layout)."""
    capture = Path(capture)
    manifest = _read_json(capture / "manifest.json")
    weights = capture / "sae" / "weights.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"{capture}: no sae/weights.safetensors (the trained SAE the picker reads)")
    config_path = capture / "sae" / "config.json"
    config = _read_json(config_path) if config_path.is_file() else {}
    k = config.get("k")
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"{config_path}: needs the SAE's k (BatchTopK) to match an untrained SAE's sparsity")
    split_path = capture / "split.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"{capture}: no split.json (the untrained SAE's threshold is set on the train split)")
    split = _read_json(split_path)
    return {
        "manifest": manifest,
        "weights": weights,
        "k": int(k),
        "train": sorted(int(item) for item in split["train"]),
        "layer": int(manifest.get("layer", PLAN_LAYER)),
    }


def train_chunks(capture: Path, train: list[int]):
    """Raw activation rows of the train-split episodes, one shard at a time (float32 tensors [n, D])."""
    import torch
    from safetensors.torch import load_file

    wanted = set(int(item) for item in train)
    shards = sorted((Path(capture) / "acts").glob("shard-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"{capture}: no acts/shard-*.safetensors")
    for shard in shards:
        tensors = load_file(str(shard))
        episode = tensors["episode_idx"].long()
        keep = torch.tensor([int(item) in wanted for item in episode.tolist()], dtype=torch.bool)
        tok = tensors.get("tok")
        if tok is not None:
            keep &= tok.long() >= 0
        if keep.any():
            yield tensors["acts"][keep].float()


def _picker_view(capture: Path, sae_dir: Path, view: Path) -> Path:
    """A run folder that is the capture with ``sae/`` swapped, so the picker reads another dictionary."""
    view.mkdir(parents=True, exist_ok=True)
    for entry in Path(capture).iterdir():
        if entry.name in ("sae", "picker"):
            continue
        link = view / entry.name
        if not link.exists() and not link.is_symlink():
            os.symlink(entry.resolve(), link)
    link = view / "sae"
    if link.is_symlink() or link.exists():
        if link.resolve() != Path(sae_dir).resolve():
            raise RuntimeError(f"{link} points at {link.resolve()}, expected {sae_dir}")
    else:
        os.symlink(Path(sae_dir).resolve(), link)
    return view


def run_picker(capture: Path, sae_dir: Path, out: Path, *, seed: int, top: int) -> dict:
    """Run the GR00T picker on ``capture`` read through ``sae_dir``. Output files go to ``out``.

    The picker sets torch's thread count; it is handed the current count so the process's BLAS setup (and with it
    float reduction order in later work) does not change.
    """
    import torch

    picker = _load_picker()
    view = _picker_view(capture, sae_dir, out / "run")
    threads = torch.get_num_threads()
    argv = [str(view), "--out", str(out), "--seed", str(int(seed)), "--top", str(int(top)), "--threads", str(int(threads)), *PICKER_ARGS]
    args = picker.args_parser().parse_args(argv)
    grad = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(False)
        result = picker.pick(str(view), args)
    finally:
        torch.set_grad_enabled(grad)
        if torch.get_num_threads() != threads:
            torch.set_num_threads(threads)
    return {"result": result, "argv": argv[1:]}


def ranked_features(out: Path, result: dict) -> list[int]:
    """Gated candidates first, then every other tested feature in contrast order (see the module docstring)."""
    order = [int(item) for item in result.get("candidate_ids") or []]
    seen = set(order)
    for line in (Path(out) / "contrast.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        feature = int(json.loads(line)["feature_id"])
        if feature not in seen:
            order.append(feature)
            seen.add(feature)
    return order


def matched_controls(out: Path, picks: list[int], count: int) -> list[int]:
    """The trained picker's firing-rate matched controls of ``picks``, round robin, distinct, ``count`` of them."""
    offline = {}
    for line in (Path(out) / "offline.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            offline[int(row["feature_id"])] = [int(item) for item in row["controls"]]
    missing = [item for item in picks if item not in offline]
    if missing:
        raise ValueError(f"picker has no matched controls for {missing}; raise --top")
    chosen: list[int] = []
    depth = 0
    while len(chosen) < count and depth < max(len(offline[item]) for item in picks):
        for item in picks:
            pool = offline[item]
            if depth < len(pool) and pool[depth] not in chosen and pool[depth] not in picks:
                chosen.append(pool[depth])
                if len(chosen) == count:
                    break
        depth += 1
    if len(chosen) < count:
        raise ValueError(f"only {len(chosen)} matched controls available, asked for {count}")
    return chosen


def _subset(full: Path, latents: list[int], out: Path) -> str:
    from interpretableintelligence.adapters.groot.v9 import ablate_v9

    out.parent.mkdir(parents=True, exist_ok=True)
    return ablate_v9.write_subset(full, sorted(set(latents)), out)


def build_pick_plan(
    run_dir: Path,
    capture: Path,
    *,
    n: int,
    random_seeds: list[int],
    controls: int,
    episodes: int,
    seed: int,
    log=print,
) -> dict:
    """Pick with every dictionary, write SAE subsets, and write ``<run>/rollouts/ablate_plan.json``. Returns the plan.

    A plan already on disk is reused when it was built from the same capture SAE and arguments; otherwise this refuses
    to overwrite it.
    """
    from interpretableintelligence.adapters.groot import random_sae
    from interpretableintelligence.adapters.groot.v9 import ablate_v9

    if n < 1:
        raise ValueError(f"--pick must be >= 1, got {n}")
    rollouts = Path(run_dir) / "rollouts"
    plan_path = rollouts / PLAN_FILE
    info = capture_info(capture)
    request = {
        "capture": str(Path(capture).resolve()),
        "capture_sae_sha256": _sha256(info["weights"]),
        "n": int(n),
        "random_sae_seeds": [int(item) for item in random_seeds],
        "controls": int(controls),
        "episodes": int(episodes),
        "seed": int(seed),
        "picker": PICKER,
        "picker_args": list(PICKER_ARGS),
    }
    if plan_path.is_file():
        plan = _read_json(plan_path)
        if plan.get("pick", {}).get("request") != request:
            raise ValueError(f"{plan_path} was built with other arguments; use a new run folder")
        log(f"reusing {plan_path}")
        return plan

    top = max(20, int(n))
    picks_dir = rollouts / PICK_DIR
    log(f"picking {n} with the trained SAE from {capture}")
    trained_out = picks_dir / "trained"
    trained = run_picker(capture, info["weights"].parent, trained_out, seed=seed, top=top)
    trained_ranked = ranked_features(trained_out, trained["result"])
    if len(trained_ranked) < n:
        raise ValueError(f"the picker tested only {len(trained_ranked)} trained features, fewer than --pick {n}")
    candidates = sorted(trained_ranked[:n])
    control_ids = matched_controls(trained_out, candidates, int(controls)) if controls else []

    dictionaries: dict[str, dict] = {}
    sae_dir = rollouts / SAE_DIR
    trained_subset = sae_dir / "trained.safetensors"
    dictionaries["trained"] = {
        "subset_file": f"{SAE_DIR}/trained.safetensors",
        "subset_sha256": _subset(info["weights"], candidates + control_ids, trained_subset),
        "full_sha256": request["capture_sae_sha256"],
        "picks": candidates,
        "picker_candidate_ids": [int(item) for item in trained["result"].get("candidate_ids") or []],
    }
    random_picks: list[list[int]] = []
    for item in random_seeds:
        name = f"random:{int(item)}"
        log(f"building untrained SAE {name} and picking {n} with the same picker")
        weights, record = random_sae.build_untrained_sae(
            info["weights"], int(item), lambda: train_chunks(capture, info["train"]), k=info["k"]
        )
        own_dir = picks_dir / f"random-{int(item)}"
        full = random_sae.save_sae_dir(weights, own_dir / "sae", {"arch": "batchtopk_untrained", **record})
        result = run_picker(capture, full.parent, own_dir, seed=seed, top=top)
        ranked = ranked_features(own_dir, result["result"])
        if len(ranked) < n:
            raise ValueError(f"the picker tested only {len(ranked)} features of {name}, fewer than --pick {n}")
        picks = sorted(ranked[:n])
        random_picks.append(picks)
        subset = sae_dir / f"random-{int(item)}.safetensors"
        dictionaries[name] = {
            "subset_file": f"{SAE_DIR}/random-{int(item)}.safetensors",
            "subset_sha256": _subset(full, picks, subset),
            "full_sha256": _sha256(full),
            "picks": picks,
            "picker_candidate_ids": [int(x) for x in result["result"].get("candidate_ids") or []],
            **{key: record[key] for key in ("seed", "k", "threshold", "calibration_rows", "init", "trained_threshold")},
        }
        full.unlink()  # the subset is what runs; the full untrained SAE is rebuilt from the seed if needed

    manifest = info["manifest"]
    sprt = rise_sprt(capture_success_rate(manifest), int(episodes))
    shortest = min_rise_episodes(sprt)
    if shortest is None:
        log(f"warning: with {episodes} episodes per arm the rule 1 SPRT (p0 {sprt['p0']}) can never call a rise, "
            "so nothing can be a finding; raise --episodes")
    else:
        log(f"rule 1 SPRT: p0 {sprt['p0']}, p1_rise {sprt['p1_rise']}; a rise call needs at least {shortest} episodes")
    conditions = [
        {"id": "baseline", "arm": "baseline", "kind": "baseline", "latents": None, "n_max": int(episodes), "sprt": False, "dir": "baseline"},
        {"id": "recon-only", "arm": "baseline", "kind": "recon_only", "latents": [], "n_max": min(int(episodes), RECON_EPISODES),
         "sprt": False, "dir": "baseline/recon-only", "dictionary": "trained"},
    ]
    for feature in candidates:
        conditions.append({"id": f"cand-{feature}", "arm": "pick", "kind": "candidate", "latents": [feature], "n_max": int(episodes),
                           "sprt": False, "dir": f"pick/cand-{feature}", "dictionary": "trained"})
    for index, feature in enumerate(control_ids):
        conditions.append({"id": f"ctrl{index + 1:02d}-{feature}", "arm": "pick", "kind": "control", "latents": [feature],
                           "n_max": int(episodes), "sprt": False, "dir": f"pick/ctrl{index + 1:02d}-{feature}", "dictionary": "trained"})
    for item, picks in zip(random_seeds, random_picks, strict=True):
        for feature in picks:
            conditions.append({"id": f"rsae-{int(item)}-{feature}", "arm": "random_sae", "kind": "random_sae", "latents": [feature],
                               "n_max": int(episodes), "sprt": False, "dir": f"random-sae/{int(item)}/{feature}",
                               "dictionary": f"random:{int(item)}"})
    plan = {
        "schema": ablate_v9.PLAN_SCHEMA,
        "version": ablate_v9.ABLATE_VERSION,
        "unvalidated": True,
        "real": False,
        "run_id": Path(run_dir).name,
        "task": {"suite": "plus", "index": int(manifest["task_index"]), "id": int(manifest["task_id"]),
                 "name": manifest.get("task_name"), "n_init_states": int(manifest["n_init_states"])},
        "hook": {"module": f"_groot_model.action_head.model.transformer_blocks.{info['layer']}", "layer": info["layer"],
                 "denoise_steps": "final", "tokens": "all", "stat_tokens": manifest.get("token_indices"),
                 "rule": "out = x - norm_scale * sum_{i in S} f_i W_dec[i]; f = JumpReLU_thr(((x-norm_mu)/norm_scale - b_dec) W_enc + b_enc); "
                         "the same rule for every dictionary"},
        "sae": {"subset_file": dictionaries["trained"]["subset_file"], "subset_sha256": dictionaries["trained"]["subset_sha256"],
                "full_sha256": request["capture_sae_sha256"], "d_in": int(manifest.get("d_model") or 0) or None,
                "dictionaries": dictionaries},
        "seeds": {"rule": "seed = base + i; init_state_id = i % n_init_states", "base": int(seed),
                  "noise": "v9 per-episode torch.Generator(seed) routed into GR00T sample_noise"},
        # No condition stops early (fixed budget, every condition has sprt: false). The design is read afterwards on
        # each arm's outcomes in episode order for research rule 1 (summarize_groot's sprt_verdict).
        "sprt": {**sprt, "use": "rule 1 rise call, read on each arm's outcomes in episode order; no arm stops early",
                 "p0_source": "capture success rate"},
        "labels": "pot_events rule pot0-v1 via physint_inproc_v9.label_episode",
        "conditions": conditions,
        "schedule": {"rule": "lockstep: each free env slot gets the next episode index of the active condition with the fewest "
                             "episodes dispatched (ties: `order`)",
                     "order": [item["id"] for item in conditions]},
        "pick": {"request": request, "picker": PICKER, "picker_path": "picker/pick_candidates.py",
                 "ranking": "candidate_ids, then contrast.jsonl order; first n", "trained_argv": trained["argv"],
                 "candidates": candidates, "controls": control_ids,
                 "random_sae_seeds": [int(item) for item in random_seeds], "random_sae_feature_ids": random_picks},
    }
    rollouts.mkdir(parents=True, exist_ok=True)
    tmp = plan_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(plan, indent=1) + "\n")
    tmp.replace(plan_path)
    log(f"wrote {plan_path}: candidates {candidates}, controls {control_ids}, random-SAE picks {random_picks}")
    return plan


def dictionary_paths(run_dir: Path, plan: dict) -> dict[str, Path]:
    """{dictionary: subset file} for ``ablate_v9.run_ablation``; ``trained`` is passed as ``sae_path``."""
    rollouts = Path(run_dir) / "rollouts"
    return {name: rollouts / entry["subset_file"] for name, entry in plan["sae"]["dictionaries"].items()}
