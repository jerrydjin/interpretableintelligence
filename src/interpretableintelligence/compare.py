"""Match SAE features across models that watched the same clips.

``ii compare RUN_A RUN_B [RUN_C ...] --out DIR`` reads finished run folders
whose clips overlap (matched by file sha256, so the pixels are identical) and
asks which features in one model track the same moments as a feature in
another.

The unit is the clip. For each clip and feature we take the feature's largest
activation over every token the run captured for that clip, so a feature's
profile is "how strongly it fired in each moment". Two features match when
their profiles correlate (Pearson over the shared clips), each is the other's
best partner (mutual nearest neighbours), and the correlation clears a
shuffle null: the 99th percentile, over clip-order permutations, of the
largest correlation anywhere in that pair of dictionaries. That null is
family-wise, so a match is unlikely to be a lucky pairing among millions.

What a match does and does not say
    A match says two models carve out the same set of moments. It is
    correlational. Labels come from each run's own scoreboard, and only
    scoreboard-``real`` (verified) features may be used for headline claims;
    every other feature is shown as exploratory. "No counterpart" means no
    feature in the other dictionary passed the null on these clips, not that
    the other model cannot represent the concept.

Nothing here edits a run folder.
"""

from __future__ import annotations

import base64
import html
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from interpretableintelligence.artifacts import read_json, write_json
from interpretableintelligence.sae import load_sae
from interpretableintelligence.store import RowSource
from interpretableintelligence.thresholds import DEAD_ACT_EPS

COMPARE_VERSION = 1
# A feature must fire in at least this many shared clips to be compared.
MIN_ACTIVE_CLIPS = 3
N_PERMUTATIONS = 200
NULL_QUANTILE = 0.99
MIN_SHARED_CLIPS = 20


@dataclass
class RunProfile:
    """One run's per-clip feature profile over the shared clips."""

    name: str
    path: Path
    adapter: str
    layer: int
    profile: np.ndarray  # [n_shared_clips, n_features], max activation per clip
    features: list[dict]  # scoreboard records, by feature id
    usable: bool | None
    clips_dir: str | None


def short_name(adapter: str) -> str:
    """A readable model name for an adapter string."""
    known = {
        "vjepa2": "V-JEPA 2",
        "hf:facebook/dinov2-small": "DINOv2",
        "hf:facebook/dinov2-base": "DINOv2",
        "groot:nvidia/gr00t17-lerobot-libero_10-640": "GR00T N1.7",
    }
    if adapter in known:
        return known[adapter]
    return adapter.split(":", 1)[-1].split("/")[-1]


def shared_clips(manifests: list[dict]) -> list[str]:
    """sha256 of the clips every run saw, in the first run's order."""
    sets = [{clip["sha256"] for clip in manifest["clips"]} for manifest in manifests]
    common = set.intersection(*sets)
    return [clip["sha256"] for clip in manifests[0]["clips"] if clip["sha256"] in common]


def clip_profile(run: Path, manifest: dict, shas: list[str], device=None) -> np.ndarray:
    """``[len(shas), n_features]``: each feature's max activation over each clip's tokens."""
    device = device or torch.device("cpu")
    model, _config = load_sae(run / "sae", device)
    n_features = int(model.n_features)
    position = {sha: index for index, sha in enumerate(shas)}
    order = np.array([position.get(clip["sha256"], -1) for clip in manifest["clips"]], dtype=np.int64)
    out = np.zeros((len(shas), n_features), dtype=np.float32)
    with RowSource(run / "acts") as source:
        for start, stop in source.chunks(4096):
            rows = np.array(source.rows[start:stop], dtype=np.float32, copy=True)
            target = order[np.array(source.clip_idx[start:stop], dtype=np.int64)]
            keep = target >= 0
            if not np.any(keep):
                continue
            with torch.no_grad():
                acts = model.encode(torch.from_numpy(rows[keep]).to(device)).cpu().numpy()
            np.maximum.at(out, target[keep], acts)
    return out


def load_profile(run: Path, shas: list[str], device=None) -> RunProfile:
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    adapter = str(manifest["model"]["adapter"])
    profile = clip_profile(run, manifest, shas, device)
    board = run / "scoreboard.json"
    if board.is_file():
        scoreboard = read_json(board)
        features = list(scoreboard["features"])
        usable: bool | None = bool(scoreboard["sae"]["usable"])
    else:
        # Not scored yet: every feature is exploratory and unlabelled.
        features = [{"id": i, "label": "", "real": False} for i in range(profile.shape[1])]
        usable = None
    return RunProfile(
        name=short_name(adapter),
        path=run,
        adapter=adapter,
        layer=int(manifest["layer"]),
        profile=profile,
        features=features,
        usable=usable,
        clips_dir=manifest.get("clips_dir"),
    )


def _standardise(profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Columns with zero mean and unit norm, and the ids of features active often enough."""
    active = (profile > DEAD_ACT_EPS).sum(axis=0) >= MIN_ACTIVE_CLIPS
    ids = np.flatnonzero(active)
    x = profile[:, ids].astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(x, axis=0, keepdims=True)
    ok = norm[0] > 1e-12
    return x[:, ok] / norm[:, ok], ids[ok]


def match_pair(a: np.ndarray, b: np.ndarray, seed: int = 0, n_perm: int = N_PERMUTATIONS) -> dict:
    """Pearson matrix between two clip profiles, mutual best matches, and a shuffle null."""
    za, ids_a = _standardise(a)
    zb, ids_b = _standardise(b)
    if za.shape[1] == 0 or zb.shape[1] == 0:
        return {"threshold": None, "matches": [], "best_a": {}, "best_b": {}}
    corr = za.T @ zb
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for index in range(n_perm):
        null[index] = float((za[rng.permutation(za.shape[0])].T @ zb).max())
    threshold = float(np.quantile(null, NULL_QUANTILE))
    best_b_for_a = corr.argmax(axis=1)
    best_a_for_b = corr.argmax(axis=0)
    matches = []
    for i, j in enumerate(best_b_for_a):
        r = float(corr[i, j])
        if best_a_for_b[j] == i and r > threshold:
            matches.append({"a": int(ids_a[i]), "b": int(ids_b[j]), "r": r})
    matches.sort(key=lambda item: -item["r"])
    best_a = {int(ids_a[i]): {"partner": int(ids_b[j]), "r": float(corr[i, j])} for i, j in enumerate(best_b_for_a)}
    best_b = {int(ids_b[j]): {"partner": int(ids_a[i]), "r": float(corr[i, j])} for j, i in enumerate(best_a_for_b)}
    return {
        "threshold": threshold,
        "null_median": float(np.median(null)),
        "n_compared": [int(za.shape[1]), int(zb.shape[1])],
        "matches": matches,
        "best_a": best_a,
        "best_b": best_b,
    }


def _feature_card(run: RunProfile, feature_id: int) -> dict:
    record = run.features[feature_id]
    return {
        "model": run.name,
        "feature": int(feature_id),
        "label": record.get("label") or "",
        "label_holds": bool(record.get("label_holds_heldout")),
        "verified": bool(record.get("real")),
    }


def _groups(runs: list[RunProfile], pairs: dict) -> list[dict]:
    """Connected components of the mutual-match graph: one row per shared concept."""
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    edges = []
    for (i, j), result in pairs.items():
        for match in result["matches"]:
            a, b = (i, match["a"]), (j, match["b"])
            parent[find(a)] = find(b)
            edges.append((a, b, match["r"]))
    members: dict = {}
    for node in list(parent):
        members.setdefault(find(node), set()).add(node)
    groups = []
    for nodes in members.values():
        rs = [r for a, b, r in edges if a in nodes]
        cards = [_feature_card(runs[m], f) for m, f in sorted(nodes)]
        models = sorted({runs[m].name for m, _f in nodes})
        groups.append(
            {
                "models": models,
                "n_models": len(models),
                "min_r": float(min(rs)),
                "features": cards,
                "verified_in": sorted({card["model"] for card in cards if card["verified"]}),
                "all_verified": all(card["verified"] for card in cards),
            }
        )
    groups.sort(key=lambda g: (-g["n_models"], -int(g["all_verified"]), -g["min_r"]))
    return groups


def _unmatched(runs: list[RunProfile], pairs: dict) -> list[dict]:
    """Verified features with no counterpart above the null in any other model."""
    out = []
    for index, run in enumerate(runs):
        for feature_id, record in enumerate(run.features):
            if not record.get("real"):
                continue
            best = []
            matched = False
            for (i, j), result in pairs.items():
                if index not in (i, j) or result["threshold"] is None:
                    continue
                side, other = ("best_a", j) if index == i else ("best_b", i)
                partner = result[side].get(feature_id)
                if partner is None:
                    continue
                if partner["r"] > result["threshold"]:
                    matched = True
                best.append({"model": runs[other].name, "r": partner["r"], "threshold": result["threshold"]})
            if not matched:
                out.append({**_feature_card(run, feature_id), "closest": best})
    return out


def compare_runs(run_paths: list[Path], seed: int = 0, n_perm: int = N_PERMUTATIONS, log=print) -> dict:
    if len(run_paths) < 2:
        raise ValueError("compare needs at least two run folders")
    manifests = [read_json(Path(run) / "manifest.json") for run in run_paths]
    shas = shared_clips(manifests)
    if len(shas) < MIN_SHARED_CLIPS:
        raise ValueError(f"runs share {len(shas)} clips; compare needs at least {MIN_SHARED_CLIPS} identical clips")
    log(f"compare: {len(run_paths)} runs share {len(shas)} clips")
    runs = []
    for run in run_paths:
        profile = load_profile(Path(run), shas)
        log(f"compare: {profile.name} layer {profile.layer}, {profile.profile.shape[1]} features")
        runs.append(profile)
    pairs = {}
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            result = match_pair(runs[i].profile, runs[j].profile, seed=seed, n_perm=n_perm)
            log(
                f"compare: {runs[i].name} x {runs[j].name}: {len(result['matches'])} mutual matches "
                f"above r={result['threshold']}"
            )
            pairs[(i, j)] = result
    clip_ids = {clip["sha256"]: clip["clip_id"] for clip in manifests[0]["clips"]}
    groups = _groups(runs, pairs)
    for group in groups:
        group["top_clips"] = _top_clips(runs, group, shas, clip_ids)
    return {
        "compare_version": COMPARE_VERSION,
        "method": {
            "unit": "clip; a feature's value is its max activation over the clip's tokens",
            "similarity": "Pearson r over shared clips",
            "match": "mutual nearest neighbours above a shuffle null",
            "null": f"{NULL_QUANTILE:.2f} quantile of the max r over {n_perm} clip-order permutations",
            "min_active_clips": MIN_ACTIVE_CLIPS,
            "seed": seed,
        },
        "n_shared_clips": len(shas),
        "runs": [
            {
                "name": run.name,
                "path": str(run.path),
                "adapter": run.adapter,
                "layer": run.layer,
                "n_features": int(run.profile.shape[1]),
                "sae_usable": run.usable,
                "scored": run.usable is not None,
                "n_verified": int(sum(bool(f.get("real")) for f in run.features)),
            }
            for run in runs
        ],
        "pairs": [
            {
                "models": [runs[i].name, runs[j].name],
                "threshold": result["threshold"],
                "null_median": result.get("null_median"),
                "n_compared": result.get("n_compared"),
                "n_matches": len(result["matches"]),
            }
            for (i, j), result in pairs.items()
        ],
        "shared": groups,
        "unmatched_verified": _unmatched(runs, pairs),
        "clips": [clip_ids[sha] for sha in shas],
        "_profiles": runs,
        "_shas": shas,
    }


def _top_clips(runs: list[RunProfile], group: dict, shas: list[str], clip_ids: dict, k: int = 4) -> list[str]:
    """Clips where the group's features fire together most (mean rank across members)."""
    by_name = {run.name: run for run in runs}
    ranks = []
    for card in group["features"]:
        column = by_name[card["model"]].profile[:, card["feature"]]
        ranks.append(np.argsort(np.argsort(-column)))
    order = np.argsort(np.mean(ranks, axis=0))
    return [clip_ids[shas[int(index)]] for index in order[:k]]


def describe_groups(report: dict, clips_dir: Path | None, names: list[str] | None = None, top: int = 25) -> None:
    """Add a CLIP hint to each shared group: the concept whose whole-frame score best
    ranks the clips where the group fires (rank correlation with the group's mean
    rank profile). A hint is a reading aid, not a label: it is not held out and
    never makes a feature verified."""
    if clips_dir is None:
        return
    from interpretableintelligence.labelers import WindowConceptSource, load_concepts

    concepts = names or load_concepts(None)
    frames = []
    for clip_id in report["clips"]:
        path = Path(clips_dir) / f"{clip_id}.npy"
        if not path.is_file():
            return
        clip = np.load(path, mmap_mode="r")
        frames.append(np.asarray(clip[len(clip) // 2]))
    if frames[0].ndim != 3 or frames[0].shape[-1] != 3:
        return
    source = WindowConceptSource(adapter=None, names=concepts)
    scores = np.concatenate([source.frame_scores(np.stack(frames[i : i + 32])) for i in range(0, len(frames), 32)])
    concept_ranks = np.argsort(np.argsort(scores, axis=0), axis=0).astype(np.float64)
    concept_ranks -= concept_ranks.mean(axis=0)
    concept_ranks /= np.linalg.norm(concept_ranks, axis=0) + 1e-12
    by_name = {run.name: run for run in report["_profiles"]}
    for group in report["shared"][:top]:
        ranks = np.mean(
            [np.argsort(np.argsort(by_name[c["model"]].profile[:, c["feature"]])) for c in group["features"]], axis=0
        )
        ranks = ranks - ranks.mean()
        ranks /= np.linalg.norm(ranks) + 1e-12
        rho = ranks @ concept_ranks
        best = int(np.argmax(rho))
        group["clip_hint"] = {"concept": concepts[best], "rho": float(rho[best])}
        group["top_task"] = _top_task(Path(clips_dir), report["clips"], ranks)


def _top_task(clips_dir: Path, clip_ids: list[str], ranks: np.ndarray, k: int = 12) -> dict | None:
    """The instruction most of the group's top clips share, from GR00T sidecars when present."""
    from collections import Counter

    tasks = []
    for index in np.argsort(-ranks)[:k]:
        sidecar = clips_dir / f"{clip_ids[int(index)]}.groot.json"
        if not sidecar.is_file():
            return None
        tasks.append(str(json.loads(sidecar.read_text(encoding="utf-8")).get("task", "")))
    task, count = Counter(tasks).most_common(1)[0]
    return {"task": task, "share": count / len(tasks)} if task else None


# ----------------------------------------------------------------- report


def _thumb(clips_dir: Path | None, clip_id: str, size: int = 112) -> str:
    if clips_dir is None:
        return ""
    path = Path(clips_dir) / f"{clip_id}.npy"
    if not path.is_file():
        return ""
    frames = np.load(path, mmap_mode="r")
    frame = np.asarray(frames[len(frames) // 2])
    if frame.ndim != 3 or frame.shape[-1] not in (1, 3):
        return ""
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    try:
        from PIL import Image
    except ImportError:  # Pillow is optional; the page works without thumbnails
        return ""
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    image = Image.fromarray(frame).resize((size, size), Image.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def write_compare(report: dict, out: Path, clips_dir: Path | None = None) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    public = {key: value for key, value in report.items() if not key.startswith("_")}
    write_json(out / "compare.json", public)
    (out / "index.html").write_text(render_html(public, clips_dir), encoding="utf-8")


def _chip(card: dict) -> str:
    label = html.escape(card["label"]) if card["label"] else "no label"
    badge = '<span class="badge">verified</span>' if card["verified"] else '<span class="badge muted">exploratory</span>'
    return (
        f'<div class="chip"><div class="model">{html.escape(card["model"])}</div>'
        f'<div class="feat">feature {card["feature"]}</div><div class="label">{label}</div>{badge}</div>'
    )


def render_html(report: dict, clips_dir: Path | None = None) -> str:
    runs = report["runs"]
    names = [run["name"] for run in runs]
    shared = report["shared"]
    all_models = [g for g in shared if g["n_models"] == len(runs)]
    verified_groups = [g for g in shared if g["verified_in"]]
    thumbs = {}
    for group in shared[:12]:
        for clip_id in group["top_clips"]:
            thumbs.setdefault(clip_id, _thumb(clips_dir, clip_id))

    def group_html(group: dict) -> str:
        strip = "".join(
            f'<img src="{thumbs.get(c, "")}" alt="{html.escape(c)}">' for c in group["top_clips"] if thumbs.get(c)
        )
        chips = "".join(_chip(card) for card in group["features"])
        hint = group.get("clip_hint")
        task = group.get("top_task")
        if task and task["share"] >= 0.5:
            hint_html = (
                f'<p class="hint">Fires on <b>{html.escape(task["task"])}</b>'
                f'<span> ({round(task["share"] * 100)}% of its top moments)</span></p>'
            )
        elif hint and hint["rho"] > 0.2:
            hint_html = (
                f'<p class="hint">Looks like <b>{html.escape(hint["concept"])}</b>'
                f'<span> (CLIP hint, rank r = {hint["rho"]:.2f})</span></p>'
            )
        else:
            hint_html = ""
        return (
            f'<article class="group"><div class="strip">{strip}</div>{hint_html}'
            f'<div class="chips">{chips}</div>'
            f'<p class="r">Weakest link r = {group["min_r"]:.2f}</p></article>'
        )

    pair_rows = "".join(
        f"<tr><td>{html.escape(p['models'][0])} and {html.escape(p['models'][1])}</td>"
        f"<td>{p['n_matches']}</td><td>{(p['threshold'] or 0):.2f}</td></tr>"
        for p in report["pairs"]
    )
    run_rows = "".join(
        f"<tr><td>{html.escape(r['name'])}</td><td>{r['layer']}</td><td>{r['n_features']}</td>"
        f"<td>{'not scored' if r['sae_usable'] is None else ('yes' if r['sae_usable'] else 'no')}</td>"
        f"<td>{r['n_verified'] if r.get('scored', True) else 'not scored'}</td></tr>"
        for r in runs
    )
    unmatched = report["unmatched_verified"]
    unmatched_html = "".join(
        f'<li>{_chip(card)}<span class="note">closest: '
        + ", ".join(f"{html.escape(c['model'])} r = {c['r']:.2f} (needs {c['threshold']:.2f})" for c in card["closest"])
        + "</span></li>"
        for card in unmatched
    ) or "<li class='note'>None. Every verified feature has a counterpart, or no feature is verified yet.</li>"
    scope = "both models" if len(runs) == 2 else f"all {len(runs)} models"
    if not all(run.get("scored", True) for run in runs):
        tail = "None is verified yet: at least one run has not finished scoring, so treat every match as a lead."
    else:
        tail = f"{len(verified_groups)} of them include a verified feature."
    headline = f"{len(all_models)} concepts show up in {scope}. {tail}"
    shown = shared[:12]
    groups_html = "".join(group_html(g) for g in shown) or "<p class='note'>No feature pairs cleared the shuffle null.</p>"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>What robot models share</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Newsreader:opsz@6..72&family=Geist:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {{ --paper:#faf9f5; --ink:#1f1e1b; --soft:#6b6860; --line:#e6e3da; --ember:#d9653b; --cobalt:#3557c8; --card:#ffffff; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --paper:#1b1a17; --ink:#f1efe8; --soft:#a8a49a; --line:#34322d; --card:#24221f; }} }}
:root[data-theme="dark"] {{ --paper:#1b1a17; --ink:#f1efe8; --soft:#a8a49a; --line:#34322d; --card:#24221f; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font:400 16px/1.55 Geist, system-ui, sans-serif; }}
main {{ max-width:1040px; margin:0 auto; padding:56px 16px 96px; }}
h1 {{ font:400 52px/1.08 Newsreader, Georgia, serif; margin:0 0 16px; letter-spacing:-0.01em; }}
h2 {{ font:400 30px/1.2 Newsreader, Georgia, serif; margin:64px 0 16px; }}
.lede {{ font-size:21px; color:var(--soft); max-width:720px; margin:0; }}
.lede b {{ color:var(--ink); font-weight:500; }}
.groups {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(300px, 1fr)); gap:16px; }}
.group {{ background:var(--card); border:1px solid var(--line); border-radius:20px; padding:16px; }}
.strip {{ display:grid; grid-template-columns:repeat(4, 1fr); gap:4px; margin-bottom:16px; }}
.strip img {{ width:100%; aspect-ratio:1; border-radius:10px; object-fit:cover; display:block; }}
.chips {{ display:grid; gap:8px; }}
.chip {{ display:grid; grid-template-columns:auto 1fr auto; grid-template-areas:"model label badge" "feat label badge"; column-gap:12px; align-items:center; padding:8px 12px; border-radius:12px; background:var(--paper); }}
.model {{ grid-area:model; font-weight:500; font-size:13px; }}
.feat {{ grid-area:feat; color:var(--soft); font-size:13px; }}
.label {{ grid-area:label; font-family:Newsreader, Georgia, serif; font-size:16px; }}
.badge {{ grid-area:badge; font-size:13px; padding:2px 10px; border-radius:999px; background:var(--ember); color:#fff; }}
.badge.muted {{ background:transparent; color:var(--soft); border:1px solid var(--line); }}
.hint {{ font-family:Newsreader, Georgia, serif; font-size:21px; margin:0 0 12px; }}
.hint b {{ font-weight:400; color:var(--ember); }}
.hint span {{ font-family:Geist, system-ui, sans-serif; font-size:13px; color:var(--soft); }}
.r, .note {{ color:var(--soft); font-size:13px; margin:12px 0 0; }}
ul {{ list-style:none; padding:0; display:grid; gap:12px; }}
table {{ border-collapse:collapse; width:100%; font-size:16px; }}
td, th {{ text-align:left; padding:8px 12px 8px 0; border-bottom:1px solid var(--line); font-weight:400; }}
th {{ color:var(--soft); }}
.wrap {{ overflow-x:auto; }}
@media (max-width:600px) {{ h1 {{ font-size:30px; }} .lede {{ font-size:16px; }} }}
</style></head>
<body><main>
<h1>Do {html.escape(", ".join(names[:-1]))} and {html.escape(names[-1])} see the same things?</h1>
<p class="lede"><b>{headline}</b> We ran each model on the same {report["n_shared_clips"]} robot moments, trained a sparse dictionary on each, and paired features that fire on the same moments.</p>

<h2>Concepts more than one model learned</h2>
<p class="note">Each card is a set of features, one per model, that fire on the same moments more tightly than any pairing in shuffled data. The frames are the moments where they fire together most. Only verified features support a claim; exploratory ones are leads to test.</p>
<div class="groups">{groups_html}</div>

<h2>Verified features with no counterpart</h2>
<ul>{unmatched_html}</ul>

<h2>How it was measured</h2>
<div class="wrap"><table><tr><th>Model</th><th>Layer</th><th>Features</th><th>Usable dictionary</th><th>Verified</th></tr>{run_rows}</table></div>
<div class="wrap"><table style="margin-top:24px"><tr><th>Pair</th><th>Matches</th><th>Null r (99th pct)</th></tr>{pair_rows}</table></div>
<p class="note">A feature's value in a clip is its largest activation there. Two features match when each is the other's best partner by Pearson r over the shared clips and r beats the 99th percentile of the best r found after shuffling clip order. A match is correlational. "No counterpart" means nothing passed on these clips, not that the model lacks the concept.</p>
</main></body></html>
"""
