"""``ii try``: a guided first run for someone who has never used the tool.

One command, no download and no setup. It writes the practice clips for the ``fake`` model, runs
the whole pipeline on them, and opens the viewer with the playground on, already pointed at the
feature most worth looking at. A second run reuses everything and starts in a couple of seconds.

The fake model is a small synthetic video model with three planted features that really drive its
predictions and three decoys that only look busy. Because the answer is known, a newcomer can check
what the tool finds against the truth before pointing it at a real model.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.parse import quote

from interpretableintelligence.thresholds import DEFAULT_LAYER

N_CLIPS = 30
CLIP_SEED = 0

# The features planted in the practice model. Nothing else in it changes the prediction.
PLANTED = ("stripe", "pulse", "corner")


def cli_command() -> str:
    """How to type this CLI in the shell the newcomer is using.

    ``uv sync`` installs ``ii`` into the project's ``.venv``, which is not on the shell's PATH, so
    a newcomer who started with ``uv run ii`` gets "command not found" from a bare ``ii``. ``uv run``
    marks the processes it starts with ``UV_RUN_RECURSION_DEPTH``.
    """
    return "uv run ii" if os.environ.get("UV_RUN_RECURSION_DEPTH") else "ii"


def pick_feature(scoreboard: dict) -> dict | None:
    """The feature to open first: the verified one with the biggest ablation effect.

    Without a verified feature, fall back to the one with the biggest effect, so the first screen
    always has something on it.
    """
    features = [
        f for f in scoreboard.get("features") or [] if isinstance(f, dict) and "id" in f
    ]
    if not features:
        return None

    def effect(feature: dict) -> float:
        value = feature.get("ablation_effect")
        return float(value) if isinstance(value, int | float) else float("-inf")

    real = [f for f in features if f.get("real") is True]
    return max(real or features, key=lambda f: (effect(f), -int(f["id"])))


def feature_route(feature_id: int, top_clip: dict | None) -> str:
    """Viewer route that opens ``feature_id`` on the clip and frame where it fires hardest."""
    if not top_clip:
        return f"#/features/{feature_id}"
    clip_id = quote(str(top_clip["clip_id"]), safe="")
    return (
        f"#/clips/{clip_id}?frame={int(top_clip.get('frame', 0))}&feature={feature_id}"
    )


def best_clip(tops: list[dict]) -> dict | None:
    """The strongest held-out moment, else the strongest moment.

    Ablation effects are measured on held-out clips, so a held-out clip shows the feature's effect
    next to its activation.
    """
    for top in tops:
        if top.get("split") == "heldout":
            return top
    return tops[0] if tops else None


def unverified_on_frame(model, top: dict | None) -> dict | None:
    """The strongest unverified feature in the list the page opens on, to steer next for contrast.

    Naming one that is not on the screen sends a newcomer looking for something that is not there.
    """
    if not top:
        return None
    try:
        clip = model.clips.get(str(top["clip_id"])) or {}
        frame = (clip.get("frames") or [])[int(top.get("frame", 0))]
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    for feature in frame.get("features") or []:
        if feature.get("badge") != "real":
            return feature
    return None


def next_steps(chosen: dict | None, contrast: dict | None) -> list[str]:
    """What to do on the page that just opened, then how to go on to a real model."""
    lines = []
    if chosen is not None:
        label = chosen.get("label") or f"feature {chosen['id']}"
        lines.append(
            f"Start here: the page opens on feature {chosen['id']} ({label}) at the moment it fires hardest."
        )
        lines.append("The orange overlay on the frame marks where it fires.")
    lines += [
        "Things to try:",
        "  1. In the Playground panel, Remove is already chosen: press Steer. The blue squares show",
        "     where the model's prediction moved without that feature, and the panel says by how much.",
    ]
    if contrast is not None:
        name = contrast.get("label") or f"feature {contrast['id']}"
        lines.append(
            f"  2. In the list above the Playground, click {name} (marked unverified) and press Steer again."
        )
    else:
        lines.append(
            "  2. Open a feature marked unverified (the 'Features' tab lists them) and steer it the same way."
        )
    ii = cli_command()
    lines += [
        "     Compare the numbers: an unverified feature can fire just as hard yet barely move the",
        "     prediction. That is why it is not verified, however busy it looks.",
        "  3. Open 'Features' to see every feature and the checks each one passed.",
        "",
        "When you are ready for a real model, try one on real robot video (no clips needed):",
        f"  {ii} get-clips --out clips/droid",
        f"  {ii} run hf --model facebook/dinov2-small --clips clips/droid --out runs/droid --no-score --steer",
        "Or point --clips at a folder of your own videos (at least 20 clips).",
        "",
        "Press Ctrl+C to stop.",
    ]
    return lines


def _clips_ready(clips_dir: Path) -> bool:
    index = clips_dir / "clips.json"
    if not index.is_file():
        return False
    try:
        entries = json.loads(index.read_text())
    except (OSError, ValueError):
        return False
    entries = entries.get("clips", entries) if isinstance(entries, dict) else entries
    return (
        isinstance(entries, list)
        and len(entries) == N_CLIPS
        and all(
            isinstance(e, dict) and (clips_dir / str(e.get("file", ""))).is_file()
            for e in entries
        )
    )


def guided_run(
    out_dir: Path,
    clips_dir: Path,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = True,
    serve: bool = True,
    verbose: bool = False,
    say=print,
) -> dict:
    """Make clips, run the pipeline, then serve the playground on the most interesting feature.

    Returns ``{"feature": ..., "route": ...}`` (useful with ``serve=False``).
    """
    out_dir = Path(out_dir)
    clips_dir = Path(clips_dir)
    started = time.perf_counter()
    reusing = (out_dir / "scoreboard.json").is_file()

    say("")
    say("Welcome to interpretableintelligence.")
    say("")
    say(
        "You are about to look inside a small video model. It watches short clips and predicts the"
    )
    say(
        "parts of each frame it cannot see. We will find the features it uses, test which ones"
    )
    say("actually matter, and then let you switch them off yourself.")
    say("")

    if _clips_ready(clips_dir):
        say(f"1/3  Practice clips are already in {clips_dir}")
    else:
        from interpretableintelligence.adapters.fake import generate_clips

        say(f"1/3  Making {N_CLIPS} short practice clips in {clips_dir}")
        generate_clips(clips_dir, n=N_CLIPS, seed=CLIP_SEED)

    if reusing:
        say(f"2/3  Reusing your earlier run in {out_dir}")
    else:
        say("2/3  Finding features and testing them (about 15 seconds on a laptop)")
        say(
            "     We record the model's activations, train a sparse autoencoder to split them into"
        )
        say(
            "     features, then switch each feature off to see whether the prediction changes."
        )

    from interpretableintelligence.pipeline import execute_pipeline

    lines: list[str] = []
    log = say if verbose else lines.append
    try:
        execute_pipeline(
            adapter_name="fake",
            clips_dir=clips_dir,
            out_dir=out_dir,
            layer=DEFAULT_LAYER,
            serve=False,
            log=log,
        )
    except Exception:
        if lines:
            say("     The run stopped. Its log so far:")
            for line in lines:
                say(f"       {line}")
        raise

    scoreboard = json.loads((out_dir / "scoreboard.json").read_text())
    features = scoreboard.get("features") or []
    real = [f for f in features if f.get("real") is True]
    if not reusing:
        say(f"     Done in {time.perf_counter() - started:.0f}s.")
    say("")
    say(
        f"     The model has {len(features)} features. {len(real)} passed every check, so they are verified"
    )
    say("     (the page marks them 'real'):")
    for feature in sorted(real, key=lambda f: int(f["id"])):
        say(f"       feature {feature['id']}  {feature.get('label') or 'unlabelled'}")
    decoys = len(features) - len(real)
    if decoys:
        say(
            f"     The other {decoys} did not pass every check (the page marks them 'unverified'). Some fire a lot"
        )
        say(
            "     but barely change the prediction."
        )
    say(
        f"     This practice model has {len(PLANTED)} features planted on purpose ({', '.join(PLANTED)}), so you"
    )
    say(
        "     can check the tool found the right ones before you trust it on a real model."
    )
    say("")

    chosen = pick_feature(scoreboard)
    from interpretableintelligence.view import load_run

    model = load_run(out_dir, clips_dir)
    top = None
    if chosen is not None:
        detail = model.feature(int(chosen["id"])) or {}
        top = best_clip(detail.get("top_clips") or [])
    route = feature_route(int(chosen["id"]), top) if chosen is not None else ""
    contrast = unverified_on_frame(model, top)
    result = {
        "feature": chosen,
        "route": route,
        "real": [int(f["id"]) for f in real],
        "next_steps": next_steps(chosen, contrast),
    }
    if not serve:
        return result

    say("3/3  Opening the playground")
    from interpretableintelligence.steer import Steerer  # noqa: PLC0415 (loads the model)
    from interpretableintelligence.view_launch import open_in_browser, start_viewer  # noqa: PLC0415

    viewer = start_viewer(
        out_dir,
        clips_dir,
        host=host,
        port=port,
        steerer=Steerer(out_dir, clips_dir),
        model=model,
        log=lambda line: say(f"     {line}"),
    )
    url = f"{viewer.url}{route}"
    say(f"     {url}")
    say("")
    for line in result["next_steps"]:
        say(line)
    if open_browser:
        open_in_browser(url)
    viewer.wait()
    return result
