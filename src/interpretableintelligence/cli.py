"""interpretableintelligence command line."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from interpretableintelligence import __version__
from interpretableintelligence.thresholds import (
    DEFAULT_HELDOUT_FRAC,
    DEFAULT_LAYER,
    DEFAULT_PREFILTER_K,
    DEFAULT_ROLLOUT_ALPHA,
    DEFAULT_ROLLOUT_BATCH,
    DEFAULT_ROLLOUT_CONTROLS,
    DEFAULT_ROLLOUT_EPISODES,
    DEFAULT_ROLLOUT_MIN_EPISODES,
    DEFAULT_ROLLOUT_WORKERS,
    DEFAULT_RANDOM_SAE,
    DEFAULT_SEED,
    MIN_CLIPS,
    PROBE_BOOTSTRAP,
    PROBE_L2,
    PROBE_MIN_LEAD_S,
    PICK_DISCOVERY_EPISODES,
    STEER_BOOST_FACTOR,
    STEER_BOOTSTRAP,
    STEER_CONTROLS,
    STEER_TOP_FEATURES,
)

if TYPE_CHECKING:
    from interpretableintelligence.explain import ExplainError

# Command modules are imported inside each command. torch and transformers take
# over a second to import, and `ii --help`, `ii view` and `ii validate` on a
# finished run should not pay for them.

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Mechanistic interpretability for physical AI models.")


def _version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Print the tool version and exit."),
    ] = False,
) -> None:
    """Shared run format and scoring core."""


@app.command("try")
def try_cmd(
    out: Annotated[Path, typer.Option("--out", help="Run folder for the practice run.")] = Path("runs/try"),
    clips: Annotated[Path, typer.Option("--clips", help="Where to write the practice clips.")] = Path("clips/try"),
    host: Annotated[str, typer.Option("--host", help="Viewer interface.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Viewer port. 0 chooses a free port.")] = 0,
    no_open: Annotated[bool, typer.Option("--no-open", help="Do not open a browser.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", help="Show the full pipeline log.")] = False,
) -> None:
    """New here? Start with this. Finds features in a small practice model and opens the playground.

    No download and no setup. Takes about 15 seconds the first time and a couple of seconds after.
    """
    from interpretableintelligence.first_run import guided_run

    try:
        guided_run(out, clips, host=host, port=port, open_browser=not no_open, verbose=verbose, say=typer.echo)
    except (ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("generate-clips")
def generate_clips_cmd(
    out: Annotated[Path, typer.Option("--out", help="Directory to write synthetic clips into.")],
    n: Annotated[int, typer.Option("--n", min=1, help="Number of clips.")] = 30,
    seed: Annotated[int, typer.Option("--seed", help="Generator seed.")] = 0,
    failure_labels: Annotated[
        str | None,
        typer.Option(
            "--failure-labels",
            help="Also write failures.json for `ii probe`: 'planted' (fails when the stripe peak is in the first half) or 'null' (seeded coin, no signal).",
        ),
    ] = None,
) -> None:
    """Write deterministic synthetic clips for the fake adapter."""
    from interpretableintelligence.adapters.fake import FAILURE_LABELS_FILE, generate_clips
    from interpretableintelligence.adapters.fake import failure_labels as make_labels
    from interpretableintelligence.artifacts import write_json

    try:
        labels = None if failure_labels is None else make_labels(n, seed, failure_labels)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    generate_clips(out, n=n, seed=seed)
    typer.echo(f"wrote {n} clips to {out}")
    if labels is not None:
        write_json(Path(out) / FAILURE_LABELS_FILE, labels)
        failed = sum(1 for episode in labels["episodes"] if episode["failure"])
        typer.echo(f"wrote {failure_labels} failure labels ({failed}/{n} fail) to {Path(out) / FAILURE_LABELS_FILE}")


@app.command("get-clips")
def get_clips_cmd(
    out: Annotated[Path, typer.Option("--out", help="Directory to write the clips into.")] = Path("clips/droid"),
    n: Annotated[int, typer.Option("--n", min=1, max=100, help="Number of clips, one per robot episode. `ii run` needs at least 20.")] = 100,
    seconds: Annotated[float, typer.Option("--seconds", min=0.5, help="Length of each clip.")] = 4.0,
    with_states: Annotated[
        bool,
        typer.Option(
            "--with-states",
            help="Also save each clip's per-frame arm pose as <clip>.states.npy (for `ii action-effects`).",
        ),
    ] = False,
) -> None:
    """Download real robot video to try a real model on: short clips from DROID (MIT licence).

    One 175 MB download, cached after the first time. Each clip is cut from the middle of a robot
    episode in a real kitchen or office and saved as an mp4.
    """
    from interpretableintelligence.first_run import cli_command
    from interpretableintelligence.sample_clips import DATASET, LICENSE, fetch_droid_clips

    try:
        entries = fetch_droid_clips(out, n=n, seconds=seconds, say=typer.echo, with_states=with_states)
    except (ValueError, RuntimeError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        if "connection" in str(exc).lower() or "internet" in str(exc).lower():
            typer.echo(
                "The first download needs internet access to huggingface.co. Later runs read the cached copy.",
                err=True,
            )
        raise typer.Exit(code=1) from exc
    ii = cli_command()
    typer.echo(f"wrote {len(entries)} clips from {DATASET} ({LICENSE} licence) to {out}")
    typer.echo("")
    if len(entries) < MIN_CLIPS:
        typer.echo(f"Note: `{ii} run` needs at least {MIN_CLIPS} clips. Run this again with --n {MIN_CLIPS} or more first.")
        typer.echo("")
    typer.echo("Next, find the features a real vision model uses on them, and open the playground:")
    typer.echo(f"  {ii} run hf --model facebook/dinov2-small --clips {out} --out runs/droid --no-score --steer")
    typer.echo("That takes about 10 minutes on a laptop CPU (it downloads DINOv2-small, about 90 MB, once).")
    typer.echo("Every feature is explorable, but none is verified yet: --no-score skips the checks.")
    typer.echo("To run the checks that mark features verified, run the same command without --no-score.")
    typer.echo("It reuses what is done and adds the scoring, which takes longer (about 40 minutes on a 4-core CPU).")


def _cli() -> str:
    from interpretableintelligence.first_run import cli_command

    return cli_command()


def _adapter_name(adapter: str, model: str | None, blocks: str | None, tiny: bool = False) -> str:
    """``hf`` plus ``--model``/``--blocks`` becomes the self-describing ``hf:<model>[#<blocks>]`` name.

    ``groot`` becomes ``groot:<checkpoint>`` (``--model``, default the LIBERO checkpoint), or
    ``groot:tiny`` with ``--tiny``.
    """
    if adapter == "groot":
        from interpretableintelligence.adapters.groot.adapter import make_spec as groot_spec

        if blocks:
            raise ValueError("--blocks only applies to the hf adapter")
        if model and Path(model).expanduser().is_dir():
            model = str(Path(model).expanduser().resolve())
        return groot_spec(model, tiny=tiny)
    if adapter == "hf":
        from interpretableintelligence.adapters.hf import make_spec

        if not model:
            raise ValueError("the hf adapter needs --model, e.g. --model facebook/dinov2-small")
        local = Path(model).expanduser()
        if local.is_dir():
            model = str(local.resolve())  # explain and the playground reload it from any cwd
        return make_spec(model, blocks)
    if model or blocks:
        raise ValueError("--model and --blocks only apply to the hf and groot adapters")
    return adapter


def _default_layer(adapter: str) -> int:
    if adapter.startswith("groot:"):
        from interpretableintelligence.adapters.groot.adapter import DEFAULT_LAYER as GROOT_LAYER

        return GROOT_LAYER
    if adapter.startswith("hf:"):
        from interpretableintelligence.adapters.hf import middle_block

        return middle_block(adapter)
    return DEFAULT_LAYER


@app.command()
def run(
    adapter: Annotated[
        str,
        typer.Argument(help="Adapter: fake, vjepa2, hf (any Hugging Face vision model; pass --model), or groot (NVIDIA GR00T N1.7)."),
    ],
    clips: Annotated[Path, typer.Option("--clips", file_okay=False, help="Clip directory (at least 20 clips).")],
    out: Annotated[Path, typer.Option("--out", help="Run folder, e.g. runs/demo.")],
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="hf: Hugging Face model id or local directory, e.g. facebook/dinov2-small. "
            "groot: LeRobot GR00T checkpoint (default nvidia/gr00t17-lerobot-libero_10-640).",
        ),
    ] = None,
    blocks: Annotated[
        str | None,
        typer.Option("--blocks", help="hf only: dotted path to the transformer block list, when auto-detection picks the wrong one."),
    ] = None,
    layer: Annotated[
        int | None,
        typer.Option("--layer", min=0, help=f"Layer to instrument. Default {DEFAULT_LAYER}; for hf, the middle block; for groot, DiT block 5."),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Split, SAE, and random-feature seed.")] = DEFAULT_SEED,
    heldout_frac: Annotated[
        float,
        typer.Option("--heldout-frac", help="Fraction of clips held out."),
    ] = DEFAULT_HELDOUT_FRAC,
    prefilter: Annotated[
        int | None,
        typer.Option(
            "--prefilter",
            min=1,
            help="Ablate and run random-feature controls only for the K features with the largest attribution. Omit to score every feature.",
        ),
    ] = None,
    extra_effects: Annotated[
        str,
        typer.Option(
            "--extra-effects",
            help="Comma-separated clip_effects modes to add: loss, group. The cosine scoreboard stays.",
        ),
    ] = "",
    sae_features: Annotated[
        int | None,
        typer.Option(
            "--sae-features",
            min=1,
            help="SAE dictionary size. Default 12 for fake and vjepa2; 8 per activation dimension for hf. Above 64, a TopK SAE.",
        ),
    ] = None,
    concepts: Annotated[
        Path | None,
        typer.Option(
            "--concepts",
            exists=True,
            dir_okay=False,
            help="hf, vjepa2, groot: text file of concepts to label features with, one per line. Default: a built-in list of everyday objects, parts and textures.",
        ),
    ] = None,
    sae_from: Annotated[
        Path | None,
        typer.Option(
            "--sae-from",
            exists=True,
            file_okay=False,
            help="Copy this frozen SAE into the run instead of training one, so feature ids from elsewhere keep their meaning: "
            "a run folder, a GR00T v9 capture or retrain-sae folder, or their sae/ directory. Needs --no-score.",
        ),
    ] = None,
    tiny: Annotated[
        bool,
        typer.Option(
            "--tiny",
            help="CPU smoke path, no weight download. vjepa2: tiny random config (pass --layer 1). "
            "groot: a GR00T-shaped random network (not GR00T).",
        ),
    ] = False,
    gpu_cost_per_hour: Annotated[
        float | None,
        typer.Option(
            "--gpu-cost-per-hour",
            help="Price of one GPU-hour. When set, the report includes cost per hour of footage.",
        ),
    ] = None,
    no_score: Annotated[
        bool,
        typer.Option("--no-score", help="Stop after feature candidates. The viewer still starts."),
    ] = False,
    no_serve: Annotated[
        bool,
        typer.Option("--no-serve", help="Write the run folder and exit without starting the viewer."),
    ] = False,
    host: Annotated[str, typer.Option("--host", help="Viewer interface.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Viewer port. 0 chooses a free port.")] = 0,
    no_open: Annotated[bool, typer.Option("--no-open", help="Do not open a browser.")] = False,
    steer: Annotated[
        bool,
        typer.Option("--steer", help="Turn on the playground in the viewer: edit features and watch the prediction move."),
    ] = False,
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache",
            help="Run every forward pass even when the activation cache has it (see `ii cache`). Nothing is read from or written to the cache.",
        ),
    ] = False,
    remote: Annotated[
        bool,
        typer.Option(
            "--remote",
            help="Run on a hosted GPU (Modal, your account) and stream results into the local folder and viewer. "
            "Needs `uv sync --extra remote` and `modal setup` once. See docs/REMOTE.md.",
        ),
    ] = False,
    gpu: Annotated[
        str | None,
        typer.Option("--gpu", help="--remote only: GPU type, e.g. T4, L4, A10, L40S, A100-80GB, H100. Default L4."),
    ] = None,
    max_cost: Annotated[
        float | None,
        typer.Option(
            "--max-cost",
            help="--remote only: spending cap in USD. Becomes the container's timeout at the GPU's price. Default 2.00.",
        ),
    ] = None,
) -> None:
    """Capture, train, extract features, score what is missing, then serve the viewer.

    A stage whose outputs are already valid is skipped. The run prints a cost report
    from measured stage time. Pass --gpu-cost-per-hour to price an hour of footage.
    With --remote the same stages run on a hosted GPU and the folder fills in as each finishes.
    """
    from interpretableintelligence.remote.backends import RemoteSetupError

    modes = tuple(part.strip() for part in extra_effects.split(",") if part.strip())
    if not clips.is_dir():
        typer.echo(f"error: clips folder not found: {clips}", err=True)
        maker = "generate-clips" if adapter == "fake" else "get-clips"
        typer.echo(
            f"Make one with `{_cli()} {maker} --out {clips}`, or pass a folder of your own videos (at least 20).",
            err=True,
        )
        raise typer.Exit(code=2)
    try:
        if not remote and (gpu is not None or max_cost is not None):
            raise ValueError("--gpu and --max-cost only apply with --remote")
        adapter = _adapter_name(adapter, model, blocks, tiny=tiny)
        if adapter.startswith("groot:"):
            tiny = False  # groot:tiny names the stand-in itself, so the run folder can rebuild it
        layer = _default_layer(adapter) if layer is None else layer
        if remote and sae_from is not None:
            raise ValueError("--sae-from is not available with --remote yet")
        if remote:
            from interpretableintelligence.remote.client import execute_remote
            from interpretableintelligence.remote.pricing import DEFAULT_GPU, DEFAULT_MAX_COST

            execute_remote(
                adapter_name=adapter,
                clips_dir=clips,
                out_dir=out,
                layer=layer,
                seed=seed,
                heldout_frac=heldout_frac,
                tiny=tiny,
                prefilter=prefilter,
                extra_effects=modes,
                gpu_cost_per_hour=gpu_cost_per_hour,
                sae_features=sae_features,
                concepts_file=concepts,
                score=not no_score,
                serve=not no_serve,
                host=host,
                port=port,
                open_browser=not no_open,
                gpu=gpu or DEFAULT_GPU,
                max_cost=DEFAULT_MAX_COST if max_cost is None else max_cost,
                use_cache=not no_cache,
                log=typer.echo,
            )
            return
        from interpretableintelligence.pipeline import execute_pipeline

        execute_pipeline(
            adapter_name=adapter,
            clips_dir=clips,
            out_dir=out,
            layer=layer,
            seed=seed,
            heldout_frac=heldout_frac,
            tiny=tiny,
            prefilter=prefilter,
            extra_effects=modes,
            gpu_cost_per_hour=gpu_cost_per_hour,
            sae_features=sae_features,
            concepts_file=concepts,
            score=not no_score,
            serve=not no_serve,
            host=host,
            port=port,
            open_browser=not no_open,
            steer=steer,
            use_cache=not no_cache,
            log=typer.echo,
            sae_from=sae_from,
        )
    except RemoteSetupError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except (ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("cache")
def cache_cmd(
    clear: Annotated[bool, typer.Option("--clear", help="Delete every cached activation.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the summary as JSON.")] = False,
) -> None:
    """Show (or clear) the activation cache that lets reruns and the playground skip forward passes.

    Entries are keyed by model weights, layer, clip hash, window and preprocessing, so a stale
    entry is never read. Set II_ACTS_CACHE to move the cache, or II_ACTS_CACHE=off to disable it.
    The size is capped at II_ACTS_CACHE_MAX_GB (default 20, 0 for no cap); the least recently
    used windows are deleted first.
    """
    from interpretableintelligence.act_cache import cache_usage, clear_cache, resolve_max_bytes, resolve_root

    root = resolve_root()
    try:
        cap = resolve_max_bytes()
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if clear:
        removed = clear_cache(root)
        typer.echo(f"removed {removed} cached window(s) from {root}" if root is not None else "activation cache off")
        return
    usage = cache_usage(root)
    usage["max_bytes"] = cap
    if as_json:
        typer.echo(json.dumps(usage, indent=2))
        return
    if root is None:
        typer.echo("activation cache off (II_ACTS_CACHE=off)")
        return
    typer.echo(f"activation cache: {root}")
    for kind, info in usage["kinds"].items():
        typer.echo(f"  {kind:<6} {info['entries']:>7} window(s)  {info['bytes'] / 1e6:10.1f} MB")
    typer.echo(f"  total  {usage['entries']:>7} window(s)  {usage['bytes'] / 1e6:10.1f} MB")
    typer.echo(f"  cap    {'none' if cap is None else f'{cap / 1e9:g} GB'} (II_ACTS_CACHE_MAX_GB; least recently used windows go first)")


@app.command()
def rank(
    adapter: Annotated[str, typer.Argument(help="Adapter name. This lane ships 'fake'.")],
    clips: Annotated[Path, typer.Option("--clips", exists=True, file_okay=False, help="Clip directory.")],
    out: Annotated[Path, typer.Option("--out", help="Directory for attribution.parquet and candidates.json.")],
    layer: Annotated[int, typer.Option("--layer", min=0, help="Layer to instrument.")] = DEFAULT_LAYER,
    seed: Annotated[int, typer.Option("--seed", help="Split and SAE seed. Ignored when --from-run has a split.")] = DEFAULT_SEED,
    heldout_frac: Annotated[
        float,
        typer.Option("--heldout-frac", help="Fraction of clips held out when training an SAE."),
    ] = DEFAULT_HELDOUT_FRAC,
    prefilter: Annotated[
        int,
        typer.Option(
            "--prefilter",
            min=1,
            help="How many top-ranked features to keep as ablation candidates.",
        ),
    ] = DEFAULT_PREFILTER_K,
    from_run: Annotated[
        Path | None,
        typer.Option(
            "--from-run",
            exists=True,
            file_okay=False,
            help="Reuse the SAE in this run folder instead of training one.",
        ),
    ] = None,
    objective: Annotated[
        str | None,
        typer.Option(
            "--objective",
            help=(
                "prediction_loss, output_sensitivity, action_loglik, or activation (magnitude baseline). "
                "Default: the adapter's own (output_sensitivity for hf and groot, else prediction_loss). "
                "The result is a shortlist."
            ),
        ),
    ] = None,
) -> None:
    """Rank SAE features. Does not ablate; every pick still needs a real ablation."""
    from interpretableintelligence.run import execute_rank

    try:
        execute_rank(
            adapter_name=adapter,
            clips_dir=clips,
            out_dir=out,
            layer=layer,
            seed=seed,
            heldout_frac=heldout_frac,
            prefilter=prefilter,
            from_run=from_run,
            objective=objective,
            log=typer.echo,
        )
    except (ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _emit_explain_error(exc: ExplainError) -> None:
    typer.echo(f"error: {exc}", err=True)
    typer.echo(f"run_id: {exc.run_id}", err=True)
    typer.echo(f"model_sha: {exc.model_sha}", err=True)
    typer.echo(f"layer: {exc.layer}", err=True)


def _print_explain(payload: dict) -> None:
    typer.echo(f"run_id: {payload['run_id']}")
    typer.echo(f"model_sha: {payload['model_sha']}")
    typer.echo(f"layer: {payload['layer']}")
    typer.echo(f"clip_id: {payload['clip_id']}")
    typer.echo(f"clip_sha256: {payload['clip_sha256']}")
    typer.echo("answer:")
    rank = 1
    for group in payload["answer"]:
        for feature in group["features"]:
            label = feature["label"] or "-"
            typer.echo(
                f"  {rank}. window={group['window']} t={group['t']}  "
                f"feature {feature['feature_id']}  {label}  "
                f"effect={feature['effect']:.6f}  ablation"
            )
            rank += 1
    if rank == 1:
        typer.echo("  (no cited features)")
    typer.echo("unverified:")
    if not payload["unverified"]:
        typer.echo("  (none)")
    for row in payload["unverified"]:
        label = row["label"] or "-"
        typer.echo(
            f"  window={row['window']} t={row['t']}  feature {row['feature_id']}  {label}  "
            f"effect={row['effect']:.6f}  {row['effect_kind']}  "
            f"real={str(row['real']).lower()}  {row['reason']}"
        )


@app.command("explain")
def explain_cmd(
    run: Annotated[Path, typer.Option("--run", exists=True, file_okay=False, help="Run folder, e.g. runs/<run_id>.")],
    clip: Annotated[Path, typer.Option("--clip", exists=True, dir_okay=False, help="New clip (.npy) to explain.")],
    as_json: Annotated[bool, typer.Option("--json", help="Print the response as JSON.")] = False,
) -> None:
    """Ablate the features active in a clip and cite only real ablation effects."""
    from interpretableintelligence.explain import ExplainError, explain_clip

    try:
        payload = explain_clip(run, clip)
    except ExplainError as exc:
        _emit_explain_error(exc)
        raise typer.Exit(code=1) from exc
    except (ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return
    _print_explain(payload)


@app.command("serve")
def serve_cmd(
    runs: Annotated[Path, typer.Option("--runs", exists=True, file_okay=False, help="Directory of run folders.")],
    host: Annotated[str, typer.Option("--host", help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535, help="Bind port.")] = 8000,
) -> None:
    """Serve POST /explain for the Rust read API (interpretableintelligence-server --explain-url)."""
    import uvicorn

    from interpretableintelligence.api import create_app

    uvicorn.run(create_app(runs), host=host, port=port)


@app.command()
def rollouts(
    run_dir: Annotated[Path, typer.Argument(help="Run folder. Rollouts are written under rollouts/.")],
    policy: Annotated[
        str,
        typer.Option(
            "--policy",
            help="Policy+simulator: fake, fake-null, groot-1028 (LIBERO-plus v9 ablation), groot-mock (CI replay).",
        ),
    ] = "fake",
    episodes: Annotated[
        int,
        typer.Option("--episodes", min=1, help="Paired episodes per arm. Controls may stop before this."),
    ] = DEFAULT_ROLLOUT_EPISODES,
    features: Annotated[
        str,
        typer.Option("--features", help="Comma-separated SAE feature ids to test, e.g. 3 or 3,4."),
    ] = "",
    controls: Annotated[
        int,
        typer.Option("--controls", min=0, help="Matched random-feature controls."),
    ] = DEFAULT_ROLLOUT_CONTROLS,
    workers: Annotated[
        int,
        typer.Option("--workers", min=1, help="Parallel episode workers."),
    ] = DEFAULT_ROLLOUT_WORKERS,
    seed: Annotated[
        int, typer.Option("--seed", help="Base seed. Episode i uses seed+i on every arm (GR00T --pick: 0 means 40000).")
    ] = 0,
    kind: Annotated[str, typer.Option("--kind", help="ablate, scale, or clamp.")] = "ablate",
    value: Annotated[float, typer.Option("--value", help="Scale factor, clamp level, or 0 for ablate.")] = 0.0,
    layer: Annotated[
        int | None,
        typer.Option("--layer", help="Layer whose SAE latents are edited. Defaults to the policy layer."),
    ] = None,
    task: Annotated[str, typer.Option("--task", help="Task id passed to reset.")] = "reach",
    alpha: Annotated[
        float,
        typer.Option(
            "--alpha",
            help="Family-wise type I level of the control SPRT, and the causal McNemar level.",
        ),
    ] = DEFAULT_ROLLOUT_ALPHA,
    min_episodes: Annotated[
        int,
        typer.Option("--min-episodes", min=1, help="Do not stop a control before this many episodes."),
    ] = DEFAULT_ROLLOUT_MIN_EPISODES,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", min=1, help="New episodes per active arm between early-stop checks."),
    ] = DEFAULT_ROLLOUT_BATCH,
    plan: Annotated[
        Path | None,
        typer.Option("--plan", exists=True, dir_okay=False, help="groot ablate_plan.json (default: bundled 1028 plan)."),
    ] = None,
    verify_slice: Annotated[
        bool,
        typer.Option(
            "--verify-slice",
            help="GR00T only: run the 8-episode verification slice (recon-only, baseline, cand-720, 2 controls).",
        ),
    ] = False,
    only: Annotated[
        str,
        typer.Option(
            "--only",
            help="GR00T only: comma-separated v9 condition ids (e.g. baseline,recon-only,A-cand-720).",
        ),
    ] = "",
    sae: Annotated[
        Path | None,
        typer.Option("--sae", exists=True, dir_okay=False, help="GR00T SAE subset weights for live v9 runs."),
    ] = None,
    v9_out: Annotated[
        Path | None,
        typer.Option(
            "--v9-out",
            file_okay=False,
            help="Import existing v9 ablation logs from this directory instead of re-running.",
        ),
    ] = None,
    pick: Annotated[
        int | None,
        typer.Option(
            "--pick",
            min=1,
            help="Pick this many candidates with the policy's picker on the trained SAE, instead of --features.",
        ),
    ] = None,
    random_sae: Annotated[
        int | None,
        typer.Option(
            "--random-sae",
            min=0,
            help="Random-weights SAE control: untrained SAEs sent through the same picker, one arm per pick. "
            "Needs --pick. A candidate is a finding only if it beats every one of these arms. "
            f"Default: {DEFAULT_RANDOM_SAE} with --pick, 0 without; 0 turns it off.",
        ),
    ] = None,
    pick_episodes: Annotated[
        int,
        typer.Option("--pick-episodes", min=1, help="Discovery episodes the default picker screens each feature on."),
    ] = PICK_DISCOVERY_EPISODES,
    capture: Annotated[
        Path | None,
        typer.Option(
            "--capture",
            exists=True,
            file_okay=False,
            help="GR00T with --pick: capture run folder (acts, failure labels, trained SAE) the picker reads, "
            "e.g. groot-1028-v9-n100.",
        ),
    ] = None,
) -> None:
    """Compare baseline and intervened task success. Resume skips episodes already written."""
    from interpretableintelligence.rollout import execute_rollouts, format_report, parse_feature_ids
    from interpretableintelligence.rollout_groot import execute_groot_rollouts, format_groot_report, is_groot_policy

    try:
        if is_groot_policy(policy) and pick is not None:
            from interpretableintelligence.rollout_groot import GROOT_SEED_BASE, execute_groot_pick

            if capture is None:
                raise ValueError("GR00T --pick needs --capture: the capture run the picker reads")
            if features.strip() or verify_slice or only or plan is not None:
                raise ValueError("GR00T --pick builds its own plan: drop --features, --verify-slice, --only and --plan")
            summary = execute_groot_pick(
                run_dir,
                policy=policy,
                capture=capture,
                pick=int(pick),
                random_sae=DEFAULT_RANDOM_SAE if random_sae is None else int(random_sae),
                controls=controls,
                episodes=episodes,
                seed=seed if seed else GROOT_SEED_BASE,
                alpha=alpha,
                workers=workers,
                sae_path=sae,
                v9_out=v9_out,
            )
            typer.echo(format_groot_report(summary))
            typer.echo(f"wrote {run_dir / 'rollouts'}")
            return
        if is_groot_policy(policy):
            if random_sae:
                raise ValueError("--random-sae compares picker against picker on GR00T: pass --pick N with --capture")
            only_ids = [part.strip() for part in only.split(",") if part.strip()] or None
            summary = execute_groot_rollouts(
                run_dir,
                policy=policy,
                plan_path=plan,
                verify_slice=verify_slice,
                only=only_ids,
                workers=workers,
                alpha=alpha,
                min_episodes=min_episodes,
                batch_size=batch_size,
                sae_path=sae,
                v9_out=v9_out,
            )
            typer.echo(format_groot_report(summary))
            typer.echo(f"wrote {run_dir / 'rollouts'}")
            return
        feature_ids = parse_feature_ids(features) if features.strip() or pick is None else []
        summary = execute_rollouts(
            run_dir,
            policy=policy,
            episodes=episodes,
            features=feature_ids,
            controls=controls,
            workers=workers,
            seed=seed,
            kind=kind,
            value=value,
            layer=layer,
            task_id=task,
            alpha=alpha,
            min_episodes=min_episodes,
            batch_size=batch_size,
            pick=pick,
            random_sae=random_sae,
            pick_episodes=pick_episodes,
        )
    except (ValueError, RuntimeError, FileNotFoundError, NotImplementedError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(format_report(summary))
    typer.echo(f"wrote {run_dir / 'rollouts'}")


@app.command("record-libero")
def record_libero_cmd(
    out: Annotated[Path, typer.Option("--out", file_okay=False, help="Clip folder to write (resumable), e.g. clips/groot-1028.")],
    task: Annotated[int, typer.Option("--task", min=1, help="LIBERO-plus task id (libero_10 suite).")] = 1028,
    episodes: Annotated[int, typer.Option("--episodes", min=1, help="Episodes to record. `ii run` needs at least 20.")] = 24,
    seed0: Annotated[int, typer.Option("--seed0", help="Episode i uses seed seed0 + i and init state i mod the task's init states.")] = 60000,
    checkpoint: Annotated[str, typer.Option("--checkpoint", help="LeRobot GR00T checkpoint.")] = "nvidia/gr00t17-lerobot-libero_10-640",
    plus_dir: Annotated[
        Path | None,
        typer.Option("--plus-dir", file_okay=False, help="LIBERO-plus checkout; writes its LIBERO config (assets, bddl, init states)."),
    ] = None,
    max_steps: Annotated[int | None, typer.Option("--max-steps", min=1, help="Cap steps per episode. Default: the env's cap (520).")] = None,
    no_parity: Annotated[bool, typer.Option("--no-parity", help="Skip the input parity check (debugging only).")] = False,
    trim_out: Annotated[
        Path | None,
        typer.Option("--trim-out", file_okay=False, help="Also write every clip cut to the shortest episode's length (whole 16-step chunks), which ii run needs."),
    ] = None,
) -> None:
    """Record closed-loop GR00T episodes on LIBERO-plus as clips the groot adapter reads (main camera, wrist, state, task).

    Needs a GPU box with lerobot[groot]==0.7.0 and LIBERO-plus (scripts/groot_gpu_setup.sh). Checks that the adapter
    rebuilds the policy's own inputs from the recorded arrays before it records anything else.
    """
    from interpretableintelligence.adapters.groot.record_libero import LeRobotLibero, episode_id, record, trim_to_common_length

    try:
        index = out / "clips.json"
        have = {c["clip_id"] for c in json.loads(index.read_text(encoding="utf-8"))["clips"]} if index.is_file() else set()
        if all(episode_id(i) in have for i in range(episodes)):
            typer.echo(f"all {episodes} episodes are already in {out}; not loading the policy")
        else:
            runtime = LeRobotLibero(checkpoint, task, plus_dir=plus_dir)
            report = record(
                runtime, out, task_id=task, episodes=episodes, seed0=seed0, max_steps=max_steps, check_parity=not no_parity, log=typer.echo
            )
            typer.echo(f"recorded {len(report['episodes'])} episodes, {report['n_success']} successes, into {out}")
        if trim_out is not None:
            trim_to_common_length(out, trim_out, log=typer.echo)
    except (ImportError, ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("steer-scorecard")
def steer_scorecard_cmd(
    run_dir: Annotated[Path, typer.Argument(help="Run folder. The scorecard is written under steering/.")],
    clips: Annotated[
        Path | None,
        typer.Option("--clips", help="Directory of clip files. Overrides a clip path recorded in the manifest."),
    ] = None,
    features: Annotated[
        str,
        typer.Option(
            "--features",
            help="Comma-separated feature ids. Default: the top verified (real) features. Unverified ids are measured and recorded as unverified.",
        ),
    ] = "",
    top: Annotated[int, typer.Option("--top", min=1, help="How many verified features to score when --features is omitted.")] = STEER_TOP_FEATURES,
    boost: Annotated[float, typer.Option("--boost", help="Scale factor for the boost edit. Remove is always scale 0.")] = STEER_BOOST_FACTOR,
    controls: Annotated[int, typer.Option("--controls", min=1, help="Matched random-feature controls per edit.")] = STEER_CONTROLS,
    seed: Annotated[int, typer.Option("--seed", help="Seed for control directions, placements and the bootstrap.")] = 0,
    bootstrap: Annotated[int, typer.Option("--bootstrap", min=1, help="Bootstrap resamples of held-out episodes for the intervals.")] = STEER_BOOTSTRAP,
    max_episodes: Annotated[
        int | None,
        typer.Option("--max-episodes", min=1, help="Score only the first N held-out clips (sorted by id)."),
    ] = None,
    controls_from: Annotated[
        str,
        typer.Option(
            "--controls-from",
            help="random: random directions at the feature's decoder norm. features: decoder directions of other live SAE "
            "features matched on train-split firing rate, mean activation and decoder norm (a stronger bar).",
        ),
    ] = "random",
) -> None:
    """Steering scorecard: remove and boost top features on held-out clips, against matched random controls.

    For GR00T this measures how much, and how reliably, the planned action chunk moves. It never changes
    scoreboard.json or a badge, and it is not a rollout finding.
    """
    from interpretableintelligence.explain import ExplainError
    from interpretableintelligence.steer_scorecard import execute_steer_scorecard, format_report, parse_feature_ids

    try:
        card = execute_steer_scorecard(
            run_dir,
            clips_dir=clips,
            features=parse_feature_ids(features) or None,
            top=top,
            boost=boost,
            controls=controls,
            seed=seed,
            bootstrap=bootstrap,
            max_episodes=max_episodes,
            controls_from=controls_from,
            log=typer.echo,
        )
    except (ValueError, RuntimeError, FileNotFoundError, OSError, KeyError, ExplainError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(format_report(card))
    typer.echo(f"wrote {run_dir / 'steering'}")


@app.command("probe")
def probe_cmd(
    run_dir: Annotated[Path, typer.Argument(help="Run folder. The result is written under probe/.")],
    labels: Annotated[
        Path | None,
        typer.Option(
            "--labels",
            exists=True,
            dir_okay=False,
            help="Episode failure labels (JSON: episodes[] of clip_id, failure, onset_s). Default: failures.json in the run's clips_dir.",
        ),
    ] = None,
    lead: Annotated[
        float,
        typer.Option("--lead-s", help=f"Seconds before onset the probe must stop looking. The gate needs at least {PROBE_MIN_LEAD_S:g}."),
    ] = PROBE_MIN_LEAD_S,
    seed: Annotated[int, typer.Option("--seed", help="Bootstrap seed.")] = 0,
    bootstrap: Annotated[int, typer.Option("--bootstrap", min=1, help="Paired bootstrap resamples of held-out episodes.")] = PROBE_BOOTSTRAP,
    l2: Annotated[float, typer.Option("--l2", help="L2 penalty of the logistic probe (same for every input).")] = PROBE_L2,
    random_seed: Annotated[
        int | None,
        typer.Option("--random-seed", help="Seed of the random-weights SAE control. Default: the trained SAE's seed (its own untrained start)."),
    ] = None,
) -> None:
    """Held-out failure probe: trained SAE latents vs a random-weights SAE and raw activations.

    Fits one logistic probe per input on split.json's train clips, scores held-out clips, and passes
    sae_gate only if the trained SAE beats both baselines on AUROC (paired bootstrap p < 0.05) at a lead
    of at least 1 s. Never changes scoreboard.json, never sets real, and is not a rollout finding.
    """
    from interpretableintelligence.probe import execute_probe, format_report

    try:
        result = execute_probe(
            run_dir,
            labels_path=labels,
            lead_s=lead,
            seed=seed,
            bootstrap=bootstrap,
            l2=l2,
            random_seed=random_seed,
            log=typer.echo,
        )
    except (ValueError, RuntimeError, FileNotFoundError, OSError, KeyError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(format_report(result))
    typer.echo(f"wrote {run_dir / 'probe'}")


@app.command("action-effects")
def action_effects_cmd(
    clips: Annotated[
        Path,
        typer.Option("--clips", exists=True, file_okay=False, help="Clips from `ii get-clips --with-states` (video plus arm poses)."),
    ],
    out: Annotated[Path, typer.Option("--out", help="Directory for action_effects.json, transitions.parquet and regions.png.")],
    code_dir: Annotated[
        Path | None,
        typer.Option("--code-dir", help="Checkout of github.com/facebookresearch/vjepa2 (Meta's model code). Not needed with --tiny."),
    ] = None,
    checkpoint: Annotated[
        Path | None,
        typer.Option("--checkpoint", help="Local vjepa2-ac-vitg.pt. Default: download it to the torch hub cache."),
    ] = None,
    random_predictor: Annotated[
        bool,
        typer.Option("--random-predictor", help="Control: trained encoder, untrained predictor (seeded init)."),
    ] = False,
    tiny: Annotated[bool, typer.Option("--tiny", help="Random CPU stand-in whose actions move only the top row of patches (tests).")] = False,
    segmenter: Annotated[
        str,
        typer.Option("--segmenter", help="Arm masks: grounded-sam (Grounding DINO boxes refined by SAM) or clipseg (weaker on DROID)."),
    ] = "grounded-sam",
    max_clips: Annotated[int | None, typer.Option("--max-clips", min=1, help="Use only the first N clips.")] = None,
    seed: Annotated[int, typer.Option("--seed", help="Seed for shuffled actions, matched patches, bootstrap and the random predictor.")] = 0,
    bootstrap: Annotated[int, typer.Option("--bootstrap", min=1, help="Episode bootstrap resamples.")] = 2000,
) -> None:
    """Does V-JEPA 2-AC predict the world moving, or only its own arm? (Offline, recorded DROID video.)

    Predicts each next latent frame under the real, zero, shuffled and reversed action, and compares
    how much the action explains on arm patches, on non-arm patches that moved, and on matched still
    patches. The verdict rule is fixed in the code. Never a rollout finding; touches no scoreboard.
    """
    import numpy as np

    from interpretableintelligence.adapters.vjepa2ac.model import load_vjepa2ac, tiny_world_model
    from interpretableintelligence.device import pick_device
    from interpretableintelligence.worldmodel.action_effects import load_trajectories, run_action_effects

    try:
        if tiny:
            model = tiny_world_model(seed=seed)
            side = model.crop // model.patch

            def arm_masks(frames):
                # Stub segmenter: the top row of patches is the arm.
                h, w = frames.shape[1:3]
                m = np.zeros(frames.shape[:3], dtype=bool)
                m[:, : max(1, h // side), :] = True
                return m

        else:
            if code_dir is None:
                raise ValueError("--code-dir is required: a checkout of github.com/facebookresearch/vjepa2")
            from interpretableintelligence.worldmodel.segment import ClipSegArm, GroundedSamArm

            segmenters = {"grounded-sam": GroundedSamArm, "clipseg": ClipSegArm}
            if segmenter not in segmenters:
                raise ValueError(f"--segmenter must be one of {', '.join(segmenters)}")
            device = pick_device()
            model = load_vjepa2ac(device, code_dir=code_dir, checkpoint=checkpoint, random_predictor=random_predictor, seed=seed)
            arm_masks = segmenters[segmenter](device)
        trajectories = load_trajectories(clips, max_clips=max_clips)
        if not trajectories:
            raise ValueError(f"no usable clips in {clips}")
        typer.echo(f"     {len(trajectories)} clips, model {model.name}")
        extra = {"segmenter": "stub" if tiny else segmenter}
        result = run_action_effects(model, trajectories, arm_masks, out, seed=seed, n_boot=bootstrap, say=typer.echo, extra=extra)
    except (ValueError, RuntimeError, FileNotFoundError, OSError, KeyError, ImportError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    moving = result["moving"]
    s = moving["stats"]
    typer.echo(
        f"verdict: {moving['verdict']} on {moving['n_transitions']} transitions with moving objects "
        f"({moving['n_episodes']} episodes)"
    )
    for key in ("benefit_arm", "benefit_moved", "benefit_matched", "moved_minus_matched"):
        v = s[key]
        typer.echo(f"  {key:20s} {v['value']:+.4f}  [{v['lo']:+.4f}, {v['hi']:+.4f}]")
    typer.echo(f"wrote {out}")


@app.command("plan-probe")
def plan_probe_cmd(
    clips: Annotated[
        Path,
        typer.Option("--clips", exists=True, file_okay=False, help="Clips from `ii get-clips --with-states` (video plus arm poses)."),
    ],
    out: Annotated[Path, typer.Option("--out", help="Directory for plan_probe.json and transitions.parquet.")],
    code_dir: Annotated[
        Path | None,
        typer.Option("--code-dir", help="Checkout of github.com/facebookresearch/vjepa2 (Meta's model code). Not needed with --tiny."),
    ] = None,
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint", help="Local vjepa2-ac-vitg.pt.")] = None,
    random_predictor: Annotated[bool, typer.Option("--random-predictor", help="Control: trained encoder, untrained predictor.")] = False,
    tiny: Annotated[bool, typer.Option("--tiny", help="Random CPU stand-in (tests).")] = False,
    max_clips: Annotated[int | None, typer.Option("--max-clips", min=1, help="Use only the first N clips.")] = None,
    stride: Annotated[int, typer.Option("--stride", min=1, help="Probe every Nth transition of each clip.")] = 4,
    samples: Annotated[int, typer.Option("--samples", min=10, help="CEM samples per iteration.")] = 200,
    steps: Annotated[int, typer.Option("--steps", min=1, help="CEM iterations.")] = 8,
    wide_xyz: Annotated[float, typer.Option("--wide-xyz", min=0.08, help="Arm step limit (m) of the wide search.")] = 0.5,
    seed: Annotated[int, typer.Option("--seed", help="Seed for the searches, bootstrap and the random predictor.")] = 0,
    bootstrap: Annotated[int, typer.Option("--bootstrap", min=1, help="Episode bootstrap resamples.")] = 2000,
) -> None:
    """Can V-JEPA 2-AC's planner be fooled by impossible actions? (Offline, recorded DROID video.)

    For each transition, plans toward the next real frame with CEM inside Meta's action range and with
    a much wider range, and counts how often the wide search returns an out-of-range action that scores
    better than both the real action and the best in-range one. The verdict rule is fixed in the code.
    """
    from interpretableintelligence.adapters.vjepa2ac.model import load_vjepa2ac, tiny_world_model
    from interpretableintelligence.device import pick_device
    from interpretableintelligence.worldmodel.action_effects import load_trajectories
    from interpretableintelligence.worldmodel.plan_probe import run_plan_probe

    try:
        if tiny:
            model = tiny_world_model(seed=seed)
        else:
            if code_dir is None:
                raise ValueError("--code-dir is required: a checkout of github.com/facebookresearch/vjepa2")
            model = load_vjepa2ac(pick_device(), code_dir=code_dir, checkpoint=checkpoint, random_predictor=random_predictor, seed=seed)
            model.predict_batch = 64
        trajectories = load_trajectories(clips, max_clips=max_clips)
        if not trajectories:
            raise ValueError(f"no usable clips in {clips}")
        typer.echo(f"     {len(trajectories)} clips, model {model.name}")
        result = run_plan_probe(
            model, trajectories, out, stride=stride, samples=samples, steps=steps, wide_xyz=wide_xyz,
            seed=seed, n_boot=bootstrap, say=typer.echo,
        )
    except (ValueError, RuntimeError, FileNotFoundError, OSError, KeyError, ImportError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"verdict: {result['verdict']} on {result['n_transitions']} transitions ({result['n_episodes']} episodes)")
    for key, v in result["stats"].items():
        typer.echo(f"  {key:30s} {v['value']:+.4f}  [{v['lo']:+.4f}, {v['hi']:+.4f}]")
    typer.echo(f"wrote {out}")


@app.command("gripper-probe")
def gripper_probe_cmd(
    clips: Annotated[
        Path,
        typer.Option("--clips", exists=True, file_okay=False, help="Clips from `ii get-clips --with-states` (video plus arm poses)."),
    ],
    out: Annotated[Path, typer.Option("--out", help="Directory for gripper_probe.json.")],
    code_dir: Annotated[Path | None, typer.Option("--code-dir", help="Checkout of github.com/facebookresearch/vjepa2. Not needed with --tiny.")] = None,
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint", help="Local vjepa2-ac-vitg.pt.")] = None,
    random_predictor: Annotated[bool, typer.Option("--random-predictor", help="Control: trained encoder, untrained predictor.")] = False,
    tiny: Annotated[bool, typer.Option("--tiny", help="Random CPU stand-in (tests).")] = False,
    max_clips: Annotated[int | None, typer.Option("--max-clips", min=1, help="Use only the first N clips.")] = None,
    samples: Annotated[int, typer.Option("--samples", min=10, help="CEM samples per iteration.")] = 100,
    steps: Annotated[int, typer.Option("--steps", min=1, help="CEM iterations.")] = 6,
    seed: Annotated[int, typer.Option("--seed", help="Seed for the search, bootstrap and the random predictor.")] = 0,
    bootstrap: Annotated[int, typer.Option("--bootstrap", min=1, help="Episode bootstrap resamples.")] = 2000,
) -> None:
    """Does V-JEPA 2-AC's goal score read the gripper? (Offline, recorded DROID video.)

    At moments where the gripper opens or closes while the arm stays still, scores the right and the
    wrong gripper command toward the next frame and plans one step. The verdict rule is fixed in the code.
    """
    from interpretableintelligence.adapters.vjepa2ac.model import load_vjepa2ac, tiny_world_model
    from interpretableintelligence.device import pick_device
    from interpretableintelligence.worldmodel.action_effects import load_trajectories
    from interpretableintelligence.worldmodel.gripper_probe import run_gripper_probe

    try:
        if tiny:
            model = tiny_world_model(seed=seed)
        else:
            if code_dir is None:
                raise ValueError("--code-dir is required: a checkout of github.com/facebookresearch/vjepa2")
            model = load_vjepa2ac(pick_device(), code_dir=code_dir, checkpoint=checkpoint, random_predictor=random_predictor, seed=seed)
            model.predict_batch = 64
        trajectories = load_trajectories(clips, max_clips=max_clips)
        if not trajectories:
            raise ValueError(f"no usable clips in {clips}")
        result = run_gripper_probe(model, trajectories, out, samples=samples, steps=steps, seed=seed, n_boot=bootstrap, say=typer.echo)
    except (ValueError, RuntimeError, FileNotFoundError, OSError, KeyError, ImportError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"verdict: {result['verdict']} on {result['n_events']} gripper events ({result['n_episodes']} episodes)")
    for key, v in result["stats"].items():
        typer.echo(f"  {key:20s} {v['value']:+.4f}  [{v['lo']:+.4f}, {v['hi']:+.4f}]")
    typer.echo(f"wrote {out}")


@app.command("impossible-events")
def impossible_events_cmd(
    clips: Annotated[
        Path,
        typer.Option("--clips", exists=True, file_okay=False, help="Clips from `ii get-clips --with-states` (video plus arm poses)."),
    ],
    out: Annotated[Path, typer.Option("--out", help="Directory for impossible_events.json and heatmaps/.")],
    code_dir: Annotated[Path | None, typer.Option("--code-dir", help="Checkout of github.com/facebookresearch/vjepa2. Not needed with --tiny.")] = None,
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint", help="Local vjepa2-ac-vitg.pt.")] = None,
    random_predictor: Annotated[bool, typer.Option("--random-predictor", help="Control: trained encoder, untrained predictor.")] = False,
    tiny: Annotated[bool, typer.Option("--tiny", help="Random CPU stand-in (tests).")] = False,
    max_clips: Annotated[int | None, typer.Option("--max-clips", min=1, help="Use only the first N clips.")] = None,
    seed: Annotated[int, typer.Option("--seed", help="Seed for the swapped squares, bootstrap and the random predictor.")] = 0,
    bootstrap: Annotated[int, typer.Option("--bootstrap", min=1, help="Episode bootstrap resamples.")] = 2000,
    context: Annotated[int, typer.Option("--context", min=1, max=7, help="Real frames of context fed to the predictor (Meta trained with up to 7).")] = 1,
    start: Annotated[int, typer.Option("--start", min=1, help="First transition scored (raised to context - 1); match it across runs.")] = 1,
) -> None:
    """Which impossible next frames does V-JEPA 2-AC notice? (Offline, recorded DROID video.)

    Compares the predictor's surprise at the real next frame with five impossible ones (frozen world,
    frozen moving square, time reversed, swapped squares, another episode), against a copy-the-frame
    baseline and a zero action. The verdict rule is fixed in the code.
    """
    from interpretableintelligence.adapters.vjepa2ac.model import load_vjepa2ac, tiny_world_model
    from interpretableintelligence.device import pick_device
    from interpretableintelligence.worldmodel.action_effects import load_trajectories
    from interpretableintelligence.worldmodel.impossible_events import run_impossible_events

    try:
        if tiny:
            model = tiny_world_model(seed=seed)
        else:
            if code_dir is None:
                raise ValueError("--code-dir is required: a checkout of github.com/facebookresearch/vjepa2")
            model = load_vjepa2ac(pick_device(), code_dir=code_dir, checkpoint=checkpoint, random_predictor=random_predictor, seed=seed)
            model.predict_batch = 64
        trajectories = load_trajectories(clips, max_clips=max_clips)
        if not trajectories:
            raise ValueError(f"no usable clips in {clips}")
        result = run_impossible_events(model, trajectories, out, seed=seed, context=context, start=start, n_boot=bootstrap, say=typer.echo)
    except (ValueError, RuntimeError, FileNotFoundError, OSError, KeyError, ImportError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"{result['n_transitions']} moving transitions ({result['n_episodes']} episodes)")
    for e, v in result["edits"].items():
        d = v["detected"]
        typer.echo(
            f"  {e:13s} {v['verdict']:17s} model {d['model']['value']:.2f} [{d['model']['lo']:.2f}, {d['model']['hi']:.2f}]"
            f"  zero-action {d['zero_action']['value']:.2f}  copy {d['copy']['value']:.2f}  (n={v['n_transitions']})"
        )
    typer.echo(f"wrote {out}")


@app.command("plan-stall")
def plan_stall_cmd(
    clips: Annotated[
        Path,
        typer.Option("--clips", exists=True, file_okay=False, help="Clips from `ii get-clips --with-states` (video plus arm poses)."),
    ],
    out: Annotated[Path, typer.Option("--out", help="Directory for plan_stall.json.")],
    code_dir: Annotated[Path | None, typer.Option("--code-dir", help="Checkout of github.com/facebookresearch/vjepa2. Not needed with --tiny.")] = None,
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint", help="Local vjepa2-ac-vitg.pt.")] = None,
    random_predictor: Annotated[bool, typer.Option("--random-predictor", help="Control: trained encoder, untrained predictor.")] = False,
    tiny: Annotated[bool, typer.Option("--tiny", help="Random CPU stand-in (tests).")] = False,
    max_clips: Annotated[int | None, typer.Option("--max-clips", min=1, help="Use only the first N clips.")] = None,
    seed: Annotated[int, typer.Option("--seed", help="Seed for the random token sets, bootstrap and the random predictor.")] = 0,
    bootstrap: Annotated[int, typer.Option("--bootstrap", min=1, help="Episode bootstrap resamples.")] = 2000,
) -> None:
    """Does V-JEPA 2-AC's planner rather stand still, and does scoring the moving parts help? (Offline.)

    On moving DROID transitions the planner picks from the real step, its reverse, doing nothing and
    26 other directions, under Meta's goal score, a score on the most-changed tokens, and a score on
    random tokens. The verdict rules are fixed in the code.
    """
    from interpretableintelligence.adapters.vjepa2ac.model import load_vjepa2ac, tiny_world_model
    from interpretableintelligence.device import pick_device
    from interpretableintelligence.worldmodel.action_effects import load_trajectories
    from interpretableintelligence.worldmodel.plan_stall import run_plan_stall

    try:
        if tiny:
            model = tiny_world_model(seed=seed)
        else:
            if code_dir is None:
                raise ValueError("--code-dir is required: a checkout of github.com/facebookresearch/vjepa2")
            model = load_vjepa2ac(pick_device(), code_dir=code_dir, checkpoint=checkpoint, random_predictor=random_predictor, seed=seed)
            model.predict_batch = 32
        trajectories = load_trajectories(clips, max_clips=max_clips)
        if not trajectories:
            raise ValueError(f"no usable clips in {clips}")
        result = run_plan_stall(model, trajectories, out, seed=seed, n_boot=bootstrap, say=typer.echo)
    except (ValueError, RuntimeError, FileNotFoundError, OSError, KeyError, ImportError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for h, v in result["horizons"].items():
        typer.echo(f"goal {h} step(s) ahead: {v['n_transitions']} transitions, hurt {v['hurt']}, fix {v['fix']}")
        for s, st in v["scores"].items():
            typer.echo("  " + f"{s:7s}" + "  ".join(f"{k} {x['value']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}]" for k, x in st.items()))
    typer.echo(f"wrote {out}")


@app.command("novelty-trace")
def novelty_trace_cmd(
    clips: Annotated[
        Path,
        typer.Option("--clips", exists=True, file_okay=False, help="Clips from `ii get-clips --with-states` (video plus arm poses)."),
    ],
    out: Annotated[Path, typer.Option("--out", help="Directory for novelty_trace.json and transitions.npz.")],
    code_dir: Annotated[Path | None, typer.Option("--code-dir", help="Checkout of github.com/facebookresearch/vjepa2. Not needed with --tiny.")] = None,
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint", help="Local vjepa2-ac-vitg.pt.")] = None,
    random_predictor: Annotated[bool, typer.Option("--random-predictor", help="Control: trained encoder, untrained predictor.")] = False,
    random_encoder: Annotated[
        bool, typer.Option("--random-encoder", help="Control for the encoder trace: untrained encoder, trained predictor.")
    ] = False,
    tubelet: Annotated[
        str,
        typer.Option("--tubelet", help="repeat: each frame fills its own 2-frame tubelet (Meta's 2-AC); pair: the previous and the current frame as one real 2-frame clip."),
    ] = "repeat",
    control: Annotated[
        Path | None,
        typer.Option("--control", exists=True, file_okay=False, help="Output of the same command with --random-predictor: the trained peak gain must beat it."),
    ] = None,
    tiny: Annotated[bool, typer.Option("--tiny", help="Random CPU stand-in (tests).")] = False,
    max_clips: Annotated[int | None, typer.Option("--max-clips", min=1, help="Use only the first N clips.")] = None,
    seed: Annotated[int, typer.Option("--seed", help="Seed for the episode folds, bootstrap and the random predictor.")] = 0,
    bootstrap: Annotated[int, typer.Option("--bootstrap", min=1, help="Episode bootstrap resamples.")] = 2000,
) -> None:
    """Where does V-JEPA 2-AC lose what is new in the next frame? (Offline, recorded DROID video.)

    Splits each next frame's latent into what the current frame predicts and what is new, and reads
    the new part from every predictor layer with cross-fitted linear readouts, and from every encoder
    block without fitting. The verdict rules are fixed in the code.
    """
    from interpretableintelligence.adapters.vjepa2ac.model import load_vjepa2ac, tiny_world_model
    from interpretableintelligence.device import pick_device
    from interpretableintelligence.worldmodel.action_effects import load_trajectories
    from interpretableintelligence.worldmodel.novelty_trace import run_novelty_trace

    try:
        if tiny:
            model = tiny_world_model(seed=seed)
        else:
            if code_dir is None:
                raise ValueError("--code-dir is required: a checkout of github.com/facebookresearch/vjepa2")
            model = load_vjepa2ac(pick_device(), code_dir=code_dir, checkpoint=checkpoint, random_predictor=random_predictor, random_encoder=random_encoder, seed=seed)
        if tubelet not in ("repeat", "pair"):
            raise ValueError("--tubelet must be repeat or pair")
        model.tubelet = tubelet
        trajectories = load_trajectories(clips, max_clips=max_clips)
        if not trajectories:
            raise ValueError(f"no usable clips in {clips}")
        result = run_novelty_trace(model, trajectories, out, control_dir=control, seed=seed, n_boot=bootstrap, say=typer.echo)
    except (ValueError, RuntimeError, FileNotFoundError, OSError, KeyError, ImportError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    st = result["stats"]
    typer.echo(f"{result['n_transitions']} moving transitions ({result['n_episodes']} episodes), peak layer {result['peak']}")
    for k, v in st.items():
        typer.echo(f"  {k:24s} {v['value']:+.4f}  [{v['lo']:+.4f}, {v['hi']:+.4f}]")
    typer.echo(f"verdict: {result['verdict']}; encoder: {result['encoder_verdict']}")
    typer.echo(f"wrote {out}")


@app.command("gripper-trace")
def gripper_trace_cmd(
    clips: Annotated[
        Path,
        typer.Option("--clips", exists=True, file_okay=False, help="Clips from `ii get-clips --with-states` (video plus arm poses)."),
    ],
    out: Annotated[Path, typer.Option("--out", help="Directory for gripper_trace.json.")],
    code_dir: Annotated[Path | None, typer.Option("--code-dir", help="Checkout of github.com/facebookresearch/vjepa2. Not needed with --tiny.")] = None,
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint", help="Local vjepa2-ac-vitg.pt.")] = None,
    random_predictor: Annotated[bool, typer.Option("--random-predictor", help="Control: trained encoder, untrained predictor.")] = False,
    tiny: Annotated[bool, typer.Option("--tiny", help="Random CPU stand-in (tests).")] = False,
    max_clips: Annotated[int | None, typer.Option("--max-clips", min=1, help="Use only the first N clips.")] = None,
    seed: Annotated[int, typer.Option("--seed", help="Seed for the episode folds, bootstrap and the random predictor.")] = 0,
    bootstrap: Annotated[int, typer.Option("--bootstrap", min=1, help="Episode bootstrap resamples.")] = 2000,
) -> None:
    """Where does V-JEPA 2-AC lose the gripper command? (Offline, recorded DROID video.)

    Reads gripper state from the encoder's latents, traces the right and wrong gripper command through
    every predictor block with activation patching, and reads the predicted latent with the same probe.
    The verdict rules are fixed in the code.
    """
    from interpretableintelligence.adapters.vjepa2ac.model import load_vjepa2ac, tiny_world_model
    from interpretableintelligence.device import pick_device
    from interpretableintelligence.worldmodel.action_effects import load_trajectories
    from interpretableintelligence.worldmodel.gripper_trace import run_gripper_trace

    try:
        if tiny:
            model = tiny_world_model(seed=seed)
        else:
            if code_dir is None:
                raise ValueError("--code-dir is required: a checkout of github.com/facebookresearch/vjepa2")
            model = load_vjepa2ac(pick_device(), code_dir=code_dir, checkpoint=checkpoint, random_predictor=random_predictor, seed=seed)
        trajectories = load_trajectories(clips, max_clips=max_clips)
        if not trajectories:
            raise ValueError(f"no usable clips in {clips}")
        result = run_gripper_trace(model, trajectories, out, seed=seed, n_boot=bootstrap, say=typer.echo)
    except (ValueError, RuntimeError, FileNotFoundError, OSError, KeyError, ImportError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    v = result["verdict"]
    typer.echo(f"verdict: encoder {v['encoder']}, command {v['command']} on {result['n_events']} gripper events ({result['n_episodes']} episodes)")
    for name in ("encoder", "pooled", "pixels"):
        r = result["readout"][name]
        typer.echo(f"  {name + ' AUROC':20s} {r['value']:.4f}  [{r['lo']:.4f}, {r['hi']:.4f}]")
    for key, r in result["stats"].items():
        typer.echo(f"  {key:20s} {r['value']:+.4f}  [{r['lo']:+.4f}, {r['hi']:+.4f}]")
    typer.echo(f"wrote {out}")


@app.command("gripper-circuit")
def gripper_circuit_cmd(
    clips: Annotated[
        Path,
        typer.Option("--clips", exists=True, file_okay=False, help="Clips from `ii get-clips --with-states` (video plus arm poses)."),
    ],
    out: Annotated[Path, typer.Option("--out", help="Directory for gripper_circuit.json.")],
    code_dir: Annotated[Path | None, typer.Option("--code-dir", help="Checkout of github.com/facebookresearch/vjepa2. Not needed with --tiny.")] = None,
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint", help="Local vjepa2-ac-vitg.pt.")] = None,
    tiny: Annotated[bool, typer.Option("--tiny", help="CPU stand-in with a planted gripper eraser (tests).")] = False,
    max_clips: Annotated[int | None, typer.Option("--max-clips", min=1, help="Use only the first N clips.")] = None,
    components: Annotated[int, typer.Option("--components", min=1, help="Attention or MLP parts knocked out together.")] = 4,
    heads: Annotated[int, typer.Option("--heads", min=1, help="Attention heads knocked out together.")] = 8,
    random_sets: Annotated[int, typer.Option("--random-sets", min=1, help="Random sets of the same size, the knockout control.")] = 20,
    seed: Annotated[int, typer.Option("--seed", help="Seed for the episode folds, random sets, directions and bootstrap.")] = 0,
    bootstrap: Annotated[int, typer.Option("--bootstrap", min=1, help="Episode bootstrap resamples.")] = 2000,
) -> None:
    """Which predictor parts shrink V-JEPA 2-AC's gripper command, and does undoing them help? (Offline.)

    Scores every attention and MLP output (and head) on how much of the gripper difference it
    removes, knocks the top ones out on held-out episodes against random sets, and re-injects the
    command's peak-layer effect against random directions. The verdict rules are fixed in the code.
    """
    from interpretableintelligence.adapters.vjepa2ac.model import load_vjepa2ac, tiny_circuit_model
    from interpretableintelligence.device import pick_device
    from interpretableintelligence.worldmodel.action_effects import load_trajectories
    from interpretableintelligence.worldmodel.gripper_circuit import run_gripper_circuit

    try:
        if tiny:
            model = tiny_circuit_model(seed=seed)
        else:
            if code_dir is None:
                raise ValueError("--code-dir is required: a checkout of github.com/facebookresearch/vjepa2")
            model = load_vjepa2ac(pick_device(), code_dir=code_dir, checkpoint=checkpoint, seed=seed)
        trajectories = load_trajectories(clips, max_clips=max_clips)
        if not trajectories:
            raise ValueError(f"no usable clips in {clips}")
        result = run_gripper_circuit(model, trajectories, out, k_comp=components, k_head=heads, n_rand=random_sets, seed=seed,
                                     n_boot=bootstrap, say=typer.echo)
    except (ValueError, RuntimeError, FileNotFoundError, OSError, KeyError, ImportError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    v = result["verdict"]
    typer.echo(f"verdict: components {v['components']}, heads {v['heads']}, steer {v['steer']} on {result['n_tested']} held-out events "
               f"({result['n_episodes']} episodes)")
    for f, p in result["picks"].items():
        if p:
            typer.echo(f"  fold {f} picks: peak layer {p['peak']}, parts {', '.join(p['comps'])}, heads {', '.join(p['heads'])}")
    for kind, k in result["knockout"].items():
        r = k["recovered"]
        ctrl = f"  random mean {k['random_mean']:+.3f}, max {k['random_max']:+.3f}" if "random_mean" in k else "  (descriptive)"
        typer.echo(f"  knockout {kind:8s} recovered {r['value']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]{ctrl}")
    b = result["baseline"]["prefers_correct"]
    typer.echo(f"  prefers_correct     baseline {b['value']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]")
    for a, s in result["steer"].items():
        p = s["prefers_correct"]
        typer.echo(f"  steer alpha {a:4s}    {p['value']:.3f} [{p['lo']:.3f}, {p['hi']:.3f}]  arm_pref {s['arm_pref']['value']:.3f}  damage {s['damage']['value']:+.3f}")
    typer.echo(f"wrote {out}")


@app.command("gripper-sae")
def gripper_sae_cmd(
    clips: Annotated[
        Path,
        typer.Option("--clips", exists=True, file_okay=False, help="Clips from `ii get-clips --with-states` (video plus arm poses)."),
    ],
    out: Annotated[Path, typer.Option("--out", help="Directory for gripper_sae.json and the SAE.")],
    layer: Annotated[int, typer.Option("--layer", min=0, help="Predictor layer (0 = input, i = block i's output), e.g. where `ii gripper-trace` peaks.")],
    code_dir: Annotated[Path | None, typer.Option("--code-dir", help="Checkout of github.com/facebookresearch/vjepa2. Not needed with --tiny.")] = None,
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint", help="Local vjepa2-ac-vitg.pt.")] = None,
    tiny: Annotated[bool, typer.Option("--tiny", help="Random CPU stand-in (tests).")] = False,
    max_clips: Annotated[int | None, typer.Option("--max-clips", min=1, help="Use only the first N clips.")] = None,
    n_latents: Annotated[int, typer.Option("--latents", min=2, help="SAE dictionary size.")] = 4096,
    k: Annotated[int, typer.Option("--k", min=1, help="BatchTopK k.")] = 32,
    steps: Annotated[int, typer.Option("--steps", min=1, help="SAE training steps.")] = 3000,
    stride: Annotated[int, typer.Option("--stride", min=1, help="Use every Nth real transition for SAE training rows.")] = 2,
    random_saes: Annotated[int, typer.Option("--random-saes", min=1, help="Untrained SAEs as the control.")] = 5,
    seed: Annotated[int, typer.Option("--seed", help="Seed for the folds, SAE, controls and bootstrap.")] = 0,
    bootstrap: Annotated[int, typer.Option("--bootstrap", min=1, help="Episode bootstrap resamples.")] = 2000,
) -> None:
    """Is there a gripper-command feature inside V-JEPA 2-AC's predictor? (Offline, recorded DROID video.)

    Trains an SAE on the predictor's frame tokens at one layer, picks the latents that move most with
    the gripper command on half the episodes, and tests them on the other half against untrained SAEs.
    The verdict rule is fixed in the code.
    """
    from interpretableintelligence.adapters.vjepa2ac.model import load_vjepa2ac, tiny_world_model
    from interpretableintelligence.device import pick_device
    from interpretableintelligence.worldmodel.action_effects import load_trajectories
    from interpretableintelligence.worldmodel.gripper_sae import run_gripper_sae

    try:
        if tiny:
            model = tiny_world_model(seed=seed)
        else:
            if code_dir is None:
                raise ValueError("--code-dir is required: a checkout of github.com/facebookresearch/vjepa2")
            model = load_vjepa2ac(pick_device(), code_dir=code_dir, checkpoint=checkpoint, seed=seed)
            model.predict_batch = 32
        trajectories = load_trajectories(clips, max_clips=max_clips)
        if not trajectories:
            raise ValueError(f"no usable clips in {clips}")
        result = run_gripper_sae(model, trajectories, out, layer=layer, n_latents=n_latents, k=k, steps=steps, stride=stride,
                                 random_saes=random_saes, seed=seed, n_boot=bootstrap, say=typer.echo)
    except (ValueError, RuntimeError, FileNotFoundError, OSError, KeyError, ImportError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"verdict: {result['verdict']} at layer {layer} on {result['n_test_events']} held-out gripper events")
    for name, arm in result["arms"].items():
        c, r = arm["consistency"], arm["recovered"]
        typer.echo(f"  {name:24s} consistency {c['value']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]  recovered {r['value']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]")
    typer.echo(f"wrote {out}")


@app.command("export")
def export_cmd(
    run_dir: Annotated[Path, typer.Argument(help="Run folder to export.")],
    out: Annotated[Path, typer.Option("--out", help="Directory to write the static site into.")],
    clips: Annotated[
        Path | None,
        typer.Option("--clips", help="Directory of clip files. Overrides a clip path recorded in the manifest."),
    ] = None,
    steer: Annotated[
        bool,
        typer.Option("--steer/--no-steer", help="Precompute playground edits so the static page can still steer features."),
    ] = True,
    steer_clips: Annotated[
        int, typer.Option("--steer-clips", min=0, help="How many clips get a precomputed playground.")
    ] = 12,
    steer_features: Annotated[
        int, typer.Option("--steer-features", min=1, help="Strongest features per clip window to precompute.")
    ] = 6,
    title: Annotated[str | None, typer.Option("--title", help="Page title, also used on the link preview.")] = None,
    max_clips: Annotated[
        int | None,
        typer.Option(
            "--max-clips",
            min=1,
            help="Carry only this many clips (top examples of verified features first), to keep a large run small enough to host.",
        ),
    ] = None,
) -> None:
    """Export a run as a static site: host the folder anywhere and share the link. No server needed."""
    from interpretableintelligence.explain import ExplainError
    from interpretableintelligence.export_run import export_run

    try:
        summary = export_run(
            run_dir,
            out,
            clips_dir=clips,
            steer=steer,
            steer_clips=steer_clips,
            steer_features=steer_features,
            title=title,
            max_clips=max_clips,
            log=typer.echo,
        )
    except (ValueError, RuntimeError, FileNotFoundError, OSError, KeyError, ExplainError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    playground = (
        f"playground on {len(summary.steer_clips)} clips ({summary.steer_results} edits)"
        if summary.steer_clips
        else "playground off"
    )
    typer.echo(f"wrote {summary.out_dir}: {summary.clips} clips, {summary.features} features, {playground}, {summary.bytes_total / 1e6:.1f} MB")
    typer.echo(f"preview: python -m http.server -d {summary.out_dir} 8000, then open http://localhost:8000")


@app.command("export-live")
def export_live_cmd(
    run_dir: Annotated[Path, typer.Argument(help="Run folder to pack (an hf image model with a trained SAE).")],
    out: Annotated[Path, typer.Option("--out", help="Directory to write the pack into, e.g. demo-site/live/packs/<name>.")],
    clips: Annotated[
        Path | None,
        typer.Option("--clips", help="Directory of clip files, for the sample frames. Overrides a clip path recorded in the manifest."),
    ] = None,
    meta_from: Annotated[
        Path | None,
        typer.Option(
            "--meta-from",
            help="Copy labels and badges from an ii export of the same run (checked against this SAE) instead of the run's scoreboard.",
        ),
    ] = None,
    samples: Annotated[int, typer.Option("--samples", min=0, help="Sample frames to ship with the pack.")] = 16,
    title: Annotated[str | None, typer.Option("--title", help="Name shown in the playground.")] = None,
    readout: Annotated[
        str,
        typer.Option(
            "--readout",
            help="tokens: how far an edit turns the model's final tokens. words (CLIP models): what the model sees, in words, and which features make it see each one.",
        ),
    ] = "tokens",
    words: Annotated[Path | None, typer.Option("--words", help="With --readout words: text file of words, one per line (default: a built-in list of everyday things).")] = None,
    sample_dir: Annotated[
        Path | None,
        typer.Option("--sample-dir", help="Ship the pictures in this folder (optional credits.json: file name to credit line) instead of clip frames."),
    ] = None,
) -> None:
    """Pack a run for the live in-browser playground: the model split at the SAE's layer, the SAE, labels and badges."""
    from interpretableintelligence.live import LiveExportError, export_live

    try:
        summary = export_live(
            run_dir, out, clips_dir=clips, meta_from=meta_from, n_samples=samples, title=title, log=typer.echo,
            readout=readout, words_path=words, sample_dir=sample_dir,
        )
    except (LiveExportError, FileNotFoundError, OSError, KeyError, ValueError, RuntimeError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"wrote {summary.out_dir}: {summary.n_features} features ({summary.n_real} verified, from the {summary.meta_source}), "
        f"{summary.samples} samples, {summary.bytes_total / 1e6:.1f} MB"
    )


@app.command("name-live")
def name_live_cmd(
    pack: Annotated[Path, typer.Argument(help="A words pack (ii export-live --readout words), e.g. demo-site/live/packs/clip-b16-words.")],
    pictures: Annotated[int, typer.Option("--pictures", min=2, help="Sample pictures to write each feature onto (split in two halves that must agree).")] = 8,
    z_min: Annotated[float, typer.Option("--z-min", help="How far above other features a word must rise to name a feature (z, averaged over pictures).")] = 3.0,
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint", help="An .npz file to save progress to and resume from.")] = None,
) -> None:
    """Name each feature of a words pack by the word it makes the model see, measured through the pack's own model."""
    from interpretableintelligence.live_names import name_pack

    try:
        out = name_pack(pack, n_pictures=pictures, z_min=z_min, log=typer.echo, checkpoint=checkpoint)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"named {out['named']} of {out['features']} features on {len(out['pictures'])} pictures")


@app.command("gallery-live")
def gallery_live_cmd(
    pack: Annotated[Path, typer.Argument(help="A pack, e.g. demo-site/live/packs/clip-b16-words.")],
    images: Annotated[Path, typer.Option("--images", help="A folder of pictures to look through (jpg, png, webp).")],
    credits: Annotated[
        Path | None, typer.Option("--credits", help='A JSON file {"<file stem>": "credit line"}; shown on each picture.')
    ] = None,
    per_feature: Annotated[int, typer.Option("--per-feature", min=1, help="Pictures kept for each feature.")] = 6,
    min_frac: Annotated[float, typer.Option("--min-frac", help="Keep a picture only where the feature reaches this share of its training peak.")] = 0.25,
    limit: Annotated[int | None, typer.Option("--limit", min=1, help="Read only the first N pictures (sorted by name).")] = None,
) -> None:
    """Find where each feature of a pack lights up most on a folder of pictures, through the pack's own model (gallery/ in the pack)."""
    from interpretableintelligence.live_gallery import build_gallery

    try:
        names = json.loads(credits.read_text(encoding="utf-8")) if credits else None
        out = build_gallery(pack, images, names, per_feature=per_feature, min_frac=min_frac, limit=limit, log=typer.echo)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"read {out['read']} pictures; {out['features']} of {out['of']} features have some, from {out['pictures']} pictures kept")


@app.command("bake-live")
def bake_live_cmd(
    pack: Annotated[Path, typer.Argument(help="A words pack inside the playground folder, e.g. demo-site/live/packs/clip-b16-words.")],
    page: Annotated[Path | None, typer.Option("--page", help="The playground folder (default: the folder holding packs/).")] = None,
    ort_dir: Annotated[
        Path | None,
        typer.Option("--ort-dir", help="Serve ONNX Runtime Web from this local dist/ folder instead of cdn.jsdelivr.net."),
    ] = None,
) -> None:
    """Work out the playground's first answer for each example picture ahead of time (instant.json), so the page answers before the model loads."""
    from interpretableintelligence.live_bake import BakeError, bake_live

    try:
        summary = bake_live(pack, page=page, ort_dir=ort_dir, log=typer.echo)
    except (BakeError, OSError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"wrote {summary.path}: {summary.samples} pictures, {summary.bytes / 1e3:.0f} KB (baked on {summary.backend})")


@app.command("export-text")
def export_text_cmd(
    pack: Annotated[Path, typer.Argument(help="A words pack, e.g. demo-site/live/packs/clip-b16-words.")],
    model: Annotated[str | None, typer.Option("--model", help="Hugging Face id or local folder of the CLIP model (default: the one pack.json names).")] = None,
) -> None:
    """Add CLIP's text half to a words pack, so the playground can read any word, not only its list (ii export-live does this for new packs)."""
    from interpretableintelligence.live_words import WordsError, add_text_encoder

    try:
        files = add_text_encoder(pack, model, log=typer.echo)
    except (WordsError, OSError, ValueError, KeyError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"wrote {', '.join(str(v) for v in files.values())} into {pack}")


LIVE_DIR = Path("demo-site/live")


@app.command("serve-live")
def serve_live_cmd(
    packs: Annotated[Path, typer.Option("--packs", help="Directory of live packs (each with pack.json and sae.safetensors).")] = LIVE_DIR / "packs",
    pack: Annotated[list[str] | None, typer.Option("--pack", help="Serve only this pack (repeatable). Default: every pack.")] = None,
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8787,
    origins: Annotated[str, typer.Option("--origins", help="Comma-separated origins allowed to call the API (CORS).")] = "*",
    builds: Annotated[
        Path | None,
        typer.Option("--builds", help="Let the page build packs for Hugging Face models it names, kept in this directory (bring your own model)."),
    ] = None,
    build_clips: Annotated[Path, typer.Option("--build-clips", help="Clips a build trains its dictionary on (ii get-clips --n 40).")] = Path("clips/droid"),
    builds_per_day: Annotated[int, typer.Option("--builds-per-day", min=1, help="Most new builds a day.")] = 10,
    max_params: Annotated[int, typer.Option("--max-params", min=1, help="Largest model a build accepts, in millions of parameters.")] = 350,
    allow_local_models: Annotated[bool, typer.Option("--allow-local-models", help="Also build from model directories on this machine.")] = False,
) -> None:
    """Run the live playground's model server on this machine (the API in docs/PLAYGROUND-API.md).

    Open the feature playground (/live/features.html) with ?api=http://127.0.0.1:8787 to use it. ii deploy-live hosts the same thing on Modal.
    With --builds, the page can also name a Hugging Face vision model and get a dictionary for it;
    each build runs ii run hf --no-score on the --build-clips clips here, then packs it.
    """
    import uvicorn

    from interpretableintelligence.playground.compute import PackLoadError, load_packs
    from interpretableintelligence.playground.http import create_app

    try:
        models = load_packs(packs, only=pack or None)
    except (PackLoadError, OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for model in models.values():
        typer.echo(f"loaded {model.id} ({model.pack['model']['name']}, {model.n_features} features) on {model.device.type} in {model.load_ms / 1000:.1f} s")
    table = models
    builder = None
    if builds is not None:
        from interpretableintelligence.clips import list_clip_meta
        from interpretableintelligence.playground.builds import Builds, JobStore, Limits, thread_runner
        from interpretableintelligence.playground.registry import PackRegistry

        n_clips = len(list_clip_meta(build_clips)) if build_clips.is_dir() else 0
        if n_clips < MIN_CLIPS:
            typer.echo(f"error: --builds needs at least {MIN_CLIPS} clips in {build_clips}; run `ii get-clips --n 40` first", err=True)
            raise typer.Exit(code=1)
        store = JobStore(builds)
        limits = Limits(max_params=max_params * 1_000_000, builds_per_day=builds_per_day, n_clips=n_clips, allow_local=allow_local_models)
        builder = Builds(store, thread_runner(store, build_clips), limits)
        table = PackRegistry(models, built_dir=store.root / "packs")
        typer.echo(f"builds on: packs for new models go to {store.root / 'packs'} (at most {builds_per_day} a day)")
    typer.echo(f"playground API on http://{host}:{port}/v1/health")
    app_ = create_app(table, allow_origins=[o.strip() for o in origins.split(",") if o.strip()], builds=builder)
    uvicorn.run(app_, host=host, port=port, log_level="warning")


@app.command("deploy-live")
def deploy_live_cmd(
    packs: Annotated[Path, typer.Option("--packs", help="Directory of live packs to serve.")] = LIVE_DIR / "packs",
    gpu: Annotated[str, typer.Option("--gpu", help="Modal GPU type, or none for CPU only (enough for DINOv2-small).")] = "T4",
    idle: Annotated[int, typer.Option("--idle", min=2, help="Seconds a container stays warm after the last request.")] = 300,
    max_containers: Annotated[int, typer.Option("--max-containers", min=1, help="Most containers at once; caps the bill.")] = 2,
    origins: Annotated[str, typer.Option("--origins", help="Comma-separated origins allowed to call the API (CORS).")] = "*",
    live_dir: Annotated[Path, typer.Option("--live-dir", help="The page's folder; its config.json gets the API URL.")] = LIVE_DIR,
    api: Annotated[
        str | None,
        typer.Option("--api", help="Skip deploying: just point the page at this API URL (or 'none' to run in the browser)."),
    ] = None,
    builds: Annotated[
        bool,
        typer.Option("--builds/--no-builds", help="Let visitors name a Hugging Face vision model and get a dictionary for it (GPU jobs on your account, capped below)."),
    ] = True,
    build_gpu: Annotated[str, typer.Option("--build-gpu", help="Modal GPU type for builds.")] = "L4",
    builds_per_day: Annotated[int, typer.Option("--builds-per-day", min=1, help="Most new builds a day (UTC).")] = 10,
    build_timeout: Annotated[int, typer.Option("--build-timeout", min=120, help="Seconds a build may run before it is stopped.")] = 1200,
    max_params: Annotated[int, typer.Option("--max-params", min=1, help="Largest model a build accepts, in millions of parameters.")] = 350,
) -> None:
    """Host the playground backend on Modal (your account; scales to zero) and point the page at it.

    With builds on (the default), the page can name a Hugging Face vision model; one GPU build at a
    time trains a dictionary for it on bundled robot clips. Builds are capped per day and in length,
    so the most they can cost a day is printed before deploying.
    """
    from interpretableintelligence.playground.deploy import DeployError, build_cost_line, deploy, write_config

    try:
        if api is None:
            build_env = None
            if builds:
                build_env = {
                    "II_BUILD_GPU": build_gpu,
                    "II_BUILD_PER_DAY": str(builds_per_day),
                    "II_BUILD_TIMEOUT": str(build_timeout),
                    "II_BUILD_MAX_PARAMS": str(max_params * 1_000_000),
                }
                typer.echo(build_cost_line(build_gpu, builds_per_day, build_timeout))
            api = deploy(packs, gpu=gpu, idle_s=idle, max_containers=max_containers, origins=origins, log=typer.echo, build_env=build_env)
        url = None if api.lower() == "none" else api
        path = write_config(live_dir, url)
    except (DeployError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"wrote {path}: the page now {'calls ' + url if url else 'runs the model in the browser'}. Commit it to publish.")


@app.command("export-site")
def export_site_cmd(
    out: Annotated[Path, typer.Option("--out", help="Directory to write the static site into.")] = Path("demo-site"),
    groot: Annotated[
        Path,
        typer.Option("--groot", help="Directory with groot-1028-ablate-summary.json and its method checks."),
    ] = Path("demo-data/groot"),
) -> None:
    """Regenerate the static site (home, your-model page, blog), keeping the live playground."""
    from interpretableintelligence.export_site import export_site

    try:
        dest = export_site(out, groot_dir=groot)
    except (ValueError, RuntimeError, FileNotFoundError, OSError, KeyError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"wrote {dest}")


@app.command("import")
def import_cmd(
    source: Annotated[Path, typer.Argument(help="Capture to import: a directory, or a .tar / .tar.xz / .tar.gz of one.")],
    out: Annotated[Path, typer.Argument(help="Run folder to write. Must not exist or must be empty.")],
    format_name: Annotated[str, typer.Option("--format", help="Source layout. Known: groot-v9.")],
    model_sha: Annotated[
        str | None,
        typer.Option(
            "--model-sha",
            help="manifest.model.sha to record. Default: the same sha ii run groot computes (needs the Hugging Face Hub), "
            "else a digest of the declared model; the manifest records which.",
        ),
    ] = None,
    fps: Annotated[float, typer.Option("--fps", help="Env steps per second, recorded as clips[].fps (LIBERO: 20).")] = 20.0,
) -> None:
    """Import rollout logs captured outside ii into a run folder that ii validate accepts.

    split.json is copied byte for byte. Metrics that need the model in the loop are null, each with a reason in
    manifest.import.nulls, and every feature stays real: false.
    """
    from interpretableintelligence.importers import import_run

    try:
        import_run(format_name, source, out, log=typer.echo, model_sha=model_sha, fps=fps)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"next: ii validate {out}")


@app.command("retrain-sae")
def retrain_sae_cmd(
    capture: Annotated[Path, typer.Argument(help="Unpacked GR00T capture folder (groot-1028-v9-n100 layout: acts/, split.json, sae/).")],
    out: Annotated[Path, typer.Option("--out", help="Folder to write: the new sae/, retrain.json, and links to the capture's SAE-independent files.")],
    n_latents: Annotated[int, typer.Option("--n-latents", min=1, help="Dictionary size.")] = 2048,
    k: Annotated[int, typer.Option("--k", min=1, help="BatchTopK: active latents per token on average.")] = 32,
    steps: Annotated[int, typer.Option("--steps", min=1, help="Optimiser steps.")] = 6000,
    batch_tokens: Annotated[int, typer.Option("--batch-tokens", min=1, help="Tokens per batch.")] = 4096,
    lr: Annotated[float, typer.Option("--lr", help="Adam learning rate.")] = 4e-4,
    revive: Annotated[
        str, typer.Option("--revive", help="Dead-latent revival: auxk (AuxK loss), resample (reinitialise dead latents on high-error tokens), or none.")
    ] = "auxk",
    aux_coeff: Annotated[float, typer.Option("--aux-coeff", help="AuxK loss weight.")] = 1 / 32,
    aux_k: Annotated[int, typer.Option("--aux-k", min=0, help="Dead latents per token in the AuxK reconstruction.")] = 512,
    dead_tokens: Annotated[
        int, typer.Option("--dead-tokens", min=0, help="Training tokens without firing before a latent counts as dead. 0: one pass over the train split.")
    ] = 0,
    resample_every: Annotated[int, typer.Option("--resample-every", min=0, help="--revive resample: steps between resamples (first 60% of training). 0: steps // 10.")] = 0,
    seed: Annotated[int, typer.Option("--seed", help="Init and batch-order seed. random:<seed> in the rollout control is this SAE's init.")] = 0,
    norm: Annotated[str, typer.Option("--norm", help="keep: reuse the capture SAE's norm_mu / norm_scale. fit: recompute from train rows. channel: per-channel standardisation (norm_scale becomes a [D] vector).")] = "keep",
    device: Annotated[str, typer.Option("--device", help="auto, cpu, or cuda.")] = "auto",
) -> None:
    """Retrain the GR00T capture SAE with dead-latent revival and compare it with the old one.

    BatchTopK with AuxK or resampling, trained on split.json's train episodes only, written in the capture's sae/ layout
    so the picker, ii import and ii rollouts --pick read it unchanged. Reports alive, dead and dense latents, L0
    and explained variance on train and held-out for both SAEs. Offline: never a finding, never touches a scoreboard.
    """
    import torch

    from interpretableintelligence.adapters.groot.retrain_sae import RetrainConfig, format_report, retrain

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    config = RetrainConfig(
        n_latents=n_latents,
        k=k,
        steps=steps,
        batch_tokens=batch_tokens,
        lr=lr,
        aux_coeff=aux_coeff,
        aux_k=aux_k,
        dead_tokens=dead_tokens,
        revive=revive,
        resample_every=resample_every,
        seed=seed,
        norm=norm,
    )
    try:
        report = retrain(capture, out, config, torch.device(device), log=typer.echo)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(format_report(report))
    typer.echo(f"wrote {out}")


@app.command("map-feature")
def map_feature_cmd(
    capture: Annotated[Path, typer.Argument(help="Unpacked GR00T capture folder (acts/, split.json) both SAEs were trained on.")],
    from_sae: Annotated[Path, typer.Option("--from-sae", help="SAE folder the feature id belongs to (e.g. the capture's sae/).")],
    to_sae: Annotated[Path, typer.Option("--to-sae", help="SAE folder to find its counterpart in (e.g. an ii retrain-sae output's sae/).")],
    feature: Annotated[int, typer.Option("--feature", min=0, help="Feature id in --from-sae.")],
    out: Annotated[Path, typer.Option("--out", help="Folder for mapping.json (and the picker runs with --direction).")],
    direction: Annotated[
        bool, typer.Option("--direction/--no-direction", help="Also run the GR00T picker through both SAEs and check each mapped feature fires more before failure.")
    ] = True,
) -> None:
    """Find a GR00T SAE feature's counterpart in another dictionary trained on the same activations.

    Activation correlation and raw-unit decoder cosine on the capture's held-out tokens decide one counterpart, a split
    into up to 3 features, or none, by fixed rules (adapters/groot/map_feature.py). Offline: never a finding.
    """
    from interpretableintelligence.adapters.groot.map_feature import map_feature

    try:
        report = map_feature(capture, from_sae, to_sae, feature, out, direction=direction, log=typer.echo)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    decision = report["decision"]
    typer.echo(f"{decision['outcome']}: {decision['features']}")
    if report["direction"] is not None:
        typer.echo(f"direction check: {'pass' if report['direction']['passed'] else 'fail'}")
    typer.echo(f"wrote {Path(out) / 'mapping.json'}")


@app.command("compare")
def compare_cmd(
    runs: Annotated[list[Path], typer.Argument(help="Two or more finished run folders that captured the same clips.")],
    out: Annotated[Path, typer.Option("--out", help="Folder for compare.json and index.html.")],
    clips: Annotated[
        Path | None,
        typer.Option("--clips", help="Clip directory for thumbnails. Defaults to the first run's recorded clips_dir."),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Shuffle-null seed.")] = 0,
) -> None:
    """Match SAE features across models on the same clips: which concepts they share and which are unique."""
    from interpretableintelligence.compare import compare_runs, describe_groups, write_compare

    report = compare_runs(list(runs), seed=seed, log=typer.echo)
    clips_dir = clips
    if clips_dir is None and report["_profiles"][0].clips_dir:
        clips_dir = Path(report["_profiles"][0].clips_dir)
    describe_groups(report, clips_dir)
    write_compare(report, out, clips_dir)
    shared = report["shared"]
    every = sum(1 for group in shared if group["n_models"] == len(runs))
    typer.echo(
        f"{len(shared)} shared concepts ({every} in all {len(runs)} models), "
        f"{len(report['unmatched_verified'])} verified features with no counterpart. Wrote {out / 'index.html'}"
    )


@app.command()
def validate(
    run_dir: Annotated[Path, typer.Argument(help="Run folder to check against the schema.")],
) -> None:
    """Validate a run folder. Exits 0 when the contract holds."""
    from interpretableintelligence.validate import validate_run

    errors = validate_run(run_dir)
    if errors:
        for error in errors:
            typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"valid: {run_dir}")


@app.command()
def view(
    run_dir: Annotated[Path, typer.Argument(help="Run folder to open.")],
    clips: Annotated[
        Path | None,
        typer.Option(
            "--clips",
            help="Directory of clip files. Overrides a clip path recorded in the manifest.",
        ),
    ] = None,
    host: Annotated[str, typer.Option("--host", help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port. 0 chooses a free port.")] = 7860,
    no_open: Annotated[bool, typer.Option("--no-open", help="Do not open a browser.")] = False,
    steer: Annotated[
        bool,
        typer.Option(
            "--steer",
            help="Load the run's model and SAE and turn on the playground: steer features and see the prediction move.",
        ),
    ] = False,
) -> None:
    """Start a local viewer for a run folder and open it in a browser."""
    from interpretableintelligence.view import serve_run

    errors: tuple[type[Exception], ...] = (ValueError, FileNotFoundError, OSError)
    if steer:
        from interpretableintelligence.explain import ExplainError

        errors = (*errors, ExplainError)
    try:
        serve_run(run_dir, clips_dir=clips, host=host, port=port, open_browser=not no_open, steer=steer)
    except errors as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
