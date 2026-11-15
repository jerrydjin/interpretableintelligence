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
    n: Annotated[int, typer.Option("--n", min=1, max=100, help="Number of clips, one per robot episode.")] = 100,
    seconds: Annotated[float, typer.Option("--seconds", min=0.5, help="Length of each clip.")] = 4.0,
) -> None:
    """Download real robot video to try a real model on: short clips from DROID (MIT licence).

    One 175 MB download, cached after the first time. Each clip is cut from the middle of a robot
    episode in a real kitchen or office and saved as an mp4.
    """
    from interpretableintelligence.sample_clips import DATASET, LICENSE, fetch_droid_clips

    try:
        entries = fetch_droid_clips(out, n=n, seconds=seconds, say=typer.echo)
    except (ValueError, RuntimeError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"wrote {len(entries)} clips from {DATASET} ({LICENSE} licence) to {out}")
    typer.echo("")
    typer.echo("Next, find the features a real vision model uses on them, and open the playground:")
    typer.echo(f"  ii run hf --model facebook/dinov2-small --clips {out} --out runs/droid --no-score --steer")
    typer.echo("That takes about 10 minutes on a laptop CPU. Every feature is explorable but none is verified yet.")
    typer.echo("To run the checks that mark features verified, run the same command without --no-score.")
    typer.echo("It reuses what is done and adds the scoring, which takes longer (up to an hour on a CPU).")


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
    clips: Annotated[Path, typer.Option("--clips", exists=True, file_okay=False, help="Clip directory.")],
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
    try:
        if not remote and (gpu is not None or max_cost is not None):
            raise ValueError("--gpu and --max-cost only apply with --remote")
        adapter = _adapter_name(adapter, model, blocks, tiny=tiny)
        if adapter.startswith("groot:"):
            tiny = False  # groot:tiny names the stand-in itself, so the run folder can rebuild it
        layer = _default_layer(adapter) if layer is None else layer
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
    clips: Annotated[
        Path | None,
        typer.Option("--clips", exists=True, file_okay=False, help="Directory of clip files (.npy)."),
    ] = None,
    host: Annotated[str, typer.Option("--host", help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535, help="Bind port.")] = 8000,
) -> None:
    """Serve the read API and POST /explain. Reads stay available for contract tests; the Rust server is the one to run."""
    import uvicorn

    from interpretableintelligence.api import create_app

    uvicorn.run(create_app(runs, clips), host=host, port=port)


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
    seed: Annotated[int, typer.Option("--seed", help="Base seed. Episode i uses seed+i on every arm.")] = 0,
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
) -> None:
    """Compare baseline and intervened task success. Resume skips episodes already written."""
    from interpretableintelligence.rollout import execute_rollouts, format_report, parse_feature_ids
    from interpretableintelligence.rollout_groot import execute_groot_rollouts, format_groot_report, is_groot_policy

    try:
        if is_groot_policy(policy):
            if pick is not None or random_sae:
                raise ValueError("--pick and --random-sae are not available for GR00T policies yet (research rule 4 known gap)")
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


@app.command("export-site")
def export_site_cmd(
    out: Annotated[Path, typer.Option("--out", help="Directory to write the static site into.")] = Path("demo-site"),
    vjepa: Annotated[
        Path,
        typer.Option("--vjepa", help="V-JEPA slim bundle: scoreboard, clips, viz, random-init."),
    ] = Path("demo-data/vjepa-slim"),
    picker: Annotated[
        Path,
        typer.Option("--picker", help="Directory with synth_results.json and synth_results_extra.json."),
    ] = Path("demo-data/picker"),
    groot: Annotated[
        Path,
        typer.Option("--groot", help="Directory with groot-1028-ablate-summary.json."),
    ] = Path("demo-data/groot"),
) -> None:
    """Regenerate the static demo site from a fresh fake run and the vendored measurements."""
    from interpretableintelligence.export_site import export_site

    try:
        dest = export_site(out, vjepa_dir=vjepa, picker_dir=picker, groot_dir=groot)
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
