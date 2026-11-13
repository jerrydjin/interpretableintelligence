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
    DEFAULT_SEED,
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


@app.command("generate-clips")
def generate_clips_cmd(
    out: Annotated[Path, typer.Option("--out", help="Directory to write synthetic clips into.")],
    n: Annotated[int, typer.Option("--n", min=1, help="Number of clips.")] = 30,
    seed: Annotated[int, typer.Option("--seed", help="Generator seed.")] = 0,
) -> None:
    """Write deterministic synthetic clips for the fake adapter."""
    from interpretableintelligence.adapters.fake import generate_clips

    generate_clips(out, n=n, seed=seed)
    typer.echo(f"wrote {n} clips to {out}")


def _adapter_name(adapter: str, model: str | None, blocks: str | None) -> str:
    """``hf`` plus ``--model``/``--blocks`` becomes the self-describing ``hf:<model>[#<blocks>]`` name."""
    if adapter == "hf":
        from interpretableintelligence.adapters.hf import make_spec

        if not model:
            raise ValueError("the hf adapter needs --model, e.g. --model facebook/dinov2-small")
        local = Path(model).expanduser()
        if local.is_dir():
            model = str(local.resolve())  # explain and the playground reload it from any cwd
        return make_spec(model, blocks)
    if model or blocks:
        raise ValueError("--model and --blocks only apply to the hf adapter")
    return adapter


def _default_layer(adapter: str) -> int:
    if adapter.startswith("hf:"):
        from interpretableintelligence.adapters.hf import middle_block

        return middle_block(adapter)
    return DEFAULT_LAYER


@app.command()
def run(
    adapter: Annotated[
        str,
        typer.Argument(help="Adapter: fake, vjepa2, or hf (any Hugging Face vision model; pass --model)."),
    ],
    clips: Annotated[Path, typer.Option("--clips", exists=True, file_okay=False, help="Clip directory.")],
    out: Annotated[Path, typer.Option("--out", help="Run folder, e.g. runs/demo.")],
    model: Annotated[
        str | None,
        typer.Option("--model", help="hf only: Hugging Face model id or local directory, e.g. facebook/dinov2-small."),
    ] = None,
    blocks: Annotated[
        str | None,
        typer.Option("--blocks", help="hf only: dotted path to the transformer block list, when auto-detection picks the wrong one."),
    ] = None,
    layer: Annotated[
        int | None,
        typer.Option("--layer", min=0, help=f"Layer to instrument. Default {DEFAULT_LAYER}; for hf, the middle block."),
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
            help="SAE dictionary size. Default 12 for fake and vjepa2; 8 per activation dimension for hf.",
        ),
    ] = None,
    concepts: Annotated[
        Path | None,
        typer.Option(
            "--concepts",
            exists=True,
            dir_okay=False,
            help="hf only: text file of concepts to label features with, one per line. Default: a built-in list of everyday objects, parts and textures.",
        ),
    ] = None,
    tiny: Annotated[
        bool,
        typer.Option(
            "--tiny",
            help="V-JEPA 2 CPU smoke path: tiny random config, no weight download. Pass --layer 1.",
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
) -> None:
    """Capture, train, extract features, score what is missing, then serve the viewer.

    A stage whose outputs are already valid is skipped. The run prints a cost report
    from measured stage time. Pass --gpu-cost-per-hour to price an hour of footage.
    """
    from interpretableintelligence.pipeline import execute_pipeline

    modes = tuple(part.strip() for part in extra_effects.split(",") if part.strip())
    try:
        adapter = _adapter_name(adapter, model, blocks)
        layer = _default_layer(adapter) if layer is None else layer
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
            log=typer.echo,
        )
    except (ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc




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
        str,
        typer.Option(
            "--objective",
            help="prediction_loss, action_loglik, or activation (magnitude baseline). The result is a shortlist.",
        ),
    ] = "prediction_loss",
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
        typer.Option("--policy", help="Policy+simulator name. This lane ships 'fake' and 'fake-null'."),
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
) -> None:
    """Compare baseline and intervened task success. Resume skips episodes already written."""
    from interpretableintelligence.rollout import execute_rollouts, format_report, parse_feature_ids

    try:
        feature_ids = parse_feature_ids(features)
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
        )
    except (ValueError, RuntimeError, FileNotFoundError, NotImplementedError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(format_report(summary))
    typer.echo(f"wrote {run_dir / 'rollouts'}")


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
