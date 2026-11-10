"""physint command line."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from physint import __version__
from physint.adapters.fake import generate_clips
from physint.explain import ExplainError, explain_clip
from physint.pipeline import execute_pipeline
from physint.rollout import execute_rollouts, format_report, parse_feature_ids
from physint.thresholds import (
    DEFAULT_HELDOUT_FRAC,
    DEFAULT_LAYER,
    DEFAULT_ROLLOUT_ALPHA,
    DEFAULT_ROLLOUT_BATCH,
    DEFAULT_ROLLOUT_CONTROLS,
    DEFAULT_ROLLOUT_EPISODES,
    DEFAULT_ROLLOUT_MIN_EPISODES,
    DEFAULT_ROLLOUT_WORKERS,
    DEFAULT_SEED,
)
from physint.validate import validate_run
from physint.view import serve_run

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
    generate_clips(out, n=n, seed=seed)
    typer.echo(f"wrote {n} clips to {out}")


@app.command()
def run(
    adapter: Annotated[str, typer.Argument(help="Adapter name: fake or vjepa2.")],
    clips: Annotated[Path, typer.Option("--clips", exists=True, file_okay=False, help="Clip directory.")],
    out: Annotated[Path, typer.Option("--out", help="Run folder, e.g. runs/demo.")],
    layer: Annotated[int, typer.Option("--layer", min=0, help="Layer to instrument.")] = DEFAULT_LAYER,
    seed: Annotated[int, typer.Option("--seed", help="Split, SAE, and random-feature seed.")] = DEFAULT_SEED,
    heldout_frac: Annotated[
        float,
        typer.Option("--heldout-frac", help="Fraction of clips held out."),
    ] = DEFAULT_HELDOUT_FRAC,
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
    try:
        execute_pipeline(
            adapter_name=adapter,
            clips_dir=clips,
            out_dir=out,
            layer=layer,
            seed=seed,
            heldout_frac=heldout_frac,
            tiny=tiny,
            gpu_cost_per_hour=gpu_cost_per_hour,
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

    from physint.api import create_app

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


@app.command()
def validate(
    run_dir: Annotated[Path, typer.Argument(help="Run folder to check against the schema.")],
) -> None:
    """Validate a run folder. Exits 0 when the contract holds."""
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
) -> None:
    """Start a local viewer for a run folder and open it in a browser."""
    try:
        serve_run(run_dir, clips_dir=clips, host=host, port=port, open_browser=not no_open)
    except (ValueError, FileNotFoundError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
