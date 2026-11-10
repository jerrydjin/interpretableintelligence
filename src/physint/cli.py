"""physint command line."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from physint import __version__
from physint.adapters.fake import generate_clips
from physint.rollout import execute_rollouts, format_report, parse_feature_ids
from physint.run import execute_run
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
    adapter: Annotated[str, typer.Argument(help="Adapter name. This lane ships 'fake'.")],
    clips: Annotated[Path, typer.Option("--clips", exists=True, file_okay=False, help="Clip directory.")],
    out: Annotated[Path, typer.Option("--out", help="Run folder, e.g. runs/demo.")],
    layer: Annotated[int, typer.Option("--layer", min=0, help="Layer to instrument.")] = DEFAULT_LAYER,
    seed: Annotated[int, typer.Option("--seed", help="Split, SAE, and random-feature seed.")] = DEFAULT_SEED,
    heldout_frac: Annotated[
        float,
        typer.Option("--heldout-frac", help="Fraction of clips held out."),
    ] = DEFAULT_HELDOUT_FRAC,
) -> None:
    """Capture activations, train an SAE, label features, score them, write a run folder."""
    try:
        execute_run(
            adapter_name=adapter,
            clips_dir=clips,
            out_dir=out,
            layer=layer,
            seed=seed,
            heldout_frac=heldout_frac,
        )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


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
