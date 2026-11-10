"""physint command line."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from physint import __version__
from physint.adapters.fake import generate_clips
from physint.run import execute_rank, execute_run
from physint.thresholds import DEFAULT_HELDOUT_FRAC, DEFAULT_LAYER, DEFAULT_PREFILTER_K, DEFAULT_SEED
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
    prefilter: Annotated[
        int | None,
        typer.Option(
            "--prefilter",
            min=1,
            help="Ablate and run random-feature controls only for the K features with the largest attribution. Omit to score every feature.",
        ),
    ] = None,
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
            prefilter=prefilter,
        )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
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
        )
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


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
