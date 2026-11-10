"""physint command line."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from physint import __version__
from physint.adapters.fake import generate_clips
from physint.pipeline import execute_pipeline
from physint.thresholds import DEFAULT_HELDOUT_FRAC, DEFAULT_LAYER, DEFAULT_SEED
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
    adapter: Annotated[str, typer.Argument(help="Adapter name. This lane ships 'fake'.")],
    clips: Annotated[Path, typer.Option("--clips", exists=True, file_okay=False, help="Clip directory.")],
    out: Annotated[Path, typer.Option("--out", help="Run folder, e.g. runs/demo.")],
    layer: Annotated[int, typer.Option("--layer", min=0, help="Layer to instrument.")] = DEFAULT_LAYER,
    seed: Annotated[int, typer.Option("--seed", help="Split, SAE, and random-feature seed.")] = DEFAULT_SEED,
    heldout_frac: Annotated[
        float,
        typer.Option("--heldout-frac", help="Fraction of clips held out."),
    ] = DEFAULT_HELDOUT_FRAC,
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
