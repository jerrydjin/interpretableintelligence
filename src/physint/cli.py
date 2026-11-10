"""physint command line."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from physint import __version__
from physint.adapters.fake import generate_clips
from physint.explain import ExplainError, explain_clip
from physint.run import execute_run
from physint.thresholds import DEFAULT_HELDOUT_FRAC, DEFAULT_LAYER, DEFAULT_SEED
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
    host: Annotated[str, typer.Option("--host", help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535, help="Bind port.")] = 8000,
) -> None:
    """Serve GET /runs, GET /features/{id}, and POST /explain on localhost."""
    import uvicorn

    from physint.api import create_app

    uvicorn.run(create_app(runs), host=host, port=port)


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
