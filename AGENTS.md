# Agent instructions (source of truth)

Shared rules for Claude Code, Codex, and other coding agents working in this repo.

## What this is

**interpretableintelligence** (`ii`) is infrastructure for mechanistic interpretability on video world models and robot / VLA policies: a shared **run folder** format, SAE training, offline scoreboard gates, local viewers, and causal rollouts.

The main library and CLI live under `src/interpretableintelligence/`. Adapters shipped on main are `fake` (acceptance test) and `vjepa2`. Robot work is **dogfooded** on real models in `src/interpretableintelligence/adapters/groot/`—currently **NVIDIA GR00T** checkpoint `nvidia/gr00t17-lerobot-libero_10-640` on **LIBERO-plus** (task 1028 is the active rise-test task).

Contract details: `SCHEMA.md`, `ARCHITECTURE.md`, `README.md`. Do not drift from them in agent docs.

## Setup and tests

From the repo root (Python ≥ 3.11; [uv](https://docs.astral.sh/uv/) recommended):

```bash
uv sync
uv run ii --help
uv run pytest
```

`interpretableintelligence` is the long command name; **`ii` is the same CLI**. PyTorch is pinned to CPU wheels in `pyproject.toml` for CI and local dev; GR00T/LIBERO jobs use separate GPU environments (Colab), not this default install.

On a fresh clone of `main`, expect **138 passed, 13 skipped** (`pytest` excludes `slow` and two deselected tests by default). Run `uv run pytest -m slow` only when you intend to download V-JEPA weights or run large synthetic paths.

## `ii` CLI (public surface)

All experiments that belong in this repo should be reachable through these commands (add a new subcommand before relying on a one-off script):

| Command | Purpose |
| --- | --- |
| `generate-clips` | Write deterministic synthetic clips for the `fake` adapter. |
| `run` | Capture activations, train SAE, label features, score, optionally serve the viewer (`fake`, `vjepa2`). |
| `rank` | Attribution prefilter shortlist; does not replace ablation. |
| `explain` | Explain a new clip; cites only scoreboard-`real` ablation effects. |
| `serve` | Python read API + `POST /explain` (Rust server in `server/` is preferred for production reads). |
| `rollouts` | Paired policy rollouts + matched random-feature controls → `rollouts/` (schema v1). |
| `export-site` | Regenerate static `demo-site/` from fake run + vendored measurements. |
| `validate` | Check a run folder against `SCHEMA.md`. |
| `view` | Local browser viewer for a finished run. |

World-model path: `ii run fake --clips … --out runs/…` or `vjepa2`. Policy path on main: `ii rollouts runs/… --policy fake` (LIBERO/GR00T integration is not wired into `ii rollouts` yet; see groot adapter lane).

## Run folder layout

Runs write under a directory you choose (convention: `runs/<run_id>/`). **Never commit run outputs** (`runs/` and `/clips/` are gitignored).

Version **0** (from `ii run`):

```
runs/<run_id>/
  manifest.json
  split.json
  acts/shard-NNNNN.safetensors
  sae/weights.safetensors
  sae/config.json
  features/feature_acts.parquet
  features/clip_effects.parquet
  features/attribution.parquet   # after rank or run
  features/candidates.json
  features/features.jsonl
  scoreboard.json
```

Version **1** adds `rollouts/episodes.parquet`, `rollouts/summary.json`, and `manifest.schema_version: 1` after `ii rollouts`. Activation stream caches under `acts/stream/` are resume artifacts, not part of the public contract. Clip pixels stay outside the run folder; the manifest records hashes and optional `clips_dir`.

## Research rules (hard)

These override intuition, offline plots, and demo-site copy.

**When a feature or edit counts as a real finding (robot / rollout lane):** all of the following must hold on shared seeds—otherwise it is **not** a finding, however large the offline effect:

1. The **Sobel–Wald SPRT** stop rule calls a **rise** (not merely a fall or cap).
2. The rise over the measured baseline on those seeds has **McNemar p < 0.05**.
3. The effect **beats every matched random control** arm in the plan.
4. It **beats the random-weights SAE control** (the trained SAE hook sanity arm; empty latent edit / recon-only path in groot ablation plans).

**Scoreboard (`scoreboard.json`):** changes only through the automated scorer (`ii run` scoring stage or equivalent validated pipeline), **never by hand**. Every feature stays `real: false` until its gates pass (`SCHEMA.md`: usable SAE plus diversity, label holdout, random-feature percentile, random-init absence). Offline `real` does **not** imply rollout causality.

**How to run work:**

- Go through the **public `ii` CLI** and write into the standard run layout. No ad-hoc scripts for production experiments. If tests need a capability the CLI lacks, **add a reusable `ii` command** first—build only what the experiment needs.
- **Never edit `split.json` or held-out labels** in an existing run.
- **No paid compute** on behalf of this repo (no cloud GPU rental, etc.). GR00T rollouts use existing Colab/box paths documented under `adapters/groot/`.

## Git workflow

- **Never push to `main`.** Branch from latest `main`, one topic per branch (`cursor/<topic>-bedd`).
- Open a PR; **Michael Truell** reviews and merges.
- Keep PRs small; **`uv run pytest` must stay green** before you push.

Docs-only PRs are fine; do not touch `scoreboard.json`, `split.json`, held-out labels, or committed run artifacts.

## Current state (GR00T task 1028)

The **single-feature** ablation on LIBERO-plus task **1028** was a formal **null**: **641** rollouts, **20** random controls, candidates **720** and **7081+5852**—candidate “falls” were **not distinguishable** from controls (`demo-data/groot/groot-1028-ablate-summary.json`, `real: false`). Do **not** re-run single-feature rise sweeps on 1028 without a **new experimental design** approved in the research thread.

Offline world-model acceptance remains `ii run fake` → exactly three `real` features (`stripe`, `pulse`, `corner`).

## Touch points

| Need | Read |
| --- | --- |
| Run schema / gates | `SCHEMA.md`, `src/interpretableintelligence/thresholds.py` |
| Pipeline stages | `ARCHITECTURE.md`, `src/interpretableintelligence/pipeline.py` |
| GR00T v9 ablation / SPRT | `src/interpretableintelligence/adapters/groot/v9/ablate_v9.py` |
| Demo bundles | `demo-data/`, `ii export-site` |
