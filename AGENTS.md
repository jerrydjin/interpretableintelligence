# Agent instructions (source of truth)

Shared rules for Claude Code, Codex, and other coding agents working in this repo.

## What this is

**interpretableintelligence** (`ii`) is infrastructure for mechanistic interpretability on video world models and robot / VLA policies: a shared **run folder** format, SAE training, offline scoreboard gates, local viewers, and causal rollouts.

The main library and CLI live under `src/interpretableintelligence/`. Adapters shipped on main are `fake` (acceptance test), `vjepa2`, the generic `hf:<model>` adapter (any ViT-style Hugging Face vision model, `ii run hf --model <id>`), and `groot:<checkpoint>` (NVIDIA GR00T N1.7 through LeRobot, `ii run groot`; needs `lerobot[groot]==0.7.0`, and `--tiny` is a random-weights stand-in for tests). Robot work is **dogfooded** on real models in `src/interpretableintelligence/adapters/groot/`—currently **NVIDIA GR00T** checkpoint `nvidia/gr00t17-lerobot-libero_10-640` on **LIBERO-plus** (task 1028 is the active rise-test task).

Contract details: `SCHEMA.md`, `ARCHITECTURE.md`, `README.md`. Do not drift from them in agent docs.

## Setup and tests

From the repo root (Python ≥ 3.11; [uv](https://docs.astral.sh/uv/) recommended):

```bash
uv sync
uv run ii --help
uv run pytest
```

`interpretableintelligence` is the long command name; **`ii` is the same CLI**. PyTorch is pinned to CPU wheels in `pyproject.toml` for CI and local dev; GR00T/LIBERO jobs use separate GPU environments (Colab), not this default install.

On a fresh clone of `main`, expect **210 passed, 13 skipped** (`pytest` excludes `slow` and two deselected tests by default). Run `uv run pytest -m slow` only when you intend to download V-JEPA weights or run large synthetic paths.

## `ii` CLI (public surface)

All experiments that belong in this repo should be reachable through these commands (add a new subcommand before relying on a one-off script):

| Command | Purpose |
| --- | --- |
| `try` | Guided first run for newcomers: practice clips, a `fake` run under `runs/try`, then the playground opened on the strongest verified feature. |
| `generate-clips` | Write deterministic synthetic clips for the `fake` adapter. |
| `run` | Capture activations, train SAE, label features, score, optionally serve the viewer (`fake`, `vjepa2`, `hf --model <id>`). |
| `rank` | Attribution prefilter shortlist; does not replace ablation. |
| `explain` | Explain a new clip; cites only scoreboard-`real` ablation effects. |
| `serve` | Python read API + `POST /explain` (Rust server in `server/` is preferred for production reads). |
| `rollouts` | Paired policy rollouts + matched random-feature controls → `rollouts/` (schema v1). |
| `steer-scorecard` | Remove and boost the top verified features on held-out clips against matched random-direction controls; effect sizes with episode-bootstrap intervals → `steering/`, shown in `ii view`. Offline; never changes the scoreboard and is not a rollout finding. |
| `export` | Static, hostable copy of one run's viewer, with a precomputed playground (no server). |
| `export-site` | Regenerate static `demo-site/` from fake run + vendored measurements. |
| `validate` | Check a run folder against `SCHEMA.md`. |
| `view` | Local browser viewer for a finished run. `--steer` adds the playground (steer SAE features, see the prediction move; exploration only, never evidence). |

World-model path: `ii run fake --clips … --out runs/…` or `vjepa2`. Policy path on main: `ii rollouts runs/… --policy fake`.

**Known gap:** GR00T is on `ii` for the offline lane (`ii run groot`: capture at DiT block 5 on the final denoise step, SAE, scoring, `ii view --steer`), but **closed-loop LIBERO-plus rollouts are not**. They still run through `src/interpretableintelligence/adapters/groot/v9/` (Colab, `ablate_v9.py`, ablation plans), the current pre-CLI route, not a violation of the long-term CLI rule. Steering GR00T in the playground is exploration, never evidence; the research rules below still decide findings.

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
4. It **beats the random-weights SAE control**: the same picker pipeline, but an SAE with **untrained (random) weights**—the feature must beat what a random dictionary would produce. **Not implemented in the repo yet** (no `ii` command; known gap).

**`recon_only` (groot ablation plans):** sanity check only—the hook applies SAE reconstruction with **no latent ablation** (empty edit set). On task 1028 it was **8/8 bit-identical to baseline**. This verifies the hook path; it is **not** the random-weights SAE control and does not satisfy rule 4.

**Scoreboard (`scoreboard.json`):** changes only through the automated scorer (`ii run` scoring stage or equivalent validated pipeline), **never by hand**. Every feature stays `real: false` until its gates pass (`SCHEMA.md`: usable SAE plus diversity, label holdout, random-feature percentile, random-init absence). Offline `real` does **not** imply rollout causality.

**How to run work:**

- Target the **public `ii` CLI** and the standard run layout. If a capability is missing, **add a reusable `ii` command** first—build only what the experiment needs. Until GR00T rollouts and the random-weights SAE control land on `ii`, the v9 adapter path above is the documented exception (known gap).
- **Never edit `split.json` or held-out labels** in an existing run.
- **No paid compute** on behalf of this repo (no cloud GPU rental, etc.). GR00T rollouts use existing Colab/box paths under `adapters/groot/`.

## Git workflow

- **Never push to `main`.** Branch from latest `main`, one topic per branch (for example `claude/<topic>`).
- Open a PR against `main`; the repo owner (**@jerrydjin**) reviews and merges.
- Keep PRs small; **`uv run pytest` must stay green** before you push. CI (`.github/workflows/ci.yml`) runs the same suite on every PR.

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
