# Agent instructions (source of truth)

Shared rules for Claude Code, Codex, and other coding agents working in this repo.

## What this is

**interpretableintelligence** (`ii`) is infrastructure for mechanistic interpretability on video world models and robot / VLA policies: a shared **run folder** format, SAE training, offline scoreboard gates, local viewers, and causal rollouts.

The main library and CLI live under `src/interpretableintelligence/`. Adapters shipped on main are `fake` (acceptance test), `vjepa2`, the generic `hf:<model>` adapter (any Hugging Face vision model: ViT-style, CNN, the vision tower of CLIP/SigLIP/VLMs, or a SmolVLA-style policy's frozen encoder; `ii run hf --model <id>`), and `groot:<checkpoint>` (NVIDIA GR00T N1.7 through LeRobot, `ii run groot`; needs `lerobot[groot]==0.7.0`, and `--tiny` is a random-weights stand-in for tests). Robot work is **dogfooded** on real models in `src/interpretableintelligence/adapters/groot/`—currently **NVIDIA GR00T** checkpoint `nvidia/gr00t17-lerobot-libero_10-640` on **LIBERO-plus** (task 1028 is the active rise-test task).

Contract details: `SCHEMA.md`, `ARCHITECTURE.md`, `README.md`. Do not drift from them in agent docs.

## Setup and tests

From the repo root (Python ≥ 3.11; [uv](https://docs.astral.sh/uv/) recommended):

```bash
uv sync
uv run ii --help
uv run pytest
```

`interpretableintelligence` is the long command name; **`ii` is the same CLI**. PyTorch is pinned to CPU wheels in `pyproject.toml` for CI and local dev; GR00T/LIBERO jobs use separate GPU environments (Colab), not this default install.

On a fresh clone of `main` with `cargo`, Chrome and `node` on the machine (as in CI, on Ubuntu and macOS), expect **484 passed, 11 skipped, 4 deselected** (the `slow` tests, excluded by default). Without `cargo` (and no `II_SERVER`) the Rust server tests download the prebuilt binary pinned in `server_release.json`; with no binary at all (no network either) the 16 Rust-backed tests in `tests/test_contract.py`, `tests/test_view_rust.py` and `tests/test_view_launch.py` also skip (**468 passed, 27 skipped**). Without Chrome or Chromium the 10 browser tests skip, and without `node` 4 guide tests skip. Every skip prints its reason with `pytest -rs`. Run `uv run pytest -m slow` only when you intend to download V-JEPA weights or run large synthetic paths.

## `ii` CLI (public surface)

All experiments that belong in this repo should be reachable through these commands (add a new subcommand before relying on a one-off script):

| Command | Purpose |
| --- | --- |
| `try` | Guided first run for newcomers: practice clips, a `fake` run under `runs/try`, then the playground opened on the strongest verified feature. |
| `generate-clips` | Write deterministic synthetic clips for the `fake` adapter (`--failure-labels planted\|null` also writes `failures.json` for `ii probe`). |
| `get-clips` | Download short real robot clips (DROID, MIT licence) to run a real model on, for learners with no video of their own. `--with-states` also saves each clip's per-frame arm pose (`<clip>.states.npy`) for the V-JEPA 2-AC commands. |
| `record-libero` | Record closed-loop GR00T episodes on one LIBERO-plus task (default 1028, seeds `60000 + i`) as clips the `groot` adapter reads: main camera, wrist camera, state and instruction per env step. Checks first that the adapter rebuilds the policy's own backbone inputs from the recorded arrays. GPU box with `lerobot[groot]` and LIBERO-plus (`scripts/groot_gpu_setup.sh`). |
| `run` | Capture activations, train SAE, label features, score, optionally serve the viewer (`fake`, `vjepa2`, `hf --model <id>`). `--remote` runs it on a Modal GPU billed to the invoking user's account and streams results back (`docs/REMOTE.md`). `--sae-from` copies a frozen SAE in (ii run folder or GR00T v9 capture) instead of training one, so its feature ids keep their meaning; such a run is never scored (`SCHEMA.md`, "Reused SAEs"). |
| `rank` | Attribution prefilter shortlist; does not replace ablation. |
| `explain` | Explain a new clip; cites only scoreboard-`real` ablation effects. |
| `serve` | `POST /explain` only, for the Rust read API in `server/` to proxy (`--explain-url`). All reads are Rust. |
| `rollouts` | Paired policy rollouts + matched random-feature controls → `rollouts/` (schema v1). |
| `steer-scorecard` | Remove and boost the top verified features on held-out clips against matched random-direction controls (or, with `--controls-from features`, other live SAE features matched on firing rate, activation and decoder norm); effect sizes with episode-bootstrap intervals → `steering/`, shown in `ii view`. Offline; never changes the scoreboard and is not a rollout finding. |
| `probe` | Held-out failure probe on a run's SAE latents, trained on `split.json`'s train clips only, against a seeded random-weights SAE (same architecture, width, `k`) and raw activations → `probe/`. `sae_gate` passes only if the trained SAE beats both on AUROC (paired episode bootstrap p < 0.05) at a lead of ≥ 1 s; `not_run` when skipped. Offline; never changes the scoreboard, never sets `real`, not a rollout finding. |
| `retrain-sae` | Retrain the GR00T capture SAE on its train split (BatchTopK with AuxK or resampling revival, any dictionary size) into a new capture folder whose `sae/` the picker, `ii import` and `ii rollouts --pick` read unchanged; reports alive, dead and dense latents and EV for old and new SAEs (`docs/groot-retrain-sae.md`). A new dictionary: old feature ids and findings do not carry over. Offline; never a finding. |
| `map-feature` | Find a GR00T SAE feature's counterpart in another dictionary trained on the same capture (activation correlation and raw-unit decoder cosine on held-out tokens): one feature, a split into up to 3, or none, by fixed rules; `--direction` also runs the picker through both SAEs and checks the counterpart fires more before failure. Offline; never a finding. |
| `export` | Static, hostable copy of one run's viewer, with a precomputed playground (no server). |
| `export-site` | Regenerate the static site in `demo-site/`: home, Explore (a page per scored V-JEPA 2 feature, read from `demo-data/vjepa-slim/`), the run-it-on-your-model page and the Research blog (the GR00T 1028 post reads every number from `demo-data/groot/`), plus redirects for cut pages. Keeps `demo-site/live/` and `vercel.json`. |
| `export-live` | Pack an `hf` image-model run for the live in-browser playground (`demo-site/live/`): the model split at the SAE's layer as ONNX, the SAE, labels and badges copied from the scoreboard (or, with `--meta-from`, from an `ii export` after checking the dictionary matches). Also writes `sae.safetensors` for the backend (`serve-live`, `deploy-live`); without a configured backend the browser runs everything. Exploration only, never evidence. |
| `serve-live` | Run the live playground's backend locally: each pack's model and SAE in PyTorch behind playground API v1 (`docs/PLAYGROUND-API.md`; read a picture, steer features). Open the page with `?api=http://127.0.0.1:8787`. `--builds DIR` also lets the page name a Hugging Face vision transformer and get a dictionary for it (bring your own model: `ii run hf --no-score` on `--build-clips`, then a backend-only pack; `playground/builds.py`). Exploration only, never evidence. |
| `deploy-live` | Host that backend on Modal (the deployer's own account; scales to zero, `--max-containers` caps cost, `--gpu none` for CPU; bring-your-own-model builds on by default, one GPU build at a time with daily, size and timeout caps, `--no-builds` turns them off) and write its URL into `demo-site/live/config.json`. Agents do not deploy unless the user asks. `--api <url>` only rewrites the config. |
| `compare` | Match SAE features across runs on the same clips (activation correlation, shuffle null): which concepts models share and which are unique. Writes `compare.json` and a static page. |
| `action-effects` | V-JEPA 2-AC (Meta's action-conditioned world model; code from a vjepa2 checkout, `--tiny` for tests): predicts each next latent frame of DROID clips under the real, zero, shuffled and reversed action and compares how much the action explains on arm patches, moving non-arm patches and matched still patches. Fixed verdict rule; `--random-predictor` control. Offline; never a finding. |
| `plan-probe` | V-JEPA 2-AC planner (one-step CEM toward the next real frame) inside Meta's action range and far outside it: how often an impossible action scores better than the real one and the best in-range one. Fixed verdict rule; `--random-predictor` control. Offline; never a finding. |
| `gripper-probe` | V-JEPA 2-AC goal score at moments the gripper opens or closes with the arm still: does it prefer the right gripper command? Fixed verdict rule; `--random-predictor` control. Offline; never a finding. |
| `gripper-trace` | V-JEPA 2-AC inside the model at the same gripper events: a probe reads gripper open or closed from the encoder's latents (episode folds, pixel baseline), the right and wrong gripper command are traced through every predictor block with activation patching of the action token and the frame tokens, and the same probe reads the predicted latent. Fixed verdict rules; `--random-predictor` control. Offline; never a finding. |
| `gripper-sae` | V-JEPA 2-AC: BatchTopK SAE on the predictor's frame tokens at one layer (`--layer`, e.g. where `gripper-trace` peaks), trained on one half of the episodes; picks the latents that move most with the gripper command there and tests them on the other half's gripper events (does the top latent move the right way; how much of the prediction gap patching the picked latents closes) against untrained SAEs (rule 4's dictionary) and random live latents. Fixed verdict rule. Offline; never a finding. |
| `gripper-circuit` | V-JEPA 2-AC: which predictor parts shrink the gripper command after `gripper-trace`'s peak. Scores every attention and MLP output and every head on how much of the right-vs-wrong gripper difference it removes (against the same for an arm step); one episode fold picks the top parts and heads after its peak and the other fold knocks them out (they get the right command's output in the wrong command's run) against random sets of the same size; then re-injects each candidate's own peak-layer gripper effect at the last block and re-reads `gripper-probe`'s goal score against random directions, with arm and damage checks. `--tiny` plants a known eraser. Fixed verdict rules. Offline; never a finding. |
| `impossible-events` | V-JEPA 2-AC screen: on moving DROID transitions, is the predictor more surprised (planner L1) by an impossible next frame than by the real one? Five edits (frozen world, frozen moving square, time reversed, two squares swapped, another episode), each against a copy-the-frame baseline and a zero action; per-edit verdict `caught` / `trivial` / `missed` / `mixed`, heatmaps for the local edits. `--random-predictor` control. Offline; never a finding. |
| `import` | Turn rollout logs captured elsewhere into a run folder (`--format groot-v9`: the v9 rollcap-v0 capture, directory or tarball). Copies `split.json` byte for byte; uncomputable metrics are `null` with a reason in `manifest.import.nulls` (`SCHEMA.md`, "Imported runs"). |
| `validate` | Check a run folder against `SCHEMA.md`. |
| `cache` | Show or clear the shared activation cache (keyed by model, layer, clip hash, window, preprocessing) that lets reruns and the playground skip forward passes. Not part of the run contract. |
| `view` | Local browser viewer for a finished run. `--steer` adds the playground (steer SAE features, see the prediction move; exploration only, never evidence). |

World-model path: `ii run fake --clips … --out runs/…` or `vjepa2`. Policy path on main: `ii rollouts runs/… --policy fake`.

**Known gap:** GR00T is on `ii` for the offline lane (`ii run groot`: capture at DiT block 5 on the final denoise step, SAE, scoring, `ii view --steer`). Closed-loop LIBERO-plus rollouts go through `ii rollouts --policy groot-1028` (plans, including `--pick` with the random-weights SAE control), but they still execute on the v9 runner in `src/interpretableintelligence/adapters/groot/v9/` on Colab, not on a local simulator. Steering GR00T in the playground is exploration, never evidence; the research rules below still decide findings.

## Language: Rust for engineering and infra

New technical engineering and infrastructure (servers, storage, file formats and readers, tooling that does not run a model) is written in **Rust**, in the Cargo workspace (`server/` is the example).

The exception is code that runs inside the model process and touches model tensors: adapters, the pipeline stages, scoring, steering, and the glue that hands tensors to and from PyTorch (for example the activation cache in `act_cache.py`, whose file I/O already goes through the Rust-backed `safetensors` package). That stays Python, so the package still installs with `uv sync` and no Rust toolchain. When a change is unclear, prefer Rust for anything that reads or serves run folders or caches without loading a model, and say which side of the line it is on in the PR.

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
4. It **beats the random-weights SAE control**: the same picker pipeline, but an SAE with **untrained (random) weights**—the feature must beat what a random dictionary would produce. On `ii`: `ii rollouts --pick N` (with `--random-sae K`, default 5 with `--pick`) picks the candidates and each untrained SAE's features with the same picker and marks a candidate `finding: true` only when it is `causal` and strictly above every random-SAE arm (`SCHEMA.md`, "Random-weights SAE control"). It runs on any policy that sets `random_dictionaries`; the `fake` policies do. On GR00T it is `ii rollouts --policy groot-1028 --pick N --capture <capture run>` (`groot-mock` in CI): `picker/pick_candidates.py` picks from the trained SAE and from each untrained SAE (the trainer's init, threshold set for the same `k`, `sae_batchtopk.py`), the plan runs every arm through the v9 hook, and the summary scores a **rise** (`direction: rise`). Those runs use a fixed budget (no arm stops early), but the plan carries a Sobel–Wald design (`p0` = the capture's success rate, `p1 = p0 ± 0.30`) and the summary reads it on each arm's outcomes in episode order (`sprt_verdict`), so rule 1's rise call is part of `finding` there too (`SCHEMA.md`, "GR00T"). Offline, `ii probe` uses the same untrained dictionary (`sae.random_weights_sae`) as a baseline for its held-out failure probe; that gate is a separate check and does not satisfy rule 4 on its own.

**`recon_only` (groot ablation plans):** sanity check only—the hook applies SAE reconstruction with **no latent ablation** (empty edit set). On task 1028 it was **8/8 bit-identical to baseline**. This verifies the hook path; it is **not** the random-weights SAE control and does not satisfy rule 4.

**Scoreboard (`scoreboard.json`):** changes only through the automated scorer (`ii run` scoring stage or equivalent validated pipeline), **never by hand**. Every feature stays `real: false` until its gates pass (`SCHEMA.md`: usable SAE plus diversity, label holdout, random-feature percentile, random-init absence). Offline `real` does **not** imply rollout causality.

**How to run work:**

- Target the **public `ii` CLI** and the standard run layout. If a capability is missing, **add a reusable `ii` command** first—build only what the experiment needs. GR00T rollouts with the random-weights SAE control go through `ii rollouts --pick` and execute on the v9 runner (Colab), the documented exception above.
- **Never edit `split.json` or held-out labels** in an existing run.
- **No paid compute** on behalf of this repo (no cloud GPU rental, etc.). GR00T rollouts use existing Colab/box paths under `adapters/groot/`. `ii run --remote` is a user feature that bills the user's own Modal account; agents do not run it unless the user asks for that run. Its tests use the in-process `LocalBackend` and never reach Modal.

## Git workflow

- **Never push to `main`.** Branch from latest `main`, one topic per branch (for example `claude/<topic>`).
- Open a PR against `main`; the repo owner (**@jerrydjin**) reviews and merges.
- Keep PRs small; **`uv run pytest` must stay green** before you push. CI (`.github/workflows/ci.yml`) runs the same suite on every PR. `.github/workflows/real-model.yml` also runs one real Hugging Face model (DeiT-tiny) through `ii run hf` end to end on every PR (`uv run pytest -m real_model`; downloads weights, so it is not in the default suite).

Docs-only PRs are fine; do not touch `scoreboard.json`, `split.json`, held-out labels, or committed run artifacts.

## Current state (GR00T task 1028)

The **single-feature** ablation on LIBERO-plus task **1028** was a formal **null**: **641** rollouts, **20** random controls, candidates **720** and **7081+5852**—candidate “falls” were **not distinguishable** from controls (`demo-data/groot/groot-1028-ablate-summary.json`, `real: false`). Do **not** re-run single-feature rise sweeps on 1028 without a **new experimental design** approved in the research thread.

Offline world-model acceptance remains `ii run fake` → exactly three `real` features (`stripe`, `pulse`, `corner`).

## Touch points

| Need | Read |
| --- | --- |
| Run schema / gates | `SCHEMA.md`, `src/interpretableintelligence/thresholds.py` |
| Steering scorecard on real GR00T | `docs/groot-steer-scorecard.md`, `scripts/groot_steer_scorecard.sh` |
| Pipeline stages | `ARCHITECTURE.md`, `src/interpretableintelligence/pipeline.py` |
| GR00T v9 ablation / SPRT | `src/interpretableintelligence/adapters/groot/v9/ablate_v9.py` (runner); the SPRT and its exact inference are in `src/interpretableintelligence/rollout_stats.py` |
| Demo bundles | `demo-data/`, `ii export-site` |
| Retrain the GR00T SAE (`ii retrain-sae`) | `docs/groot-retrain-sae.md`, `src/interpretableintelligence/adapters/groot/retrain_sae.py` (capture I/O); the BatchTopK/AuxK trainer and its untrained twin are in core, `src/interpretableintelligence/sae_batchtopk.py` |
| Import a capture (`ii import`) | `SCHEMA.md` ("Imported runs"), `src/interpretableintelligence/importers/` |
