# Architecture

This lane is the contract, the scoring core, and a local clip viewer. It does not load V-JEPA 2 or pi0.5, and it does not talk to a database. `physint view` reads a finished run folder only: the manifest, plus clip effects, `features.jsonl`, and the scoreboard when those files are present. Feature acts stay in parquet and are queried per clip and per feature; the file is not loaded into memory. It does not import adapters. mp4 and webm play natively; other containers are transcoded with ffmpeg. Clip files match the manifest `name` (exact filename), then the stem, then sha256. A token `(window, t)` maps to source frame `window_start + t * tubelet * stride`. The patch heatmap is drawn on the centre square of a V-JEPA frame (`side = min(H, W)` from the clip's `hw`), unless `manifest.extra.crop` gives the rectangle. When the manifest has no per-clip starts, `manifest.extra` (`frames`, `stride`, `windows`) places the windows; a clip shorter than `frames * stride` uses stride 1.

## Pipeline

`physint run <adapter> --clips DIR --out runs/<run_id>` does one job:

1. **Split** clips with a fixed seed. Holdout is by clip.
2. **Capture** the instrumented layer through `hook(..., mode="capture")` and write activation shards.
3. **Train** a ReLU SAE on train rows only.
4. **Label** each feature. The fake adapter uses `fake-cue-v1`, which reads planted cue channels back out of the clip. `LLMLabeller` is a stub and raises rather than calling a network.
5. **Score** explained variance, spliced predictor loss, dead features, per-feature ablations, matched random features, and a random-init control.
6. **Write** the run folder in `SCHEMA.md`. `physint validate` checks that folder, including that `real` follows the four gates.

Thresholds are constants in `src/physint/thresholds.py`. Changing them changes the meaning of `real`.

## Adapter protocol

`physint.adapters.base.Adapter`:

| method | role |
| --- | --- |
| `load(random_init=False)` | Build weights. Random init must not reproduce trained firing patterns. |
| `windows(clip)` | One or more temporal windows. The fake adapter uses a single window. |
| `hook(layer, mode, fn=None)` | `capture`, `edit`, or `zero`. Context manager. |
| `encode(clip, window)` | Run the window and return layer activations `[t, h, w, d_model]`. |
| `predict_masked(clip, window, edit=None)` | `(pred, target)` on masked cells, row-major `(t, h, w)`. |
| `mask(window)` | Bool grid of those cells. |
| `grid()` | `{t, h, w, tubelet, patch_px}`. |

The scorer only talks to this surface. It does not know whether the model is a world model or a policy.

## Fake adapter

`fake` is a position-wise predictor with a fixed dictionary of six orthonormal directions in activation space.

- Causal cues `stripe`, `pulse`, `corner` are read by the predictor. Zeroing the matching SAE feature moves held-out predictions.
- Decoy cues `flicker`, `band`, `speckle` are painted the same way, fire on every clip, and receive stable labels, but the readout is orthogonal to them. Their ablation effect sits under the random-feature baseline.
- `load(random_init=True)` applies a fixed token permutation before a random encoder, so cue identity no longer lines up with position. An SAE on that model does not reproduce the planted patterns (`absent_in_random_init`).

Clips are float32 arrays shaped `[16, 16, 16, 8]` (frames, height, width, channels), one channel per cue plus two distractor channels. `physint generate-clips` writes them. The head adds a bias orthogonal to the cue subspace so cosine distance measures a real rotation of the output rather than a collapse to zero.

The reference run (`--layer 16 --seed 0 --heldout-frac 0.2`, 30 clips) is the acceptance test: exactly the three causal labels are `real`, and each decoy fails only the causal percentile while passing diversity, label transfer, and the random-init check.

## Next lanes

1. **Real adapters.** Implement the protocol for V-JEPA 2 and pi0.5. They must run on MPS and CPU. They replace `fake`; they do not change the run folder or the gates.
2. **Clip-first viewer.** Shipped as `physint view runs/<run_id>`. Video on the left, features that fire on the current frame on the right, activation maps from `feature_acts.parquet`, effects from `clip_effects.parquet` when that file exists. Clip files come from a path on the manifest or from `--clips` (filename stem, then sha256). The patch heatmap is drawn over the video frame. It only reads the contract.
3. **Hosted ingest.** Upload a run folder into Postgres plus object storage. Shards and weights go to objects; manifest, scoreboard, and feature rows go to tables. Ingest reuses `physint validate` and refuses a folder that fails it.
4. **Failed-clip API.** `upload a clip that the policy missed, return the features that drove it`. The response may cite only features with `real: true`, and only effects with `effect_kind: "ablation"`. Estimates are a ranking aid inside a run, not something the API serves as evidence.
