# Architecture

This lane is the contract, the scoring core, the V-JEPA 2 adapter, and a local clip viewer. It loads a model only through an adapter. The adapters here are `fake` and `vjepa2`. It does not talk to a database. `physint view` reads a finished run folder only: the manifest, plus clip effects, `features.jsonl`, and the scoreboard when those files are present. Feature acts stay in parquet and are queried per clip and per feature; the file is not loaded into memory. The viewer does not import adapters. mp4 and webm play natively; other containers are transcoded with ffmpeg. Clip files match the manifest `name` (exact filename), then the stem, then sha256. A token `(window, t)` maps to source frame `window_start + t * tubelet * stride`. The patch heatmap is drawn on the centre square of a V-JEPA frame (`side = min(H, W)` from the clip's `hw`), unless `manifest.extra.crop` gives the rectangle. When the manifest has no per-clip starts, `manifest.extra` (`frames`, `stride`, `windows`) places the windows; a clip shorter than `frames * stride` uses stride 1.

## Pipeline

`physint run <adapter> --clips DIR --out runs/<run_id> [--gpu-cost-per-hour X]` does one job, then serves the viewer:

1. **Split** clips with a fixed seed. Holdout is by clip.
2. **Capture** the instrumented layer through `hook(..., mode="capture")` and write activation shards. Rows are appended to a float32 cache and contract shards are emitted in slices. The full activation matrix is not a tensor.
3. **Train** a ReLU SAE on train rows only, one batch at a time from that cache.
4. **Features.** Encode in chunks, write sparse `feature_acts.parquet`, and write `candidates.json`.
5. **Score** (skipped with `--no-score`, or when `scoreboard.json` is already valid). Explained variance, spliced predictor loss, dead features, per-feature ablations, matched random features, and a random-init control. The fake adapter uses `fake-cue-v1`, which reads planted cue channels back out of the clip. `LLMLabeller` is a stub and raises rather than calling a network.
6. **Serve** the folder. `GET /api/run` is the health check. `--no-serve` stops after the report.

Each stage records wall time, peak RSS, device, and row/token counts on `manifest.stages` and is skipped when its outputs still match the clips, seed, and upstream weights. `physint validate` checks the folder in `SCHEMA.md`, including that `real` follows the four gates. The printed cost per hour of footage is `stage_seconds * gpu_cost_per_hour / video_seconds`.

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
| `raw_activations(clip, window)` | Layer residual before hooks, `[t, h, w, d_model]`. The straight-through estimate reads it. |
| `visible_mean_baseline(clip, window)` | Masked-token guess equal to the mean of the visible patches. Same order as `predict_masked`. |

The scorer only talks to this surface. It does not know whether the model is a world model or a policy.

## Fake adapter

`fake` is a position-wise predictor with a fixed dictionary of six orthonormal directions in activation space.

- Causal cues `stripe`, `pulse`, `corner` are read by the predictor. Zeroing the matching SAE feature moves held-out predictions.
- Decoy cues `flicker`, `band`, `speckle` are painted the same way, fire on every clip, and receive stable labels, but the readout is orthogonal to them. Their ablation effect sits under the random-feature baseline.
- `load(random_init=True)` applies a fixed token permutation before a random encoder, so cue identity no longer lines up with position. An SAE on that model does not reproduce the planted patterns (`absent_in_random_init`).

Clips are float32 arrays shaped `[16, 16, 16, 8]` (frames, height, width, channels), one channel per cue plus two distractor channels. `physint generate-clips` writes them. The head adds a bias orthogonal to the cue subspace so cosine distance measures a real rotation of the output rather than a collapse to zero.

The reference run (`--layer 16 --seed 0 --heldout-frac 0.2`, 30 clips) is the acceptance test: exactly the three causal labels are `real`, and each decoy fails only the causal percentile while passing diversity, label transfer, and the random-init check.

## Rollouts

`physint rollouts` asks a different question from the scoreboard: if this SAE feature is edited inside a policy, does task success move? The run folder gains `rollouts/` and `manifest.schema_version` becomes 1. Scoreboard files are not rewritten.

`PolicyEnv` is `reset`, `act`, `step`, and `intervene`. `act` encodes the instrumented layer, applies the intervention to SAE latents, and decodes before the controller runs. Kinds are `ablate` (set to 0), `scale` (multiply by `value`), and `clamp` (set to `value`).

The shipped pair is `fake`: a point-mass reacher with one planted coordinate. Baseline success is about 0.80. Ablating the planted feature drops it to about 0.30. Ablating any other feature copies the baseline trajectory. `fake-null` turns the planted effect off: the controller ignores latents and the success bit is a fair coin of the post-intervention code, so arms are exchangeable and not identical. `libero`, `openvla`, and `pi0.5` are stubs behind the optional `robots` extra, which depends on nothing.

The schedule is paired on seed. Baseline and candidate features run all `N` episodes. After each batch a control is dropped only for efficacy: a one-sided SPRT on discordant pairs (control succeeds and the candidate fails) crosses a Bonferroni share of `--alpha`. Per-control beta is about 0.022 when there are 10 controls, so family-wise power stays about 0.8. A control that does not cross runs to `N`. The summary reports the success delta, a paired Wald interval, an exact McNemar p-value, the average-rank percentile against controls (ties score 50, not 100), episodes used against the fixed `N` budget, and rollouts per feature flagged causal.

## Next lanes

1. **pi0.5.** V-JEPA 2 is `src/physint/adapters/vjepa2`. The remaining world-model adapter is pi0.5, and the policy stubs for LIBERO / OpenVLA / pi0.5 still need real simulators. They must run on MPS and CPU. They do not change the run folder or the gates.
2. **Clip-first viewer.** Shipped as `physint view runs/<run_id>`. Video on the left, features that fire on the current frame on the right, activation maps from `feature_acts.parquet`, effects from `clip_effects.parquet` when that file exists. Clip files come from a path on the manifest or from `--clips` (filename stem, then sha256). The patch heatmap is drawn over the video frame. It only reads the contract.
3. **Hosted ingest.** Upload a run folder into Postgres plus object storage. Shards and weights go to objects; manifest, scoreboard, and feature rows go to tables. Ingest reuses `physint validate` and refuses a folder that fails it.
4. **Failed-clip API.** `POST /explain` still runs in Python (`physint.explain`): it has to capture the layer and ablate features. `physint explain` and `physint serve` share that module. The read API is `server/` (`physint-server`), which queries the run folder's parquet with DuckDB and matches `physint.api`. An explanation may cite only features with `real: true`, and only effects with `effect_kind: "ablation"`. Estimates rank which features to ablate; they are not evidence. A run whose SAE is not usable is refused. A feature that is not real is labelled `unverified`. Hosting is a proposal in `docs/HOSTING.md`, not a deployment.
