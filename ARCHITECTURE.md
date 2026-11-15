# Architecture

This lane is the contract, the scoring core, the V-JEPA 2 adapter, and a local clip viewer. It loads a model only through an adapter. The adapters here are `fake`, `vjepa2`, and the generic `hf:<model>` adapter for Hugging Face vision models (ViT-style, CNN, or the vision tower of an image-text model, VLM or SmolVLA-style policy). It does not talk to a database. `interpretableintelligence view` reads a finished run folder only: the manifest, plus clip effects, `features.jsonl`, and the scoreboard when those files are present. Feature acts stay in parquet and are queried per clip and per feature; the file is not loaded into memory. The viewer does not import adapters, except through the opt-in playground: `interpretableintelligence view --steer` loads the run's adapter and SAE in `steer.py` and serves `POST /api/steer`, which edits chosen SAE features and reports how the masked prediction moved. It writes nothing to the run folder. mp4 and webm play natively; other containers are transcoded with ffmpeg. Clip files match the manifest `name` (exact filename), then the stem, then sha256. A token `(window, t)` maps to source frame `window_start + t * tubelet * stride`. The patch heatmap is drawn on the centre square of a V-JEPA frame (`side = min(H, W)` from the clip's `hw`), unless `manifest.extra.crop` gives the rectangle. When the manifest has no per-clip starts, `manifest.extra` (`frames`, `stride`, `windows`) places the windows; a clip shorter than `frames * stride` uses stride 1.

## Pipeline

`interpretableintelligence run <adapter> --clips DIR --out runs/<run_id> [--gpu-cost-per-hour X]` does one job, then serves the viewer:

1. **Split** clips with a fixed seed. Holdout is by clip.
2. **Capture** the instrumented layer through `hook(..., mode="capture")` and write activation shards. Rows are appended to a float32 cache and contract shards are emitted in slices. The full activation matrix is not a tensor. Each window's layer first goes through the shared activation cache (`act_cache.py`): one safetensors file per window, keyed by adapter, model sha, random-init flag, layer, device type, clip sha256, window and frame hash, `adapter.cache_spec(clip)`, and tool version. A hit skips the forward pass; the random-init control capture in the score stage uses the same cache. `--no-cache` bypasses it. The cache is capped at `$II_ACTS_CACHE_MAX_GB` (default 20 GB) and evicts the least recently used windows.
3. **Train** a ReLU SAE on train rows only, one batch at a time from that cache.
4. **Features.** Encode in chunks, write sparse `feature_acts.parquet`, and write a frequency list that scoring replaces.
5. **Attribute.** One forward and backward per window scores every SAE feature (`activation × ∂objective/∂activation`, summed over tokens and windows) and writes `features/attribution.parquet` plus `features/candidates.json`. The adapter supplies the scalar: `prediction_loss`, or `action_loglik` when success is the log-likelihood of the taken action. Adapters scored against their own clean output (hf, groot) set `attribution_default = "output_sensitivity"`, a random-probe estimate of how far each ablation turns the output, because their loss has a zero gradient at the clean point. `activation` is the magnitude baseline and does not backprop. `interpretableintelligence rank` stops at the shortlist. `interpretableintelligence run --prefilter K` sends only that shortlist to the ablation sweep, and every one of those picks is confirmed by a real ablation.
6. **Score** (skipped with `--no-score`, or when `scoreboard.json` is already valid). Explained variance, spliced predictor loss, dead features, per-feature ablations, matched random features, and a random-init control. The default ablation is single-feature cosine distance. Without `--prefilter`, every feature is ablated. With `--prefilter K`, held-out ablations and the matched random controls run only for the top K; the others keep a null ablation and cannot be `real`. `--extra-effects loss,group` also writes full-video loss changes for single features and for co-active groups, each against matched controls of the same size and activation mass. The cosine scoreboard is unchanged. The fake adapter uses `fake-cue-v1`, which reads planted cue channels back out of the clip. The `hf` adapter uses `clip-concepts-v2`: per-token CLIP similarity to a concept list (built in, or `--concepts FILE`), scored by ROC AUC of firing against non-firing tokens, so a sparse real-video feature can earn a label that has to hold on held-out clips. `LLMLabeller` is a stub and raises rather than calling a network.
7. **Serve** the folder. `GET /api/run` is the health check. `--no-serve` stops after the report.

`scripts/compare_attribution_vs_ablation.py` reads a run folder that has both the attribution file and measured ablations and prints Spearman correlation and top-K overlap for the attribution, next to the same numbers for an activation-magnitude baseline.

Each stage records wall time, peak RSS, device, and row/token counts on `manifest.stages` and is skipped when its outputs still match the clips, seed, and upstream weights. `interpretableintelligence validate` checks the folder in `SCHEMA.md`, including that `real` follows the four gates. The printed cost per hour of footage is `stage_seconds * gpu_cost_per_hour / video_seconds`.

`--remote` runs these stages unchanged on a hosted GPU (`remote/`, `docs/REMOTE.md`). `execute_pipeline`'s `on_stage` callback fires once each stage's outputs are on disk; the container-side worker turns it into an event listing the run folder's files and hashes, and the client downloads what changed into the local `--out`. The local viewer serves the folder as a `LiveRun`, which reloads on each new stage. The cap is the container's timeout: `--max-cost` divided by the container's price per hour.

Thresholds are constants in `src/interpretableintelligence/thresholds.py`. Changing them changes the meaning of `real`.

## Adapter protocol

`interpretableintelligence.adapters.base.Adapter`:

| method | role |
| --- | --- |
| `load(random_init=False)` | Build weights. Random init must not reproduce trained firing patterns. |
| `windows(clip)` | One or more temporal windows. The fake adapter uses a single window. |
| `hook(layer, mode, fn=None)` | `capture`, `edit`, or `zero`. Context manager. |
| `encode(clip, window)` | Run the window and return layer activations `[t, h, w, d_model]`. |
| `predict_masked(clip, window, edit=None)` | `(pred, target)` on masked cells, row-major `(t, h, w)`. |
| `attribution_objective(clip, window, objective, edit=None)` | Scalar the prefilter backprops into: `prediction_loss` or `action_loglik`. |
| `mask(window)` | Bool grid of those cells. |
| `grid()` | `{t, h, w, tubelet, patch_px}`. |
| `raw_activations(clip, window)` | Layer residual before hooks, `[t, h, w, d_model]`. The straight-through estimate reads it. |
| `visible_mean_baseline(clip, window)` | Masked-token guess equal to the mean of the visible patches. Same order as `predict_masked`. |

Optional, for the activation cache: `cache_spec(clip)` returns what besides weights, layer and frames decides the adapter's tensors (input size, normalisation, frame count; the fake adapter adds the clip id, which seeds its noise). An adapter without it is never cached. The key cannot see adapter code, so `cache_spec` starts with a version tag (`"hf-v1"`, `"vjepa2-v1"`, `"fake-v1"`): **bump it in the same change whenever the adapter returns different tensors for the same weights and frames** (preprocessing, windowing, which tensor the hook reads), or old entries keep being read. `clean_state(clip, window)` and `load_clean_state(clip, window, state)` export and restore the unedited intermediate tensors of one window, so the playground answers a cold edit without the clean forward passes. `hf`, `vjepa2` and `fake` implement `cache_spec`; `hf` and `vjepa2` implement the clean state. GR00T implements neither yet.

The scorer only talks to this surface. It does not know whether the model is a world model or a policy.

## Fake adapter

`fake` is a position-wise predictor with a fixed dictionary of six orthonormal directions in activation space.

- Causal cues `stripe`, `pulse`, `corner` are read by the predictor. Zeroing the matching SAE feature moves held-out predictions.
- Decoy cues `flicker`, `band`, `speckle` are painted the same way, fire on every clip, and receive stable labels, but the readout is orthogonal to them. Their ablation effect sits under the random-feature baseline.
- `load(random_init=True)` applies a fixed token permutation before a random encoder, so cue identity no longer lines up with position. An SAE on that model does not reproduce the planted patterns (`absent_in_random_init`).
- `plant="group"` is a separate fixture: three co-active features share a `max` readout, so each marginal ablation is small and the joint ablation is not. It uses two windows. `plant="null"` fires features the predictor does not read. Neither fixture is the default acceptance model.

Clips are float32 arrays shaped `[16, 16, 16, 8]` (frames, height, width, channels), one channel per cue plus two distractor channels. `interpretableintelligence generate-clips` writes them. The head adds a bias orthogonal to the cue subspace so cosine distance measures a real rotation of the output rather than a collapse to zero.

The reference run (`--layer 16 --seed 0 --heldout-frac 0.2`, 30 clips) is the acceptance test: exactly the three causal labels are `real`, and each decoy fails only the causal percentile while passing diversity, label transfer, and the random-init check.

## Generic Hugging Face adapter

`hf:<model>[#<blocks>]` (`adapters/hf.py`) wraps any `AutoModel` vision encoder. The name is the whole configuration, so `get_adapter` can rebuild it from `manifest.model.adapter` for `explain` and the playground. A forward hook on block `layer` exposes the patch tokens as the `[t, h, w, d]` grid; prefix tokens are untouched. For a CNN the block is a stage and its `[b, c, h, w]` map is the grid; the final map is resized to that grid for the readout. Composite configs (`vision_config`) load the whole model and keep only the vision tower. Capture stops the forward at the hooked block. `predict_masked` returns the final patch tokens against the clean final tokens, with every token in the mask, and `visible_mean_baseline` is the mean output token. The random-init control is `AutoModel.from_config` with seed 0.

## Rollouts

`interpretableintelligence rollouts` asks a different question from the scoreboard: if this SAE feature is edited inside a policy, does task success move? The run folder gains `rollouts/` and `manifest.schema_version` becomes 1. Scoreboard files are not rewritten.

`PolicyEnv` is `reset`, `act`, `step`, and `intervene`. `act` encodes the instrumented layer, applies the intervention to SAE latents, and decodes before the controller runs. Kinds are `ablate` (set to 0), `scale` (multiply by `value`), and `clamp` (set to `value`).

The shipped pair is `fake`: a point-mass reacher with one planted coordinate. Baseline success is about 0.80. Ablating the planted feature drops it to about 0.30. Ablating any other feature copies the baseline trajectory. `fake-null` turns the planted effect off: the controller ignores latents and the success bit is a fair coin of the post-intervention code, so arms are exchangeable and not identical. `libero`, `openvla`, and `pi0.5` are stubs behind the optional `robots` extra, which depends on nothing.

The schedule is paired on seed. Baseline and candidate features run all `N` episodes. After each batch a control is dropped only for efficacy: a one-sided SPRT on discordant pairs (control succeeds and the candidate fails) crosses a Bonferroni share of `--alpha`. Per-control beta is about 0.022 when there are 10 controls, so family-wise power stays about 0.8. A control that does not cross runs to `N`. The summary reports the success delta, a paired Wald interval, an exact McNemar p-value, the average-rank percentile against controls (ties score 50, not 100), episodes used against the fixed `N` budget, and rollouts per feature flagged causal.

## Steering scorecard

`interpretableintelligence steer-scorecard` asks how much, and how reliably, editing a feature moves the model's output, compared with matched random directions. It reuses the playground's edit (`steer.steer_edit`) and the run's held-out clips. For GR00T the output is the planned action chunk. Results go to `steering/` (see `SCHEMA.md`); `validate` rebuilds the scorecard from the effects parquet. Nothing here reads or writes the scoreboard gates, and it is not a closed-loop rollout.

## Failure probe

`interpretableintelligence probe` asks whether the trained SAE's latents predict episode failure ahead of time better than a random-weights SAE of the same shape (`sae.random_weights_sae`: the trainer's initial weights, zero steps) and better than the raw activations. It reads only the run folder (shards, `sae/`, `split.json`) plus a labels file, fits one logistic probe per input on the train clips, and gates on held-out AUROC with a paired episode bootstrap at a lead of at least 1 s. Results go to `probe/` (see `SCHEMA.md`); `validate` rebuilds `probe.json` from the scores parquet. It never touches the scoreboard gates or sets `real`. It is Python because it encodes activations through the SAE in torch.

## Next lanes

1. **pi0.5.** V-JEPA 2 is `src/interpretableintelligence/adapters/vjepa2`. The remaining world-model adapter is pi0.5, and the policy stubs for LIBERO / OpenVLA / pi0.5 still need real simulators. They must run on MPS and CPU. They do not change the run folder or the gates.
2. **Clip-first viewer.** Shipped as `interpretableintelligence view runs/<run_id>`. Video on the left, features that fire on the current frame on the right, activation maps from `feature_acts.parquet`, effects from `clip_effects.parquet` when that file exists. Clip files come from a path on the manifest or from `--clips` (filename stem, then sha256). The patch heatmap is drawn over the video frame. It only reads the contract.
3. **Hosted ingest.** Upload a run folder into Postgres plus object storage. Shards and weights go to objects; manifest, scoreboard, and feature rows go to tables. Ingest reuses `interpretableintelligence validate` and refuses a folder that fails it.
4. **Failed-clip API.** `POST /explain` still runs in Python (`interpretableintelligence.explain`): it has to capture the layer and ablate features. `interpretableintelligence explain` and `interpretableintelligence serve` share that module. The read API is `server/` (`interpretableintelligence-server`), which queries the run folder's parquet with DuckDB and matches `interpretableintelligence.api`. An explanation may cite only features with `real: true`, and only effects with `effect_kind: "ablation"`. Estimates rank which features to ablate; they are not evidence. A run whose SAE is not usable is refused. A feature that is not real is labelled `unverified`. Hosting is a proposal in `docs/HOSTING.md`, not a deployment.
