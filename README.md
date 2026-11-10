# interpretableintelligence

`interpretableintelligence` is a Python library and CLI for mechanistic interpretability of physical AI models: video world models and robot policies. This package is the shared run format and the scoring core. Drop in an adapter and a directory of clips; it captures a layer, trains a sparse autoencoder, labels features, and writes a scoreboard. The CLI is `interpretableintelligence`, and `ii` is the short form of the same command.

A feature is marked `real` only when the SAE is good enough to splice back into the predictor and the feature passes four checks: it fires across clips, its label holds on held-out clips, its ablation effect beats matched random features, and a randomly initialised copy of the model does not reproduce its firing pattern. A missing check is `null`, and `null` never counts as a pass.

`fake` is a deterministic synthetic model with three planted causal features and three decoys. It is the acceptance test for the scorer. `vjepa2` is Facebook V-JEPA 2 through Hugging Face transformers. pi0.5 is still a later lane.

`interpretableintelligence rollouts` is the causal check on a policy. It runs paired baseline and intervention episodes, plus matched random-feature controls, and writes `rollouts/` into the run folder (schema version 1). The fake policy is a reacher whose planted feature drops success from about 0.80 to about 0.30 when ablated; other features do not move it. LIBERO, OpenVLA, and pi0.5 sit behind the optional `robots` extra as stubs. That extra installs no simulator and no GPU package.

Runs on CPU and Apple Silicon (MPS). PyTorch is the only tensor library; nothing here requires CUDA.

## Quickstart

```bash
uv sync
uv run interpretableintelligence generate-clips --out clips/demo --n 30 --seed 0
uv run interpretableintelligence run fake --clips clips/demo --layer 16 --seed 0 --heldout-frac 0.2 --out runs/demo --gpu-cost-per-hour 2.50
```

That command captures the layer, trains the SAE, writes feature candidates, scores them, prints a cost report, and serves the viewer. A second invocation skips every stage whose outputs are already valid and serves the same folder. `--no-serve` writes the folder and exits so a script can `interpretableintelligence validate runs/demo` before opening it with `interpretableintelligence view`.

The report's cost per hour of footage is measured stage wall time times `--gpu-cost-per-hour`, divided by the hours of footage in the clips. It is omitted when the flag is absent. Stage wall time, peak RSS, device, and row/token counts are stored on `manifest.stages`. Activation rows are streamed from an on-disk cache; a stage does not load the full activation matrix.

`interpretableintelligence rank` is the cheap prefilter. One forward and backward per window scores every SAE feature by `activation × ∂objective/∂activation`, summed over tokens and windows. `--objective prediction_loss` is the masked predictor MSE. `--objective action_loglik` is the log-likelihood of the action the policy took, for a robot policy whose success is not differentiable. `--objective activation` ranks by activation magnitude only, the baseline the attribution has to beat. The objective is stored in the parquet metadata. The output is a shortlist: every pick is confirmed by a real ablation, and the score itself is not evidence.

```bash
uv run interpretableintelligence rank fake --clips clips/demo --prefilter 3 --out runs/rank
uv run interpretableintelligence run fake --clips clips/demo --prefilter 3 --out runs/short --no-serve
uv run python scripts/compare_attribution_vs_ablation.py runs/demo --k 3
```

`--prefilter K` on `interpretableintelligence run` still trains, labels, and checks the random-init gate for every feature, and runs the held-out ablation plus the 100 matched random controls only for the top K. Omit it to score every feature. `--extra-effects loss,group` adds full-video loss rows for single features and for co-active groups; the cosine scoreboard stays. A finished run always writes the attribution files. `scripts/compare_attribution_vs_ablation.py` prints Spearman and top-K overlap for that attribution next to the same numbers for activation magnitude. Attribution has to beat the magnitude baseline.

`runs/demo/scoreboard.json` should mark exactly three features real, labelled `stripe`, `pulse`, and `corner`. The decoys `flicker`, `band`, and `speckle` fire on many clips and keep their labels on held-out clips, but they do not move the predictor, so they are not real.

`interpretableintelligence view` serves that folder locally and opens a browser. It reads the manifest, and queries feature acts per clip and per feature instead of loading the parquet into memory. Clip effects, `features.jsonl`, and the scoreboard are shown when they are present. A missing scoreboard leaves every feature unverified, and a missing `clip_effects.parquet` shows activations with no effect column. This manifest does not store clip paths, so pass `--clips` (or a `path` / `clips_dir` recorded on a manifest that has one). Files in `--clips` match the clip's `name` (exact filename, such as `droid_ep000.mp4`) first, then the filename stem, then sha256. mp4, webm, and m4v play in a `<video>` element; avi, mov, and mkv are transcoded with ffmpeg. `--no-open` skips the browser.

On a V-JEPA run the heatmap sits on the centre square of the displayed frame (`hw` is `[height, width]`, side `min(H, W)`), and the cropped-out side bands are dimmed. `manifest.extra.crop` sets that rectangle explicitly.

A token `(window, t)` is the source frame `window_start + t * tubelet * stride`. `tubelet` is `manifest.grid.tubelet`. `stride` is the capture's temporal stride, so the source frames inside one step of `tubelet * stride` share a feature list. An explicit per-clip `window_starts` and `stride` win. Otherwise, when `manifest.extra` has `frames`, `stride`, or `windows`, starts are Jeff's placement over the clip's `n_frames` and `n_windows`: `span = frames * stride`, windows spread across the clip, one window centered, and stride 1 when the clip is shorter than `span`. With neither, windows tile from frame 0 and `stride` is 1, which is `window_start + t * tubelet`. A V-JEPA window recorded as `extra.frames` 16 and `extra.stride` 2 covers 32 source frames.

Explain a new clip with that run. The cited answer is per window and time, and it includes only real features and only ablation effects. Anything else is listed as unverified. If the run's SAE is not usable, the command refuses.

The read API is the Rust binary `interpretableintelligence-server` (`server/`). It queries the run folder's parquet with DuckDB. `POST /explain` stays in Python, because it has to run the model: start `interpretableintelligence serve` and point the Rust server at it with `--explain-url` (or `II_EXPLAIN_URL`). Without that, `POST /explain` returns 501.

```bash
cargo build -p interpretableintelligence-server
cargo run -p interpretableintelligence-server -- serve --runs runs --clips clips
uv run interpretableintelligence serve --runs runs --port 8000
```

```bash
uv run interpretableintelligence explain --run runs/demo --clip clips/demo/clip-000.npy
uv run interpretableintelligence serve --runs runs
```

Force a device with `II_DEVICE=cpu` or `II_DEVICE=mps`. On MPS the run sets `PYTORCH_ENABLE_MPS_FALLBACK=1`.

V-JEPA 2 weights are `facebook/vjepa2-vitl-fpc64-256`. The download allow-list is `*.json`, `model.safetensors`, and `README.md`, so the 5.1 GB `original/model.pth` is not fetched. `--tiny` skips the download and builds a small random config for a CPU smoke test. Pass `--layer 1` with `--tiny` (the tiny encoder has 2 blocks; the pretrained ViT-L has 24, and layer 16 is the usual instrumented block).

Clip directories may contain `.npy` arrays or `.mp4`, `.webm`, and `.mov` files. Video is decoded with PyAV: the short edge is resampled to 292 at decode time, display rotation is applied, and the container frame rate is kept. The adapter then reads one centred window, with stride 2 when the clip is long enough.

```bash
uv run python -c "from interpretableintelligence.adapters.vjepa2 import generate_tiny_clips; generate_tiny_clips('clips/tiny', 20, seed=3)"
uv run interpretableintelligence run vjepa2 --clips clips/tiny --layer 1 --tiny --out runs/vjepa-tiny
uv run interpretableintelligence validate runs/vjepa-tiny
```

```bash
uv run interpretableintelligence rollouts runs/demo --policy fake --episodes 32 --features 3 --controls 10 --workers 2
uv run pytest
```

Feature `3` is the planted reacher feature. The command prints the success delta, a paired Wald interval, the exact McNemar p-value, the percentile against the controls, and rollouts per validated feature. Controls stop early only for efficacy: a one-sided sequential probability ratio test on discordant pairs, with `--alpha` split across the controls. A control that does not cross that boundary runs to `--episodes`. The percentile splits ties, so a control whose delta matches the other controls scores 50. `--policy fake-null` plants nothing: success is a fair coin of the edited latents. A second invocation skips episodes already written.

## Layout

| Path | What it is |
| --- | --- |
| `src/interpretableintelligence/thresholds.py` | Frozen scoring thresholds |
| `src/interpretableintelligence/adapters/base.py` | Adapter protocol |
| `src/interpretableintelligence/adapters/fake.py` | Synthetic model, clips, and planted cues |
| `src/interpretableintelligence/adapters/vjepa2/` | V-JEPA 2 adapter |
| `docs/FRICTION-vjepa2.md` | Jeff's setup log for the V-JEPA 2 port |
| `src/interpretableintelligence/sae.py` | ReLU SAE trainer |
| `src/interpretableintelligence/score.py` | Explained variance, loss recovered, ablations, random controls |
| `src/interpretableintelligence/attribute.py` | Attribution-patching prefilter and the ablation comparison |
| `src/interpretableintelligence/pipeline.py` | Resumable capture, SAE, features, score, and viewer |
| `src/interpretableintelligence/run.py` | `execute_run` entry that does not start the viewer, and `interpretableintelligence rank` |
| `scripts/compare_attribution_vs_ablation.py` | Spearman and top-K overlap for a finished run |
| `src/interpretableintelligence/view.py` | Local clip viewer over a finished run folder |
| `src/interpretableintelligence/contract_view.py` | Read-API payloads matched by `interpretableintelligence-server` |
| `src/interpretableintelligence/policies/` | Policy + simulator protocol, fake reacher, robot stubs |
| `src/interpretableintelligence/rollout.py` | Paired rollouts, control early stopping, schema v1 summary |
| `SCHEMA.md` | Run-folder contract |
| `ARCHITECTURE.md` | What this lane owns, and the next lanes |
| `server/` | Rust read API (`interpretableintelligence-server`) over the run folder |
| `docs/HOSTING.md` | Hosting proposal and the DuckDB latency notes |

The run refuses fewer than 20 clips. The holdout is a fraction of **clips**, not frames, using `random.Random(seed)`.
