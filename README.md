# interpretableintelligence

`interpretableintelligence` is a Python library and CLI for mechanistic interpretability of physical AI models: video world models and robot policies. This package is the shared run format and the scoring core. Drop in an adapter and a directory of clips; it captures a layer, trains a sparse autoencoder, labels features, and writes a scoreboard. The CLI is `interpretableintelligence`, and `ii` is the short form of the same command.

A feature is marked `real` only when the SAE is good enough to splice back into the predictor and the feature passes four checks: it fires across clips, its label holds on held-out clips, its ablation effect beats matched random features, and a randomly initialised copy of the model does not reproduce its firing pattern. A missing check is `null`, and `null` never counts as a pass.

`fake` is a deterministic synthetic model with three planted causal features and three decoys. It is the acceptance test for the scorer. `vjepa2` is Facebook V-JEPA 2 through Hugging Face transformers. `hf` brings a Hugging Face vision model by name: ViT-style and video encoders, CNNs, the vision tower of CLIP, SigLIP and VLMs, and SmolVLA-style robot policies (see below). `groot` is NVIDIA's GR00T N1.7 robot policy through LeRobot. pi0.5 is still a later lane.

`interpretableintelligence rollouts` is the causal check on a policy. It runs paired baseline and intervention episodes, plus matched random-feature controls, and writes `rollouts/` into the run folder (schema version 1). The fake policy is a reacher whose planted feature drops success from about 0.80 to about 0.30 when ablated; other features do not move it. LIBERO, OpenVLA, and pi0.5 sit behind the optional `robots` extra as stubs. That extra installs no simulator and no GPU package.

Runs on CPU and Apple Silicon (MPS). PyTorch is the only tensor library; nothing here requires CUDA.

## Quickstart

New here? Clone the repo, then one command. No model download, no GPU, no setup:

```bash
git clone https://github.com/jerrydjin/interpretableintelligence.git
cd interpretableintelligence
uv sync
uv run ii try
```

You need git and [uv](https://docs.astral.sh/uv/getting-started/installation/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`, or `brew install uv`). uv downloads Python 3.11+ itself if your machine has an older one. It works on Linux (x86_64 and arm64) and on macOS 14 or later on Apple Silicon; PyTorch no longer publishes wheels for Intel Macs. Nothing else is needed: no ffmpeg (video goes through PyAV, which ships its own), no Rust toolchain, no Hugging Face account. The first `uv sync` fills about 1.5 GB of uv's package cache, mostly PyTorch's CPU build. CI runs exactly these commands on fresh Ubuntu and macOS runners with empty caches (`scripts/quickstart_smoke.sh`, which you can also run yourself).

`ii try` writes 30 practice clips to `clips/try`, runs the whole pipeline on the `fake` model into `runs/try` (about 15 seconds on a laptop), and opens the viewer with the playground on. The page starts on the verified feature with the biggest effect, at the held-out moment where it fires hardest, and the terminal suggests what to try first: remove that feature, then remove a decoy and compare. The practice model has three features planted on purpose, so you can check what the tool finds against the truth before you point it at a real model. A second `ii try` reuses the run and opens in a couple of seconds. `--no-open` skips the browser and `--verbose` prints the full pipeline log.

Then try a real model on real video. You do not need clips of your own:

```bash
uv run ii get-clips --out clips/droid          # 100 short robot clips from DROID (MIT), one 175 MB download
uv run ii run hf --model facebook/dinov2-small --clips clips/droid --out runs/droid --no-score --steer
```

`--no-score --steer` gets you to the playground fastest (about 10 minutes on a laptop CPU): every feature is explorable, but none is verified yet. Run the same command without `--no-score` to add the checks; it reuses what is done. Scoring a real model is the slow part and prints progress with a time estimate as it goes. On 100 DROID clips with DINOv2-small it takes about 40 minutes on a 4-core CPU, most of it the random controls: each feature that reaches them costs about 6 minutes there, which is why they only run for features that can still be verified. The controls are plain batched forward passes, so a GPU (a Mac's is picked automatically; `--remote` rents one) is the way to shorten them further; that has not been timed here.

`ii get-clips` cuts a 4-second mp4 from the middle of each of the 100 episodes in [`lerobot/droid_100`](https://huggingface.co/datasets/lerobot/droid_100), real robot arms in real kitchens and offices, and keeps each episode's instruction in `clips/droid/clips.json`. `--n` and `--seconds` take fewer or longer clips. Any folder of your own mp4s works in its place.

The rest of this section is the fake pipeline step by step.

```bash
uv sync
uv run interpretableintelligence generate-clips --out clips/demo --n 30 --seed 0
uv run interpretableintelligence run fake --clips clips/demo --layer 16 --seed 0 --heldout-frac 0.2 --out runs/demo --gpu-cost-per-hour 2.50
```

That command captures the layer, trains the SAE, writes feature candidates, scores them, prints a cost report, and serves the viewer. A second invocation skips every stage whose outputs are already valid and serves the same folder. `--no-serve` writes the folder and exits so a script can `interpretableintelligence validate runs/demo` before opening it with `interpretableintelligence view`.

The report's cost per hour of footage is measured stage wall time times `--gpu-cost-per-hour`, divided by the hours of footage in the clips. It is omitted when the flag is absent. Stage wall time, peak RSS, device, and row/token counts are stored on `manifest.stages`. Activation rows are streamed from an on-disk cache; a stage does not load the full activation matrix.

Captured layers also go into a content-addressed activation cache shared by every run on the machine (`~/.cache/interpretableintelligence/acts`, or `$II_ACTS_CACHE`; `II_ACTS_CACHE=off` disables it). An entry is keyed by the adapter, the weights' sha (the random-init control has its own), the layer, the device type, the clip's sha256, the window and a hash of its frames, the adapter's preprocessing, and the tool version, so a rerun into a new folder, with a new seed or split, or with a new SAE size reads the layer back instead of running the model. The playground (`view --steer`, `export`) caches each window's unedited pass the same way, so only the edited pass runs. Entries are safetensors files whose stored key is checked on read; a mismatched or damaged entry is recomputed. `interpretableintelligence cache` shows its size, `cache --clear` empties it, and `run --no-cache` bypasses it. The cache is capped at `$II_ACTS_CACHE_MAX_GB` (default 20, `0` for no cap); past the cap the least recently used windows are deleted. It holds activations only, never results. GR00T is not cached yet: its activations also depend on per-clip sidecars and the denoising noise.

`interpretableintelligence rank` is the cheap prefilter. One forward and backward per window scores every SAE feature by `activation × ∂objective/∂activation`, summed over tokens and windows. `--objective prediction_loss` is the masked predictor MSE. `--objective action_loglik` is the log-likelihood of the action the policy took, for a robot policy whose success is not differentiable. `--objective output_sensitivity` (the default for hf and groot, whose loss against their own clean output has a zero gradient) ranks by a random-probe estimate of how far ablating each feature turns the output. `--objective activation` ranks by activation magnitude only, the baseline the attribution has to beat. The objective is stored in the parquet metadata. The output is a shortlist: every pick is confirmed by a real ablation, and the score itself is not evidence.

```bash
uv run interpretableintelligence rank fake --clips clips/demo --prefilter 3 --out runs/rank
uv run interpretableintelligence run fake --clips clips/demo --prefilter 3 --out runs/short --no-serve
uv run python scripts/compare_attribution_vs_ablation.py runs/demo --k 3
```

`--prefilter K` on `interpretableintelligence run` still trains, labels, and checks the random-init gate for every feature, and runs the held-out ablation plus the 100 matched random controls only for the top K, and among those only for the features that can still be verified: one that already fails a check needing no ablation (label does not hold, too few distinct clips, present at random init, unusable SAE) is left unmeasured, because the controls are the expensive part and could not change its verdict. If none of the K can be verified, all K are measured. Omit `--prefilter` to score every feature. `--extra-effects loss,group` adds full-video loss rows for single features and for co-active groups; the cosine scoreboard stays. A finished run always writes the attribution files. `scripts/compare_attribution_vs_ablation.py` prints Spearman and top-K overlap for that attribution next to the same numbers for activation magnitude. Attribution has to beat the magnitude baseline.

`runs/demo/scoreboard.json` should mark exactly three features real, labelled `stripe`, `pulse`, and `corner`. The decoys `flicker`, `band`, and `speckle` fire on many clips and keep their labels on held-out clips, but they do not move the predictor, so they are not real.

`interpretableintelligence view` serves that folder locally and opens a browser. It reads the manifest, and queries feature acts per clip and per feature instead of loading the parquet into memory. Clip effects, `features.jsonl`, and the scoreboard are shown when they are present. A missing scoreboard leaves every feature unverified, and a missing `clip_effects.parquet` shows activations with no effect column. This manifest does not store clip paths, so pass `--clips` (or a `path` / `clips_dir` recorded on a manifest that has one). Files in `--clips` match the clip's `name` (exact filename, such as `droid_ep000.mp4`) first, then the filename stem, then sha256. mp4, webm, and m4v play in a `<video>` element; avi, mov, and mkv are transcoded with ffmpeg. `--no-open` skips the browser.

`interpretableintelligence view runs/demo --steer` also loads the run's model and SAE and turns on the playground under the video. Pick a feature, choose an edit (remove, scale, clamp to a value, or add), set the strength, and press Steer. The viewer redraws the patch grid in blue to show where the model's masked prediction moved, and reports the mean cosine shift and the change in prediction error for that window. Steering is exploration only: it writes nothing to the run folder and never marks a feature real. The edit keeps the SAE's reconstruction error and changes only the chosen features (`src/interpretableintelligence/steer.py`). `POST /api/steer` takes `{clip_id, window, features, kind, value}`.

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

### Any Hugging Face vision model

```bash
uv run ii run hf --model facebook/dinov2-small --clips clips/mine --out runs/dinov2
uv run ii view runs/dinov2 --clips clips/mine --steer
```

`--model` is a Hub id or a local directory. What works:

| Kind | Examples | What gets hooked |
| --- | --- | --- |
| ViT-style image or video encoder | `facebook/dinov2-small`, `WinKawaks/vit-tiny-patch16-224`, `MCG-NJU/videomae-base` | Patch tokens of a transformer block |
| Image-text model or VLM | `openai/clip-vit-base-patch32`, `google/siglip-base-patch16-224`, `HuggingFaceTB/SmolVLM-256M-Instruct` | The vision tower only; the text side is dropped |
| Convolutional model | `microsoft/resnet-18`, `facebook/convnext-tiny-224` | A stage's feature map, one token per map cell; the readout is the final map resized to that grid |
| LeRobot policy with a frozen VLM vision encoder | `lerobot/smolvla_base` | The vision tower of its backbone (`vlm_model_name`), which is exactly what the policy sees |

Anything else stops before downloading weights with the reason and what to try: text-only models, wrong ids, gated repos (accept the licence, then `hf auth login`), and other LeRobot policies (use `ii run groot`, or point `--model` at the policy's vision backbone). The run records the adapter as `hf:<model>`, so `explain` and the playground reload the same model later. `--layer` is the 1-indexed block output (`hidden_states[layer]`) and defaults to the middle block (for CNNs, the second-to-last stage, 14x14 at 224 px). The block list is detected from the vision config's depth; `--blocks encoder.layer` picks it by hand.

A generic encoder has no masked-prediction head, so for `hf` the "prediction" is the model's own final patch tokens. Every token is scored, the target is the clean output, and the visible-mean baseline replaces every token with the average token. An ablation effect therefore reads as "how far does removing this feature move the model's output". Image models see each sampled frame (8 by default) as its own image; video models see the window at once. Frames are resized to the model's square input without cropping, so the heatmap covers the whole frame. The input side comes from the model's preprocessor config when the model accepts it (DINOv2: 224 px, although its config says 518), else from the model config. Prefix tokens (CLS, registers) pass through unedited.

The SAE for `hf` is a TopK SAE (32 active features per token) with 8 features per activation dimension (3072 for DINOv2-small); `fake` and `vjepa2` keep the 12-feature L1 acceptance dictionary, whose orthogonality penalty sums over every feature pair and would swamp reconstruction at that width. `--sae-features N` overrides either. When the SAE has more than 64 features, `ii run` scores the 10 features with the largest attribution unless `--prefilter K` says otherwise, because each scored feature costs a held-out ablation plus its matched random controls. Unscored features stay unverified and are still explorable in the playground. Finished controls are kept in the run folder (`control/random_controls.json`), so scoring again with other labels (`--concepts`, another label model) or a different `--prefilter`, or after an interruption, does not pay for them twice.

Labels for `hf` come from a local CLIP model (`openai/clip-vit-base-patch16`, downloaded on first use). Each concept in a list of everyday objects, parts, surfaces and textures gets a per-token similarity map, and a feature is labelled with the concept that best separates the tokens where it fires from those where it does not (ROC AUC ≥ 0.80 on train clips). The label holds only if that AUC is also ≥ 0.80 on held-out clips. Pass `--concepts my_concepts.txt` (one per line) to label with your own vocabulary.

A fixed word list rarely fits what a real model learns (parts of things, edges, the robot itself), so with a vision-model key set, `ii run hf` labels the features it scores by explain-then-simulate instead. A vision-language model sees the training frames where a feature fires most, boxed, and names what they share. It is asked three times, and the name that best predicts the feature on other training clips is kept, because a single reply varied from run to run. It then rates held-out boxes (half where the feature fires, half where it does not, some in the same frames) given only that label. The label holds only if those ratings reach ROC AUC ≥ 0.80, the same bar. The keys are `ANTHROPIC_API_KEY`, `FIREWORKS_API_KEY`, or `OPENAI_API_KEY` with `II_LABEL_BASE_URL` for any OpenAI-compatible endpoint. `II_LABEL_PROVIDER` picks one when several are set (`none` turns this off), and `II_LABEL_MODEL` changes the model. Replies are cached under `~/.cache/interpretableintelligence/autointerp` (or `II_LABEL_CACHE`). The evidence for every label is in `features/autointerp.json`. With no key, the CLIP labeller runs as before.

### NVIDIA GR00T N1.7 (robot policy)

```bash
uv pip install "lerobot[groot]==0.7.0"     # LeRobot's GR00T port; not in the default install
uv run ii run groot --clips clips/libero --out runs/groot
uv run ii view runs/groot --clips clips/libero --steer
```

The adapter is `groot:<checkpoint>`, and `--model` defaults to `nvidia/gr00t17-lerobot-libero_10-640` (the checkpoint behind the task 1028 work). It hooks DiT block `--layer` (default 5, 0-indexed) of the action head on the final flow-matching step, the same tensor as the Colab capture and ablation runs. The grid is one state token plus one token per planned action step (40), so in the viewer and playground the activation strip under the frame reads left to right as "now" to "40 steps ahead", and steering shows which future action steps move. The "prediction" is the action chunk under fixed noise, so an edit is the only thing that changes between the clean and edited chunk. A window is one observation, spanning the 16 steps the robot executes from it.

GR00T also needs a wrist camera, the robot state and an instruction. Put them in a sidecar next to each clip, `<stem>.groot.json`: `{"task": "...", "state": [[8 floats per frame]], "wrist": "<stem>.wrist.npy"}`. Without one the model still runs, on the main frame twice, a zero state and an empty instruction, and warns that its actions are off-distribution.

Weights stream in at half precision (6.3 GB). On a 4-core CPU the model loads in about a minute and each window takes a few seconds; a GPU is much faster. `ii run groot --tiny` runs a GR00T-shaped random network instead, with no download; it is for tests and is not GR00T. Closed-loop LIBERO rollouts for GR00T still go through `adapters/groot/v9/`.

#### Import an existing GR00T capture

```bash
uv run ii import --format groot-v9 groot-1028-v9-n100.tar.xz runs/groot-1028
uv run ii validate runs/groot-1028
```

`ii import` turns rollout logs captured outside `ii` into a run folder. `groot-v9` reads the v9 Colab capture layout (rollcap-v0), such as the task-1028 release: 100 episodes, block-5 activations, and the BatchTopK SAE trained on them. It takes the directory or the tarball and needs no GPU and no model download (about a minute on a laptop for 1028). The mapping is the one `ii run groot` uses: one clip per episode, one window per policy call, `w` = DiT token. The SAE keeps its feature ids. `split.json` is copied byte for byte. Failure labels are the pot-0 detector's (`fail_rule` `pot0-v1`), kept per clip in `manifest.import.clips`. Metrics that need the model in the loop (loss recovered, ablations, matched random features, the random-init control, effect rows) are `null`, each with a reason in `manifest.import.nulls`. So `usable` is null and every feature stays `real: false`. `SCHEMA.md` ("Imported runs") is the contract.

#### Steering scorecard

```bash
uv run ii steer-scorecard runs/groot --clips clips/libero
```

For the top verified (`real`) features, strongest ablation effect first (`--top`, default 5), it removes each feature (scale 0) and boosts it (`--boost`, default ×3) on every held-out window where the feature fires. It then measures how far the planned action chunk moved: the relative L2 change against the clean chunk. Each edit is paired with `--controls` (default 32, at least 19) matched random-feature controls on the same window. A control is a random direction at the feature's decoder norm, carrying the same per-token edit sizes, on a random permutation of that window's tokens. The scorecard reports, per feature and edit:

- how much: the mean change, the controls' mean change, the excess, and the ratio to controls;
- how reliably: the share of windows where the edit moved the actions more than every control did (chance is `1 / (controls + 1)`);
- a Monte Carlo p-value against the control arms;
- 95% intervals from a bootstrap that resamples held-out episodes, since windows of one episode are not independent.

It writes `steering/effects.parquet` and `steering/scorecard.json`, and `ii view` shows it under **Steering** and on each feature's page. When no feature is `real`, the command stops; `--features 3,7` measures named features and records them as unverified. The scorecard never touches `scoreboard.json` or a badge. It is offline, on recorded clips, so it is not a rollout finding under the research rules. It works on any run (`fake`, `hf`, `vjepa2`), where it measures the masked prediction instead.

```bash
uv run interpretableintelligence rollouts runs/demo --policy fake --episodes 32 --features 3 --controls 10 --workers 2
uv run pytest
```

Feature `3` is the planted reacher feature. The command prints the success delta, a paired Wald interval, the exact McNemar p-value, the percentile against the controls, and rollouts per validated feature. Controls stop early only for efficacy: a one-sided sequential probability ratio test on discordant pairs, with `--alpha` split across the controls. A control that does not cross that boundary runs to `--episodes`. The percentile splits ties, so a control whose delta matches the other controls scores 50. `--policy fake-null` plants nothing: success is a fair coin of the edited latents. A second invocation skips episodes already written.

`uv run pytest` needs nothing beyond `uv sync`. A few groups skip themselves, with the reason, when a tool is missing: the browser tests need Chrome or Chromium (an installed one, `uv run playwright install chromium`, or a path in `II_CHROME`); the guide tests need `node`; the Rust read-API parity tests need `cargo` (https://rustup.rs); and the GR00T box scripts under `adapters/groot/tests/` need a GPU box's checkpoint and trees (`box_paths.py` there says where they look).

### Compare models on the same clips

```bash
uv run python scripts/make_libero_clips.py --out clips/libero-moments   # 148 LIBERO moments with GR00T sidecars
uv run ii run hf --model facebook/dinov2-small --clips clips/libero-moments --out runs/cmp-dinov2 --no-serve
uv run ii run groot --clips clips/libero-moments --out runs/cmp-groot --sae-features 512 --no-serve
uv run ii compare runs/cmp-dinov2 runs/cmp-groot --out site/compare
```

Each LIBERO clip is 16 frames, so a vision model and GR00T read it as exactly one window and see the same moment. `ii compare` takes each feature's largest activation in each shared clip (clips are matched by file hash), pairs features across models that are each other's best partner by Pearson r, and keeps a pair only if r beats the 99th percentile of the best r after shuffling clip order. It writes `compare.json` and a static `index.html`. A match is correlational. Only scoreboard-`real` features are marked verified; a run that has not been scored can still be compared, and all its features show as exploratory. When clips carry GR00T sidecars, each shared concept is described by the instruction most of its top moments come from; otherwise by a CLIP hint, which is a reading aid and never a label.

`vjepa2` and `groot` runs now get CLIP concept labels too (`groot` at window level, since its tokens are actions, not patches), so their features can pass the label gate. `--sae-features` above 64 trains a TopK SAE for any adapter. Real models use every CPU core by default and the `fake` run keeps one thread so its scoreboard is reproducible; `II_THREADS=N` overrides both.

### No GPU? Run on a hosted one

```bash
uv sync --extra remote && uv run modal setup     # once
uv run ii run hf --model facebook/dinov2-small --clips clips/mine --out runs/dinov2 --remote
```

`--remote` runs the same pipeline on a [Modal](https://modal.com) GPU on your own account (default `--gpu L4`, about $1.12/h with its CPU and memory) and mirrors each finished stage into `--out`. The local viewer opens once features exist and refreshes itself when scores land, with a status pill showing the GPU, the stage, and the spend so far. `--max-cost` (default $2.00) becomes the container's timeout, so the cap holds on Modal's side even if this machine disconnects. The estimate and the run's details are written to `runs/<run_id>/remote.json`. Clips upload once, weights are cached, and a rerun resumes remotely. Setup, prices, and limits: [docs/REMOTE.md](docs/REMOTE.md).

### Share a run as a static page

```bash
uv run ii export runs/dinov2 --clips clips/mine --out site/dinov2
python -m http.server -d site/dinov2 8000
```

`ii export` writes the viewer and every file it would ask the local server for into one folder. Paths are relative, so the folder works from any static host or subpath (GitHub Pages, Netlify, Cloudflare Pages, an S3 bucket), and the page carries link-preview tags for X and other sites. Nothing in it calls a server.

The playground needs the model, so the export precomputes it: remove, scale, clamp and add at a few strengths, for the strongest features in each window, on `--steer-clips` clips (default 12, starting with the top clips of `real` and candidate features). The slider steps through those strengths. `--steer-features` sets features per window (default 6) and `--no-steer` turns the playground off. Like the live playground this is exploration only and never touches the run folder or the scoreboard. The fake run exports in about 10 seconds to 3 MB.

`make demo-site` (or `ii export-site --out demo-site`) rebuilds the static site in `demo-site/`. Serve that directory with any static host. It does not start the viewer API.

## Layout

| Path | What it is |
| --- | --- |
| `src/interpretableintelligence/thresholds.py` | Frozen scoring thresholds |
| `src/interpretableintelligence/adapters/base.py` | Adapter protocol |
| `src/interpretableintelligence/adapters/fake.py` | Synthetic model, clips, and planted cues |
| `src/interpretableintelligence/adapters/vjepa2/` | V-JEPA 2 adapter |
| `src/interpretableintelligence/adapters/hf.py` | Generic Hugging Face vision adapter (`hf:<model>`) |
| `src/interpretableintelligence/adapters/groot/adapter.py` | GR00T N1.7 adapter (`groot:<checkpoint>`) and its random `groot:tiny` stand-in |
| `docs/FRICTION-vjepa2.md` | Jeff's setup log for the V-JEPA 2 port |
| `src/interpretableintelligence/sae.py` | ReLU / TopK SAE trainer and the seeded random-weights SAE control |
| `src/interpretableintelligence/score.py` | Explained variance, loss recovered, ablations, random controls |
| `src/interpretableintelligence/attribute.py` | Attribution-patching prefilter and the ablation comparison |
| `src/interpretableintelligence/pipeline.py` | Resumable capture, SAE, features, score, and viewer |
| `src/interpretableintelligence/act_cache.py` | Content-addressed activation cache shared by reruns and the playground |
| `scripts/bench_act_cache.py` | Rerun and playground timings with and without the activation cache |
| `demo-site/` | Static demo. Regenerated by `make demo-site` |
| `src/interpretableintelligence/run.py` | `execute_run` entry that does not start the viewer, and `interpretableintelligence rank` |
| `scripts/compare_attribution_vs_ablation.py` | Spearman and top-K overlap for a finished run |
| `src/interpretableintelligence/view.py` | Local clip viewer over a finished run folder |
| `src/interpretableintelligence/steer.py` | Playground steering behind `view --steer` |
| `src/interpretableintelligence/steer_scorecard.py` | `ii steer-scorecard`: remove and boost features against matched controls, `steering/` |
| `src/interpretableintelligence/probe.py` | `ii probe`: held-out failure probe on trained-SAE latents vs a random-weights SAE and raw activations, `probe/` |
| `src/interpretableintelligence/contract_view.py` | Read-API payloads matched by `interpretableintelligence-server` |
| `src/interpretableintelligence/policies/` | Policy + simulator protocol, fake reacher, robot stubs |
| `src/interpretableintelligence/rollout.py` | Paired rollouts, control early stopping, schema v1 summary |
| `src/interpretableintelligence/importers/` | `ii import`: rollout logs captured elsewhere to a run folder (`groot-v9` first) |
| `SCHEMA.md` | Run-folder contract |
| `ARCHITECTURE.md` | What this lane owns, and the next lanes |
| `server/` | Rust read API (`interpretableintelligence-server`) over the run folder |
| `docs/HOSTING.md` | Hosting proposal and the DuckDB latency notes |

The run refuses fewer than 20 clips. The holdout is a fraction of **clips**, not frames, using `random.Random(seed)`.
