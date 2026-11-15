# Hosted GPU runs: `ii run --remote`

No GPU on your machine? `--remote` runs the same pipeline on a [Modal](https://modal.com) GPU and fills your local run folder as each stage finishes. The viewer opens on your machine as soon as features exist and refreshes itself when scores land. Runs bill to your own Modal account. The cost is shown before the run starts, while it runs, and in a record next to the run when it ends.

## Setup (once)

```bash
uv sync --extra remote
```

```bash
uv run modal setup
```

`modal setup` opens a browser to log in and stores a token in `~/.modal.toml`. `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` work instead, for CI. Nothing else needs to be configured: the first run creates the Modal app, the image, and one Volume named `interpretableintelligence`.

## Run

Add `--remote` to any `ii run`:

```bash
uv run ii run hf --model facebook/dinov2-small --clips clips/kitchen --out runs/dinov2-kitchen --remote
```

```bash
uv run ii run vjepa2 --clips clips/droid --out runs/vjepa-droid --remote --gpu L40S --max-cost 5
```

What you see:

```
remote: modal L4 at $1.12/h: GPU $0.80/h + 4 CPU cores and 16 GiB $0.32/h (Modal list price (2026))
remote: spending cap $2.00 (--max-cost) -> the container is stopped after 1h47m. A first run also builds the image, which is not counted here.
remote: uploading 31 clip files (3.8 MB)
  remote | run capture
remote: capture ready, pulled 4 files (538.1 KB); ~$0.004 so far of $2.00
  remote | run sae
  ...
remote: features ready, pulled 7 files (61.0 KB); ~$0.011 so far of $2.00
viewer http://127.0.0.1:54012/  (updates live as the remote run finishes stages)
  remote | run score
  ...
remote: done, GPU container 1m02s ~$0.019, wall 1m40s (at most ~$0.031 with startup), cap $2.00. The provider's usage page has the exact bill.
```

Lines starting `  remote |` are the pipeline's own log, unchanged. Every other `ii run` flag works the same way (`--layer`, `--prefilter`, `--sae-features`, `--concepts`, `--tiny`, `--no-score`, `--no-serve`, `--no-open`, ...). The adapters are the usual ones: `fake`, `vjepa2`, `hf --model <hub id>`, and `groot` (its image adds `lerobot[groot]==0.7.0`; the first GR00T build takes longer).

| Flag | Default | Meaning |
| --- | --- | --- |
| `--remote` | off | Run on Modal instead of this machine. |
| `--gpu` | `L4` | `T4`, `L4`, `A10`, `L40S`, `A100-40GB`, `A100-80GB`, `H100`, `H200`, `B200`. |
| `--max-cost` | `2.00` | Spending cap in USD for the GPU container. |
| `--gpu-cost-per-hour` | list price | Replaces the price per hour, if you know your own (credits, contract pricing). |

## Costs

The container's price per hour is fixed before it starts: the GPU's list price plus the 4 CPU cores and 16 GiB of memory it reserves (`src/interpretableintelligence/remote/pricing.py`). That rate is:

- **printed first**, with the cap,
- **enforced**: `--max-cost` divided by the rate becomes the Modal function's timeout, so Modal stops the container at the cap even if your laptop sleeps or the connection drops (24 h at most),
- **shown live**: each finished stage prints the spend so far, and the viewer shows a status pill with the GPU, the stage, and a ticking estimate,
- **recorded**: `runs/<run_id>/remote.json` holds the GPU, rate, cap, timeout, container time, wall time, status, and two numbers: `estimated_cost_usd` (time inside the GPU function times the rate) and `upper_bound_cost_usd` (wall time since launch times the rate, which also covers cold start).
- **used by the run's own report**: the pipeline's cost report prices an hour of footage at the remote rate, as if you had passed `--gpu-cost-per-hour`.

These are estimates from list prices. Plans, free credits and region multipliers change what you pay, and the Modal dashboard's usage page is the bill. The first run builds the container image (a few minutes of CPU, before the GPU starts) and is not part of the estimate. Later runs reuse the image.

Ctrl-C stops the Modal app, which stops the container and its billing. `remote.json` then says `interrupted`.

Things that save money: clips are uploaded once per content (a second run on the same clips uploads nothing), Hugging Face weights are cached on the Volume, a rerun into the same `--out` resumes on the Volume and skips every stage that is still valid, exactly like a local rerun, and the activation cache (`ii cache`) lives on the Volume too, so a remote run with a new seed, split, SAE size or `--out` reads captured layers instead of running the model again. `--no-cache` bypasses it, as it does locally.

## Where things live

| Where | What |
| --- | --- |
| Volume `clips/<key>/` | Uploaded clips, keyed by the sha256 of every file. |
| Volume `runs/<key>/<run_id>/` | The remote run folder. `<key>` comes from the absolute path of your `--out`, stored in `remote.json`. |
| Volume `hf-cache/` | Hugging Face downloads. |
| Volume `acts-cache/` | The activation cache, capped like the local one (`II_ACTS_CACHE_MAX_GB`, default 20 GB). |
| Your `--out` | A mirror of the remote run folder, plus `remote.json`. |

The local folder holds every contract file (`ii validate` passes on it) but not the `acts/stream/` resume caches, which are large and rebuilt on demand. The manifest's `clips_dir` points at your local clips, so `ii view runs/<run_id>`, `ii view --steer` and `ii explain` work on the mirror like on any run. A local `ii run` into the same folder would rebuild the caches and redo the work locally; to keep resuming remotely, keep passing `--remote`.

To free Volume space:

```bash
uv run modal volume rm -r interpretableintelligence runs
```

## Limits

- `--out` has to be new, empty, or a folder an earlier `--remote` run filled. A folder holding a local run is refused, so the mirror never overwrites it.
- The model has to come from the Hugging Face Hub. A model directory on your disk is refused. For a gated or private model, set `HF_TOKEN` before the run; it is passed to your own Modal account as a secret for that run only.
- One GPU per run, and one run at a time per `--out`.
- `ii rollouts` and the GR00T LIBERO-plus closed-loop rollouts are not on `--remote`.
- Steering in the playground (`ii view --steer`) runs the model on your machine.

## How it works

`src/interpretableintelligence/remote/`:

- `client.py` prices the run, uploads clips, starts the worker, and mirrors each stage: every file whose sha256 differs is downloaded, checked, and moved into place. A stage can rewrite a file the previous stage listed before it arrives; that file is picked up from the next listing, and the last listing, taken after the pipeline's final write, must arrive complete.
- `worker.py` is the container side. It runs `execute_pipeline` unchanged and turns its `on_stage` callback into events. It does not import Modal.
- `backends.py` has `ModalBackend` (a Volume and a GPU generator function) and `LocalBackend`, which runs the same worker in-process against a directory. The tests use `LocalBackend`; a new provider has to provide the same five calls.
- `modal_app.py` defines the Modal app, image, and Volume. `image.py` lists what the image installs (the same dependencies as `pyproject.toml`; a test keeps them equal).
- `pricing.py` is the price table, the cap-to-timeout conversion, and nothing else.

The viewer side is `LiveRun` in `view.py`: it forwards to the loaded run, reloads when new files arrive, and adds `live` to `/api/run`. The page polls `/api/run` while `live.state` is `running`.

Nothing in this lane changes what a run means. The scoreboard comes from the same scorer, the gates are the same, and a feature found on a remote GPU is held to the same research rules as one found locally.
