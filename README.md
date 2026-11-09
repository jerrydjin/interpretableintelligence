# physint

`physint` is a Python library and CLI for mechanistic interpretability of physical AI models: video world models and robot policies. This package is the shared run format and the scoring core. Drop in an adapter and a directory of clips; it captures a layer, trains a sparse autoencoder, labels features, and writes a scoreboard.

A feature is marked `real` only when the SAE is good enough to splice back into the predictor and the feature passes four checks: it fires across clips, its label holds on held-out clips, its ablation effect beats matched random features, and a randomly initialised copy of the model does not reproduce its firing pattern. A missing check is `null`, and `null` never counts as a pass.

The only adapter in this package is `fake`: a deterministic synthetic model with three planted causal features and three decoys. It is the acceptance test for the scorer. Real V-JEPA 2 and pi0.5 adapters are a later lane.

Runs on CPU and Apple Silicon (MPS). PyTorch is the only tensor library; nothing here requires CUDA.

## Quickstart

```bash
uv sync
uv run physint generate-clips --out clips/demo --n 30 --seed 0
uv run physint run fake --clips clips/demo --layer 16 --seed 0 --heldout-frac 0.2 --out runs/demo
uv run physint validate runs/demo
```

`runs/demo/scoreboard.json` should mark exactly three features real, labelled `stripe`, `pulse`, and `corner`. The decoys `flicker`, `band`, and `speckle` fire on many clips and keep their labels on held-out clips, but they do not move the predictor, so they are not real.

Explain a new clip with that run. The cited answer is per window and time, and it includes only real features and only ablation effects. Anything else is listed as unverified. If the run's SAE is not usable, the command refuses.

```bash
uv run physint explain --run runs/demo --clip clips/demo/clip-000.npy
uv run physint serve --runs runs
```

Force a device with `PHYSINT_DEVICE=cpu` or `PHYSINT_DEVICE=mps`.

```bash
uv run pytest
```

## Layout

| Path | What it is |
| --- | --- |
| `src/physint/thresholds.py` | Frozen scoring thresholds |
| `src/physint/adapters/base.py` | Adapter protocol |
| `src/physint/adapters/fake.py` | Synthetic model, clips, and planted cues |
| `src/physint/sae.py` | ReLU SAE trainer |
| `src/physint/score.py` | Explained variance, loss recovered, ablations, random controls |
| `src/physint/run.py` | `capture → SAE → label → score → run folder` |
| `SCHEMA.md` | Run-folder contract |
| `ARCHITECTURE.md` | What this lane owns, and the next lanes |

The run refuses fewer than 20 clips. The holdout is a fraction of **clips**, not frames, using `random.Random(seed)`.
