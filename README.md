# physint

`physint` is a Python library and CLI for mechanistic interpretability of physical AI models: video world models and robot policies. This package is the shared run format and the scoring core. Drop in an adapter and a directory of clips; it captures a layer, trains a sparse autoencoder, labels features, and writes a scoreboard.

A feature is marked `real` only when the SAE is good enough to splice back into the predictor and the feature passes four checks: it fires across clips, its label holds on held-out clips, its ablation effect beats matched random features, and a randomly initialised copy of the model does not reproduce its firing pattern. A missing check is `null`, and `null` never counts as a pass.

The only world-model adapter in this package is `fake`: a deterministic synthetic model with three planted causal features and three decoys. It is the acceptance test for the scorer. Real V-JEPA 2 and pi0.5 world-model adapters are a later lane.

`physint rollouts` is the causal check on a policy. It runs paired baseline and intervention episodes, plus matched random-feature controls, and writes `rollouts/` into the run folder (schema version 1). The fake policy is a reacher whose planted feature drops success from about 0.80 to about 0.30 when ablated; other features do not move it. LIBERO, OpenVLA, and pi0.5 sit behind the optional `robots` extra as stubs. That extra installs no simulator and no GPU package.

Runs on CPU and Apple Silicon (MPS). PyTorch is the only tensor library; nothing here requires CUDA.

## Quickstart

```bash
uv sync
uv run physint generate-clips --out clips/demo --n 30 --seed 0
uv run physint run fake --clips clips/demo --layer 16 --seed 0 --heldout-frac 0.2 --out runs/demo
uv run physint validate runs/demo
```

`runs/demo/scoreboard.json` should mark exactly three features real, labelled `stripe`, `pulse`, and `corner`. The decoys `flicker`, `band`, and `speckle` fire on many clips and keep their labels on held-out clips, but they do not move the predictor, so they are not real.

Force a device with `PHYSINT_DEVICE=cpu` or `PHYSINT_DEVICE=mps`.

```bash
uv run physint rollouts runs/demo --policy fake --episodes 32 --features 3 --controls 10 --workers 2
uv run pytest
```

Feature `3` is the planted reacher feature. The command prints the success delta, a paired Wald interval, the exact McNemar p-value, the percentile against the controls, and rollouts per validated feature. Controls stop early once a Wilson bound separates them from the candidate. A second invocation skips episodes already written.

## Layout

| Path | What it is |
| --- | --- |
| `src/physint/thresholds.py` | Frozen scoring thresholds |
| `src/physint/adapters/base.py` | Adapter protocol |
| `src/physint/adapters/fake.py` | Synthetic model, clips, and planted cues |
| `src/physint/sae.py` | ReLU SAE trainer |
| `src/physint/score.py` | Explained variance, loss recovered, ablations, random controls |
| `src/physint/run.py` | `capture → SAE → label → score → run folder` |
| `src/physint/policies/` | Policy + simulator protocol, fake reacher, robot stubs |
| `src/physint/rollout.py` | Paired rollouts, control early stopping, schema v1 summary |
| `SCHEMA.md` | Run-folder contract |
| `ARCHITECTURE.md` | What this lane owns, and the next lanes |

The run refuses fewer than 20 clips. The holdout is a fraction of **clips**, not frames, using `random.Random(seed)`.
