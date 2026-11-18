# Retrain the GR00T task-1028 SAE (`ii retrain-sae`)

The capture SAE in `groot-1028-v9-n100/sae` (BatchTopK, 8192 latents, `k` = 32) fires only about 460 of its
latents: 7,736 are dead on the train split and 7,730 on held-out (`demo-data/groot/sae-heldout-eval.json`). Its
explained variance is 0.998 anyway. `ii retrain-sae` trains a replacement on the same capture (train episodes of
`split.json` only), with dead-latent revival and a choice of dictionary size, and reports the old and new SAEs side
by side on the same tokens.

The output folder is a capture folder: a new `sae/` in the exact v9 layout plus links to the capture's
SAE-independent files (`manifest.json`, `split.json`, `episodes.jsonl`, `acts/`, ...). So `ii import --format
groot-v9`, the picker and `ii rollouts --pick --capture <folder>` read it unchanged, and the random-weights SAE
control (research rule 4) redraws the new SAE's own init (`random:0` is the retrained SAE before its first step).

## Run it on Colab (about 30 minutes on a T4)

1. Build the notebook (notebooks are generated, not committed):

   ```bash
   uv run python src/interpretableintelligence/adapters/groot/colab/build_retrain_notebook.py \
     --branch main --out groot_retrain_sae.ipynb
   ```

   Use `--branch <branch>` to run a branch that is not merged yet.

2. In Colab: File > Upload notebook, then Runtime > Change runtime type > T4 (or L4) GPU.
3. Put `groot-1028-v9-n100.tar.xz` at `MyDrive/groot/` in Google Drive, or add a GitHub token with access to the
   repo as the Colab secret `GH_TOKEN` so the notebook can fetch the draft release asset. The notebook checks the
   tarball's sha256 against `data/groot-1028.md`.
4. Run all. The last cell prints the comparison table and copies `sweep-summary.json`, each run's `retrain.json`,
   and each new `sae/` to `MyDrive/groot/sae-retrain/`.

## Or run it on Modal (one T4, about $1)

The same sweep runs on a Modal T4 billed to your own account, with no Drive or token setup (the container
downloads the release asset and checks its sha256):

```bash
pip install modal   # or: uv sync --extra remote
modal run src/interpretableintelligence/adapters/groot/modal_retrain_sweep.py --out runs/groot-sae-retrain
```

It writes `sweep-summary.json` and each run's `retrain.json` under `--out` and prints the table. The new SAEs
stay on the Volume: `modal volume get ii-groot-sae-retrain runs/<name>/sae <dest>`.
The sweep takes about an hour; the function has a 2-hour timeout (about $1.50 at most at list price); finished configs stay on the Modal
Volume `ii-groot-sae-retrain`, so a rerun skips them. `--only auxk-2048,none-2048` runs a subset.

The sweep (edit `SWEEP` in cell 3 to change it):

| Run | Latents | Revival | Why |
| --- | --- | --- | --- |
| `none-8192` | 8192 | none | The old recipe through this trainer, so trainer differences are not mistaken for a fix. |
| `auxk-8192` | 8192 | AuxK | Same size, dead-latent revival. |
| `auxk-4096`, `auxk-2048`, `auxk-1024` | 4096 / 2048 / 1024 | AuxK | Smaller dictionaries for about 67k train tokens. |
| `none-2048` | 2048 | none | Size alone, without revival. |
| `resample-2048` | 2048 | resampling | The other revival method. |
| `auxk-8192-chnorm` | 8192 | AuxK | `auxk-8192` with per-channel input standardisation (`--norm channel`). Experiment only, see below. |

`--steps 200 --timeout-min 15` is a smoke test (about a minute of training, kept on the Volume as
`runs/<name>-smoke200` so it is never mistaken for the real run); `--timeout-min` lowers the hard cap for any run.

One run by hand: `ii retrain-sae groot-1028-v9-n100 --out runs/groot-1028-sae-2048 --n-latents 2048 --revive auxk`.

## Read the table

- **alive tr / alive ho**: latents that fire at least once on train / held-out tokens at the inference threshold.
  The old SAE should show about 456 / 462 of 8192. If it does not, the eval is not reading the capture the way the
  original held-out eval did; stop there.
- **EV ho**: held-out explained variance. **EV/dim ho** averages each channel's own explained variance, so it drops
  when an SAE reconstructs only a few large channels.
- **dense ho**: latents firing on more than 10% of held-out tokens. Revival can turn dead latents into dense ones,
  which is not progress.
- **Input check** (printed under the table): the share of held-out variance in the 8 largest-variance channels. It
  is a property of the activations, the same for every SAE. On synthetic data with two outlier channels holding
  about 97% of the variance, plain BatchTopK left most latents dead, and neither revival nor a smaller dictionary
  recovered more of the planted features: AuxK made more latents fire without making them better. If the real
  share is near 1, the dead latents most likely come from a few outlier channels in the block-5 output, and the
  fix is per-channel input normalisation, which the v9 SAE layout (one scalar `norm_scale`) cannot express yet.
  That would be a follow-up across the hook, the picker reader and the importer.

### Per-channel normalisation (`--norm channel`)

`--norm channel` standardises each input channel by its own train std (`norm_scale` becomes a [D] vector, floored
at 1e-3 of the median std). The v9 hook (`ablate_v9.load_sae`), the picker (`picker/rollout_io.py`) and
`ii import --format groot-v9` read either shape: one value as before, or `d_in` values. The importer folds a vector
scale into `W_enc` and `b_dec` and multiplies the decoder rows by it, so the imported SAE's activations equal the
source's and its decoder rows are in raw units (not unit-norm); a scalar scale is folded as before.

On task 1028 (Nov 17 2026, Modal T4) `auxk-8192` with `--norm channel` had 4,756 of 8,192 latents alive on held-out
tokens (auxk-8192: 1,032), 57 dense, EV 0.9994 and EV/dim 0.994.

`EV` in the tables is `ev_raw`, measured in raw activation units, so SAEs with different normalisations compare
fairly. `ev_normalised` is in the SAE's own space; for a scalar `norm_scale` the two are equal.

A retrained SAE worth keeping has many more alive latents than the old one, few dense latents, held-out EV close
to the old 0.998, and EV/dim no worse. Alive count on its own is not the goal.

## What this does not change

- A retrained SAE is a new dictionary. Feature ids do not carry over (720, 7081 and 5852 belong to the old SAE),
  and nothing found or ruled out on the old SAE transfers.
- Offline only: no rollout, no scoreboard, no `real` flag. `split.json` is linked, never rewritten.
- The task-1028 single-feature null stands. Rollouts on a retrained SAE still need a new experimental design
  approved in the research thread before they run (`AGENTS.md`, "Current state").
- `loss_recovered` is not computed: it needs the GR00T action head with the reconstruction spliced in. The
  steering scorecard on the real model is the next check for that.
