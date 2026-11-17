# Steering scorecard on the real GR00T model (task 1028)

`ii steer-scorecard` has so far run only on stand-ins (`groot:tiny`, `fake`). This runs it on
`nvidia/gr00t17-lerobot-libero_10-640` on LIBERO-plus task 1028, for the features `ii rollouts --pick` chose on the
task-1028 capture SAE in PR 50: **6262, 7353, 7775**. None of them passed the offline gates; this measures how much,
and how reliably, removing or boosting each one moves the planned action chunk, against 19 matched random-direction
controls.

## Why it needs new episodes

The capture (`groot-1028-v9-n100`) stores DiT block 5's output, not the camera images, robot state and instruction
that produced it, and steering has to run the model. So the run has three steps, all public `ii` commands:

1. **`ii record-libero`** drives GR00T closed loop on task 1028 the way the v8/v9 runners do and keeps every
   observation: main and wrist camera, state, and the bddl's instruction, one frame per env step (20 Hz). Seeds are
   `60000 + i` with `init_state_id = i % n_init_states`, away from the capture (`20000 + i`, which trained the SAE) and
   the ablation runs (`40000 + i`). Before recording anything else it runs an **input parity check**: the GR00T
   backbone on the batch the policy actually got, and on the adapter's batch rebuilt from the recorded arrays, must
   give the same backbone and state features (relative difference ≤ 1e-3). If not, it stops.
2. **`ii run groot --sae-from <capture> --no-score`** captures block 5 on those clips through the `groot` adapter
   (the same hook as the capture: final denoise step) and copies the capture's SAE in frozen, folded exactly as
   `ii import` folds it, so feature 6262 is the same latent as in PR 50. The run is not scored: the scoreboard gates
   assume an SAE trained on the run's own train split (`SCHEMA.md`, "Reused SAEs").
3. **`ii steer-scorecard --features 6262,7353,7775`** on the held-out episodes (`--heldout-frac 0.75`: 18 of 24).

`scripts/groot_steer_scorecard.sh` runs all three and prints two sanity lines. `scripts/groot_gpu_setup.sh` installs
what they need. Both are plain shell, so the same run works on Colab or on any GPU machine with a shell.

## Run it on Colab (about 1.5 to 2 hours on a T4)

1. Build the notebook (notebooks are generated, not committed):

   ```bash
   uv run python src/interpretableintelligence/adapters/groot/colab/build_steer_notebook.py \
     --branch main --out groot_steer_scorecard.ipynb
   ```

   Use `--branch <branch>` to run a branch that is not merged yet.

2. In Colab: File > Upload notebook, then Runtime > Change runtime type > T4 (or L4) GPU.
3. Put `groot-1028-v9-n100.tar.xz` at `MyDrive/groot/` in Google Drive, or add a GitHub token with access to the
   repo as the Colab secret `GH_TOKEN` so the notebook can fetch the draft release asset. The script checks the
   tarball's sha256 against `data/groot-1028.md`.
4. Run all. Results land in `MyDrive/groot/steer-scorecard/` (`scorecard.json`, `effects.parquet`, `record.json`,
   `features-stats.json`, `run-manifest.json`). Every step resumes, so after a disconnect, Run all again.

## Run it anywhere else

On any Ubuntu machine with an NVIDIA GPU (16 GB is enough), a CUDA torch, and the repo checked out:

```bash
PLUS_DIR=$PWD/.libero-plus bash scripts/groot_gpu_setup.sh
MUJOCO_GL=egl PLUS_DIR=$PWD/.libero-plus RESULTS=$PWD/results \
  bash scripts/groot_steer_scorecard.sh /path/to/groot-1028-v9-n100.tar.xz runs/groot-1028-steer
```

Knobs (environment variables): `FEATURES`, `EPISODES` (24), `HELDOUT_FRAC` (0.75), `CONTROLS` (19), `BOOST`,
`SEED0` (60000), `MAX_STEPS`, `RESULTS`. To measure a retrained SAE from `ii retrain-sae`, pass its output folder as
`CAPTURE` and set `FEATURES` to ids in that dictionary.

## Reading the result

Check the sanity lines first:

- **Input parity pass=True.** Otherwise the recorder has already stopped.
- **SAE held-out EV on the new clips** should be close to the capture's 0.998, and the dead fraction near the
  capture's (about 456 of 8192 alive). The capture kept a fixed subset of the 41 DiT tokens and this run reads all of them, so small
  differences are expected. A much lower EV means the adapter's hooked tensor is not the capture's, and the
  scorecard does not mean anything until that is fixed.
- **Held-out tokens each feature fires on.** A feature that never fires has no scorecard row.

Then the table, one line per feature and edit: **change vs controls** is the mean relative L2 change of the planned
40-step action chunk; **ratio** is feature over controls with a 95% episode-bootstrap interval; **beats every control
on X%** is the share of windows where the feature moved the chunk more than all 19 controls (chance 5%);
**beats controls** needs p ≤ 0.05 against the control arms and an excess whose interval is above 0.

## Matched-feature controls (`--controls-from features`)

A random direction in DiT block 5's 1536 dimensions is mostly ignored by the network, so beating 19 of them is a weak
bar. `ii steer-scorecard --controls-from features` (`CONTROLS_FROM=features` for the shell script) swaps them for the
decoder directions of other live SAE features: for each steered feature, the 19 live features nearest to it on
train-split firing rate, mean activation when firing and decoder norm (sum of absolute log ratios), skipping the
other steered features and any feature whose decoder has |cosine| above 0.5 with the steered one's. Each control's
direction is rescaled to the steered feature's decoder norm and added with exactly the feature edit's per-token
coefficients, on the same tokens, so the only difference is the direction, and it is a direction the model uses.
The matched ids and their statistics are in `scorecard.json`'s `control_features`. On Modal, after a full random run,
`steer_smoke` checks one held-out episode and `scorecard --controls-from features` reuses the recorded clips and run
(results go to `results-features` on the Volume).

## What this does not say

- It is offline: the edited chunk is never executed. A feature that beats its controls here is a candidate for a
  rollout design, not a finding. Findings still need the rollout rules in `AGENTS.md` (SPRT rise, McNemar, matched
  controls, the random-weights SAE control), and new task-1028 rollouts need a design approved in the research thread.
- It never reads or writes `scoreboard.json`, and nothing here sets `real`.
- The adapter's flow-matching noise is fixed per window, not the noise the policy drew while acting, so the clean
  chunk it compares against is a fresh sample from the same observation.
