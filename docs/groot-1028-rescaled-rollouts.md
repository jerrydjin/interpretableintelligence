# GR00T task 1028 on the rescaled SAE: positive control and rise test (Nov 18 2026)

Two closed-loop experiments on LIBERO-plus task 1028 (`nvidia/gr00t17-lerobot-libero_10-640`, DiT block 5, final denoise step). Both use the channel-normalised `auxk-8192` SAE from `ii retrain-sae` ("rescaled SAE"), and both ran on a Modal T4 through `scripts/modal_groot_steer.py`. Neither is a finding.

## How they ran

`ii rollouts --policy groot-1028 --features IDS --control-features IDS --capture DIR` builds a fixed-budget plan for hand-chosen features (`SCHEMA.md`, "GR00T, named features"). The plan runs on the v9 runner in the same container, so it no longer needs Colab: `run_live_v9` now loads the policy with `LeRobotGroot` itself. With 8 simulator workers, a T4 runs about 330 episodes an hour. Seeds are `70000 + i` on every arm.

- `modal run scripts/modal_groot_steer.py::rollouts --name NAME --features ... --controls ... --episodes N --direction rise|fall [--exp2-random SEEDS]`
- `::action_screen` runs `ii steer-scorecard` (remove and boost against 19 matched-feature controls) on the rescaled run.
- `::exp2_pick` runs the `--pick` picker on the capture through the rescaled SAE and 5 untrained SAEs.
- `::exp2_random` runs the same selection on each untrained SAE.

## Experiment 1: positive control (passed)

The question was whether the rollout pipeline sees a fall when an edit clearly matters.

The edit was chosen offline. A remove screen covered the 8 densest rescaled features (each fires on 54 to 82% of held-out tokens), on 6 held-out episodes with 19 matched controls. Six of the 8 beat every control. The strongest was 8129, which moves the planned action chunk by 17.5%, 1.44 times as much as its controls.

Rollouts used `--direction fall` with 30 episodes per arm. The 5 controls are 8129's closest matched features.

| Arm | Successes | McNemar p vs baseline |
| --- | --- | --- |
| Baseline | 14/30 | |
| Recon-only (first 8 seeds) | 1/8, the same outcome as baseline on all 8 | 1 |
| Remove 8129 | 0/30 | 0.0001 |
| Control 4865 | 3/30 | 0.003 |
| Control 7503 | 0/30 | 0.0001 |
| Control 5555 | 15/30 | 1 |
| Control 4146 | 13/30 | 1 |
| Control 1335 | 1/30 | 0.0002 |

The pipeline does detect a large fall, and the hook path is clean. However, removing most dense features breaks the policy, whichever feature it is: 8129 does not beat its controls. Rule 3 (beat every matched control) is therefore a real hurdle on this SAE.

## Experiment 2: rise test (null)

The selection procedure was fixed before looking at outcomes, and every dictionary went through the same steps:

1. Run the `--pick` picker (`--W 48 --match time`, seed 40000) on the capture.
2. Take the features with d > 0 (more active before failures) that fire on at least 2% of held-out tokens, and keep the top 10 by d.
3. Screen a remove edit of each against 19 matched-feature controls on 6 held-out episodes.
4. Confirm the passes on all 18 held-out episodes.
5. The pick is the first feature by (confirmed, screened, screen ratio).

Offline results:

- **Picker:** no dictionary had a feature pass the picker's own gates. The trained SAE's best eligible d was 0.73 (feature 3862, q 0.40). The untrained SAEs' best were 0.53 to 0.69.
- **Screen, trained SAE:** 1 of 10 features passed and was confirmed: 4686. Its remove edit beats all 19 controls on 88% of windows, but it moves actions by only 0.33%.
- **Screen, untrained SAEs:** 2 of 5 also produced a feature that passed and was confirmed (3784 and 5185). Offline, the trained SAE does no better than a random dictionary.

Rollouts used `--direction rise` with 40 episodes per arm. The SPRT design was p0 0.56 and p1 0.86. Rule 4's arms came in through `--selection`.

| Arm | Successes | Delta vs baseline | McNemar p | SPRT |
| --- | --- | --- | --- | --- |
| Baseline | 20/40 | | | |
| Recon-only (first 8 seeds) | 2/8, the same as baseline | 0 | 1 | |
| **Remove 4686** | 19/40 | -0.025 | 1 | fall |
| Controls 826, 4083, 2799, 5812, 7359 | 19, 20, 20, 21, 17 /40 | -0.075 to +0.025 | ≥ 0.58 | no rise |
| Untrained-SAE picks 2260, 2826, 3784, 5185, 4227 | 23, 22, 17, 16, 20 /40 | -0.10 to +0.075 | ≥ 0.48 | no rise |

All four rules fail for 4686:

1. The SPRT does not call a rise.
2. The delta is negative, with McNemar p = 1.
3. It does not beat its controls (it is at the 30th percentile).
4. It does not beat the random-SAE arms (40th percentile).

Every arm is within noise of baseline.

## What this says

- On task 1028, the rescaled SAE has no single feature whose removal raises success. The offline evidence pointing to one (the picker and the action screen) is matched by untrained dictionaries. This agrees with the earlier null on the old SAE (`demo-data/groot/groot-1028-ablate-summary.json`).
- Single-feature removal of dense features is destructive in a non-specific way, and removal of sparse features does nothing measurable. A rise test with single-feature ablations on 1028 is unlikely to work without a different design.
- The infrastructure works. It is cheap at about $0.004 per episode on a T4, including CPU and memory, and the positive control shows it can detect a real effect.

Result files are in the project's `groot-1028-experiments/` folder: `positive-control-8129.json`, `dense-action-screen.json`, `exp2-picker.json`, `exp2-trained-screen.json`, `exp2-trained-confirm-4686.json`, `exp2-random-sae-selection.json` and `exp2-rise-test-4686.json`.
