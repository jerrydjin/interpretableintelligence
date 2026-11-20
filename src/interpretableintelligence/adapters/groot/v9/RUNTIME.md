# GR00T v9 runtime

The env workers import v8 helpers by the bare name `physint_inproc_v8`. Importing `interpretableintelligence.adapters.groot.v9` registers that name (and `physint_inproc_v9`, `pot_events`, `ablate_v9`) before anything else runs.

The run stops after the probe.

Jeff confirmed the L4 run: a fresh env per episode, 8 parallel envs, and zero errors.

Task 614 uses the stove region measured on 1028 (`pot_events.STOVE_REGIONS`), which hasn't been checked against 614's own successes, and all labels describe events, not a validated cause.

The ablation runner is ablate-v1 on task 1028: 24 conditions (a fixed 75-episode baseline on seeds 40000+i, an 8-episode recon-only identity check, feature 720 vs 10 matched controls, and the 7081+5852 pair vs 10 random control pairs). The edit subtracts norm_scale·Σf·W_dec at the final denoise step. The stop rule is Sobel–Wald (stop when either SPRT accepts H1, or both accept H0, or at the cap of 75). Arm B's controls aren't correlation-matched. All results are unvalidated, real=false.
