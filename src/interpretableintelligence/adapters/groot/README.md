# GR00T N1.7 / LIBERO runner

In-process runner for GR00T N1.7 on LIBERO-plus. `low_ram_policy_load.py` is the fixed loader: it rebuilds each RoPE buffer from the module's own class and raises if `inv_freq` does not match that module's `rope_theta`. `v8/inproc_v8.py` runs setup, the open-loop diagnostic gate, smoke, probe, and hooked capture; `v9/inproc_v9.py` adds the parallel env pool and `v9/ablate_v9.py` the SAE feature-ablation runner.

## How it runs

Import it as a package: `import interpretableintelligence.adapters.groot.v9` registers the bare module names the runner and its env workers use (`physint_inproc_v8`, `physint_inproc_v9`, `pot_events`, `ablate_v9`). `ii rollouts --policy groot-1028` (`docs/groot-rollouts-verify.md`) and the Modal scripts load it this way.

The generated Colab notebooks and hotfix cells (`v8/build_v8.py`, `v9/build_v9.py`, `v8_hotfix_cell.py`, `v8_diag_cell.py`, `v9/hotfix_cell.py`) pasted copies of this source into notebook cells. They and the box scripts that drove them were removed in November 2026; they are in git history before that change.

## Hardware

L4 or T4 on Colab.

## Timing

Jeff's measurements on that runtime:

- `env.step`: 0.267 s/step
- policy predict: about 0.26 s
- 1 inference per 16 steps
