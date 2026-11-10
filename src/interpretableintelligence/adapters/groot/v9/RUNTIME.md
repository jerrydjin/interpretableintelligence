# GR00T v9 on Colab

On Colab, the v9 hotfix cell needs the current `physint_inproc_v8.py` sitting next to it (`/content/physint_inproc_v8.py`), because it imports `verify_rotary_inv_freq` from it and the env workers import v8 helpers from disk.

The run stops after the probe.

Jeff confirmed the L4 run: a fresh env per episode, 8 parallel envs, and zero errors.
