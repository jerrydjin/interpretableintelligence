# GR00T N1.7 / LIBERO v8

In-process Colab pipeline for GR00T N1.7 on LIBERO. `low_ram_policy_load.py` is the fixed loader: it rebuilds each RoPE buffer from the module's own class and raises if `inv_freq` does not match that module's `rope_theta`. `v8/inproc_v8.py` runs setup, the open-loop diagnostic gate, smoke, probe, and hooked capture.

## Build the Colab notebook

Notebooks are generated, not committed. `v8/build_v8.py` reads `groot_colab_nodrive_v7.ipynb` from `/workspace/groot-colab` and writes `groot_colab_nodrive_v8.ipynb` plus a refreshed hotfix cell:

```bash
python v8/build_v8.py
```

Run that from the groot-colab workspace the script expects (`ROOT` in `build_v8.py`).

## Hardware

L4 or T4 on Colab.

## Timing

Jeff's measurements on that runtime:

- `env.step`: 0.267 s/step
- policy predict: about 0.26 s
- 1 inference per 16 steps
