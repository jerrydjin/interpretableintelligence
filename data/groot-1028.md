# GR00T task 1028 capture (`groot-1028-v9-n100`)

Published capture from the GR00T v9 Colab path: **100 LIBERO-plus rollouts** on task **1028**, activations at action-head **transformer block 5** (final denoise step), plus a trained **BatchTopK SAE** and picker outputs. This is the dataset behind the task-1028 rise-test work described in [issue #3](https://github.com/jerrydjin/interpretableintelligence/issues/3).

## What is in the archive

| Field | Value |
| --- | --- |
| Model | `nvidia/gr00t17-lerobot-libero_10-640` (GR00T / LeRobot) |
| Benchmark | LIBERO-plus, task **1028** (`KITCHEN_SCENE8_put_both_moka_pots_on_the_stove…`) |
| Episodes | **100** (`seed = 20000 + episode_idx`) |
| Successes | **56** / 100 |
| Hook | `_groot_model.action_head.model.transformer_blocks.5` output @ final denoise step (index 3 of 4), `d_model=1536` |
| SAE | BatchTopK, **k=32**, **8192** latents (`sae/weights.safetensors`, `sae/config.json`) |
| Train / held-out split | Episode-level: `numpy.default_rng(0).choice(100, 25, replace=False)` → **25** held-out episodes (see `split.json`) |

Unpacked layout is the v9 **rollcap / rollout_io** run folder (`manifest.json` format `rollcap-v0`): `acts/`, `acts_meanpool/`, `actions.safetensors`, `sim/`, `episodes.jsonl`, `sae/`, `candidates_final.json`, picker outputs, etc. See the bundled `README.md` inside the archive for full schema notes.

## Release asset

| | |
| --- | --- |
| **Release** | [data-groot-1028-v1](https://github.com/jerrydjin/interpretableintelligence/releases/tag/data-groot-1028-v1) (pre-release) |
| **Asset** | `groot-1028-v9-n100.tar.xz` |
| **URL** | https://github.com/jerrydjin/interpretableintelligence/releases/download/data-groot-1028-v1/groot-1028-v9-n100.tar.xz |
| **Size** | 359,595,840 bytes |
| **SHA-256** | `0f8954e52ae737aec8a9cd734d58efc3e43ad3725d623fae2e5a6eac858cf00e` |

## Download, verify, unpack

### GitHub CLI (private repo; authenticated `gh`)

```bash
mkdir -p runs && cd runs
gh release download data-groot-1028-v1 \
  --repo jerrydjin/interpretableintelligence \
  --pattern 'groot-1028-v9-n100.tar.xz'
echo '0f8954e52ae737aec8a9cd734d58efc3e43ad3725d623fae2e5a6eac858cf00e  groot-1028-v9-n100.tar.xz' | sha256sum -c -
tar -xJf groot-1028-v9-n100.tar.xz
```

### `curl` (private repo; fine-grained or classic token with `repo`)

```bash
mkdir -p runs && cd runs
curl -fL \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "Accept: application/octet-stream" \
  -o groot-1028-v9-n100.tar.xz \
  "https://github.com/jerrydjin/interpretableintelligence/releases/download/data-groot-1028-v1/groot-1028-v9-n100.tar.xz"
echo '0f8954e52ae737aec8a9cd734d58efc3e43ad3725d623fae2e5a6eac858cf00e  groot-1028-v9-n100.tar.xz' | sha256sum -c -
tar -xJf groot-1028-v9-n100.tar.xz
```

After unpack, the run directory is `groot-1028-v9-n100/`.

## `ii validate` (interpretableintelligence run contract)

This folder is **not** a finished `ii run` tree (`SCHEMA.md` version 0). Validation was checked on the published tarball contents:

```bash
uv run ii validate groot-1028-v9-n100
```

**Result (2026-11-12):** exit **1** — first error reported:

```text
error: missing scoreboard.json
```

Even with `scoreboard.json` present, the v9 capture would still diverge from the `ii run` contract: `manifest.json` uses `rollcap-v0` episode metadata instead of clip-grid `manifest` fields; `split.json` is episode-index lists without the clip-level `seed` / `heldout_frac` / `sha` shape; there is no `features/feature_acts.parquet`, `features/clip_effects.parquet`, `features/features.jsonl`, or automated scorer output. **Do not hand-edit** `scoreboard.json` or `split.json` to force validation.

**Conversion gap (future work, not part of this release):** a dedicated importer (or a future `ii` adapter command for GR00T v9) would need to map episodes/tokens to the standard run layout and run labeling/scoring stages to produce contract artifacts. Until then, use the v9 tooling under `src/interpretableintelligence/adapters/groot/v9/` (e.g. `make_ablate_plan.py`, Colab `ablate_v9.py`) against an unpacked copy of this folder.
