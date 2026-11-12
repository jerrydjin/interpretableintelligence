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

## Import into the `ii` run contract

Convert a rollcap-v0 folder (unmodified on disk) into a sibling run directory that passes `ii validate`:

```bash
tar -xJf groot-1028-v9-n100.tar.xz
uv run ii import --format groot-v9 groot-1028-v9-n100 runs/groot-1028-v9-n100-ii
uv run ii validate runs/groot-1028-v9-n100-ii
```

- `split.json` is copied **byte-identical** (episode-index held-out lists unchanged).
- Clip rows use POT-0 failure fields from `episodes.jsonl` (`fail_kind` / `fail_t` / `fail_rule`, not `fail_*_v9_active`).
- `scoreboard.json` is honest: SAE `usable: false`, every feature `real: false`, plus `scoreboard.groot_ablation` from `demo-data/groot/groot-1028-ablate-summary.json` (task-1028 rise test null). Override with `--ablation-summary` if needed.
- Offline cosine ablation / random-percentile gates are **not** re-run; `clip_effects` carries placeholder estimate rows for the import prefilter shortlist only.

### Known gaps vs a full `ii run`

- `manifest.split` uses `kind: episode_index` (file sha256) instead of clip-level `seed` / `heldout_frac` / `assign_split`.
- `loss_recovered` is **0.0** (not measured; GR00T head splice was not captured).
- `scoreboard.groot_ablation` is an extension field; rollouts/Sprt/McNemar live there, not in `rollouts/`.
- Attribution uses objective `activation` (sum of SAE activations), not masked-predictor `prediction_loss`.
