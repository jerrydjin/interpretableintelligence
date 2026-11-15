# GR00T task 1028 capture (`groot-1028-v9-n100`)

**Data release pending licence clarification.** The capture tarball is not offered for public download until upstream licence terms are settled. This page records what the archive contains, the formal task-1028 ablation null, and internal release metadata only.

Background: **100 LIBERO-plus rollouts** on task **1028**, activations at action-head **transformer block 5** (final denoise step), plus a trained **BatchTopK SAE** and picker outputs—the dataset behind the task-1028 rise-test work described in [issue #3](https://github.com/jerrydjin/interpretableintelligence/issues/3).

## What is in the archive (when published)

| Field | Value |
| --- | --- |
| Model | `nvidia/gr00t17-lerobot-libero_10-640` (GR00T / LeRobot) |
| Benchmark | LIBERO-plus, task **1028** (`KITCHEN_SCENE8_put_both_moka_pots_on_the_stove…`) |
| Episodes | **100** (`seed = 20000 + episode_idx`) |
| Successes | **56** / 100 |
| Hook | `_groot_model.action_head.model.transformer_blocks.5` output @ final denoise step (index 3 of 4), `d_model=1536` |
| SAE | BatchTopK, **k=32**, **8192** latents (`sae/weights.safetensors`, `sae/config.json`) |
| Train / held-out split | Episode-level: `numpy.default_rng(0).choice(100, 25, replace=False)` → **25** held-out episodes (see `split.json`) |

Unpacked layout is the v9 **rollcap / rollout_io** run folder (`manifest.json` format `rollcap-v0`): `acts/`, `acts_meanpool/`, `actions.safetensors`, `sim/`, `episodes.jsonl`, `sae/`, `candidates_final.json`, picker outputs, etc.

## Task 1028 rise test (formal null)

The single-feature ablation on LIBERO-plus task **1028** was a formal **null** (**641** rollouts, **20** random controls, `real: false`). Vendored summary: `demo-data/groot/groot-1028-ablate-summary.json` (also surfaced on the static demo site GR00T page).

| Condition | Result |
| --- | --- |
| Baseline | **39** / **75** successes |
| Feature **720** (ablated) | **1** / **9** |
| Features **7081** + **5852** (ablated together) | **23** / **63** |
| Random controls | **5** of **20** control arms also received a SPRT **fall** verdict |

Candidate falls were **not** distinguishable from matched random controls. The vendored summary also carries early-stopping-valid statistics (exact intervals and p-values under the plan's SPRT, per-arm control counts with a permutation-style p-value, Holm across the two candidates; see `src/interpretableintelligence/adapters/groot/v9/ablate/README.md`). Feature 720: 3 of 10 matched controls also fell, 1 of 10 did at least as badly, permutation p 0.18. Pair 7081+5852: 2 of 10 fell, 2 of 10 at least as badly, permutation p 0.27. Sequential rise p-values are 0.99 and 0.96; both stay `real: false`. Do not re-run single-feature rise sweeps on 1028 without a **new experimental design** approved in the research thread.

## Draft release metadata (not publicly downloadable)

These fields are recorded for provenance only. There is no public download URL while the release stays a hidden draft.

| Field | Value |
| --- | --- |
| **Release tag (draft)** | `data-groot-1028-v1` |
| **Asset name** | `groot-1028-v9-n100.tar.xz` |
| **Size** | 359,595,840 bytes |
| **SHA-256** | `0f8954e52ae737aec8a9cd734d58efc3e43ad3725d623fae2e5a6eac858cf00e` |

## Contract note

The unpacked v9 capture tree is **not** an `ii` run folder by itself (`rollcap-v0` episode metadata, no `scoreboard.json`). `ii import` maps it onto the run contract without editing the source:

```bash
uv run ii import --format groot-v9 groot-1028-v9-n100.tar.xz runs/groot-1028-v9-n100
uv run ii validate runs/groot-1028-v9-n100
```

The import takes the tarball or the unpacked directory. On the release asset above it writes 89,760 activation rows (2,805 policy calls × 32 captured tokens), reproduces the SAE's held-out EV (0.9976) and its live latents (466 of 8192 fire anywhere in the capture), and `ii validate` exits 0.

- `split.json` is copied byte for byte (episode-index held-out list unchanged), and `manifest.split.sha` is its sha256.
- Per-episode outcomes are the **pot-0** labels (`fail_kind`, `fail_t`, `fail_subkind`, `fail_rule` `pot0-v1`), in `manifest.import.clips`. The v9 active-object labels (`*_v9_active`) are not used.
- `loss_recovered`, `loss_sae`, `loss_visible_mean` and therefore `usable` are `null`, and `clip_effects.parquet` has no rows. Each has a reason in `manifest.import.nulls`: they need the GR00T action head run with the block-5 reconstruction spliced in, and the capture holds block-5 outputs only. Every feature is `real: false`.
- Feature ids match the picker and ablation work (for example 720, 7081 and 5852). Values are `norm_scale` (269.12) times the source SAE's, because the input normalisation is folded into the stored weights.
- `model.sha` is the sha `ii run groot` would record when the Hugging Face Hub is reachable (or `--model-sha`). Otherwise it is a digest of the declared model, and `manifest.import.notes.model_sha_source` says which.

The task-1028 rise test above is not affected: an import adds no rollouts and no causal claim.