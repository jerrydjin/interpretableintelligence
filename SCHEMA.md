# Run folder contract

A finished `physint run` writes one directory. Later lanes (viewers, ingest, the failed-clip API) read this directory and nothing else from the training process. `physint validate <run>` checks the rules below.

## Versions

The contract has two versions. Version 1 reads version 0.

- **Version 0.** The tree below, and no `rollouts/` directory. `manifest.json` omits `schema_version` (an explicit `0` is the same thing). `physint run` writes this.
- **Version 1.** Version 0 plus `manifest.schema_version` equal to `1` and a `rollouts/` directory. Nothing from version 0 is removed or renamed. `physint rollouts` bumps an existing manifest to version 1 after it writes `rollouts/`.

A version-0 folder still validates. A folder that grew `rollouts/` without the version bump does not.

```
runs/<run_id>/
  manifest.json
  split.json
  acts/shard-NNNNN.safetensors
  sae/weights.safetensors
  sae/config.json
  features/feature_acts.parquet
  features/clip_effects.parquet
  features/features.jsonl
  scoreboard.json
```

`run_id` is the directory name. Thresholds cited below live in `src/physint/thresholds.py` and are part of the contract.

## manifest.json

```json
{
  "run_id": "demo",
  "tool_version": "0.1.0",
  "model": {"adapter": "fake", "name": "fake", "sha": "<64 hex>", "random_init": false},
  "layer": 16,
  "d_model": 64,
  "grid": {"t": 8, "h": 4, "w": 4, "tubelet": 2, "patch_px": 4},
  "split": {"seed": 0, "heldout_frac": 0.2, "sha": "<64 hex>"},
  "clips": [
    {"clip_id": "clip-000", "sha256": "<64 hex>", "fps": 8.0, "n_frames": 16, "split": "train"}
  ]
}
```

- At least 20 clips.
- `clips[].sha256` is the sha256 of the clip file bytes at capture time. The run folder does not contain the pixels.
- `clips[].split` is `train` or `heldout`. A clip is entirely one or the other.
- `model.sha` is a sha256 of the adapter weights used for the run (`random_init` is false for a normal run).
- `split.sha` equals `split.json`'s `sha`.
- `grid` is the activation lattice: `t` time bins, `h`×`w` patches, each bin pooling `tubelet` frames and `patch_px` pixels.

## split.json

```json
{
  "seed": 0,
  "heldout_frac": 0.2,
  "train": ["clip-000"],
  "heldout": ["clip-001"],
  "sha": "<64 hex>"
}
```

Algorithm:

1. Sort clip ids.
2. Shuffle with `random.Random(seed)`.
3. Hold out `n_held = round(n * heldout_frac)`, clamped to `[1, n - 1]`.
4. The first `n_held` shuffled ids are held out. Both lists are stored sorted.

`sha` is sha256 of the UTF-8 JSON of `{seed, heldout_frac, train, heldout}` with `sort_keys=true` and separators `(',', ':')`. The `sha` field is not part of the hashed payload.

## Activation shards

`acts/shard-00000.safetensors`, then `shard-00001.safetensors`, with no gaps. Rows are one spatiotemporal token:

| tensor | dtype | shape |
| --- | --- | --- |
| `acts` | float16 | `[N, d_model]` |
| `clip_idx` | int32 | `[N]` |
| `window` | int16 | `[N]` |
| `t`, `h`, `w` | int16 | `[N]` |

`clip_idx` indexes `manifest.clips`. Order is manifest clip order, then window index, then row-major `(t, h, w)`. Every grid cell of every window appears once. Shards are concatenated in filename order. The SAE is trained on float32 copies of the train rows; float16 is the stored contract.

## SAE

`sae/weights.safetensors` holds `W_enc [n_features, d_model]`, `b_enc [n_features]`, `W_dec [d_model, n_features]`, `b_dec [d_model]`.

```
f = relu(W_enc @ (x - b_dec) + b_enc)
x_hat = W_dec @ f + b_dec
```

Decoder columns are unit-norm. `sae/config.json` records `d_model`, `n_features`, `l1_coeff`, `ortho_coeff`, `lr`, `steps`, `batch_size`, `seed`, `normalize_decoder`, and `activation` (`relu`). Training uses train-split activations only. The orthogonality term keeps one planted direction from splitting into two live features.

## features/feature_acts.parquet

Sparse nonzero SAE activations. Columns, in order:

`feature_id` int32, `clip_id` string, `window` int16, `t` int16, `h` int16, `w` int16, `value` float32.

Every `value` is > 0. A feature that never fires has no rows. "Never fires" means max activation ≤ `1e-5` (`DEAD_ACT_EPS`) on the full capture (train and held-out).

## features/clip_effects.parquet

Columns, in order:

`feature_id` int32, `clip_id` string, `window` int16, `t` int16, `effect` float32, `effect_kind` string.

`effect_kind` is `estimate` or `ablation`.

- `estimate`: activation × ∂(mean masked predictor MSE)/∂(feature activation), averaged over `h, w` in that frame. Straight-through: the forward value is the true layer activation, and the gradient flows through the decoder. One row for every feature, clip, window, and frame.
- `ablation`: the feature is actually zeroed by subtracting `f_i * decoder_i` from the layer. Rows exist for the top 10 features per clip (or every feature, if the dictionary is smaller), ranked by mean absolute estimate on that clip, and for every frame of those features. The stored effect is the mean cosine distance of the predictor output over the masked spatial cells in that frame.

## features/features.jsonl

One JSON object per SAE feature, id `0 .. n_features-1`:

```json
{
  "id": 0,
  "label": "stripe",
  "label_model": "fake-cue-v1",
  "top": [{"clip_id": "clip-003", "window": 0, "t": 1, "value": 1.04}],
  "distinct_clips_top20": 20,
  "label_holds_heldout": true,
  "ablation_effect": 0.0003,
  "random_percentile": 100.0,
  "n_random": 100,
  "absent_in_random_init": true,
  "real": true
}
```

`top` is the highest strictly positive activations, at most 20, each as `{clip_id, window, t, value}`. `distinct_clips_top20` is the number of distinct `clip_id`s in that list. The other fields match `scoreboard.json` exactly.

## scoreboard.json

One record per run. No extra keys.

```json
{
  "run": {
    "model_sha": "",
    "layer": 16,
    "split_sha": "",
    "seed": 0,
    "n_train_clips": 0,
    "n_heldout_clips": 0
  },
  "sae": {
    "ev_heldout": 0.0,
    "loss_recovered": 0.0,
    "dead_frac": 0.0,
    "usable": false
  },
  "features": [
    {
      "id": 0,
      "label": "",
      "distinct_clips_top20": 0,
      "label_holds_heldout": null,
      "ablation_effect": 0.0,
      "random_percentile": 0.0,
      "n_random": 100,
      "absent_in_random_init": null,
      "real": false
    }
  ]
}
```

`run` matches the manifest (`model.sha`, `layer`, `split.sha`, `split.seed`, clip counts).

### SAE block

- `ev_heldout`: explained variance of the SAE reconstruction on held-out activations, `1 - ||x - x_hat||^2 / ||x - mean(x)||^2`.
- `loss_recovered`: splice `x_hat` in place of the layer and measure mean squared error of `predict_masked` on held-out clips. Scale is 0 when the layer is zeroed and 1 when the layer is untouched: `(loss_zero - loss_sae) / (loss_zero - loss_clean)`.
- `usable`: `loss_recovered >= 0.70`.
- `dead_frac`: share of features that never fire on the captured tokens.

### Feature block

`ablation_effect` is the mean cosine distance between the original predictor output and the output after the feature is zeroed, over held-out masked tokens. Masked rows come from `predict_masked` in row-major `(t, h, w)` order of the adapter mask. Cosine distance is `1 - cos`. A zero vector against a nonzero vector counts as distance 1; two zero vectors count as 0.

`random_percentile` compares that effect to `n_random` (≥ 100) random features matched on held-out firing count and on `(decoder L2 norm × mean firing activation)`. Each control uses a random unit direction scaled to the same decoder norm, the same multiset of activation values, and a random subset of held-out tokens of the same size. The percentile is the percentage of control effects that are `<=` the feature's effect (ties do not count as a win for the control).

`absent_in_random_init` reruns capture and SAE training with the same hyperparameters on `load(random_init=True)`. It is true when the feature's activation pattern has variance and no random-init feature has Pearson correlation ≥ `0.45` with it. A flat pattern is not absent: dead features share that pattern.

`label_holds_heldout` is the labeller's held-out check. For the fake labeller (`fake-cue-v1`), the label is the planted cue with the highest train-token Pearson correlation, and only if that correlation is > `0.50`. The label holds when the same cue correlates ≥ `0.60` on held-out tokens. Otherwise the field is false. `null` means the check was not run.

`real` is true only when `usable` is true and all four of the following hold:

- `distinct_clips_top20 >= 5`
- `label_holds_heldout is true`
- `random_percentile >= 95`
- `absent_in_random_init is true`

`null` is not a pass. A high percentile alone is not a pass: a dead feature can tie a null control distribution and still score 100.

## Adapter obligations the schema assumes

`predict_masked(clip, window, edit=None)` returns `(pred, target)` for the masked region only, row-major over the true cells of `mask(window)`. `hook(layer, mode)` is a context manager with modes `capture`, `edit`, and `zero`. Edits replace the instrumented layer before the predictor reads it. Splicing the SAE and zeroing a feature are edits; zeroing the whole layer uses mode `zero`.

## rollouts/ (version 1)

`physint rollouts` measures whether an intervention on SAE latents changes task success. It writes two files:

```
rollouts/episodes.parquet
rollouts/summary.json
```

`episodes.parquet` is one row per episode. Columns, in order:

| column | type | meaning |
| --- | --- | --- |
| `episode_id` | string | Identity of the episode. A rerun skips ids already stored. |
| `task_id` | string | Task passed to `reset`. |
| `seed` | int64 | Simulator seed. Episode offset `i` uses `seed + i` on every arm. |
| `condition` | string | `baseline` or `intervention`. |
| `feature_ids` | list<int32> | SAE features the intervention edits. Empty for baseline. |
| `layer` | int32, null for baseline | Layer whose latents are edited. |
| `kind` | string, null for baseline | `ablate`, `scale`, or `clamp`. |
| `value` | float64, null for baseline | Scale factor, clamp level, or `0` for ablate. |
| `success` | bool | Terminal task success. |
| `steps` | int32 | Simulator steps taken. At least 1. |
| `wall_time_s` | float64 | Wall time of that episode, in seconds. |

`ablate` sets the listed latents to 0. `scale` multiplies them by `value`. `clamp` sets them to the constant `value`.

The design is paired. One baseline arm and every intervention arm see the same seeds. Candidate features are the ids passed to `--features`. Control arms are `K` other features (`--controls`), drawn from a seeded permutation of the remaining policy features, with the same layer, kind, and value. Baseline and candidates always run `N = episodes_per_arm` episodes. A control can stop early.

Early stopping is a two-sided sequential probability ratio test on the discordant pairs of each control against each candidate. Looks happen on seed prefixes. The first prefix has length `min_episodes`; later prefixes grow by `batch_size`. A pair is `+1` when the control succeeds and the candidate fails, and `-1` the other way. Concordant pairs are skipped. Under the null that the two arms are exchangeable, the signs are fair coin flips.

The test runs two one-sided likelihood-ratio martingales. The positive martingale's alternative is `P(sign = +1) = (1 + λ) / 2`, with `λ = sprt_lambda` (0.6, so the alternative discordant probability is 0.80). The negative martingale swaps the alternative. Each martingale starts at 1 and is multiplied by `(1 + λ)` or `(1 - λ)` on each discordant pair. A control is not scheduled past a prefix once, for every candidate, the larger martingale's maximum along that prefix has reached `2 / alpha`. Ville's inequality bounds each side by `alpha / 2` no matter how often the prefix is inspected, so the type I error of stopping is at most `alpha`. The control is not stopped before `min_episodes`. A crossing only on the final prefix does not stop early: `reason` stays `fixed_budget`. Otherwise `reason` is `sprt_separated`.

`summary.json` records, per arm, `episodes_run`, `episodes_budget`, `stopped_early`, and `reason`. Each control also stores `gap_vs_candidate` (its success rate minus the success rate of the candidate with the smallest path-maximum log likelihood ratio), `sprt_log_lr` (that log likelihood ratio), and `sprt_threshold` (`log(2 / alpha)`). Baseline and candidates leave those three null. The summary's `gap_bound` is `sprt_discordant` and `sprt_lambda` is `λ`.

`summary.json` is derived from the parquet and the schedule. `total_episodes` is the number of rows. `fixed_budget_episodes` is `(1 + n_candidates + n_controls) * N`, the cost if nothing stopped early. `episodes_saved` is the difference. `rollouts_per_validated_feature` is `total_episodes / n_validated`, or null when nothing was flagged.

For each candidate and each control the summary stores the paired success rates on the seeds that arm actually ran, `delta = baseline_success - intervention_success`, a Wald interval on the paired differences (`ci_low`, `ci_high`), and the two-sided exact McNemar `p_value`. `control_percentile` is the average-rank percentile of this delta among the comparison deltas: `100 * (n_strictly_below + 0.5 * n_tied) / n`. A delta tied with every comparison scores 50. A delta strictly above every comparison scores 100. A delta strictly below every comparison scores 0. Candidates are compared with the controls. A control is compared with the other controls. Values within `1e-9` count as tied. No comparisons means the percentile is null. Counting every tie as `<=` is the scoreboard's `percentile_le`, and it is not used here: that rule scores 100 when every delta is 0.

`causal` is true only when all three hold:

- `delta >= 0.15`
- `p_value <= alpha`
- `control_percentile >= 95`

Null does not pass. `validated_feature_ids` lists the feature ids with `causal: true`. These thresholds live next to the scoreboard gates in `src/physint/thresholds.py` and are not the `real` rule: a rollout never rewrites `scoreboard.json`.
