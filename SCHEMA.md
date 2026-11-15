# Run folder contract

A finished `interpretableintelligence run` writes one directory. Later lanes (viewers, ingest, the failed-clip API) read this directory and nothing else from the training process. `interpretableintelligence validate <run>` checks the rules below.

## Versions

The contract has two versions. Version 1 reads version 0.

- **Version 0.** The tree below, and no `rollouts/` directory. `manifest.json` omits `schema_version` (an explicit `0` is the same thing). `interpretableintelligence run` writes this.
- **Version 1.** Version 0 plus `manifest.schema_version` equal to `1` and a `rollouts/` directory. Nothing from version 0 is removed or renamed. `interpretableintelligence rollouts` bumps an existing manifest to version 1 after it writes `rollouts/`.

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
  features/attribution.parquet
  features/candidates.json
  features/features.jsonl
  scoreboard.json
```

`run_id` is the directory name. Thresholds cited below live in `src/interpretableintelligence/thresholds.py` and are part of the contract.

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

Two optional manifest fields are not part of the contract readers must understand:

- `clips_dir`: absolute directory the capture read. The viewer uses it when `--clips` is omitted.
- `stages`: one object per finished stage (`capture`, `sae`, `features`, `score`, `view`). Each object has exactly `wall_s` (seconds), `peak_ram_bytes`, `device`, `rows`, and `tokens`. These are measurements from that stage, not estimates.

`acts/stream/` is a float32 row cache the pipeline uses to resume training and encoding without loading every shard. It is not part of the contract. Shards remain the activation record. The shared activation cache (`interpretableintelligence cache`) lives outside run folders and is not part of the contract either.

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

Decoder columns are unit-norm. `sae/config.json` records `d_model`, `n_features`, `l1_coeff`, `ortho_coeff`, `lr`, `steps`, `batch_size`, `seed`, `normalize_decoder`, and `activation` (`relu` or `topk`). A `topk` SAE also records `k`, the number of features kept per token; its `l1_coeff` and `ortho_coeff` are 0. Training uses train-split activations only. The orthogonality term keeps one planted direction from splitting into two live features.

## features/feature_acts.parquet

Sparse nonzero SAE activations. Columns, in order:

`feature_id` int32, `clip_id` string, `window` int16, `t` int16, `h` int16, `w` int16, `value` float32.

Every `value` is > 0. A feature that never fires has no rows. "Never fires" means max activation ≤ `1e-5` (`DEAD_ACT_EPS`) on the full capture (train and held-out).

## features/clip_effects.parquet

Columns, in order:

`feature_id` int32, `clip_id` string, `window` int16, `t` int16, `effect` float32, `effect_kind` string.

`effect_kind` is one of:

- `estimate`: activation × ∂(mean masked predictor MSE)/∂(feature activation), averaged over `h, w` in that frame. Straight-through: the forward value is the true layer activation, and the gradient flows through the decoder. One row for every feature, clip, window, and frame.
- `ablation`: the feature is actually zeroed by subtracting `f_i * decoder_i` from the layer. On a full run, rows exist for the top 10 features per clip (or every feature, if the dictionary is smaller), ranked by mean absolute estimate on that clip, and for every frame of those features. `interpretableintelligence run --prefilter K` instead ablates exactly those K features on every clip and frame. The stored effect is the mean cosine distance of the predictor output over the masked spatial cells in that frame. This is the original single-feature score. The scoreboard's `ablation_effect` is the mean of these distances over held-out masked tokens.
- `loss_video`: same single-feature zeroing, scored as the change in the model's own masked prediction loss (edited MSE minus clean MSE) over every window of the clip. The stored number is the mean per-token loss increase inside that frame. The video-level effect is the token-weighted mean of the frame rows.
- `loss_window`: the same loss change restricted to one window.
- `group_loss_video`: the co-active group (default size 3) is zeroed together, and the loss change is taken over every window. Each member feature gets a copy of the group's per-frame loss increase.
- `group_loss_window`: the group loss change restricted to one window.

`estimate` and `ablation` are always present. The loss kinds are written when a run is asked for them (`--extra-effects loss,group`) and are what the loss-based scorer compares to matched controls. A loss row is a positive effect when ablation makes predictions worse. Controls match the ablated set's size, each decoder norm, and each multiset of positive activations.

## features/attribution.parquet

Cheap prefilter scores, one row per feature and clip. Columns, in order:

`feature_id` int32, `clip_id` string, `score` float32.

The objective is recorded in the parquet schema metadata as `objective`. For `prediction_loss` and `action_loglik`, `score` is the sum over that clip's windows and tokens of `activation × ∂(objective)/∂(activation)`, from one forward and one backward per window. For `activation` there is no backward, and `score` is the sum of `|activation|`.

- `prediction_loss`: mean masked predictor MSE. This is what `interpretableintelligence run` writes, and it matches `effect_kind="estimate"`.
- `action_loglik`: mean log-likelihood of the action the policy took. Use this when success is not differentiable.
- `activation`: no backward. The score is the sum of `|activation|`, a magnitude baseline.

`interpretableintelligence rank --objective` selects which one. A scored run's metadata is `prediction_loss`. The file is a shortlist, not a confirmation.

## features/candidates.json

The ranking of those scores and the shortlist expensive ablations should run on.

```json
{
  "method": "attribution_patching",
  "objective": "prediction_loss",
  "shortlist_only": true,
  "prefilter_k": 10,
  "rank_by": "abs(mean over clips of the per-clip score); ties break toward the smaller feature id",
  "ranked": [
    {"rank": 1, "feature_id": 9, "label": "stripe", "score": -0.0004, "abs_score": 0.0004}
  ],
  "candidates": [9, 0, 3],
  "seconds": {"attribution": 0.01, "ablation": 0.02, "random_controls": 1.5}
}
```

`ranked` lists every feature once, ordered by descending `|mean score over clips|`. `candidates` is the first `prefilter_k` feature ids. `shortlist_only` is true: every candidate still has to be confirmed by a real ablation before it can count as causal. `seconds.ablation` and `seconds.random_controls` are null when `interpretableintelligence rank` did not ablate. A feature outside a `--prefilter` shortlist has `ablation_effect: null`, `random_percentile: null`, and `n_random: 0` in the scoreboard; `real` is then false.

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
    "loss_sae": 0.0,
    "loss_visible_mean": 0.0,
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
- `loss_sae`: that spliced mean squared error, on the same held-out clips. The reduction is the sum of squares over the feature axis, averaged over masked tokens.
- `loss_visible_mean`: mean squared error of predicting every masked token as the mean of the visible patches, on the same held-out clips and the same reduction. This is the floor a trained V-JEPA predictor has to beat.
- `usable`: `loss_recovered >= 0.70` and `loss_sae < loss_visible_mean`. The 0.70 threshold is unchanged. A splice that is not below the mean-of-visible-patches baseline is not usable, regardless of `loss_recovered`.
- `dead_frac`: share of features that never fire on the captured tokens.

### Feature block

`ablation_effect` is the mean cosine distance between the original predictor output and the output after the feature is zeroed, over held-out masked tokens. Masked rows come from `predict_masked` in row-major `(t, h, w)` order of the adapter mask. Cosine distance is `1 - cos`. A zero vector against a nonzero vector counts as distance 1; two zero vectors count as 0. It is `null` when `interpretableintelligence run --prefilter` did not ablate that feature. `random_percentile` is `null` in that case and `n_random` is 0. `null` is not a pass.

`random_percentile` compares that effect to `n_random` (≥ 100) random features matched on held-out firing count and on `(decoder L2 norm × mean firing activation)`. Each control uses a random unit direction scaled to the same decoder norm, the same multiset of activation values, and a random subset of held-out tokens of the same size. The percentile is the percentage of control effects that are `<=` the feature's effect (ties do not count as a win for the control).

`absent_in_random_init` reruns capture and SAE training with the same hyperparameters on `load(random_init=True)`. It is true when the feature's activation pattern has variance and no random-init feature has Pearson correlation ≥ `0.45` with it. A flat pattern is not absent: dead features share that pattern.

`label_holds_heldout` is the labeller's held-out check. For the fake labeller (`fake-cue-v1`), the label is the planted cue with the highest train-token Pearson correlation, and only if that correlation is > `0.50`. The label holds when the same cue correlates ≥ `0.60` on held-out tokens. Otherwise the field is false. `null` means the check was not run. The `hf` adapter uses `clip-concepts-v2:openai/clip-vit-base-patch16`. Each concept's per-token strength is the CLIP similarity between the patch (MaskCLIP dense embedding, resized to the token grid and averaged per tubelet) and the prompt `a photo of {concept}.`. Because real-model features are sparse, the statistic is ROC AUC rather than Pearson: how well the concept's strength separates tokens where the feature fires (> `1e-5`) from tokens where it does not. The label is the concept with the highest train AUC, and only if that AUC is ≥ `0.80`; it holds when the same concept's held-out AUC is ≥ `0.80`. The concept list is built in, or one per line from `ii run hf --concepts FILE`.

`real` is true only when `usable` is true and all four of the following hold:

- `distinct_clips_top20 >= 5`
- `label_holds_heldout is true`
- `random_percentile >= 95`
- `absent_in_random_init is true`

`null` is not a pass. A high percentile alone is not a pass: a dead feature can tie a null control distribution and still score 100.

## Adapter obligations the schema assumes

`predict_masked(clip, window, edit=None)` returns `(pred, target)` for the masked region only, row-major over the true cells of `mask(window)`. `hook(layer, mode)` is a context manager with modes `capture`, `edit`, and `zero`. Edits replace the instrumented layer before the predictor reads it. Splicing the SAE and zeroing a feature are edits; zeroing the whole layer uses mode `zero`.

## rollouts/ (version 1)

`interpretableintelligence rollouts` measures whether an intervention on SAE latents changes task success. It writes two files:

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
| `dictionary` | string, null for baseline | SAE whose latents are edited: `trained` (the policy's own) or `random:<seed>`, an untrained SAE of the same shape. Files written before this column read as `trained`. |

`ablate` sets the listed latents to 0. `scale` multiplies them by `value`. `clamp` sets them to the constant `value`.

The design is paired. One baseline arm and every intervention arm see the same seeds. Candidate features are the ids passed to `--features`, or, with `--pick N`, the top `N` of the policy's picker on the trained SAE (`candidate_source` is `features` or `pick`). Control arms are `K` other features (`--controls`), drawn from a seeded permutation of the remaining policy features, with the same layer, kind, and value. Baseline and candidates always run `N = episodes_per_arm` episodes. A control can stop early.

Early stopping is a one-sided sequential probability ratio test on the discordant pairs of each control against each candidate. A pair is discordant when the candidate-ablated and control-ablated outcomes differ. It counts as efficacy only when the control succeeds and the candidate fails. The other way pulls the likelihood ratio down and never stops the control. Concordant pairs are skipped. Looks happen on seed prefixes. The first prefix has length `min_episodes`; later prefixes grow by `batch_size`. There is no futility boundary: a control that never crosses runs to `episodes_per_arm`.

The alternative is `P(efficacy pair) = (1 + λ) / 2` among discordant pairs, with `λ = sprt_lambda` (0.5, so that probability is 0.75), against the null `1/2`. The run's `alpha` is family-wise across the controls and is split by Bonferroni, `alpha / n_controls`. Per-control beta solves `(1 - beta) ^ n_controls = 0.8`, which is about 0.022 for 10 controls, so the probability of stopping every control under the design alternative is about 0.8. The boundary is `log((1 - beta) / (alpha / n_controls))`. A control is not scheduled past a prefix once that path-maximum log likelihood ratio has reached the boundary against every candidate, and not before `min_episodes`. A crossing only on the final prefix does not stop early: `reason` stays `fixed_budget`. Otherwise `reason` is `sprt_separated`.

`summary.json` records, per arm, `episodes_run`, `episodes_budget`, `stopped_early`, and `reason`. Every arm of the control family (feature controls and random-SAE arms) also stores `gap_vs_candidate` (its success rate minus the success rate of the candidate with the smallest path-maximum log likelihood ratio), `sprt_log_lr` (that one-sided log likelihood ratio), and `sprt_threshold` (the efficacy boundary above). Baseline and candidates leave those three null. The summary's `gap_bound` is `sprt_discordant` and `sprt_lambda` is `λ`.

`summary.json` is derived from the parquet and the schedule. `total_episodes` is the number of rows. `fixed_budget_episodes` is `(1 + n_candidates + n_controls + n_random_sae_arms) * N`, the cost if nothing stopped early. `episodes_saved` is the difference. `rollouts_per_validated_feature` is `total_episodes / n_validated`, or null when nothing was flagged.

For each candidate and each control the summary stores the paired success rates on the seeds that arm actually ran, `delta = baseline_success - intervention_success`, a Wald interval on the paired differences (`ci_low`, `ci_high`), and the two-sided exact McNemar `p_value`. `control_percentile` is the average-rank percentile of this delta among the comparison deltas: `100 * (n_strictly_below + 0.5 * n_tied) / n`. A delta tied with every comparison scores 50. A delta strictly above every comparison scores 100. A delta strictly below every comparison scores 0. Candidates are compared with the controls. A control is compared with the other controls. Values within `1e-9` count as tied. No comparisons means the percentile is null. Counting every tie as `<=` is the scoreboard's `percentile_le`, and it is not used here: that rule scores 100 when every delta is 0.

`causal` is true only when all three hold:

- `delta >= 0.15`
- `p_value <= alpha`
- `control_percentile >= 95`

Null does not pass. `validated_feature_ids` lists the trained-SAE feature ids (candidates and controls) with `causal: true`. These thresholds live next to the scoreboard gates in `src/interpretableintelligence/thresholds.py` and are not the `real` rule: a rollout never rewrites `scoreboard.json`.

### Random-weights SAE control

`causal` is not a finding. AGENTS.md research rule 4 also asks a candidate to beat what an untrained dictionary produces through the same picker. `--random-sae K` (it needs `--pick`; with `--pick` it defaults to 5, and `--random-sae 0` turns it off) draws `K` untrained SAEs, `random:<seed>` for the seeds `random_sae_seeds` (a seeded `SeedSequence`, so a larger `K` extends the same list). Each is `sae.untrained_sae`: the SAE that training starts from, with the trained SAE's shape and no optimiser step. The picker that chose the candidates picks as many features from each one (`random_sae_feature_ids`, one list per seed), and each pick becomes an arm with role `random_sae`, the same layer, kind and value, and its `dictionary`. An edit of a random dictionary keeps the SAE's error: `x' = x + W_dec (z' - z)`, so an edit that leaves the latents alone returns `x`. Random-SAE arms join the feature controls in the early-stopping family: `n_controls` in the SPRT and its Bonferroni split counts both.

`picker` is `action_divergence` when the policy has no picker of its own. That default screens every feature on `pick_discovery_episodes` episodes whose seeds start 1,000,000 above `seed`, so no arm reuses them, and ranks features by the mean of `sum_t ||a_edit(t) - a_clean(t)||^2` over each closed-loop episode; ties go to the lower id. `picker` is `policy` when the policy brings `pick_features`, and `pick_discovery_episodes` is then null. Both are null for `--features`.

Per feature, `random_sae_percentile` is the candidate's delta among the random-SAE arms' deltas (the same average-rank percentile), `n_random_sae` counts them, and `beats_random_sae` is true only at 100: strictly above every random-SAE arm, so a tie does not beat. All three are null (`n_random_sae` is 0) for controls, random-SAE arms, and runs without `--random-sae`. A random-SAE arm's `control_percentile` and `causal` compare it with the feature controls, which shows whether an untrained dictionary would pass the three conditions above. `finding` is true only for a candidate that is `causal` and `beats_random_sae`. `finding_feature_ids` lists them. Without `--random-sae` nothing is a finding.

Rollout summaries written before these keys fail `validate`. Rerun `interpretableintelligence rollouts` with the same arguments to refresh them: stored episodes are reused.

## steering/ (optional)

`interpretableintelligence steer-scorecard` writes an offline steering scorecard. It works on a version-0 or version-1 folder and does not change `schema_version`. It never rewrites `scoreboard.json`, and a feature's `real` does not depend on it. `validate` checks it only when `steering/` exists.

```
steering/effects.parquet
steering/scorecard.json
```

**Features.** Without `--features`, the scoreboard features with `real: true`, ordered by descending `ablation_effect` (null last, ties toward the smaller id), first `--top`. `selection` is `verified`. With no real feature the command stops. With `--features`, those ids in the given order, and `selection` is `requested`.

**Edits.** On every window of every held-out clip (`split.json`), for each feature that fires on that window (some token above `DEAD_ACT_EPS`): `remove` scales the feature by 0 and `boost` scales it by `boost_factor`. The edit adds `(factor - 1) * f_i * W_dec[:, i]` on each token and keeps the SAE's reconstruction error. Each edit has `n_controls` matched controls. Control `k` of feature `i` is one random direction, rescaled to the norm of `W_dec[:, i]`, used on every window. On a window it adds the same per-token coefficients `(factor - 1) * f_i`, permuted across that window's tokens. `n_controls` is at least 19, so a p-value can reach 0.05.

**effects.parquet.** One row per (feature, kind, arm, clip, window) where the feature fires. Columns, in order:

| column | type | meaning |
| --- | --- | --- |
| `feature_id` | int32 | SAE feature. |
| `kind` | string | `remove` or `boost`. |
| `arm` | string | `feature` or `control`. |
| `control` | int16 | Control index, `-1` for the feature arm. |
| `clip_id` | string | Held-out clip. |
| `window` | int16 | Window index. |
| `rel_change` | float32 | `‖edited − clean‖ / ‖clean‖` over the masked output (for GR00T, the planned action chunk). |
| `abs_change` | float32 | Mean per-row L2 change (per action step for GR00T, in the policy's normalised units). |

**scorecard.json.** Top-level keys: `schema_version` (1), `run_id`, `model_sha`, `layer`, `split_sha`, `adapter` (all matching the manifest), `target` (`action_chunk` for `groot:` adapters, else `masked_prediction`), `metric` (`relative_l2_change`), `selection`, `feature_ids`, `boost_factor`, `n_controls`, `control`, `seed`, `bootstrap`, `confidence` (0.95), `alpha` (0.05), `sae_usable`, `n_episodes` and `n_windows` (held-out clips and windows scored), `rows`, `rollout_finding` (always false), and `note`.

`rows` holds one object per feature and kind, features in `feature_ids` order, `remove` before `boost`. Over the rows of that feature and kind:

- `n_windows`, `n_episodes`: windows and distinct clips where the feature fired. `coverage` is `n_windows / n_windows(top level)`.
- `action_change`: mean `rel_change` of the feature arm. `control_change`: mean over windows of the per-window mean over controls. `excess`: their difference. `ratio`: `action_change / control_change`, null when the controls do not move the output. Each is `{mean, ci_low, ci_high}`.
- `abs_change`: mean `abs_change` of the feature arm.
- `reliability`: `{rate, ci_low, ci_high, chance}`. `rate` is the share of windows where the feature arm's `rel_change` is above every control's by more than `1e-9`. `chance` is `1 / (n_controls + 1)`.
- `p_value`: `(1 + #{control arms whose mean ≥ action_change}) / (n_controls + 1)`. Ties count against the feature. `control_percentile` is the scoreboard's `percentile_le` of `action_change` among the control arms' means.
- `beats_controls`: `p_value <= alpha` and `excess.ci_low > 0`. A null bound does not pass.

Intervals are percentile intervals from `bootstrap` resamples of the episodes (clips) with replacement, seeded by `(seed, feature_id, kind)`. They are null with fewer than two episodes. A feature that never fires on a held-out window has `n_windows: 0`, null statistics, and `beats_controls: false`. `validate` recomputes `scorecard.json` from `effects.parquet` and the top-level settings and requires the stored file to match.
