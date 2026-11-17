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
- Optional `sae_source`: present only on a run made with `ii run --sae-from` (see "Reused SAEs").
- `grid` is the activation lattice: `t` time bins, `h`×`w` patches, each bin pooling `tubelet` frames and `patch_px` pixels.

Two optional manifest fields are not part of the contract readers must understand:

- `clips_dir`: absolute directory the capture read. The viewer uses it when `--clips` is omitted.
- `stages`: one object per finished stage (`capture`, `sae`, `features`, `score`, `view`). Each object has exactly `wall_s` (seconds), `peak_ram_bytes`, `device`, `rows`, and `tokens`. These are measurements from that stage, not estimates.

`remote.json` is written by `ii run --remote` next to a run it mirrored: where the run executed, the price per hour it was estimated at, the `--max-cost` cap, timings, status, and the estimated cost. No reader depends on it, and it is not part of the contract (`docs/REMOTE.md`).

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

Decoder columns are unit-norm. `sae/config.json` records `d_model`, `n_features`, `l1_coeff`, `ortho_coeff`, `lr`, `steps`, `batch_size`, `seed`, `normalize_decoder`, and `activation` (`relu`, `topk`, or `jumprelu`). A `topk` SAE also records `k`, the number of features kept per token; its `l1_coeff` and `ortho_coeff` are 0. A `jumprelu` SAE records `threshold` instead: `f = r * (r > threshold)` with `r = relu(W_enc @ (x - b_dec) + b_enc)`. That is how a BatchTopK SAE runs at inference, and `ii import` writes it (see "Imported runs"). Training uses train-split activations only. The orthogonality term keeps one planted direction from splitting into two live features.

### Reused SAEs

`ii run --sae-from DIR` copies a frozen SAE into `sae/` instead of training one, so feature ids found elsewhere keep their meaning on new clips. `DIR` is an `ii` run folder, a GR00T v9 capture (or `ii retrain-sae` output), or their `sae/` directory. A v9 BatchTopK SAE is folded exactly as `ii import` folds it ("Imported runs", `groot-v9`). The manifest records where it came from:

```json
"sae_source": {"path": "/abs/groot-1028-v9-n100/sae", "format": "groot-v9", "config_sha256": "<64 hex>",
               "weights_sha256": "<64 hex>", "n_features": 8192, "notes": {"norm_scale": 3.1, "fold": "..."}}
```

`format` is `ii` or `groot-v9`; the two sha256s are of the source's `config.json` and `weights.safetensors` (the folded weights in this run differ for `groot-v9`). The SAE was not trained on this run's train split, so such a run is never scored: it has no `scoreboard.json`, and `ii steer-scorecard --features` is how its features are measured. Training the run's own SAE again removes `sae_source`.

## features/feature_acts.parquet

Sparse nonzero SAE activations. Columns, in order:

`feature_id` int32, `clip_id` string, `window` int16, `t` int16, `h` int16, `w` int16, `value` float32.

Every `value` is > 0. A feature that never fires has no rows. "Never fires" means max activation ≤ `1e-5` (`DEAD_ACT_EPS`) on the full capture (train and held-out).

## features/clip_effects.parquet

Columns, in order:

`feature_id` int32, `clip_id` string, `window` int16, `t` int16, `effect` float32, `effect_kind` string.

`effect_kind` is one of:

- `estimate`: activation × ∂(mean masked predictor MSE)/∂(feature activation), averaged over `h, w` in that frame. Straight-through: the forward value is the true layer activation, and the gradient flows through the decoder. One row for every feature, clip, window, and frame.
- `ablation`: the feature is actually zeroed by subtracting `f_i * decoder_i` from the layer. On a full run, rows exist for the top 10 features per clip (or every feature, if the dictionary is smaller), ranked by mean absolute estimate on that clip, and for every frame of those features. `interpretableintelligence run --prefilter K` instead ablates the shortlisted features that can still be verified (see `candidates.json` below), on every clip and frame. The stored effect is the mean cosine distance of the predictor output over the masked spatial cells in that frame. This is the original single-feature score. The scoreboard's `ablation_effect` is the mean of these distances over held-out masked tokens.
- `loss_video`: same single-feature zeroing, scored as the change in the model's own masked prediction loss (edited MSE minus clean MSE) over every window of the clip. The stored number is the mean per-token loss increase inside that frame. The video-level effect is the token-weighted mean of the frame rows.
- `loss_window`: the same loss change restricted to one window.
- `group_loss_video`: the co-active group (default size 3) is zeroed together, and the loss change is taken over every window. Each member feature gets a copy of the group's per-frame loss increase.
- `group_loss_window`: the group loss change restricted to one window.

`estimate` and `ablation` are always present. The loss kinds are written when a run is asked for them (`--extra-effects loss,group`) and are what the loss-based scorer compares to matched controls. A loss row is a positive effect when ablation makes predictions worse. Controls match the ablated set's size, each decoder norm, and each multiset of positive activations.

## features/attribution.parquet

Cheap prefilter scores, one row per feature and clip. Columns, in order:

`feature_id` int32, `clip_id` string, `score` float32.

The objective is recorded in the parquet schema metadata as `objective`. For `prediction_loss` and `action_loglik`, `score` is the sum over that clip's windows and tokens of `activation × ∂(objective)/∂(activation)`, from one forward and one backward per window. For `activation` there is no backward, and `score` is the sum of `|activation|`.

- `prediction_loss`: mean masked predictor MSE. This is what `interpretableintelligence run` writes for `fake` and `vjepa2`, and it matches `effect_kind="estimate"`.
- `output_sensitivity`: what `interpretableintelligence run` writes for `hf:<model>` and `groot:<checkpoint>`, whose `predict_masked` target is the model's own clean output. There `prediction_loss` and its gradient are zero at the clean point, so activation × gradient is float noise. Instead, per window, eight Gaussian probes drawn per output token orthogonal to the clean token and divided by its norm are pulled back through the network in one batched backward. For probe `r`, `s[r, i] = sum over tokens of activation_i × (Jᵀu_r)_i` estimates how far ablating feature `i` turns the output, and the window's score is `sqrt(mean_r s[r, i]²)`, which to second order tracks the mean cosine shift the scoreboard measures. Scores are non-negative. Probes are seeded from the clip id and window, so reruns match. `effect_kind="estimate"` rows hold each token's share of that score (the shares sum to it).
- `action_loglik`: mean log-likelihood of the action the policy took. Use this when success is not differentiable.
- `activation`: no backward. The score is the sum of `|activation|`, a magnitude baseline.

`interpretableintelligence rank --objective` selects which one. A scored run's metadata is `prediction_loss` or `output_sensitivity` (the adapter decides), and `candidates.json` records the same objective. The file is a shortlist, not a confirmation.

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

The ablation and its random controls are the expensive checks, so a shortlist does not spend them on a feature that can no longer be `real`. A shortlisted feature is ablated only when every check that needs no ablation already passes: the SAE is usable, `distinct_clips_top20 >= 5`, `absent_in_random_init` is true and `label_holds_heldout` is true (with a vision-model key, the shortlist is relabelled before this is decided). The others are recorded like a feature outside the shortlist. No threshold changes and the set of `real` features is the same as if every shortlisted feature had been ablated. When no shortlisted feature can be verified, all of them are ablated, so the run still has causal numbers to explore. Without a shortlist every feature is ablated, as before.

`control/random_controls.json` keeps each ablated feature's control effects, keyed by everything they depend on (model, layer, clips, split, SAE weights, seed, device and thread count). A later scoring pass over the same SAE (a new concept list or label model, a different `--prefilter`, or a run that was interrupted) reads them back instead of running the controls again, and gets the same percentile. Like `acts/stream/`, it is a resume artefact and not part of the contract.

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

`ablation_effect` is the mean cosine distance between the original predictor output and the output after the feature is zeroed, over held-out masked tokens. Masked rows come from `predict_masked` in row-major `(t, h, w)` order of the adapter mask. Cosine distance is `1 - cos`. A zero vector against a nonzero vector counts as distance 1; two zero vectors count as 0. It is `null` when `interpretableintelligence run --prefilter` did not ablate that feature (outside the shortlist, or already failing a check that needs no ablation). `random_percentile` is `null` in that case and `n_random` is 0. `null` is not a pass.

`random_percentile` compares that effect to `n_random` (≥ 100) random features matched on held-out firing count and on `(decoder L2 norm × mean firing activation)`. Each control uses a random unit direction scaled to the same decoder norm, the same multiset of activation values, and a random subset of held-out tokens of the same size. The percentile is the percentage of control effects that are `<=` the feature's effect (ties do not count as a win for the control). Control effects are means of non-negative cosine distances, so a control's running mean only grows as held-out windows are added. Scoring stops a feature's controls once more than 5% of them already exceed its effect, because the gate can no longer pass. A failing feature then records the percentile it would have if no further control overtook it, which is an upper bound on the full value. A feature that can still pass always runs every control, so a passing percentile is always exact.

`absent_in_random_init` reruns capture and SAE training with the same hyperparameters on `load(random_init=True)`. It is true when the feature's activation pattern has variance and no random-init feature has Pearson correlation ≥ `0.45` with it. A flat pattern is not absent: dead features share that pattern.

`label_holds_heldout` is the labeller's held-out check. For the fake labeller (`fake-cue-v1`), the label is the planted cue with the highest train-token Pearson correlation, and only if that correlation is > `0.50`. The label holds when the same cue correlates ≥ `0.60` on held-out tokens. Otherwise the field is false. `null` means the check was not run. The `hf` adapter uses `clip-concepts-v2:openai/clip-vit-base-patch16`. Each concept's per-token strength is the CLIP similarity between the patch (MaskCLIP dense embedding, resized to the token grid and averaged per tubelet) and the prompt `a photo of {concept}.`. Because real-model features are sparse, the statistic is ROC AUC rather than Pearson: how well the concept's strength separates tokens where the feature fires (> `1e-5`) from tokens where it does not. The label is the concept with the highest train AUC, and only if that AUC is ≥ `0.80`; it holds when the same concept's held-out AUC is ≥ `0.80`. The concept list is built in, or one per line from `ii run hf --concepts FILE`. `vjepa2` (not `--tiny`) uses the same dense labeller on the frames as the model sees them (its centre crop). `groot` (not `groot:tiny`) has no image tokens, so its concept strength is window-level: the CLIP similarity between the window's whole observation frame and the prompt, given to every token of that window; the AUC rule and thresholds are the same. `--concepts FILE` works for all three.

With a vision-model key set (README), the `hf` features that `ii run` scores (the prefilter shortlist, or the 32 with the highest attribution) are relabelled by `autointerp-v2:<provider>/<model>`. The label comes from a vision-language model shown up to 8 train frames (at most 2 per clip) where the feature fires most, with the strongest cell boxed and the activation map tinted. One reply is one sample and providers are not deterministic, so the model is asked 3 times (the same frames, in a different order) and the 3 candidate labels are compared on train clips only: each is rated on boxed cells from train frames the labels were not written from, and the one with the highest train ROC AUC becomes the label (`candidates` in `features/autointerp.json` records all 3). Only that label meets the held-out clips, once, so choosing between candidates cannot make the held-out check easier. Every reply is kept in `control/autointerp/` (a resume artefact, like `control/random_controls.json`), so scoring the same run again gives the same labels on any machine; a fresh run asks afresh, and a label near the bar can still land on either side of it. The held-out check shows the same model held-out frames with one cell boxed, given only the label: up to 20 cells where the feature fires (half its strongest held-out cells, half random) and as many where it does not (up to half in the same frames). The label holds when the model's 0 to 10 ratings separate the two with ROC AUC ≥ `0.80`. Fewer than 4 held-out firings, or fewer than 3 training frames, and it does not hold. Only spatial grids are relabelled.

### features/autointerp.json (optional)

Written by `ii run hf` when the vision labeller ran: `version`, `model`, `hold_min_auroc`, and per labelled feature its `id`, `label`, `description`, `auroc_heldout` (null when untested), `holds`, the `[clip_index, window, t]` frames it was explained from, and every simulated cell with whether the feature fires there and the score it got. It is evidence for the label, not part of the gates; the gate reads `label_holds_heldout`.

`real` is true only when `usable` is true and all four of the following hold:

- `distinct_clips_top20 >= 5`
- `label_holds_heldout is true`
- `random_percentile >= 95`
- `absent_in_random_init is true`

`null` is not a pass. A high percentile alone is not a pass: a dead feature can tie a null control distribution and still score 100.

## Imported runs

`interpretableintelligence import --format <format> <source> <out>` writes a run folder from rollout logs captured outside `ii`. `<source>` is a directory or a `.tar`, `.tar.xz` or `.tar.gz` of one. The first format is `groot-v9`, the GR00T v9 rollcap-v0 layout (the `data-groot-1028-v1` release). An imported run is the version-0 tree above plus `manifest.import`. Everything that does not need the model is computed by the same code as `interpretableintelligence run`: activation shards, `feature_acts.parquet`, top examples, `ev_heldout`, `dead_frac`, `distinct_clips_top20`, the attribution files, and `real` through the four gates. What does need the model in the loop is `null`, with a reason, and never `0.0`. A validator treats a folder as imported exactly when `manifest.import` is present; without it, every rule above applies unchanged.

### manifest.import

```json
{
  "format": "groot-v9",
  "cells": {"t": [0], "h": [0], "w": [0, 1, 3, 4]},
  "clips": [{"clip_id": "ep-00000", "source_id": 0, "n_windows": 23, "success": true, "fail_kind": null}],
  "nulls": {"scoreboard.sae.loss_recovered": "needs the GR00T action head ..."},
  "source": {"layout": "rollcap-v0", "files": {"split.json": "<64 hex>"}, "archive": {"name": "...", "sha256": "<64 hex>", "bytes": 0}},
  "notes": {},
  "wall_s": 61.2
}
```

- `cells` are the grid cells the source captured, as the product of the three sorted lists. Rows cover exactly these cells in row-major `(t, h, w)` order. A capture that kept every token lists the whole grid.
- `clips` has one entry per `manifest.clips` entry, in the same order. `source_id` is the id the source's `split.json` uses for that clip. `n_windows` is that clip's own window count: an imported clip covers windows `0 .. n_windows - 1`, and clips may differ. Other keys are per-clip outcome fields kept from the source (for `groot-v9`: `seed`, `init_state_id`, `success`, `fail_kind`, `fail_t`, `fail_subkind`, `fail_rule`).
- `nulls` maps each field the import left null to a non-empty reason. Every null listed below needs a reason, and a reason for a field that is not null is an error.
- `source` is provenance: the source layout, sha256s of its key files, and the archive's name, sha256 and size when one was imported. `notes` are format-specific (the hook, the token layout, the SAE fold, where `model.sha` came from). `wall_s` is the import's wall time.

### What changes for an imported run

- **split.json** is the source's file, copied byte for byte. It is not re-serialised, and it does not follow the clip-level algorithm. It must be a JSON object with `train` and `heldout` lists of source ids (integers or strings), disjoint, together naming every `manifest.import.clips[].source_id`; other keys are the source's. `manifest.split.sha` is the sha256 of the file's bytes. `manifest.split.seed` and `heldout_frac` are `null` (with reasons) unless the source records them, and `scoreboard.run.seed` equals `manifest.split.seed`. `clips[].split` still matches the file. Held-out labels are never edited.
- **clips[].sha256** is the sha256 of the source file the clip's rows came from (for `groot-v9`, the episode's `acts/shard-NNNNN.safetensors`), because the source has no clip pixels.
- **Activation shards** hold each clip's own windows and the captured cells only, in manifest clip order, then window, then cell. Values are the source's, cast to float16. Nothing is padded or interpolated.
- **Nullable fields.** `scoreboard.sae.loss_recovered`, `loss_sae`, and `loss_visible_mean` are null when the source cannot be spliced back into the model, and then `usable` is null too. `ev_heldout` may be null only if the source has no held-out rows (it is computed whenever it can be). `label_holds_heldout`, `ablation_effect`, `random_percentile`, and `absent_in_random_init` are null when no labeller, ablation or random-init capture ran; `n_random` is then 0. Each has a reason in `manifest.import.nulls`, under the key `scoreboard.sae.<field>` or `scoreboard.features.<field>`. `null` is still not a pass: with `usable` null, every feature is `real: false`.
- **features/clip_effects.parquet** has the usual columns and zero rows when no effect could be measured; its reason is `nulls["features/clip_effects.parquet"]`. If an imported run does carry effect rows, they follow the version-0 rules.
- **Attribution.** `attribution.parquet` may record objective `activation` (sum of |activation|, no backward), and `candidates.json` records the same objective. It is a shortlist; nothing in it was ablated.

### groot-v9

The mapping follows `interpretableintelligence run groot`. The adapter is `groot:<checkpoint>` and the layer is the hooked DiT block. Each clip is one episode, and each window is one policy call (`window = step // action_queue_steps`). The grid is `t=1, h=1, w=1+action_horizon` with `tubelet = action_queue_steps`, and `w` is the DiT token index (0 is the state token, `k` is action step `k - 1`). `cells.w` is the capture's `token_indices`. `fps` is env steps per second (LIBERO: 20, `--fps`), and `n_frames` is the env steps the episode executed.

The source SAE is BatchTopK, run with a learned JumpReLU threshold on `(x - norm_mu) / norm_scale`. The normalisation is folded into the stored weights (`W_enc' = W_enc^T`, `b_enc' = norm_scale * b_enc`, `b_dec' = norm_mu + norm_scale * b_dec`, `W_dec' = W_dec^T`), and `threshold' = norm_scale * threshold`. Feature ids and active sets equal the source's, and values are `norm_scale` times the source's, in raw activation units.

Failure labels are the pot-0 detector's (`fail_rule` starting `pot0-v1`, from `tools/derive_pot0_fail.py`). The importer refuses a capture without them and never reads the v9 active-object labels (`fail_kind_v9_active`, `fail_t_v9_active`). `model.sha` is `--model-sha` when given. Otherwise it is the sha `run groot` computes from the Hub's sha256 of the checkpoint weights, and if the Hub cannot be reached, a sha256 of the source's declared model and hook record. `notes.model_sha_source` says which.

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

`direction` is `fall` or `rise`. With `fall` (the `fake` policies) an edit counts when it lowers success, and `delta = baseline_success - intervention_success`. With `rise` (GR00T, whose candidates fire before failure, so removing them should help) `delta = intervention_success - baseline_success`. The interval, every gate above and the early-stopping test read that oriented outcome (for `rise` the efficacy pair is control fails, candidate succeeds, and `gap_vs_candidate` is a gap in failure rate), so the thresholds are the same both ways. `baseline_success`, `intervention_success` and `success_rate` are always success rates.

`picker` is `action_divergence` when the policy has no picker of its own. That default screens every feature on `pick_discovery_episodes` episodes whose seeds start 1,000,000 above `seed`, so no arm reuses them, and ranks features by the mean of `sum_t ||a_edit(t) - a_clean(t)||^2` over each closed-loop episode; ties go to the lower id. `picker` is `policy` when the policy brings `pick_features`, and `pick_discovery_episodes` is then null. Both are null for `--features`.

**GR00T.** `ii rollouts --policy groot-1028 --pick N --capture CAPTURE` (`groot-mock` replays a no-effect outcome in CI) records `picker: groot_failure_contrast`: `picker/pick_candidates.py` on the capture run (failure contrast, then offline prune; `--W 48`), once with the trained SAE and once per untrained SAE, same arguments and seed. Each run's picks are the first `N` of its ranked list: the picker's gated `candidate_ids`, then every other tested feature in `contrast.jsonl` order, so an untrained SAE always gives `N` arms. An untrained GR00T SAE (`adapters/groot/random_sae.py`) is the trainer's init drawn with its seed (unit-norm `randn(D, M)` columns as encoder, transposed as decoder, zero biases) with the trained SAE's `norm_mu` and `norm_scale`, and a threshold set so it keeps `k × n` activations over the capture's `n` train-split tokens (mean L0 `k`, as the trained SAE). Controls are the picker's firing-rate matched features of the trained SAE. `rollouts/ablate_plan.json` holds every condition with its `dictionary`, `rollouts/sae/` one subset file per dictionary (sha256 in the plan), and `rollouts/pick/` each picker run. The plan is a fixed budget: no arm stops early (`min_episodes = batch_size = episodes_per_arm`), and `direction` is `rise`.

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

## probe/ (optional)

`interpretableintelligence probe` writes an offline failure probe. It works on a version-0 or version-1 folder, does not change `schema_version`, never reads or writes `scoreboard.json`'s gates, never edits `split.json`, and never sets `real`. `validate` checks it only when `probe/` exists.

```
probe/scores.parquet
probe/probe.json
```

**Labels.** `--labels FILE` (default: `failures.json` in the manifest's `clips_dir`) is JSON with `episodes`, one per manifest clip: `{clip_id, failure, onset_s}`. A failed episode has `onset_s` in seconds from the clip start; a successful one has `onset_s: null`. Optional `source` names the labeller. Every clip needs exactly one label. `ii generate-clips --failure-labels planted|null` writes this file for fake clips.

**Inputs.** Three, from the same activation shards: `trained_sae` (the run's SAE), `random_sae`, and `raw` (the layer activations). `random_sae` is `sae.random_weights_sae`: the trained SAE's architecture, `n_features`, `activation` and `k`, with the trainer's initial weights (unit-norm Gaussian decoder columns, tied encoder, zero encoder bias, `b_dec` at the train-row mean) and zero optimiser steps. It is deterministic in its seed (`--random-seed`, default the SAE config's `seed`, which makes it the trained SAE's own starting point).

**Probe.** Bin `t` of window `k` ends at `(k * grid.t + t + 1) * grid.tubelet / fps` seconds (windows tile the clip back to back, as on `fake`). An episode's reference time is `onset_s` for a failure and `n_frames / fps` for a success; the probe sees only bins that end at or before `reference - lead_s`. An episode with no such bin is not scored (`labels.n_unscored`). Each input is max-pooled over the seen tokens, standardised with the train split's mean and standard deviation, and fit with an L2 logistic regression (`--l2`, bias unpenalised) on `split.json`'s train clips only. Held-out clips are scored once.

**scores.parquet.** One row per (input, scored episode), train and held-out. Columns, in order:

| column | type | meaning |
| --- | --- | --- |
| `input` | string | `trained_sae`, `random_sae`, or `raw`. |
| `clip_id` | string | Episode. |
| `split` | string | `train` or `heldout`, as in `split.json`. |
| `failure` | bool | Label. |
| `ref_s` | float64 | Onset (failure) or clip end (success). |
| `cutoff_s` | float64 | `ref_s - lead_s`. |
| `lead_s` | float64 | `ref_s` minus the end of the last bin the probe saw. |
| `score` | float64 | Probe logit. |

**probe.json.** Top-level keys: `schema_version` (1), `run_id`, `model_sha`, `layer`, `split_sha`, `adapter` (all matching the manifest), `target` (`failure`), `labels` (`source`, `sha256` of the file, `n_failures`, `n_episodes`, `n_unscored`), `pool` (`max`), `probe`, `lead_s`, `min_lead_s`, `bootstrap`, `confidence` (0.95), `alpha` (0.05), `seed`, `n_episodes` (held-out scored, failed, successful), `inputs`, `comparisons`, `sae_gate`, `positive`, `reason`, `real` (always false), `rollout_finding` (always false), and `note`.

- `inputs.<name>`: `n_dims`, held-out `auroc` (failures scored above successes, ties half), its percentile interval `ci_low`/`ci_high`, and `lead_time_s` (the smallest `lead_s` among held-out failures). `trained_sae` adds `sae_usable` and `loss_recovered`, copied from `scoreboard.json` when it has them and null otherwise. `random_sae` adds `seed`, `init`, `weights_sha256`, and `sae_usable`, `loss_recovered`, `clip_effects` (always null: the control is never spliced, scored or ablated, and a missing measurement is null, never 0.0) and `real` (always false).
- `comparisons.random_sae` and `comparisons.raw`: `delta` (trained minus baseline AUROC), its interval, `p_value`, and `beats`. The bootstrap resamples held-out episodes with replacement within each class, `bootstrap` times, seeded by `seed`; every input is scored on the same resample. `p_value` is `(1 + #{resamples with delta <= 0}) / (bootstrap + 1)`. `beats` is `delta > 0` and `p_value < alpha`.
- `sae_gate` is `not_run` when the gate is skipped: no probe could be fit (the train split lacks a class), fewer than `PROBE_MIN_CLASS` (2) held-out episodes of either class, `lead_s` below `min_lead_s` (1 s), or a held-out lead time below it. Otherwise it is `pass` when both comparisons beat, else `fail`. `positive` is `sae_gate == "pass"`. Null never passes.

`validate` recomputes `probe.json` from `scores.parquet` and the stored settings and requires the stored file to match.
