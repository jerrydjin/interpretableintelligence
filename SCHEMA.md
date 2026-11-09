# Run folder contract

A finished `physint run` writes one directory. Later lanes (viewers, ingest, the failed-clip API) read this directory and nothing else from the training process. `physint validate <run>` checks the rules below.

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
