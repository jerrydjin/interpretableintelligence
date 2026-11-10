# Hosting the failed-clip API

This is a proposal. Nothing here is deployed: no database, no bucket, no process left running.

The local tools stay the implementation. `physint explain` and `physint serve` call `physint.explain`. A hosted copy would serve the same three routes and the same citation rule: the answer may name only features whose scoreboard record has `real: true`, and only numbers with `effect_kind: "ablation"`. An unusable SAE is a refusal, not an empty guess. Every response still carries `run_id`, `model_sha`, and `layer`.

## Two costs

Looking up a run or a feature is a query. Explaining a new clip is not. Explanation captures the run's layer, encodes with the run's SAE, and ablates the top 10 features. That work is `explain_clip` on a CPU. The 100 ms budget below is for the lookup, not for the ablation forwards.

## DuckDB over parquet meets the budget

One zstd parquet file of 1,000,000 synthetic effect rows (`run_id`, `feature_id`, `clip_id`, `window`, `t`, `effect`, `effect_kind`), 8.6 MB, queried with DuckDB 2.0.0 on this machine:

| Query | Median | p95 | Max |
| --- | --- | --- | --- |
| Ablations for one run and clip, ordered by effect, limit 10 (5 rows hit) | 16.1 ms | 19.3 ms | 20.5 ms |
| Aggregate for one feature in one run | 7.7 ms | 9.3 ms | 9.4 ms |
| First query on a fresh connection | 19.9 ms |  |  |

All of those are under 100 ms, so the free tier should query parquet with DuckDB and skip Postgres. Weights and activation shards are binaries, not rows. They go in object storage (any S3-compatible free allotment). DuckDB reads the parquet from that bucket, or from a disk cache on the same machine.

## Ingest

A run folder is accepted only after `physint validate` returns no errors. The same check the local tool already has.

1. Validate the folder.
2. Put `sae/weights.safetensors`, `sae/config.json`, and `acts/shard-*.safetensors` in object storage at `runs/<run_id>/`.
3. Register the run's `features/*.parquet` plus `manifest.json`, `scoreboard.json`, and `features.jsonl` in a DuckDB catalog partitioned by `run_id`. Replacing a run replaces that partition. DuckDB has one writer, so ingest is a batch job. Explain workers only read.

`GET /runs` and `GET /features/{id}` read that catalog. They do not load the model.

## Explain worker

One small CPU process runs `physint serve`. On `POST /explain` it downloads the run's SAE into a local cache if it is not there, then calls `explain_clip`. The adapter weights are the ones whose sha is in the manifest. If `model_sha()` does not match, the call refuses, the same as it does locally.

There is no auth on this cut, same as the local API. Do not put the process on a public address until that exists.

## When to leave DuckDB

Move the feature and effect rows to Postgres if ingest needs many concurrent writers, or if a real query stops landing under 100 ms. Shards and SAE weights stay in object storage either way. `explain_clip` does not change.
