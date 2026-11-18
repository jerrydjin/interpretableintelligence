# Hosting the failed-clip API

This is a proposal. Nothing here is deployed: no database, no bucket, no process left running.

The local tools stay the implementation. `interpretableintelligence explain` and `interpretableintelligence serve` call `interpretableintelligence.explain`. A hosted copy would serve the same three routes and the same citation rule: the answer may name only features whose scoreboard record has `real: true`, and only numbers with `effect_kind: "ablation"`. An unusable SAE is a refusal, not an empty guess. Every response still carries `run_id`, `model_sha`, and `layer`.

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

## Rust read server, same row count

`interpretableintelligence-server bench` writes 1,000,000 synthetic rows (counted back out of the parquet) and times the queries the server actually serves, plus the two scans above. DuckDB 1.20000.1, 4 threads, release build, this machine. Floats are hashed so the files do not collapse under dictionary encoding: `clip_effects.parquet` is 3.81 MB and `feature_acts.parquet` is 3.56 MB. The Python file above is 8.6 MB and includes a `run_id` column this run-folder schema does not. Different machine, so the comparison is the shape of the result, not a speedup claim.

| Query | p50 | p99 |
| --- | --- | --- |
| Frame lookup: effects at one clip, window, and t | 10.40 ms | 13.41 ms |
| Top clips for one feature | 10.31 ms | 12.70 ms |
| Ablations for one clip, ordered by effect, limit 10 | 6.90 ms | 13.35 ms |
| Aggregate for one feature (`COUNT`, `AVG`) | 3.29 ms | 3.82 ms |
| First frame lookup on a fresh connection | 18.92 ms |  |

The Python clip scan was 16.1 ms median / 19.3 ms p95, and the feature aggregate was 7.7 ms median / 9.3 ms p95. Both Rust scans land under those figures and under the 100 ms lookup budget. The frame lookup and top-clips queries are the ones a viewer issues; they were not in the Python table.

## Ingest

A run folder is accepted only after `interpretableintelligence validate` returns no errors. The same check the local tool already has.

1. Validate the folder.
2. Put `sae/weights.safetensors`, `sae/config.json`, and `acts/shard-*.safetensors` in object storage at `runs/<run_id>/`.
3. Register the run's `features/*.parquet` plus `manifest.json`, `scoreboard.json`, and `features.jsonl` in a DuckDB catalog partitioned by `run_id`. Replacing a run replaces that partition. DuckDB has one writer, so ingest is a batch job. Explain workers only read.

`GET /runs` and `GET /features/{id}` read that catalog. They do not load the model.

## Explain worker

One small CPU process runs `interpretableintelligence serve`. On `POST /explain` it downloads the run's SAE into a local cache if it is not there, then calls `explain_clip`. The adapter weights are the ones whose sha is in the manifest. If `model_sha()` does not match, the call refuses, the same as it does locally.

There is no auth on this cut, same as the local API. Do not put the process on a public address until that exists.

## When to leave DuckDB

Move the feature and effect rows to Postgres if ingest needs many concurrent writers, or if a real query stops landing under 100 ms. Shards and SAE weights stay in object storage either way. `explain_clip` does not change.

# Hosting the live playground's model

The live playground (`demo-site/live/`) is a static page plus a small model service. The page and its pack files stay on the static host (Vercel); the service runs the model and the SAE in PyTorch behind playground API v1 (`docs/PLAYGROUND-API.md`). The service touches model tensors, so it is Python (AGENTS.md, "Language"); routing is a thin layer (`playground/http.py`) that a Rust gateway could replace without touching the compute or the page.

## Deploy on Modal (your account)

Once, on your machine:

```bash
uv sync --extra remote
uv run modal setup          # signs in and stores a token in ~/.modal.toml
```

Then:

```bash
uv run ii deploy-live       # builds the image, deploys, writes demo-site/live/config.json
git add demo-site/live/config.json && git commit -m "Point the playground at the model server"
```

The Vercel deploy of that commit makes the page use the server. `ii deploy-live --api none` points it back at the in-browser engine.

## What it costs

- Nothing while idle: no container runs until a request arrives, and one stops `--idle` seconds (default 300) after the last request.
- A cold start is a few seconds: model weights are in the image and the loaded models are restored from a memory snapshot.
- While warm, Modal bills the GPU (T4 by default, about $0.59 an hour) plus 2 CPU cores and 4 GiB. `--gpu none` runs on CPU only, about a tenth of the price; DINOv2-small reads a picture in about 70 ms and edits in about 50 ms on 4 CPU cores, so CPU is enough for this pack. Bigger models want the GPU.
- `--max-containers` (default 2) caps how many run at once, which caps the bill if the page gets busy. There is no auth in v1, so this cap is the guard.
