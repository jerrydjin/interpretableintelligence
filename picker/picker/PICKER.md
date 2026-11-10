# Candidate picker for robot-policy failure causes

Files: `pick_candidates.py` (method), `rollout_io.py` (the only file that knows run-folder formats), `picker/synth_rollouts.py`,
`picker/synth_head.py`, `picker/synth_test.py`, `picker/synth_results.json`.

## Run on real folders
```bash
# from the repository root; pick_candidates.py lives in picker/
cd picker
nice -n 19 env OMP_NUM_THREADS=2 uv run python -u pick_candidates.py '/workspace/runs/groot-*' '/workspace/runs/pi05-*'
#   optional: --downstream mymod:act   (real action head; fn(acts[B,n_tok,D] float32 torch, meta)->[B,A])
#             --match phase            (if most successes end before the failure times)
#             --fail-t-scale K         (if fail_t is in sim steps and K sim steps = 1 captured policy step)
#             --W 10 --heldout-frac 0.3 --fdr 0.1 --top 20 --n-random 10 --keep-pct 90
```
Output per run: `<run>/picker/candidates.json` (for Michael's batch), `contrast.jsonl` (every tested feature), `offline.jsonl`.

## Capture format the adapter expects (proposal for Jeff; V-JEPA schema v0 with clip -> episode)
```
runs/<id>/manifest.json   {run_id, model{...}, layer, d_model,
                           episodes:[{episode_idx, episode_id, task_id, seed, success, fail_t, n_steps}]}
                           (or episodes.jsonl / rollouts/episodes.parquet with the same columns; PR #5 baseline rows are accepted)
runs/<id>/acts/shard-NNNNN.safetensors   acts f16 [N,D]; episode_idx i32, step i32, tok i16   (one row per token per policy step)
runs/<id>/actions.safetensors            episode_idx i32 [S], step i32 [S], action f32 [S,A]  (the action the policy emitted)
runs/<id>/sae/weights.safetensors + config.json   (sae.py layout)
```
Important: `fail_t` and `step` must be in the same unit (captured policy steps). Successful episodes: `fail_t` null.
Column-name aliases (clip_idx, t, failure_step, ...) live in `rollout_io.ROW_ALIASES` / `EP_ALIASES`; anything else is a one-file fix there.
