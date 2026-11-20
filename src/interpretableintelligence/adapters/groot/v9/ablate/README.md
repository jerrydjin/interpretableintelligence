# groot-1028-v9-ablate: SAE feature-ablation rise test, task 1028 (unvalidated, real=false)

Plan: `ablate_plan.json` (written by `v9/make_ablate_plan.py`). Runner: `groot-colab/v9/ablate_v9.py`. SAE: `sae_subset.safetensors` = the 30 latents used here, exact
columns/rows of `groot-1028-v9-n100/sae/weights.safetensors` (norm_mu / norm_scale checked against `sae/norm_stats.safetensors`).
Inputs read only: `groot-1028-v9-n100/candidates_final.json` (matched controls), `picker_onset_pot0/result.json` (picker 146ee8a).

## Conditions (24), all on seeds 40000+i, init_state i % 50, v9 per-episode noise, per-pot labels (pot0-v1)
| folder | condition | latents zeroed | n |
|---|---|---|---|
| `baseline/` | baseline | none (hook not editing) | fixed 75, no SPRT |
| `baseline/recon-only/` | recon-only | {} through the same code path (output asserted bit-identical) | 8 |
| `arm-A/cand-720/` | candidate | 720 | SPRT, cap 75 |
| `arm-A/ctrlNN-<id>/` | 10 controls | matched_random_controls[720][NN-1] | SPRT, cap 75 |
| `arm-B/cand-7081+5852/` | candidate | 7081 and 5852 together | SPRT, cap 75 |
| `arm-B/ctrlNN-<a>+<b>/` | 10 control pairs | (controls[7081][NN-1], controls[5852][NN-1]) | SPRT, cap 75 |

Edit: `out = x - norm_scale * sum_{i in S} f_i W_dec[i]`, `f = JumpReLU_thr(((x - norm_mu)/norm_scale - b_dec) W_enc + b_enc)`, at every
policy call, on the **final denoise step only** (index 3 of 4, where the SAE was trained), on all 41 DiT tokens (state + 40 action
tokens; the SAE saw 32 of them, and firing stats are reported on those 32).

SPRT per condition, on the contiguous prefix of finished episodes: rise H0 p=0.56 vs H1 p=0.86, fall H0 p=0.56 vs H1 p=0.26,
alpha=beta=0.05 (bounds +/-2.944). Stop when either test accepts H1, or both accept H0, or at 75. Schedule: lockstep. Each free env
slot gets the next episode of the active condition with the fewest dispatched, in the order baseline, recon-only, cand A, cand B,
ctrl A1, ctrl B1, ... ctrl A10, ctrl B10 (header line of every `episodes.jsonl`, and `schedule.json`).

Per condition: `episodes.jsonl` (append-only, resumable), `summary.json` (n, successes, rate, Wilson 95%, SPRT, fail_kind counts,
paired delta vs baseline on shared seeds + McNemar exact p, live firing of the zeroed latents), `sim/ep-*.npz` (sim + actions).
Top: `summary.json` (table), `plan.json`, `schedule.json`, `errors.jsonl`.

The Colab output goes to `/content/physint/groot/groot-1028-v9-ablate` and `/content/ablate_snapshot.tar`
(arcname `physint/groot/groot-1028-v9-ablate/...`). To bring it back: `tar xf ablate_snapshot.tar` and copy that folder over this one
(`plan.json` must have the same sha as `ablate_plan.json`).


## Early-stopping-valid statistics (added after the run, from the existing records)

Wilson intervals assume a fixed n; the SPRT chose n. `rollout_stats.sequential_inference` (re-exported by `ablate_v9`) enumerates every stopping point of
the plan's SPRT exactly (same boundaries and stop rule as `default_plan` / `sequential_decision`) and inverts the exact
tails under the sample-proportion ordering: `seq95` (equal-tailed 95% interval, >= 95% coverage at every p, checked
exactly and by simulation in `tests/test_sequential_stats.py`) and `p_seq_rise` / `p_seq_fall` against p0. `arms` in the
top-level summary gives, per candidate, how many matched controls fell, a permutation-style p-value
`(1 + #controls with s/n <= candidate) / 11`, and Holm across the two candidates. None of it sets a verdict or `real`.

Recompute without rollouts: `python -m interpretableintelligence.adapters.groot.v9.resummarize_v9 --logs <this folder> --write`
(per-episode logs) or `--table <summary.json> --plan ablate_plan.json --write` (the vendored table; (n, successes) is
sufficient for these statistics, and every row is checked against the plan first). Either way it refuses to write if
anything points toward a rise.
