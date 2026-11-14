# GR00T LIBERO-plus rollouts (`ii rollouts --policy groot-1028`)

This lane runs the **task 1028** feature-ablation rise test through the public CLI by reusing
`src/interpretableintelligence/adapters/groot/v9/ablate_v9.py` (policy hook, LIBERO-plus env, SAE edit,
matched random controls, seeds `40000 + i`, `init_state_id = i % 50`, step cap **75**).

CI uses `--policy groot-mock`, which replays the frozen 8-episode verification slice from
`tests/fixtures/groot-1028-verify-slice.json`. On a GPU machine, run the same slice live, then compare
against the archived v9 logs.

## One-time setup (Colab L4 or box with LIBERO-plus)

Use the existing GR00T v9 Colab stack (LeRobot 0.7 `lerobot[groot]`, LIBERO-plus assets, checkpoint
`nvidia/gr00t17-lerobot-libero_10-640`). Unpack the capture run and SAE subset next to the ablation plan:

```bash
cd /content
gh release download data-groot-1028-v1 --repo jerrydjin/interpretableintelligence \
  --pattern 'groot-1028-v9-n100.tar.xz'
tar -xJf groot-1028-v9-n100.tar.xz
# sae_subset.safetensors: copy from prior ablate bundle or build via ablate_v9.write_subset
export GROOT_ABLATE_PLAN=/content/interpretableintelligence/src/interpretableintelligence/adapters/groot/v9/ablate/ablate_plan.json
export GROOT_ABLATE_SAE=/content/sae_subset.safetensors
```

Run the v9 hotfix cell once so `physint_inproc_v8` / `physint_inproc_v9` and the policy are live
(same as `groot_colab_nodrive_v9.ipynb`).

## Verification slice (8 episodes per arm)

Conditions (seeds **40000**–**40007**, init states **0**–**7** mod 50):

| Arm | v9 condition id |
| --- | --- |
| baseline | `baseline` |
| SAE recon-only (no edit) | `recon-only` |
| ablate latent **720** | `A-cand-720` |
| matched control **7774** | `A-ctrl01-7774` |
| matched control **2343** | `A-ctrl02-2343` |

### Path A — native v9 runner, then import

```bash
cd /content/interpretableintelligence
uv sync
export GROOT_ABLATE_ONLY=baseline,recon-only,A-cand-720,A-ctrl01-7774,A-ctrl02-2343
# In the hotfix / ablate cell, set only= that list and cap via shrunk plan or env stop-after.
# Or call run_ablation with only=[...] from a short Python cell after v8 setup.
RUN=/content/runs/groot-1028-cli-verify
mkdir -p "$RUN"
python -c "
import json
from pathlib import Path
import ablate_v9 as A
# assumes V8 ctx already built in the notebook as V8
plan = json.loads(Path('${GROOT_ABLATE_PLAN}').read_text())
only = ['baseline','recon-only','A-cand-720','A-ctrl01-7774','A-ctrl02-2343']
for c in plan['conditions']:
    if c['id'] in only:
        c['n_max'] = 8
plan['sprt']['cap'] = 8
A.run_ablation(V8, plan, sae_path='${GROOT_ABLATE_SAE}', only=only)
"

uv run ii rollouts "$RUN" --policy groot-1028 \
  --v9-out "$RUN/rollouts/v9-ablate" \
  --verify-slice
```

### Path B — CLI slice flag (imports v9 output if present)

After `run_ablation` wrote under `$RUN/rollouts/v9-ablate/` (v9 layout: `baseline/`, `arm-A/`, …):

```bash
uv run ii rollouts "$RUN" --policy groot-1028 --verify-slice \
  --v9-out "$RUN/rollouts/v9-ablate"
uv run ii validate "$RUN"
```

### Compare against archived v9 summaries

Copy `ablate_snapshot.final.tar` from the bot store or use the committed
`demo-data/groot/groot-1028-ablate-summary.json` for top-level rates. Per-episode check:

```bash
tar -xf ablate_snapshot.final.tar physint/groot/groot-1028-v9-ablate
uv run python scripts/compare_groot_v9_rollouts.py "$RUN" \
  --v9-dir physint/groot/groot-1028-v9-ablate \
  --policy groot-1028
```

Exit code **0** means every compared episode’s **success** flag (and step counts when present) match the
v9 `episodes.jsonl` rows; action **sha256** is checked when `sim/ep-*.npz` files exist on both sides.

## Reproduction uncertainties (v9 code)

- **Lockstep scheduler** dispatches the next episode of the condition with the fewest finished episodes;
  completion order does not change SPRT prefixes, but episode wall times can differ between runners.
- **Per-episode noise** uses `torch.Generator(seed)` inside GR00T `sample_noise`; any tokenizer,
  dtype, or checkpoint revision that changes activations can change actions and success despite matched seeds.
- **Recon-only** must be bit-identical to baseline on actions (asserted in v9); in `rollouts/episodes.parquet`
  it is stored as `kind=scale`, `value=1` on latent **720** so `episode_id` differs from a true ablation.
- **Arm B** candidate **7081+5852** uses **two** latents in one v9 condition; the `ii rollouts` schema maps
  one feature-id list per intervention row—full 24-condition plans should use `--only` per arm or a future
  multi-latent arm extension.
- **Control SPRT** in `summary.json` uses the fake-lane discordant SPRT for schema compatibility; v9 rise/fall
  SPRT lives in the imported `v9-ablate/*/summary.json` files.
- **`ii rollouts` live path** (`run_live_v9`) expects the Colab `physint_inproc_v8/v9` modules on
  `sys.path`; this repo does not install LIBERO or LeRobot GPU deps in the default `uv sync` environment.