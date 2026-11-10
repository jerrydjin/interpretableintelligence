"""Build groot_colab_nodrive_v9.ipynb + v9_hotfix_cell.py from the v8 notebook + v9 sources.

v9 = v8 + physint_inproc_v9: N LIBERO envs in spawned worker processes (GROOT_N_ENVS, default
min(cpu-1, 8)), policy calls from the main process (per-env by default, batched with GROOT_BATCH_PREDICT=1
after a batched-vs-single check), fresh env per episode, per-episode noise generators (== v8 seeds),
probe 20 eps/task (GROOT_PROBE_N) with Wilson 95% CI. Diag gate + smoke are v8's (sequential).
SAE feature ablation (ablate_v9.py): set GROOT_ABLATE_PLAN (=default for the embedded 1028 rise-test plan, or a plan
JSON path); the notebook / hotfix then run the ablation conditions instead of probe + capture.
"""
import copy, hashlib, json, sys, textwrap, uuid, warnings
from pathlib import Path

ROOT = Path("/workspace/groot-colab")
SRC8 = (ROOT / "v8" / "physint_inproc_v8.py").read_text()
SRC9 = (ROOT / "v9" / "physint_inproc_v9.py").read_text()
SRCPE = (ROOT / "v9" / "pot_events.py").read_text()  # per-pot failure detector, shared with the offline relabel
SRCAB = (ROOT / "v9" / "ablate_v9.py").read_text()   # SAE feature-ablation runner
ABL_DIR = Path("/workspace/runs/groot-1028-v9-ablate")  # v9/make_ablate_plan.py writes the plan + SAE subset here
ABL_PLAN = (ABL_DIR / "ablate_plan.json").read_text()
sys.path[:0] = [str(ROOT / "v9"), str(ROOT / "v8")]
import ablate_v9  # noqa: E402

_plan = json.loads(ABL_PLAN)
_sub = ABL_DIR / _plan["sae"]["subset_file"]
assert ablate_v9.sha256_file(_sub) == _plan["sae"]["subset_sha256"], "SAE subset does not match the plan; rerun make_ablate_plan.py"
ABL_B85 = ablate_v9.blob_from_file(_sub)
for s in (SRC8, SRC9, SRCPE, SRCAB, ABL_PLAN):
    assert "'''" not in s and not s.rstrip().endswith("\\")
nb = json.loads((ROOT / "groot_colab_nodrive_v8.ipynb").read_text())
cells = ["".join(c["source"]) for c in nb["cells"]]
assert len(cells) == 12, len(cells)
assert ("_V8_SRC = r'''" + SRC8 + "'''") in cells[6], "v8 notebook embeds a different physint_inproc_v8.py; rebuild v8 first"


def must_replace(s, old, new, count=1):
    assert s.count(old) == count, (old[:80], s.count(old))
    return s.replace(old, new)


RUN_V8, RUN_V9 = "colab-t4-libero-plus-v8", "colab-l4-libero-plus-v9"
EMBED9 = ("_POT_EVENTS_SRC = r'''" + SRCPE + "'''\n" + "_V9_SRC = r'''" + SRC9 + "'''\n"
          + "_ABLATE_SRC = r'''" + SRCAB + "'''\n" + "_ABLATE_PLAN_DEFAULT = r'''" + ABL_PLAN + "'''\n"
          + "_ABLATE_SAE_B85 = \"" + ABL_B85 + "\"  # zlib+base85 of sae_subset.safetensors (30 latents, exact rows of sae/weights)\n")
WRITE9 = '''
Path("/content/pot_events.py").write_text(_POT_EVENTS_SRC)  # imported by physint_inproc_v9 (main + env workers)
_V9_PATH = Path("/content/physint_inproc_v9.py")
_V9_PATH.write_text(_V9_SRC)
_V8_PATH_FOR_WORKERS = Path("/content/physint_inproc_v8.py")  # env workers import v8 helpers from disk
if not _V8_PATH_FOR_WORKERS.is_file():
    _V8_PATH_FOR_WORKERS.write_text(_V8_SRC)
for _p in ("/content",):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import importlib
import pot_events
importlib.reload(pot_events)
import physint_inproc_v9 as v9
importlib.reload(v9)
Path("/content/ablate_v9.py").write_text(_ABLATE_SRC)
import ablate_v9
importlib.reload(ablate_v9)


def _run_ablate_from_env(V8):
    """GROOT_ABLATE_PLAN=default (embedded 1028 plan) or a plan path; GROOT_ABLATE_SAE overrides the embedded SAE subset;
    GROOT_ABLATE_OUT the output dir; GROOT_ABLATE_ONLY=arm-A,baseline,... limits which conditions get episodes."""
    _pe = os.environ.get("GROOT_ABLATE_PLAN", "default")
    _pp = Path("/content/ablate_plan.json") if _pe in ("", "default") else Path(_pe)
    if _pe in ("", "default"):
        _pp.write_text(_ABLATE_PLAN_DEFAULT)
    _plan = json.loads(_pp.read_text())
    _sae = os.environ.get("GROOT_ABLATE_SAE")
    if not _sae:
        _sae = "/content/ablate_sae_subset.safetensors"
        ablate_v9.file_from_blob(_ABLATE_SAE_B85, _sae, _plan["sae"]["subset_sha256"])
    _out = Path(os.environ.get("GROOT_ABLATE_OUT") or f"/content/physint/groot/{_plan['run_id']}")
    _only = [x for x in os.environ.get("GROOT_ABLATE_ONLY", "").split(",") if x] or None
    VA = v9.ctx_from_v8(V8, run=_out, run_id=_plan["run_id"], started_at=time.time(),
                        snapshot_tar=Path("/content/ablate_snapshot.tar"))
    print(json.dumps({"ablate_plan": str(_pp), "plan_sha256": ablate_v9.plan_sha(_plan), "sae": _sae, "out": str(_out),
                      "only": _only, "n_envs": VA.n_envs, "conditions": len(_plan["conditions"]),
                      "hook": {k: _plan["hook"][k] for k in ("module", "denoise_steps", "tokens")}, "sprt": _plan["sprt"]}))
    return ablate_v9.run_ablation(VA, _plan, sae_path=_sae, only=_only)
'''

md = """# GR00T Colab v9 — parallel LIBERO envs (env workers + one GPU policy), probe 20/task, capture

**Run all** from a fresh runtime (L4 or T4). Run id `colab-l4-libero-plus-v9`. Download `/content/physint_snapshot.tar`.

| Stage | What |
| --- | --- |
| Diag | v8 open-loop gate (unchanged). Stops before rollouts if the policy fails open-loop |
| Smoke | 1 full vanilla moka episode, sequential (v8) |
| Probe | **20** eps each on LIBERO-plus ids **614** / **1028** on the env pool; pick success rate nearest 50%, Wilson 95% CI printed |
| Capture | **100** hooked rollouts on the env pool → PICKER.md layout, snapshot every 10 eps |

### Knobs (set before Run all with `%env`)
| Env | Default | Meaning |
| --- | --- | --- |
| `GROOT_N_ENVS` | min(cpu-1, 8), capped by free RAM | parallel env worker processes |
| `GROOT_BATCH_PREDICT` | `0` | `0`: one predict per env (bit-identical to the sequential runner). `1`: batch envs with identical input shapes after a batched-vs-single check passes (`GROOT_BATCH_TOL`, default 2e-2) |
| `GROOT_FRESH_ENV` | `1` | build a fresh LiberoEnv per episode (v8 reused one; see below) |
| `GROOT_PROBE_N` / `GROOT_CAPTURE_N` | 20 / 100 | probe episodes per task / capture rollouts |
| `GROOT_ABLATE_PLAN` | unset | `default` (embedded 1028 plan) or a plan JSON path: run the SAE ablation instead of probe + capture |
| `GROOT_ABLATE_ONLY` / `GROOT_ABLATE_OUT` / `GROOT_ABLATE_SAE` | all / `/content/physint/groot/groot-1028-v9-ablate` / embedded | limit conditions (e.g. `baseline,arm-A`) / output dir / SAE file (full `weights.safetensors` also accepted) |

### What changed in v9
- **Parallel envs.** Each worker process owns a LiberoEnv (own MuJoCo/EGL context, `MUJOCO_GL=egl` inherited) and executes one whole 16-step chunk per message, so the main process only talks to it once per chunk. The main process does the same preprocessing → `predict_action_chunk` → postprocess as v8 and keeps a pool: when an episode ends, that slot starts the next.
- **Same noise as v8.** Each episode has its own `torch.Generator(seed)` on the GPU, and GR00T's `sample_noise` is routed through it, which is the stream v8 got from `torch.manual_seed(seed)`.
- **Fresh env per episode.** A reused LiberoEnv moves the stove fixture by about 1.2–1.4 cm on every hard reset, even with identical seeds. That made v8 episodes depend on which episodes ran before them. v9 rebuilds the env (about +1.5 s per episode), so each episode depends only on (task, seed, init_state_id).
- Per-episode `TIMING` line and aggregate `THROUGHPUT` line (episodes/hour, env steps/s, GPU predict busy %).
- **Per-pot failure detector** (`pot_events.py`, rule `pot0-v1`, shared with the offline relabel). `fail_kind` is one of lifted_not_placed / never_grasped / pot0_placed_other_failure / empty_grasp, judged on pot 0 (moka_pot_1) with lift / release / stove-placement events. v8's active-object label stays in `fail_kind_active` / `fail_t_active` (it marked pot 1's normal placement as a drop in 37/40 cases). Labels only: probe, early stopping and capture are unchanged.
- **SAE feature ablation** (`ablate_v9.py`). Set `%env GROOT_ABLATE_PLAN=default` before Run all and cells 7–9 run the 1028 rise-test plan (baseline 75 fixed, recon-only 8, arm A = latent 720, arm B = 7081+5852, 10 matched controls each, SPRT per condition) instead of probe + capture. Output `/content/physint/groot/groot-1028-v9-ablate`, snapshot `/content/ablate_snapshot.tar`, resumable after a disconnect (re-run the same cells).
- Unchanged: hook `transformer_blocks.5` final denoise step, capture layout, seeds (probe 10000+k / init k; capture 20000+i / init i % n_init), RoPE verification (raises).
"""

c1 = must_replace(cells[1], f'"{RUN_V8}"', f'"{RUN_V9}"')
c1 = must_replace(c1, "(v8 in-process, PICKER layout)", "(v9 parallel envs, PICKER layout)")
c2 = cells[2]
c3 = must_replace(cells[3], f'"{RUN_V8}"', f'"{RUN_V9}"')
c3 = must_replace(c3, '"capture_target_rollouts": 100,', '"capture_target_rollouts": int(os.environ.get("GROOT_CAPTURE_N", "100")),')
c3 = must_replace(c3, '"probe_episodes": 10,', '"probe_episodes": int(os.environ.get("GROOT_PROBE_N", "20")),')
c4, c5 = cells[4], cells[5]
c6 = cells[6].rstrip("\n") + "\n\n# ----- v9 parallel runner module (env workers import it from /content) -----\n" + EMBED9 + WRITE9
c6b, c6c = cells[7], cells[8]
c7 = '''
# === Cell 7: Probe LIBERO-plus 614 & 1028 (20 eps each, parallel env pool); pick nearest 50% ===
import json, os
V9 = v9.ctx_from_v8(V8, probe_n=int(TASK["probe_episodes"]), capture_n=int(TASK["capture_target_rollouts"]))
print(json.dumps({"n_envs": V9.n_envs, "cpu_count": os.cpu_count(), "batch_predict": V9.batch_predict,
                  "fresh_env": v9.FRESH_ENV, "amp_mode": V9.amp_mode, "probe_n": V9.probe_n, "capture_n": V9.capture_n,
                  "noise_generator_vs_global_max_diff": v9.check_generator_matches_global(next(policy.parameters()).device,
                                                                                         dtype=next(policy.parameters()).dtype)}))
if os.environ.get("GROOT_ABLATE_PLAN"):  # SAE ablation mode: no probe / capture
    ABLATE_SUMMARY = _run_ablate_from_env(V8)
    CHOSEN = None
else:
    CHOSEN = v9.probe(V9)
    print(json.dumps(CHOSEN, indent=2))
'''
c8 = '''
# === Cell 8: 100 hooked captures on the env pool -> PICKER.md layout (resumable; snapshot every 10 eps) ===
import json
if os.environ.get("GROOT_ABLATE_PLAN"):
    print("ablation mode: capture skipped")
else:
    CAPTURE_SUMMARY = v9.capture(V9, CHOSEN)
    print(json.dumps(CAPTURE_SUMMARY, indent=2))
'''
c9 = must_replace(cells[11], 'v8.write_snapshot(V7, "final")', 'v8.write_snapshot(V9, "final")')
c9 = must_replace(c9, '"amp_mode": V8.amp_mode,', '"amp_mode": V9.amp_mode, "pool_stats": V9.pool_stats, "runner": v9.V9_VERSION,')
_c9_head, _c9_body = c9.split("import json\n", 1)
c9 = (_c9_head + "import json, os\nif os.environ.get(\"GROOT_ABLATE_PLAN\"):\n"
      "    print(json.dumps(ABLATE_SUMMARY[\"table\"], indent=1))\n"
      "    print(\"Download /content/ablate_snapshot.tar from the Colab Files pane (untar into /workspace/runs/ on the box).\")\n"
      "else:\n" + textwrap.indent(_c9_body, "    "))

srcs = [md, c1, c2, c3, c4, c5, c6, c6b, c6c, c7, c8, c9]
code_tmpl = copy.deepcopy(nb["cells"][6])
new = []
for i, src in enumerate(srcs):
    c = copy.deepcopy(nb["cells"][i] if i < len(nb["cells"]) else code_tmpl)
    c["source"] = src.splitlines(keepends=True)
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
    c["id"] = f"v9-cell-{i}-" + uuid.uuid5(uuid.NAMESPACE_URL, f"groot-v9-{i}").hex[:8]
    new.append(c)
nb["cells"] = new
out = ROOT / "groot_colab_nodrive_v9.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))

# ---------------------------------------------------------------- hotfix for the LIVE v8 runtime
hot = ('# === v9 HOTFIX for a LIVE v8 runtime (policy loaded; V8 = v8.Ctx after setup) ===\n'
       '# 1) Interrupt the running v8 probe/capture cell first (Runtime > Interrupt execution). Its probe.json is kept.\n'
       '# 2) Paste this into a new cell and run it. Optional first: %env GROOT_N_ENVS=8  /  %env GROOT_STOP_AFTER_PROBE=1\n'
       '# Switches to the parallel env pool, probes 614/1028 to 20 eps each in a NEW run dir\n'
       '# /content/physint/groot/colab-l4-libero-plus-v9, then (unless GROOT_STOP_AFTER_PROBE=1) captures 100 eps.\n'
       '# SAE ABLATION instead: %env GROOT_ABLATE_PLAN=default  (optional %env GROOT_ABLATE_ONLY=baseline,arm-A)\n'
       '# -> runs the embedded 1028 rise-test plan into /content/physint/groot/groot-1028-v9-ablate (resumable:\n'
       '# re-run this cell after a disconnect), snapshot /content/ablate_snapshot.tar. Works in a live v8 or v9 runtime.\n'
       'import os, sys, json, time, shutil\nfrom pathlib import Path\nimport torch\n\n'
       "_V8_SRC = r'''" + SRC8 + "'''\n" + EMBED9 + WRITE9 + '''
assert "V8" in globals() and V8.procs is not None and V8.policy is policy, "need the live v8 runtime (V8 ctx with processors)"
if "predict_action_chunk" in policy.__dict__:  # a v8 FinalDenoiseCapture wrapper is still installed
    print("[v9] removing a stale v8 FinalDenoiseCapture wrapper + its forward hook on the DiT block")
    policy.__dict__.pop("predict_action_chunk")
    v9.resolve_block(policy, int(LAYER))[1]._forward_hooks.clear()
RUN9 = Path("/content/physint/groot/colab-l4-libero-plus-v9")
for _sub in ("acts", "actions", "sim", "smoke", "probe", "logs"):
    (RUN9 / _sub).mkdir(parents=True, exist_ok=True)
for _rel in ("smoke/smoke.json", "smoke/diag_v8.json"):  # facts about the loaded policy, copied for the record
    if (V8.run / _rel).is_file() and not (RUN9 / _rel).is_file():
        shutil.copy2(V8.run / _rel, RUN9 / _rel)
if os.environ.get("GROOT_PROBE_REUSE_V8") == "1" and not (RUN9 / "probe/probe.json").is_file() and (V8.run / "probe/probe.json").is_file():
    shutil.copy2(V8.run / "probe/probe.json", RUN9 / "probe/probe.json")  # off by default: v8 reused envs (see notes)
V9 = v9.ctx_from_v8(V8, run=RUN9, run_id="colab-l4-libero-plus-v9",
                    probe_n=int(os.environ.get("GROOT_PROBE_N", "20")),
                    capture_n=int(os.environ.get("GROOT_CAPTURE_N", "100")))
V9_SETUP = v9.setup(V9)  # v8 setup: libero config + RoPE repair/verify (raises on mismatch) + hook resolve
_dev = next(policy.parameters()).device
print(json.dumps({"n_envs": V9.n_envs, "cpu_count": os.cpu_count(), "batch_predict": V9.batch_predict,
                  "fresh_env": v9.FRESH_ENV, "amp_mode": V9.amp_mode, "probe_n": V9.probe_n, "capture_n": V9.capture_n,
                  "MUJOCO_GL": os.environ.get("MUJOCO_GL"), "rope": V9_SETUP["rope"],
                  "noise_generator_vs_global_max_diff": v9.check_generator_matches_global(_dev, dtype=next(policy.parameters()).dtype)}))
if os.environ.get("GROOT_ABLATE_PLAN"):
    ABLATE_SUMMARY = _run_ablate_from_env(V8)
    print(json.dumps(ABLATE_SUMMARY["table"], indent=1))
    print("Download /content/ablate_snapshot.tar from the Colab Files pane.")
else:
    CHOSEN = v9.probe(V9)
    print(json.dumps(CHOSEN, indent=2))
if not os.environ.get("GROOT_ABLATE_PLAN") and os.environ.get("GROOT_STOP_AFTER_PROBE") != "1":
    CAPTURE_SUMMARY = v9.capture(V9, CHOSEN)
    print(json.dumps({"chosen": CHOSEN, "capture": CAPTURE_SUMMARY, "pool": V9.pool_stats}, indent=2))
''')
(ROOT / "v9_hotfix_cell.py").write_text(hot)

with warnings.catch_warnings():
    warnings.simplefilter("error")
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"<v9 cell {i}>", "exec")
    compile(SRC9, "physint_inproc_v9.py", "exec")
    compile(SRCPE, "pot_events.py", "exec")
    compile(SRCAB, "ablate_v9.py", "exec")
    compile(hot, "v9_hotfix_cell.py", "exec")
txt = out.read_text()
assert RUN_V8 not in txt and "v8.probe(" not in txt and "v8.capture(" not in txt
assert "transformer_blocks.5" in txt and "final_denoise_step_per_chunk" in txt and "v9.probe(V9)" in txt
assert txt.count("_ABLATE_SAE_B85 = ") == 1 and hot.count("_ABLATE_SAE_B85 = ") == 1 and "_run_ablate_from_env(V8)" in hot
print("built", out, "cells", len(nb["cells"]), "| hotfix lines", hot.count("\n"), "| hotfix bytes", len(hot),
      "| plan sha", ablate_v9.plan_sha(_plan)[:16])
