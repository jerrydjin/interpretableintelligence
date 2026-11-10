"""Build groot_colab_nodrive_v8.ipynb + v8_hotfix_cell.py from the v7 notebook + v8 sources.

v8 = v7 + (1) fixed low-RAM loader (RoPE), (2) inproc_v8 (RoPE repair in setup, timing,
predict-count check), (3) diag cell (open-loop vs training data, live-obs check, timing) gating smoke.
Hook (transformer_blocks.5, final denoise step) and PICKER capture layout are unchanged.
"""
import json, warnings
from pathlib import Path

ROOT = Path("/workspace/groot-colab")
SRC = (ROOT / "v8" / "inproc_v8.py").read_text()
LOADER = (ROOT / "low_ram_policy_load.py").read_text()
DIAG = (ROOT / "v8_diag_cell.py").read_text()
assert "'''" not in SRC and not SRC.rstrip().endswith("\\")
assert "def repair_rotary_buffers" in LOADER and "fn.__func__.__code__" not in LOADER, "loader RoPE fix missing"
nb = json.loads((ROOT / "groot_colab_nodrive_v7.ipynb").read_text())
cells = ["".join(c["source"]) for c in nb["cells"]]
assert len(cells) == 10


def must_replace(s, old, new, count=1):
    assert s.count(old) == count, (old[:80], s.count(old))
    return s.replace(old, new)


RUN_V7, RUN_V8 = "colab-t4-libero-plus-v7", "colab-t4-libero-plus-v8"
EMBED = "_V8_SRC = r'''" + SRC + "'''\n"
WRITE_MODULE = '''
_V8_PATH = Path("/content/inproc_v8.py")
_V8_PATH.write_text(_V8_SRC)
if str(_V8_PATH.parent) not in sys.path:
    sys.path.insert(0, str(_V8_PATH.parent))
import importlib
import inproc_v8 as v8
importlib.reload(v8)
'''

md = """# GR00T Colab T4 v8 — RoPE fix + open-loop diag gate, then in-process smoke / probe / capture

**Run all** from a fresh runtime (T4). Run id `colab-t4-libero-plus-v8`. Download `/content/physint_snapshot.tar`.

| Stage | What |
| --- | --- |
| Diag (new) | open-loop on 3 frames of the training set (`IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot`, ep 10 moka), live-obs check, 16-step timing. **Stops the run if the policy fails open-loop** (`GROOT_DIAG_GATE=0` to skip the gate) |
| Smoke | 1 full episode, vanilla libero_10 moka pots; prints `SMOKE ... timing={...}` |
| Probe | 10 eps each on LIBERO-plus ids **614** / **1028**; pick success rate nearest 50% |
| Capture | **100** hooked rollouts → PICKER.md layout, snapshot every 10 eps |

### What changed in v8
- **Bug fix (likely cause of 0% success in v7): wrong RoPE in the VLM backbone.** The low-RAM loader builds the model on `meta` and rebuilds non-persistent buffers afterwards. Its rebuild called `fn.__func__` on a staticmethod (AttributeError, swallowed) and fell back to `theta = config.rope_theta or 10000`. transformers 5 stores theta in `config.rope_parameters`, so the Qwen3-VL **text** RoPE was rebuilt with theta 1e4 instead of **5e6** (max |Δ inv_freq| 0.187) — every one of the 16 backbone LLM layers saw wrong token positions. Vision RoPE (theta 1e4) was unaffected. v8's loader rebuilds each RoPE buffer from a real instance of its own class, and `v8.setup()` re-checks/repairs every RoPE buffer (also fixes an already-loaded v7 policy in place).
- **Diag cell** before smoke: open-loop predicted vs ground-truth actions (per-dim MAE vs zero/mean baselines, gripper sign agreement, PASS/FAIL), live env obs vs dataset frame (orientation, state, preprocessed tensors), 16-step timing and inference-call counts, bf16-autocast comparison. If fp16 fails but bf16 passes, the run switches to bf16 autocast automatically.
- **Timing** per episode (`obs_pre_s / policy_s / post_s / env_s`, `n_predict` vs `ceil(n_steps/16)`) in smoke/probe/capture records.
- `GROOT_AMP=fp16|bf16_autocast` env knob (default fp16).
- Unchanged from v7: observation path (matches lerobot 0.7.0 `lerobot_eval.rollout` + `LiberoEnv`), Qwen3-VL processor substitution (all processor files are byte-identical to the gated Cosmos-Reason2-2B: same git blob ids), hook `_groot_model.action_head.model.transformer_blocks.5` final denoise step, PICKER layout, seeds.

### Capture layout (PICKER.md / rollcap-v0) — unchanged from v7
```
<run>/manifest.json   episodes[{episode_idx,episode_id,task_id,seed,init_state_id,success,fail_t,fail_kind,n_steps,n_chunks}],
                      hook{module,attn_type,denoise_step_index,...}, n_tok, d_model, token_indices, seeds[]
<run>/acts/shard-XXXXX.safetensors   acts f16 [N,1536]; episode_idx i32; step i32; tok i16 (mean = -1)
<run>/actions.safetensors            episode_idx, step, action[7]   (+ actions/shard-*.safetensors)
<run>/sim/ep-XXXXX.npz               object_z, eef_pos, gripper_width, object_xy, grasped, step (+ object_z_rel,
                                     obj_pos, obj_grasped, obj_names, active_obj, success, fail_t, fail_kind)
```
"""

c1 = must_replace(cells[1], f'"{RUN_V7}"', f'"{RUN_V8}"')
c1 = must_replace(c1, "(v7 in-process, PICKER layout)", "(v8 in-process, PICKER layout)")
c2 = cells[2]
c3 = must_replace(cells[3], f'"{RUN_V7}"', f'"{RUN_V8}"')
c3 = must_replace(c3, "physint_inproc_v7 (cell 6)", "inproc_v8 (cell 6)")
c4 = must_replace(cells[4], "v7 smoke builds it", "v8 smoke builds it")

c5 = cells[5]
first, rest = c5.split("\n", 1)
assert first.startswith("_LOADER_SRC = ")
c5 = "_LOADER_SRC = " + json.dumps(LOADER) + "\n" + rest
c5 = must_replace(c5, 'if not _LOADER_PATH.is_file() or os.environ.get("GROOT_REFRESH_LOADER") == "1":\n    _LOADER_PATH.write_text(_LOADER_SRC)',
                  '_LOADER_PATH.write_text(_LOADER_SRC)  # v8: always write (v7 file had the RoPE rebuild bug)')
c5 = must_replace(c5, "physint_inproc_v7.resolve_block", "inproc_v8.resolve_block")

c6 = ("\n# === Cell 6: v8 in-process runtime (no lerobot-eval): setup = libero config + RoPE repair + processors + hook ===\n"
      "import os, sys, json, time\nfrom pathlib import Path\nimport torch\n\n" + EMBED + WRITE_MODULE + '''
if FAMILY != "groot":
    raise SystemExit(f"v8 in-process path is GR00T-only (FAMILY={FAMILY}); GR00T failed to load in cell 5")
V8 = v8.Ctx(
    run=RUN, run_id=RUN_ID, policy=policy, ckpt=CKPT, layer=int(LAYER), plus_dir=Path(PLUS_DIR),
    snapshot_tar=SNAPSHOT_TAR, tokens_per_step=int(os.environ.get("GROOT_TOKENS_PER_STEP", "32")),
    probe_n=int(TASK["probe_episodes"]), capture_n=int(TASK["capture_target_rollouts"]),
    started_at=globals().get("STARTED_AT", time.time()), soft_cap_h=globals().get("COLAB_SOFT_CAP_H", 10.5),
)
V8_SETUP = v8.setup(V8)
print(json.dumps({"hook": V8_SETUP["hook"], "rope": V8_SETUP["rope"]}, indent=2))
''')

GATE = '''

# ----- v8 gate: do not spend Pro units on rollouts if the policy fails open-loop -----
_ol = DIAG.get("open_loop", {})
_fp16 = _ol.get("fp16_rope_fixed") or _ol.get("fp16_as_loaded")
_bf16 = _ol.get("bf16_autocast")
if _fp16 is not None and not _fp16["pass"] and _bf16 is not None and _bf16["pass"]:
    print("[v8] fp16 FAILS open-loop but bf16 autocast PASSES -> running rollouts in bf16 autocast")
    V8.amp_mode = "bf16_autocast"
    policy.config.use_bf16 = True
elif os.environ.get("GROOT_DIAG_GATE", "1") == "1" and (_fp16 is None or not _fp16["pass"]):
    raise SystemExit("[v8] OPEN-LOOP FAIL (or diag error) -> stopping before smoke so no compute is burned. "
                     "See the table above / smoke/diag_v8.json. Set GROOT_DIAG_GATE=0 to run anyway.")
'''
c6b = "\n# === Cell 6b: v8 diag (open-loop vs training data, live obs, 16-step timing) + gate ===\n" + DIAG + GATE

c6c = '''
# === Cell 6c: smoke: 1 full vanilla moka episode (hooked) ===
import json
SMOKE = v8.smoke(V8)
print(json.dumps({k: v for k, v in SMOKE.items() if k != "traceback"}, indent=2))
'''
c7 = must_replace(cells[7], "CHOSEN = v7.probe(V7)", "CHOSEN = v8.probe(V8)")
c8 = must_replace(cells[8], "CAPTURE_SUMMARY = v7.capture(V7, CHOSEN)", "CAPTURE_SUMMARY = v8.capture(V8, CHOSEN)")
c9 = cells[9].replace("v7.", "v8.").replace("V7.", "V8.")
c9 = must_replace(c9, '"amp_mode": V8.amp_mode,', '"amp_mode": V8.amp_mode, "rope": V8_SETUP.get("rope"),\n    "diag": {k: v for k, v in DIAG.items() if k in ("open_loop", "queue_ok", "timing")},')

srcs = [md, c1, c2, c3, c4, c5, c6, c6b, c6c, c7, c8, c9]
import copy
code_tmpl = copy.deepcopy(nb["cells"][6])
new = []
for i, src in enumerate(srcs):
    c = copy.deepcopy(nb["cells"][i] if i <= 6 else code_tmpl)
    c["source"] = src.splitlines(keepends=True)
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
    new.append(c)
nb["cells"] = new
import uuid
_seen = set()
for i, c in enumerate(nb["cells"]):  # nbformat 4.5: every cell needs a unique id
    if c.get("id") in _seen or not c.get("id"):
        c["id"] = f"v8-cell-{i}-" + uuid.uuid5(uuid.NAMESPACE_URL, f"groot-v8-{i}").hex[:8]
    _seen.add(c["id"])
out = ROOT / "groot_colab_nodrive_v8.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))

hot = ('# === v8 HOTFIX for a LIVE v7 runtime (cells 1-5 ran: policy, CKPT, FAMILY, LAYER, PLUS_DIR, TASK) ===\n'
       '# No reload: (1) diag on the policy AS LOADED (shows the RoPE bug), repairs RoPE in place, re-runs open-loop;\n'
       '# (2) gate; (3) smoke -> probe -> capture into /content/physint/groot/colab-t4-libero-plus-v8/.\n'
       'import os, sys, json, time\nfrom pathlib import Path\nimport torch\n\n' + EMBED + WRITE_MODULE + '''
assert FAMILY == "groot" and policy is not None, f"GR00T not loaded (FAMILY={FAMILY})"
RUN = Path("/content/physint/groot/colab-t4-libero-plus-v8")
V8 = v8.Ctx(
    run=RUN, run_id="colab-t4-libero-plus-v8",
    policy=policy, ckpt=CKPT, layer=int(LAYER), plus_dir=Path(PLUS_DIR),
    snapshot_tar=Path("/content/physint_snapshot.tar"),
    tokens_per_step=int(os.environ.get("GROOT_TOKENS_PER_STEP", "32")),
    started_at=globals().get("STARTED_AT", time.time()), soft_cap_h=globals().get("COLAB_SOFT_CAP_H", 10.5),
)
V8.procs = v8.build_processors(policy, CKPT, device="cuda" if torch.cuda.is_available() else "cpu")
''' + "\n# ----- diag (before v8.setup so the as-loaded RoPE state is measured) -----\n" + DIAG + GATE + '''
V8_SETUP = v8.setup(V8)
print(json.dumps({"hook": V8_SETUP["hook"], "rope": V8_SETUP["rope"]}, indent=2))
SMOKE = v8.smoke(V8)
CHOSEN = v8.probe(V8)
CAPTURE_SUMMARY = v8.capture(V8, CHOSEN)
print(json.dumps({"smoke": {k: SMOKE.get(k) for k in ("seconds_per_episode", "success", "n_steps", "amp_mode", "timing")},
                  "chosen": CHOSEN, "capture": CAPTURE_SUMMARY}, indent=2))
''')
(ROOT / "v8_hotfix_cell.py").write_text(hot)

with warnings.catch_warnings():
    warnings.simplefilter("error")
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"<v8 cell {i}>", "exec")
    compile(SRC, "inproc_v8.py", "exec")
    compile(hot, "v8_hotfix_cell.py", "exec")
    compile(DIAG, "v8_diag_cell.py", "exec")
txt = out.read_text()
assert RUN_V7 not in txt and "v7.probe" not in txt and "import physint_inproc_v7" not in txt
assert "transformer_blocks.5" in txt and "final_denoise_step_per_chunk" in txt
print("built", out, "cells", len(nb["cells"]), "| hotfix lines", hot.count("\n"))
