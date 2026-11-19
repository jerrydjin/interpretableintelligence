"""Closed-loop test of the backbone pot-phase direction on task 1028 (Nov 19 2026): does adding it make GR00T lift
moka_pot_1 before moka_pot_2? (``adapters/groot/v9/steer_backbone.py``; never a finding: no SAE, so no rule 4.)

    modal run scripts/modal_groot_phase_backbone.py --save-dom /vol/work/phase-backbone/dom.npy   # the fixed vector
    modal run scripts/modal_groot_steer_backbone.py --smoke                                       # 2 arms x 2 episodes
    modal run --detach scripts/modal_groot_steer_backbone.py                                      # the plan below
    modal run scripts/modal_groot_steer_backbone.py --fetch                                       # summary only
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/repo/scripts")
from modal_groot_steer import CAPTURE_DIR, GPU, PLUS_DIR, VOL, WORK, _prepare, image, volume  # noqa: E402

HOURS = float(os.environ.get("II_MODAL_HOURS", "1.25"))
DOM = f"{WORK}/phase-backbone/dom.npy"
OUT = f"{WORK}/steer-backbone"
N = 25
app = modal.App("ii-groot-steer-backbone", image=image)


def make_plan(smoke: bool) -> dict:
    man = json.loads(Path(CAPTURE_DIR, "manifest.json").read_text())
    n = 2 if smoke else N
    arms = [{"id": "baseline", "kind": "baseline", "n": n},
            {"id": "dom-x3", "kind": "direction", "vector": DOM, "scale": 3.0, "n": n},
            {"id": "dom-x6", "kind": "direction", "vector": DOM, "scale": 6.0, "n": n},
            {"id": "rand1-x3", "kind": "random", "random_seed": 1, "norm_of": DOM, "scale": 3.0, "n": n},
            {"id": "rand2-x3", "kind": "random", "random_seed": 2, "norm_of": DOM, "scale": 3.0, "n": n},
            {"id": "rand3-x6", "kind": "random", "random_seed": 3, "norm_of": DOM, "scale": 6.0, "n": n},
            {"id": "rand4-x6", "kind": "random", "random_seed": 4, "norm_of": DOM, "scale": 6.0, "n": n}]
    if smoke:
        arms = [arms[0], arms[2]]
    return {"schema": "groot-steer-backbone-v1",
            "task": {"index": int(man["task_index"]), "id": int(man["task_id"]), "name": man["task_name"],
                     "n_init_states": int(man["n_init_states"])},
            "seeds": {"base": 80000, "rule": "seed = base + i; init_state_id = i % n_init_states (shared across arms)"},
            "site": "action_head._encode_features -> backbone_features, every token, every policy call",
            "arms": arms, "outcome": "pot0_first",
            "sprt": {"p0": 0.05, "p1_rise": 0.35, "p1_fall": 0.01, "alpha": 0.05, "beta": 0.05, "cap": n},
            "criterion": "lever confirmed if a direction arm has SPRT verdict rise, McNemar p < 0.05 vs baseline on shared "
                         "seeds, and a pot0_first rate above every random arm. Not a finding (no SAE, rule 4 cannot apply)."}


@app.function(volumes={VOL: volume}, gpu=GPU, timeout=int(HOURS * 3600), cpu=8, memory=32768)
def steer_remote(smoke: bool = False) -> dict:
    import torch

    env = _prepare()
    os.environ.update({k: v for k, v in env.items() if k in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "PLUS_DIR")})
    import interpretableintelligence.adapters.groot.v9 as v9_pkg  # registers the bare v8 / v9 module names
    from interpretableintelligence.adapters.groot.adapter import DEFAULT_CHECKPOINT, LeRobotGroot
    from interpretableintelligence.adapters.groot.v9.steer_backbone import run_steer

    del v9_pkg
    v9, v8 = sys.modules["physint_inproc_v9"], sys.modules["physint_inproc_v8"]
    if not Path(DOM).is_file():
        raise SystemExit(f"{DOM} missing: run modal_groot_phase_backbone.py --save-dom {DOM} first")
    plan = make_plan(smoke)
    out = Path(OUT + ("-smoke" if smoke else ""))
    plus = Path(PLUS_DIR)
    v8.write_libero_config(plus, log=print)
    backend = LeRobotGroot(DEFAULT_CHECKPOINT)
    backend.load(random_init=False, device=torch.device("cuda"))
    ctx = v9.Ctx(run=out, run_id=out.name, policy=backend.policy, ckpt=DEFAULT_CHECKPOINT, layer=5, plus_dir=plus,
                 snapshot_tar=out.parent / f"{out.name}-snapshot.tar", soft_cap_h=HOURS - 0.05)
    v8.setup(ctx, device="cuda")
    import threading

    stop = threading.Event()

    def committer():
        while not stop.wait(180):
            volume.commit()

    threading.Thread(target=committer, daemon=True).start()
    try:
        summary = run_steer(ctx, plan, out=out)
    finally:
        stop.set()
        volume.commit()
    return summary


@app.function(volumes={VOL: volume}, timeout=300)
def fetch_remote(smoke: bool = False) -> dict:
    path = Path(OUT + ("-smoke" if smoke else ""), "summary.json")
    return json.loads(path.read_text()) if path.is_file() else {"missing": str(path)}


@app.local_entrypoint()
def main(smoke: bool = False, fetch: bool = False, out: str = "") -> None:
    result = fetch_remote.remote(smoke) if fetch else steer_remote.remote(smoke)
    Path(out or ("steer-backbone-smoke.json" if smoke else "steer-backbone.json")).write_text(json.dumps(result, indent=1) + "\n")
    for row in result.get("table", []):
        print(row["arm"], row["kind"], "n", row["n"], "pot0_first", row.get("pot0_first"), "success", row.get("successes"),
              "p", (row.get("vs_baseline") or {}).get("mcnemar_exact_p_two_sided"), "sprt", (row.get("sprt") or {}).get("verdict"))
