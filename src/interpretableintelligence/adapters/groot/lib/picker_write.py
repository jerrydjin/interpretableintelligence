"""Write PICKER.md / rollout_io capture layout (manifest, acts shards, actions.safetensors)."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np

def atomic_json(path: Path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)

def save_acts_shard_picker(path: Path, *, acts_f16: np.ndarray, episode_idx: int, steps: np.ndarray, toks: np.ndarray):
    """acts [N,D] f16 + episode_idx/step/tok per row."""
    from safetensors.numpy import save_file
    N = int(acts_f16.shape[0])
    ep = np.full((N,), int(episode_idx), dtype=np.int32)
    payload = {
        "acts": np.ascontiguousarray(acts_f16.astype(np.float16)),
        "episode_idx": ep,
        "step": np.ascontiguousarray(np.asarray(steps, dtype=np.int32)),
        "tok": np.ascontiguousarray(np.asarray(toks, dtype=np.int16)),
    }
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    save_file(payload, str(path))
    return path

def append_actions_rebuild(run_dir: Path, episode_idx: int, steps: np.ndarray, actions: np.ndarray):
    """Write actions/shard-XXXXX.safetensors for this ep and rebuild root actions.safetensors."""
    from safetensors.numpy import save_file, load_file
    run_dir = Path(run_dir)
    shard_dir = run_dir / "actions"
    shard_dir.mkdir(parents=True, exist_ok=True)
    steps = np.asarray(steps, dtype=np.int32).reshape(-1)
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)
    ep = np.full((len(steps),), int(episode_idx), dtype=np.int32)
    save_file(
        {"episode_idx": ep, "step": steps, "action": np.ascontiguousarray(actions)},
        str(shard_dir / f"shard-{int(episode_idx):05d}.safetensors"),
    )
    # rebuild combined
    eps, sts, acts = [], [], []
    for sh in sorted(shard_dir.glob("shard-*.safetensors")):
        d = load_file(str(sh))
        eps.append(d["episode_idx"]); sts.append(d["step"]); acts.append(d["action"])
    if eps:
        save_file(
            {
                "episode_idx": np.concatenate(eps).astype(np.int32),
                "step": np.concatenate(sts).astype(np.int32),
                "action": np.concatenate(acts).astype(np.float32),
            },
            str(run_dir / "actions.safetensors"),
        )

def save_sim_npz(path: Path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})
    return path

def extract_sim_from_obs(obs: dict, env=None) -> dict:
    """Best-effort LIBERO/robosuite obs -> eef_pos, gripper_width, object_z, object_xy, grasped."""
    out = {
        "eef_pos": np.full(3, np.nan, np.float32),
        "gripper_width": np.float32(np.nan),
        "object_z": np.float32(np.nan),
        "object_xy": np.full(2, np.nan, np.float32),
        "grasped": np.int8(-1),
    }
    if not isinstance(obs, dict):
        return out
    # flatten nested
    flat = dict(obs)
    for k, v in list(obs.items()):
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat.setdefault(kk, vv)
    for key in ("robot0_eef_pos", "eef_pos", "ee_pos"):
        if key in flat:
            arr = np.asarray(flat[key], dtype=np.float32).reshape(-1)
            out["eef_pos"][: min(3, arr.size)] = arr[:3]
            break
    for key in ("robot0_gripper_qpos", "gripper_qpos"):
        if key in flat:
            g = np.asarray(flat[key], dtype=np.float32).reshape(-1)
            out["gripper_width"] = np.float32(float(np.abs(g).sum()) if g.size else np.nan)
            break
    # object positions: prefer keys ending with _pos that aren't robot/eef
    obj_candidates = []
    for k, v in flat.items():
        if not k.endswith("_pos"):
            continue
        if any(s in k for s in ("eef", "robot", "gripper", "joint", "camera")):
            continue
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
        if arr.size >= 3:
            obj_candidates.append((k, arr[:3]))
    if obj_candidates:
        # pick first movable-looking object
        _, arr = obj_candidates[0]
        out["object_z"] = np.float32(arr[2])
        out["object_xy"] = arr[:2].astype(np.float32)
    # grasped / contact if present
    for key in ("grasped", "is_grasped", "contact", "touched"):
        if key in flat:
            out["grasped"] = np.int8(1 if flat[key] else 0)
            break
    # env fallbacks
    if env is not None and np.isnan(out["object_z"]):
        try:
            states = getattr(env, "object_states_dict", None) or getattr(getattr(env, "env", None), "object_states_dict", {})
            for name, st in (states or {}).items():
                if hasattr(st, "get_geom_state"):
                    pos = np.asarray(st.get_geom_state()[0], dtype=np.float32)
                    out["object_z"] = np.float32(pos[2]); out["object_xy"] = pos[:2].astype(np.float32)
                    break
                if hasattr(st, "pose"):
                    pos = np.asarray(st.pose[:3], dtype=np.float32)
                    out["object_z"] = np.float32(pos[2]); out["object_xy"] = pos[:2].astype(np.float32)
                    break
        except Exception:
            pass
    return out
