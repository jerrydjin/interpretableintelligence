# === v9 HOTFIX for a LIVE v8 runtime (policy loaded; V8 = v8.Ctx after setup) ===
# 1) Interrupt the running v8 probe/capture cell first (Runtime > Interrupt execution). Its probe.json is kept.
# 2) Paste this into a new cell and run it. Optional first: %env GROOT_N_ENVS=8  /  %env GROOT_STOP_AFTER_PROBE=1
# Switches to the parallel env pool, probes 614/1028 to 20 eps each in a NEW run dir
# /content/physint/groot/colab-l4-libero-plus-v9, then (unless GROOT_STOP_AFTER_PROBE=1) captures 100 eps.
import os, sys, json, time, shutil
from pathlib import Path
import torch

_V8_SRC = r'''"""physint in-process GR00T rollouts for Colab T4 (v8).

v8 vs v7 (see v8/V8_NOTES.md):
  * RoPE repair: v7's low-RAM loader rebuilt Qwen3-VL text RoPE inv_freq with theta=1e4 instead
    of 5e6 (staticmethod `__func__` AttributeError swallowed -> wrong fallback). setup() now
    recomputes every rotary buffer from its own class/config, verifies each against that
    module's own config rope_theta (text 5e6, vision its own), and raises on mismatch.
  * per-episode timing breakdown (obs+preprocess / policy / post / env) and predict-call count
    (asserts the 16-step action queue is used: n_predict == ceil(n_steps / queue)).
  * GROOT_AMP=fp16|bf16_autocast selects precision (default fp16, as v7).
  * Shared obs builder `build_policy_obs` used by rollouts AND the v8 diag cell.
Hook, capture layout, failure detector, seeds and probe/capture logic are unchanged from v7.

Runs smoke / probe / capture inside the notebook process with the policy that is
already on the GPU. Nothing shells out to `lerobot-eval` (that reloaded the 12.6 GB
fp32 checkpoint on CPU and hit the stale GrootConfig.base_model_path).

Observation pipeline mirrors lerobot 0.7.0 `lerobot_eval.rollout` for one env:
  LiberoEnv raw obs -> libero.canonical_observation -> batch dim ->
  prepare_canonical_observation -> obs["task"] -> env_preprocessor (identity for LIBERO)
  -> policy preprocessor (checkpoint's policy_preprocessor.json) -> policy.select_action
  -> policy postprocessor -> env_postprocessor -> env.step(action[0]).

Hook (Andrej, 2026-11-10): forward hook on the WHOLE DiT block output (residual stream)
  _groot_model.action_head.model.transformer_blocks.{LAYER}
The DiT runs once per flow-matching Euler step (num_inference_timesteps per action
chunk). One capture row-group = the FINAL denoising step of each chunk (the one whose
output becomes the executed actions). A chunk whose block call count != the number of
denoise steps raises (no silent mixing).
"""
from __future__ import annotations

import gc
import json
import os
import tarfile
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

V7_VERSION = "v8.0"  # name kept so manifest key "v7" stays stable for readers
V8_VERSION = V7_VERSION
LIBERO_CONFIG_KEYS = ("benchmark_root", "bddl_files", "init_states", "datasets", "assets")
MOKA = "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
CAMERA_MAP = {"agentview_image": "image", "robot0_eye_in_hand_image": "wrist_image"}
VLM_PROCESSOR_REPO = os.environ.get("GROOT_VLM_PROCESSOR_REPO", "Qwen/Qwen3-VL-2B-Instruct")
CLEAN_LANGUAGE = os.environ.get("GROOT_CLEAN_LANGUAGE", "1") == "1"
FAST_PREPROCESS = os.environ.get("GROOT_FAST_PREPROCESS", "1") == "1"
OBS_HW = 256  # checkpoint input_features are 256x256 (IPEC libero_10_no_noops)
FAIL_KINDS = ("drop", "empty_grasp", "stall", "timeout")


# --------------------------------------------------------------------------- config.yaml
def _read_yaml_kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if ":" in line and not line.lstrip().startswith("#"):
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    return out


def write_libero_config(plus_dir: str | Path, assets_dir: str | Path | None = None,
                        log: Callable[[str], None] = print) -> dict[str, str]:
    """Write all 5 keys LIBERO-plus asserts on (init_states was missing in v6).

    Writes to the file the already-imported libero module reads (libero.libero.config_file),
    and to $LIBERO_CONFIG_PATH/~/.libero for any subprocess.
    """
    plus_dir = Path(plus_dir)
    root = plus_dir / "libero" / "libero"
    targets = {Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero")) / "config.yaml"}
    try:
        import libero.libero as _ll  # noqa: PLC0415
        targets.add(Path(_ll.config_file))
    except Exception as e:  # libero not importable yet: only the default file
        log(f"[v8] libero not imported yet ({e!r}); writing default config path only")
    # LIBERO resolves scene/object XMLs relative to its own package (bddl_base_domain.custom_asset_dir
    # = <libero/libero>/assets), not via the config, so that is the only assets path that matters.
    assets_dir = Path(assets_dir) if assets_dir is not None else root / "assets"
    if assets_dir.resolve() != (root / "assets").resolve():
        log(f"[v8] WARNING assets {assets_dir} != package path {root / 'assets'} (LIBERO uses the latter)")
    vals = {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "datasets": str(plus_dir / "libero" / "datasets"),
        "assets": str(assets_dir),
    }
    text = "".join(f"{k}: {v}\n" for k, v in vals.items())
    for t in targets:
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(text)
    from libero.libero import get_libero_path  # noqa: PLC0415  (re-reads the file each call)

    got = {k: get_libero_path(k) for k in LIBERO_CONFIG_KEYS}
    bad = {k: (got[k], vals[k]) for k in LIBERO_CONFIG_KEYS if str(got[k]) != vals[k]}
    if bad:
        raise RuntimeError(f"libero config mismatch after write: {bad}")
    for k in ("bddl_files", "init_states"):
        if not Path(vals[k]).is_dir():
            raise FileNotFoundError(f"libero {k} dir missing: {vals[k]}")
    if not (Path(vals["assets"]) / "wall.xml").exists():
        log(f"[v8] WARNING assets has no wall.xml: {vals['assets']}")
    log(f"[v8] libero config ok -> {sorted(str(t) for t in targets)} {vals}")
    return vals


# --------------------------------------------------------------------------- RoPE repair (v8)
def _rope_theta(cfg) -> float:
    """Theta from this module's own config. No default: missing theta is a bug, not 10000."""
    if cfg is None:
        raise RuntimeError("RoPE module has no config; cannot determine rope_theta")
    rp = getattr(cfg, "rope_parameters", None)
    th = None
    if isinstance(rp, dict):
        th = rp.get("rope_theta")
    elif rp is not None:
        th = getattr(rp, "rope_theta", None)
    if th is None:
        th = getattr(cfg, "rope_theta", None)
    if th is None:
        raise RuntimeError(
            f"RoPE config {type(cfg).__name__} has no rope_theta (rope_parameters={rp!r})"
        )
    return float(th)


def _inv_freq_from_theta(theta: float, n: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    i = torch.arange(0, n, device=device, dtype=dtype)
    return 1.0 / (theta ** (i / n))


def _theta_from_inv_freq(inv_freq: torch.Tensor) -> float:
    x = inv_freq.detach().float().reshape(-1)
    n = x.numel()
    if n < 2:
        return float("nan")
    i = n - 1
    val = float(x[i])
    if val <= 0:
        return float("nan")
    return float(val ** (-n / i))


def verify_rotary_inv_freq(module, name: str = "", *, inv_freq=None, atol: float = 1e-5) -> float:
    """Raise RuntimeError unless inv_freq matches this module's own config rope_theta."""
    label = name or type(module).__name__
    inv = module.inv_freq if inv_freq is None else inv_freq
    if not isinstance(inv, torch.Tensor):
        raise RuntimeError(f"rope {label}: missing inv_freq")
    if inv.device.type == "meta":
        raise RuntimeError(f"rope {label}: inv_freq is still on the meta device")
    theta = _rope_theta(getattr(module, "config", None))
    expected = _inv_freq_from_theta(theta, int(inv.numel()), device=inv.device, dtype=torch.float32)
    max_diff = float((inv.detach().float() - expected).abs().max())
    if max_diff > atol:
        actual = _theta_from_inv_freq(inv)
        raise RuntimeError(
            f"rope {label}: inv_freq does not match config theta "
            f"(expected theta={theta}, actual theta≈{actual}, max |Δ inv_freq|={max_diff})"
        )
    return max_diff


def _reference_inv_freq(module) -> tuple[torch.Tensor, Any]:
    """Rebuild inv_freq from a fresh instance of this module's class + config. No silent fallback."""
    cfg = getattr(module, "config", None)
    label = type(module).__name__
    if cfg is None:
        raise RuntimeError(f"rope {label}: module has no config; cannot rebuild inv_freq")
    try:
        with torch.device("cpu"):
            ref = type(module)(cfg)
    except Exception as e:
        raise RuntimeError(
            f"rope {label}: failed to construct a reference RoPE from {type(cfg).__name__}: {e}"
        ) from e
    inv = getattr(ref, "inv_freq", None)
    if not isinstance(inv, torch.Tensor) or inv.device.type == "meta":
        got = None if inv is None else f"{type(inv).__name__} on {getattr(inv, 'device', None)}"
        raise RuntimeError(f"rope {label}: class init did not produce a real inv_freq (got {got})")
    inv_freq = inv.detach().float()
    verify_rotary_inv_freq(module, label, inv_freq=inv_freq)
    return inv_freq, getattr(ref, "attention_scaling", None)


def repair_rope(policy, log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """Recompute every RoPE inv_freq from its own class + config; replace if different. Idempotent.

    v7's loader left Qwen3-VL text RoPE at theta=1e4 (checkpoint config: 5e6), max |d inv_freq|
    ~0.187, i.e. wrong positions in all 16 backbone LLM layers.

    Repair of an already-loaded policy is kept; if the buffer still does not match that
    module's own config rope_theta after repair, this raises RuntimeError (no swallow).
    """
    rep: list[dict[str, Any]] = []
    for name, m in policy.named_modules():
        inv = getattr(m, "inv_freq", None)
        if not isinstance(inv, torch.Tensor) or inv.device.type == "meta" or getattr(m, "config", None) is None:
            continue
        ref, scaling = _reference_inv_freq(m)
        if tuple(ref.shape) != tuple(inv.shape):
            raise RuntimeError(
                f"rope {name}: shape {tuple(inv.shape)} != reference {tuple(ref.shape)} "
                f"(expected theta={_rope_theta(m.config)})"
            )
        diff = float((inv.detach().float().cpu() - ref).abs().max())
        if diff > 0:
            new = ref.to(device=inv.device, dtype=inv.dtype)
            m.inv_freq = torch.nn.Buffer(new, persistent=False)
            if isinstance(getattr(m, "original_inv_freq", None), torch.Tensor):
                m.original_inv_freq = torch.nn.Buffer(new.clone(), persistent=False)
            if scaling is not None:
                m.attention_scaling = scaling
        verify_rotary_inv_freq(m, name)
        rep.append({"module": name, "class": type(m).__name__, "theta": _rope_theta(m.config),
                    "max_abs_diff_before": diff, "repaired": diff > 0})
    log(f"[v8] rope check {json.dumps(rep)}")
    return rep


# --------------------------------------------------------------------------- hook target
def groot_dit(policy) -> torch.nn.Module:
    return policy._groot_model.action_head.model


def resolve_block(policy, layer: int) -> tuple[str, torch.nn.Module, dict[str, Any]]:
    """Return (qualified name, block module, info) for transformer_blocks[layer]."""
    dit = groot_dit(policy)
    blocks = dit.transformer_blocks
    n = len(blocks)
    if not 0 <= int(layer) < n:
        raise IndexError(f"LAYER={layer} out of range for {n} DiT blocks")
    layer = int(layer)
    name = f"_groot_model.action_head.model.transformer_blocks.{layer}"
    mods = dict(policy.named_modules())
    blk = blocks[layer]
    if mods.get(name) is not blk:
        raise RuntimeError(f"named_modules() does not map {name} to transformer_blocks[{layer}]")
    cfg = getattr(dit, "config", None)
    interleave = bool(getattr(cfg, "interleave_self_attention", False))
    alt = type(dit).__name__ == "AlternateVLDiT"
    every = getattr(dit, "attend_text_every_n_blocks", None)
    if (alt or interleave) and layer % 2 == 1:
        kind = "self_attn"
    elif alt:
        kind = "cross_attn_text" if layer % (2 * int(every)) == 0 else "cross_attn_image"
    else:
        kind = "cross_attn"
    attn1 = getattr(blk, "attn1", None)
    is_cross = bool(getattr(attn1, "is_cross_attention", kind != "self_attn"))
    heads = int(getattr(cfg, "num_attention_heads", 0) or 0)
    hdim = int(getattr(cfg, "attention_head_dim", 0) or 0)
    info = {
        "module": name,
        "block_index": layer,
        "n_blocks": n,
        "dit_class": type(dit).__name__,
        "attn_type": kind,
        "attn1_is_cross_attention": is_cross,
        "attend_text_every_n_blocks": every,
        "d_model": heads * hdim if heads and hdim else None,
        "hooked_output": "block output = residual stream after attn1 + ff (BasicTransformerBlock.forward return)",
    }
    if kind == "self_attn" and is_cross:
        raise RuntimeError(f"{name}: parity says self-attn but attn1.is_cross_attention=True")
    return name, blk, info


class FinalDenoiseCapture:
    """Hook the block; keep only the final denoising call of each action chunk."""

    def __init__(self, policy, layer: int):
        self.policy = policy
        self.name, self.block, self.info = resolve_block(policy, layer)
        ah = policy._groot_model.action_head
        self.n_denoise = int(ah.num_inference_timesteps)
        self.action_horizon = int(getattr(ah, "action_horizon", 0) or 0)
        self.step = 0
        self.chunks: list[tuple[int, torch.Tensor]] = []
        self._last: torch.Tensor | None = None
        self._calls = 0
        self._handle = None
        self._had_attr = False
        self._prev_attr = None

    def _hook(self, _mod, _inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        self._last = t.detach()
        self._calls += 1

    def install(self):
        self.remove()
        self._handle = self.block.register_forward_hook(self._hook)
        self._had_attr = "predict_action_chunk" in self.policy.__dict__
        self._prev_attr = self.policy.__dict__.get("predict_action_chunk")
        inner = self.policy.predict_action_chunk

        def predict_action_chunk(batch, *args, **kwargs):
            self._last, self._calls = None, 0
            out = inner(batch, *args, **kwargs)
            if self._last is None:
                raise RuntimeError(f"hook {self.name} did not fire during predict_action_chunk")
            if self._calls != self.n_denoise:
                raise RuntimeError(
                    f"hook {self.name} fired {self._calls}x, expected num_inference_timesteps={self.n_denoise}"
                )
            x = self._last
            if x.ndim == 3:
                x = x[0]  # batch=1 -> [n_tok, D]
            self.chunks.append((int(self.step), x.to(torch.float16).cpu()))
            self._last = None
            return out

        self.policy.predict_action_chunk = predict_action_chunk
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
            if self._had_attr:
                self.policy.predict_action_chunk = self._prev_attr
            else:
                self.policy.__dict__.pop("predict_action_chunk", None)

    def pop(self):
        out, self.chunks = self.chunks, []
        return out

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.remove()


def choose_token_indices(n_tok: int, k: int = 32) -> list[int]:
    """All tokens when n_tok <= k (Andrej), else k evenly spaced (deterministic)."""
    n_tok, k = int(n_tok), int(k)
    if n_tok <= k:
        return list(range(n_tok))
    idx = np.unique(np.round(np.linspace(0, n_tok - 1, k)).astype(np.int64)).tolist()
    j = 0
    while len(idx) < k:  # linspace rounding collisions (cannot happen for k < n_tok, kept for safety)
        if j not in idx:
            idx.append(j)
        j += 1
    return sorted(int(i) for i in idx)[:k]


# --------------------------------------------------------------------------- env
class OneTaskSuite:
    """Minimal suite for LiberoEnv: exposes .tasks and .get_task (vanilla moka smoke)."""

    def __init__(self, task):
        self.tasks = [task]

    def get_task(self, i):
        return self.tasks[i]


def plus_suite(name: str = "libero_10"):
    from libero.libero import benchmark  # noqa: PLC0415

    return benchmark.get_benchmark_dict()[name]()


def vanilla_moka_suite(suite=None) -> OneTaskSuite:
    """LIBERO-plus's libero_10 has no unperturbed moka task (2519 variants); build it from the
    base bddl + base pruned_init, which ship in the clone."""
    from libero.libero import get_libero_path  # noqa: PLC0415

    suite = suite or plus_suite()
    base = None
    for t in suite.tasks:
        if t.name == MOKA:
            return OneTaskSuite(t)
        if base is None and t.name.startswith(MOKA + "_"):
            base = t
    if base is None:
        raise RuntimeError("no moka task in suite")
    task = base._replace(name=MOKA, bddl_file=MOKA + ".bddl", init_states_file=MOKA + ".pruned_init")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    if not bddl.is_file() or not init.is_file():
        raise FileNotFoundError(f"vanilla moka files missing: {bddl} {init}")
    import libero.libero.envs.bddl_utils as bddl_utils  # noqa: PLC0415

    lang = bddl_utils.get_problem_info(str(bddl))["language_instruction"]
    lang = " ".join(lang) if isinstance(lang, (list, tuple)) else str(lang)
    return OneTaskSuite(task._replace(language=lang.strip().strip('"')))


def make_libero_env(suite, task_id: int):
    from lerobot.sim.integrations.libero import LiberoEnv  # noqa: PLC0415

    return LiberoEnv(
        task_suite=suite,
        task_id=int(task_id),
        task_suite_name="libero_10",  # 520-step cap, same as lerobot-eval --env.task=libero_10
        obs_type="pixels_agent_pos",
        observation_width=OBS_HW,
        observation_height=OBS_HW,
        camera_name_mapping=dict(CAMERA_MAP),
        init_states=True,
        is_libero_plus=True,
        control_freq=20,
        hard_reset=True,
    )


def _movable_objects(base) -> list[str]:
    names = list(getattr(base, "obj_of_interest", []) or [])
    objs = getattr(base, "objects_dict", {}) or {}
    ids = getattr(base, "obj_body_id", {}) or {}
    mov = [n for n in names if n in objs and n in ids]
    if not mov:  # fall back to every movable object in the scene
        mov = [n for n in objs if n in ids]
    return mov


def read_sim(env, raw_obs: dict, names: list[str] | None = None) -> dict[str, Any]:
    """Per-step state BEFORE the action of this step is applied."""
    rs = raw_obs["robot_state"]
    eef = np.asarray(rs["eef"]["pos"], np.float32).reshape(3)
    gq = np.asarray(rs["gripper"]["qpos"], np.float32).reshape(-1)
    base = env._env.env
    names = names if names is not None else _movable_objects(base)
    pos = np.full((len(names), 3), np.nan, np.float32)
    grasp = np.full((len(names),), -1, np.int8)
    gripper = base.robots[0].gripper
    for i, n in enumerate(names):
        try:
            pos[i] = base.sim.data.body_xpos[base.obj_body_id[n]]
        except Exception:
            pass
        try:
            grasp[i] = 1 if base._check_grasp(gripper=gripper, object_geoms=base.objects_dict[n]) else 0
        except Exception:
            pass
    return {"eef_pos": eef, "gripper_width": np.float32(np.abs(gq).sum()), "obj_pos": pos,
            "obj_grasped": grasp, "names": names}


# --------------------------------------------------------------------------- failure
def derive_failure(*, success: bool, object_z: np.ndarray, eef_pos: np.ndarray, gripper_width: np.ndarray,
                   grasped: np.ndarray | None = None, obj_xy: np.ndarray | None = None,
                   lift_z: float = 0.05, fall_delta: float = 0.03, close_w: float = 0.02,
                   grasp_near: float = 0.05, stall_n: int = 25, stall_eps: float = 0.002) -> dict[str, Any]:
    """v6 rules; object_z must be RELATIVE (object z minus its z at step 0), so lift_z means
    'lifted 5 cm'. fail_t is a policy step index; success -> (None, None)."""
    if success:
        return {"fail_t": None, "fail_kind": None, "details": {}}
    z = np.asarray(object_z, np.float64).reshape(-1)
    eef = np.asarray(eef_pos, np.float64).reshape(len(z), -1) if len(z) else np.zeros((0, 3))
    gw = np.asarray(gripper_width, np.float64).reshape(-1)
    T = min(len(z), len(gw), len(eef))
    if T <= 0:
        return {"fail_t": 0, "fail_kind": "timeout", "details": {"reason": "empty"}}
    z, eef, gw = z[:T], eef[:T], gw[:T]
    zf = np.nan_to_num(z, nan=0.0)
    g = None if grasped is None else np.asarray(grasped).reshape(-1)[:T]
    lifted = zf >= lift_z
    if lifted.any():
        first = int(np.argmax(lifted))
        peak_i = first + int(np.argmax(zf[first:]))
        drop_rel = zf[peak_i] - zf[peak_i:]
        hit = drop_rel >= fall_delta
        # a drop only counts if the gripper is no longer holding it
        if g is not None and (g >= 0).any():
            hit = hit & (g[peak_i:] != 1)
        if hit.any():
            t = peak_i + int(np.argmax(hit))
            return {"fail_t": t, "fail_kind": "drop",
                    "details": {"peak_z_rel": float(zf[peak_i]), "peak_i": peak_i, "z_rel_at_fail": float(zf[t])}}
    was_open = np.concatenate([[True], gw[:-1] > close_w])
    for t in np.flatnonzero((gw <= close_w) & was_open).tolist():
        if g is not None and g[t] >= 0:
            holding = bool(g[t] == 1)
        elif obj_xy is not None:
            oxy = np.asarray(obj_xy, np.float64).reshape(-1, 2)[:T]
            holding = bool(np.linalg.norm(oxy[t] - eef[t, :2]) <= grasp_near)
        else:
            holding = False
        if not holding:
            return {"fail_t": int(t), "fail_kind": "empty_grasp",
                    "details": {"gripper_width": float(gw[t]), "z_rel": float(zf[t])}}
    if T >= stall_n:
        eef_d = np.concatenate([[0.0], np.linalg.norm(np.diff(eef, axis=0), axis=1)])
        if obj_xy is not None:
            oxy = np.nan_to_num(np.asarray(obj_xy, np.float64).reshape(-1, 2)[:T])
            obj_d = np.concatenate([[0.0], np.linalg.norm(np.diff(oxy, axis=0), axis=1)])
        else:
            obj_d = np.concatenate([[0.0], np.abs(np.diff(zf))])
        for s in range(0, T - stall_n + 1):
            sl = slice(s + 1, s + stall_n)
            if eef_d[sl].max() < stall_eps and obj_d[sl].max() < stall_eps:
                return {"fail_t": int(s + stall_n - 1), "fail_kind": "stall", "details": {"start": s}}
    return {"fail_t": int(T - 1), "fail_kind": "timeout", "details": {"n_steps": T}}


# --------------------------------------------------------------------------- episode
class NonFiniteAction(RuntimeError):
    pass


@dataclass
class Episode:
    success: bool
    n_steps: int
    seconds: float
    actions: np.ndarray
    sim: dict[str, np.ndarray]
    chunks: list
    seed: int
    init_state_id: int
    task_name: str
    n_chunks: int = 0
    first_action: list = field(default_factory=list)
    language: str = ""
    language_suite: str = ""
    timing: dict = field(default_factory=dict)


def build_processors(policy, ckpt: str, device: str = "cuda"):
    from lerobot.envs.configs import LiberoEnvConfig  # noqa: PLC0415
    from lerobot.policies import make_pre_post_processors  # noqa: PLC0415

    overrides = {
        "device_processor": {"device": device},
        "rename_observations_processor": {"rename_map": {}},
        # nvidia/Cosmos-Reason2-2B is gated (401 without an accepted licence + token). Its tokenizer,
        # chat template, image/video preprocessor configs are byte-identical (same git blob ids) to the
        # ungated Apache-2.0 Qwen/Qwen3-VL-2B-Instruct, so load the text/image processor from there.
        "groot_n1_7_vlm_encode_v1": {"device": device, "model_name": VLM_PROCESSOR_REPO},
    }
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=ckpt,
                                         preprocessor_overrides=overrides)
    vlm = [st for st in pre.steps if type(st).__name__ == "GrootN17VLMEncodeStep" or hasattr(st, "model_name")]
    if not vlm or any(getattr(st, "model_name", None) != VLM_PROCESSOR_REPO for st in vlm):
        raise RuntimeError(f"vlm encode step override failed: {[getattr(st, 'model_name', None) for st in vlm]}")
    env_pre, env_post = LiberoEnvConfig(camera_name_mapping=dict(CAMERA_MAP)).get_env_processors()
    return pre, post, env_pre, env_post


def build_policy_obs(canon: dict, task: str, procs) -> dict:
    """canonical (HWC uint8 images already upright, state float32) -> policy-ready batch.

    Same path as lerobot_eval.rollout: prepare_canonical_observation -> obs["task"] ->
    env_preprocessor -> policy preprocessor. Used by run_episode and the v8 diag cell.
    """
    from lerobot.envs.utils import prepare_canonical_observation  # noqa: PLC0415

    pre, _post, env_pre, _env_post = procs
    obs = prepare_canonical_observation({k: np.asarray(v)[None] for k, v in canon.items()})
    obs["task"] = [task]
    obs = env_pre(obs)
    return pre(obs)


def postprocess_action(a, procs):
    from lerobot.utils.constants import ACTION  # noqa: PLC0415

    _pre, post, _env_pre, env_post = procs
    a = post(a)
    return env_post({ACTION: a})[ACTION]


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def check_obs_keys(policy, obs_keys) -> None:
    want = set((policy.config.input_features or {}).keys())
    have = set(obs_keys)
    if not want <= have:
        raise RuntimeError(f"env obs keys {sorted(have)} miss policy inputs {sorted(want - have)}")


def run_episode(policy, procs, env, *, seed: int, init_state_id: int, capture: FinalDenoiseCapture | None = None,
                max_steps: int | None = None, log: Callable[[str], None] = print) -> Episode:
    from lerobot.processor import policy_output_to_transition, transition_to_policy_action  # noqa: PLC0415
    from lerobot.sim.integrations.libero import canonical_observation  # noqa: PLC0415

    pre, post, env_pre, env_post = procs
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    policy.reset()
    pre.reset()
    post.reset()
    if capture is not None:
        capture.chunks = []
    env.init_state_id = int(init_state_id)
    raw, _ = env.reset(seed=int(seed))
    if not hasattr(env, "_v7_lang_suite"):
        env._v7_lang_suite = env.task_description
    lang_suite = env._v7_lang_suite
    if CLEAN_LANGUAGE:
        # LIBERO-plus builds Task.language from the FILE NAME for non-language variants, e.g.
        # "put both moka pots on the stove view 0 0 100 0 0 initstate 229". The env itself loads the
        # base bddl, whose (:language ...) is the real instruction; feed that to the policy.
        li = getattr(env._env, "language_instruction", None)
        if isinstance(li, (list, tuple)):
            li = " ".join(li)
        if li:
            env.task_description = str(li).strip().strip('"')
    names = _movable_objects(env._env.env)
    T = int(max_steps or os.environ.get("GROOT_MAX_STEPS") or env._max_episode_steps)  # env var = debug only
    acts, eefs, gws, objp, objg = [], [], [], [], []
    success = False
    first = []
    tm = {"obs_pre_s": 0.0, "policy_s": 0.0, "post_s": 0.0, "env_s": 0.0, "sim_read_s": 0.0}
    n_predict = 0
    t0 = time.time()
    for t in range(T):
        ta = time.perf_counter()
        s = read_sim(env, raw, names)
        eefs.append(s["eef_pos"]); gws.append(s["gripper_width"]); objp.append(s["obj_pos"]); objg.append(s["obj_grasped"])
        tb = time.perf_counter(); tm["sim_read_s"] += tb - ta
        # The policy only reads the observation when its action queue is empty (once per chunk).
        # Skipping the (CPU-heavy Qwen image/text) preprocessing on the other steps changes nothing
        # for this checkpoint (use_relative_actions=False); FAST_PREPROCESS=0 restores per-step parity.
        fast = FAST_PREPROCESS and not getattr(policy.config, "use_relative_actions", False)
        if t == 0 or not fast or len(getattr(policy, "_action_queue", ())) == 0:
            canon = canonical_observation(raw)
            if t == 0:
                check_obs_keys(policy, canon.keys())
            obs = build_policy_obs(canon, env.task_description, procs)
        if capture is not None:
            capture.step = t
        if len(policy._action_queue) == 0:
            n_predict += 1
        _sync(); tc = time.perf_counter(); tm["obs_pre_s"] += tc - tb
        with torch.inference_mode():
            a = transition_to_policy_action(policy_output_to_transition(policy.select_action(obs)))
        _sync(); td = time.perf_counter(); tm["policy_s"] += td - tc
        a = postprocess_action(a, procs)
        a_np = a.to("cpu").float().numpy()
        tm["post_s"] += time.perf_counter() - td
        if a_np.ndim != 2 or a_np.shape[0] != 1:
            raise RuntimeError(f"unexpected action shape {a_np.shape}")
        if not np.isfinite(a_np).all():
            raise NonFiniteAction(f"non-finite action at step {t}: {a_np}")
        if t == 0:
            first = a_np[0].tolist()
        acts.append(a_np[0].astype(np.float32))
        te = time.perf_counter()
        raw, _r, term, trunc, info = env.step(a_np[0].astype(np.float32))
        tm["env_s"] += time.perf_counter() - te
        if bool(info.get("is_success", False)):
            success = True
        if term or trunc or success:
            break
    n = len(acts)
    q = int(getattr(policy, "_action_queue_steps", 16))
    expect = -(-n // q)
    if n_predict != expect:
        log(f"[v8] WARNING predict calls {n_predict} != ceil({n}/{q})={expect}: action queue not used as expected")
    timing = {k: round(v, 2) for k, v in tm.items()}
    timing.update({"n_predict": n_predict, "expected_predict": expect, "queue": q,
                   "policy_s_per_predict": round(tm["policy_s"] / max(1, n_predict), 3),
                   "env_s_per_step": round(tm["env_s"] / max(1, n), 4)})
    objp_a = np.asarray(objp, np.float32).reshape(n, len(names), 3)
    objg_a = np.asarray(objg, np.int8).reshape(n, len(names))
    eef_a = np.asarray(eefs, np.float32).reshape(n, 3)
    # active object per step = movable object of interest nearest the EEF
    if len(names):
        d = np.linalg.norm(np.nan_to_num(objp_a, nan=1e3) - eef_a[:, None, :], axis=-1)
        active = d.argmin(axis=1).astype(np.int16)
        rows = np.arange(n)
        az = objp_a[rows, active, 2]
        z0 = objp_a[0, :, 2]
        object_z_rel = az - z0[active]
        object_xy = objp_a[rows, active, :2]
        grasped = objg_a[rows, active]
    else:
        active = np.full((n,), -1, np.int16)
        az = object_z_rel = np.full((n,), np.nan, np.float32)
        object_xy = np.full((n, 2), np.nan, np.float32)
        grasped = np.full((n,), -1, np.int8)
    sim = {
        "object_z": az.astype(np.float32),
        "object_z_rel": object_z_rel.astype(np.float32),
        "eef_pos": eef_a,
        "gripper_width": np.asarray(gws, np.float32),
        "object_xy": object_xy.astype(np.float32),
        "grasped": grasped.astype(np.int8),
        "step": np.arange(n, dtype=np.int32),
        "active_obj": active,
        "obj_names": np.asarray(names, dtype=str),
        "obj_pos": objp_a,
        "obj_grasped": objg_a,
    }
    chunks = capture.pop() if capture is not None else []
    return Episode(success=success, n_steps=n, seconds=round(time.time() - t0, 2),
                   actions=np.asarray(acts, np.float32).reshape(n, -1), sim=sim, chunks=chunks,
                   seed=int(seed), init_state_id=int(init_state_id), task_name=env.task,
                   n_chunks=len(chunks), first_action=first, language=env.task_description,
                   language_suite=lang_suite, timing=timing)


# --------------------------------------------------------------------------- run context + IO
@dataclass
class Ctx:
    run: Path
    run_id: str
    policy: Any
    ckpt: str
    layer: int
    plus_dir: Path
    snapshot_tar: Path = Path("/content/physint_snapshot.tar")
    tokens_per_step: int = 32
    probe_ids: tuple = ((613, 614), (1027, 1028))  # (0-based index, LIBERO-plus id)
    probe_n: int = int(os.environ.get("GROOT_PROBE_N", "10"))
    capture_n: int = int(os.environ.get("GROOT_CAPTURE_N", "100"))
    snapshot_every: int = 10
    started_at: float = field(default_factory=time.time)
    soft_cap_h: float = 10.5
    procs: Any = None
    amp_mode: str = os.environ.get("GROOT_AMP", "fp16")  # fp16 | bf16_autocast
    _log: Callable[[str], None] | None = None

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
        print(line, flush=True)
        try:
            with open(self.run / "progress.log", "a") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def hours_left(self) -> float:
        return self.soft_cap_h - (time.time() - self.started_at) / 3600.0


def atomic_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def load_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def write_snapshot(ctx: Ctx, reason: str) -> None:
    """Uncompressed tar of the whole run dir (small files + shards), atomic replace."""
    ctx.snapshot_tar.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="physint_snapshot_", suffix=".tar", dir=str(ctx.snapshot_tar.parent))
    os.close(fd)
    try:
        with tarfile.open(tmp, "w") as tf:
            tf.add(str(ctx.run), arcname=f"physint/groot/{ctx.run_id}")
        os.replace(tmp, ctx.snapshot_tar)
        ctx.log(f"SNAPSHOT {ctx.snapshot_tar} reason={reason} bytes={ctx.snapshot_tar.stat().st_size}")
    except Exception as e:
        Path(tmp).unlink(missing_ok=True)
        ctx.log(f"WARN snapshot failed ({reason}): {e!r}")


def setup(ctx: Ctx, *, device: str = "cuda") -> dict[str, Any]:
    """Fix config.yaml, pick precision, build processors, resolve hook. Idempotent."""
    for sub in ("acts", "actions", "sim", "smoke", "probe", "logs"):
        (ctx.run / sub).mkdir(parents=True, exist_ok=True)
    cfgvals = write_libero_config(ctx.plus_dir, log=ctx.log)
    pol = ctx.policy
    rope = repair_rope(pol, log=ctx.log)
    q = int(pol._action_queue_steps)
    if q != int(pol.config.n_action_steps) or not hasattr(pol, "_action_queue"):
        raise RuntimeError(f"action queue {q} != n_action_steps {pol.config.n_action_steps}")
    # Weights are fp16 (low-RAM stream). Pure fp16 is native on T4; GR00T casts float inputs to
    # the param dtype in prepare_input. bf16 autocast (checkpoint default use_bf16=True) is the
    # fallback if fp16 gives non-finite actions.
    pol.config.use_bf16 = ctx.amp_mode == "bf16_autocast"
    if ctx.procs is None:
        ctx.procs = build_processors(pol, ctx.ckpt, device=device)
    name, _blk, info = resolve_block(pol, ctx.layer)
    ah = pol._groot_model.action_head
    info.update({
        "capture": "final_denoise_step_per_chunk",
        "num_inference_timesteps": int(ah.num_inference_timesteps),
        "denoise_step_index": int(ah.num_inference_timesteps) - 1,
        "action_horizon": int(ah.action_horizon),
        "action_queue_steps": int(pol._action_queue_steps),
    })
    rep = {"libero_config": cfgvals, "hook": info, "amp_mode": ctx.amp_mode, "v7": V7_VERSION, "rope": rope}
    atomic_json(ctx.run / "smoke" / "setup_v8.json", rep)
    ctx.log(f"[v8] hook {name} attn={info['attn_type']} denoise_steps={info['num_inference_timesteps']} "
            f"queue={info['action_queue_steps']} amp={ctx.amp_mode}")
    return rep


def _episode_with_amp_fallback(ctx: Ctx, env, **kw) -> Episode:
    try:
        return run_episode(ctx.policy, ctx.procs, env, log=ctx.log, **kw)
    except NonFiniteAction as e:
        if ctx.amp_mode != "fp16":
            raise
        ctx.log(f"[v8] {e} -> switching to bf16 autocast and retrying this episode")
        ctx.amp_mode = "bf16_autocast"
        ctx.policy.config.use_bf16 = True
        return run_episode(ctx.policy, ctx.procs, env, log=ctx.log, **kw)


def smoke(ctx: Ctx) -> dict[str, Any]:
    """1 full episode of vanilla libero_10 moka-pots, hooked (hook must fire every chunk)."""
    path = ctx.run / "smoke" / "smoke.json"
    prev = load_json(path, {}) or {}
    if prev.get("ok"):
        print(f"SMOKE seconds_per_episode={prev['seconds_per_episode']} success={prev['success']} "
              f"n_steps={prev['n_steps']} (cached)", flush=True)
        ctx.amp_mode = prev.get("amp_mode", ctx.amp_mode)
        ctx.policy.config.use_bf16 = ctx.amp_mode == "bf16_autocast"
        return prev
    suite = vanilla_moka_suite()
    t_env = time.time()
    env = make_libero_env(suite, 0)
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    rec: dict[str, Any] = {"task": MOKA, "suite": "vanilla (LIBERO-plus base bddl + pruned_init)", "seed": 0,
                           "init_state_id": 0}
    try:
        with FinalDenoiseCapture(ctx.policy, ctx.layer) as cap:
            ep = _episode_with_amp_fallback(ctx, env, seed=0, init_state_id=0, capture=cap)
        rec.update({
            "ok": True, "success": ep.success, "n_steps": ep.n_steps, "seconds_per_episode": ep.seconds,
            "env_build_plus_episode_s": round(time.time() - t_env, 2),
            "seconds_per_policy_step": round(ep.seconds / max(1, ep.n_steps), 4),
            "n_chunks": ep.n_chunks, "seconds_per_chunk_incl_sim": round(ep.seconds / max(1, ep.n_chunks), 3),
            "chunk_act_shape": list(ep.chunks[0][1].shape) if ep.chunks else None,
            "chunk_steps": [c[0] for c in ep.chunks][:8],
            "first_action": ep.first_action, "amp_mode": ctx.amp_mode,
            "language": ep.language, "language_suite": ep.language_suite,
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None,
            "obj_names": ep.sim["obj_names"].tolist(),
            "timing": ep.timing,
        })
        if not ep.chunks:
            raise RuntimeError("hook captured no chunks")
    except Exception as e:
        rec.update({"ok": False, "error": repr(e), "traceback": traceback.format_exc()[-4000:]})
        atomic_json(path, rec)
        write_snapshot(ctx, "smoke_failed")
        raise
    finally:
        env.close()
    atomic_json(path, rec)
    print(f"SMOKE seconds_per_episode={rec['seconds_per_episode']} success={rec['success']} n_steps={rec['n_steps']} "
          f"timing={json.dumps(rec['timing'])}", flush=True)
    ctx.log(f"smoke {json.dumps({k: v for k, v in rec.items() if k != 'traceback'})}")
    write_snapshot(ctx, "smoke")
    return rec


def probe(ctx: Ctx) -> dict[str, Any]:
    """~probe_n eps per Plus id (init_state_id k, seed 10000+k); pick success rate nearest 0.5."""
    path = ctx.run / "probe" / "probe.json"
    pr = load_json(path, {"instances": {}}) or {"instances": {}}
    suite = plus_suite()
    for index, pid in ctx.probe_ids:
        inst = pr["instances"].setdefault(str(index), {"index": index, "id": pid, "name": suite.tasks[index].name,
                                                       "results": []})
        if len(inst["results"]) >= ctx.probe_n:
            continue
        env = make_libero_env(suite, index)
        try:
            for k in range(len(inst["results"]), ctx.probe_n):
                if ctx.hours_left() < 0.5:
                    ctx.log("probe stop: soft cap")
                    break
                ep = _episode_with_amp_fallback(ctx, env, seed=10000 + k, init_state_id=k)
                inst["results"].append({"ep": k, "seed": 10000 + k, "init_state_id": k, "success": ep.success,
                                        "n_steps": ep.n_steps, "s": ep.seconds, "timing": ep.timing})
                inst["language"], inst["language_suite"] = ep.language, ep.language_suite
                inst["successes"] = sum(r["success"] for r in inst["results"])
                inst["episodes_done"] = len(inst["results"])
                inst["sr"] = inst["successes"] / inst["episodes_done"]
                atomic_json(path, pr)
                print(f"EPISODE probe id={pid} ep={k} success={ep.success} n_steps={ep.n_steps} "
                      f"seconds_per_episode={ep.seconds} sr={inst['sr']:.2f}", flush=True)
        finally:
            env.close()
        write_snapshot(ctx, f"probe_{pid}")
    cands = [v for v in pr["instances"].values() if v.get("episodes_done")]
    order = {idx: i for i, (idx, _) in enumerate(ctx.probe_ids)}
    best = min(cands, key=lambda v: (abs(v["sr"] - 0.5), order.get(int(v["index"]), 99)))
    chosen = {"index": int(best["index"]), "id": int(best["id"]), "name": best["name"], "sr": best["sr"],
              "episodes_done": best["episodes_done"], "reason": "closest_to_0.5 (tie -> primary 614)"}
    pr["chosen"] = chosen
    atomic_json(path, pr)
    atomic_json(ctx.run / "probe" / "chosen.json", chosen)
    ctx.log(f"CHOSEN {chosen}")
    write_snapshot(ctx, "probe")
    return chosen


def _save_episode(ctx: Ctx, ep_idx: int, ep: Episode, manifest: dict, fail: dict) -> int:
    from safetensors.numpy import load_file, save_file  # noqa: PLC0415

    # sim npz (fields Michael's PR #11 reader consumes: eef_pos, object_z, gripper_width, success)
    sim = dict(ep.sim)
    np.savez_compressed(
        ctx.run / "sim" / f"ep-{ep_idx:05d}.npz", **sim,
        success=np.array([int(ep.success)], np.int8),
        fail_t=np.array([-1 if fail["fail_t"] is None else int(fail["fail_t"])], np.int32),
        fail_kind=np.array([fail["fail_kind"] or ""], dtype=str),
        seed=np.array([ep.seed], np.int64), init_state_id=np.array([ep.init_state_id], np.int32),
    )
    # actions
    ad = ctx.run / "actions"
    save_file({"episode_idx": np.full((ep.n_steps,), ep_idx, np.int32), "step": np.arange(ep.n_steps, dtype=np.int32),
               "action": np.ascontiguousarray(ep.actions)}, str(ad / f"shard-{ep_idx:05d}.safetensors"))
    parts = [load_file(str(p)) for p in sorted(ad.glob("shard-*.safetensors"))]
    save_file({k: np.concatenate([p[k] for p in parts]) for k in ("episode_idx", "step", "action")},
              str(ctx.run / "actions.safetensors"))
    # acts: one row-group per chunk (final denoise step), rows = token subsample + mean (tok=-1)
    xs, steps, toks = [], [], []
    for step, x in ep.chunks:
        n_tok, d = int(x.shape[0]), int(x.shape[1])
        if manifest.get("token_indices") is None:
            manifest["n_tok"] = n_tok
            manifest["d_model"] = d
            manifest["token_indices"] = choose_token_indices(n_tok, ctx.tokens_per_step)
            manifest["token_subsample"] = ("all_tokens" if n_tok <= ctx.tokens_per_step
                                           else f"fixed_even_linspace_{ctx.tokens_per_step}_of_{n_tok}")
        if n_tok != manifest["n_tok"] or d != manifest["d_model"]:
            raise RuntimeError(f"activation shape changed: {(n_tok, d)} vs {(manifest['n_tok'], manifest['d_model'])}")
        idx = manifest["token_indices"]
        xf = x.float()
        rows = torch.cat([xf[idx], xf.mean(0, keepdim=True)], 0).to(torch.float16).numpy()
        xs.append(rows)
        steps.append(np.full((rows.shape[0],), step, np.int32))
        toks.append(np.asarray(list(idx) + [-1], np.int16))
    if xs:
        save_file({"acts": np.ascontiguousarray(np.concatenate(xs)),
                   "episode_idx": np.full((sum(len(s) for s in steps),), ep_idx, np.int32),
                   "step": np.concatenate(steps), "tok": np.concatenate(toks)},
                  str(ctx.run / "acts" / f"shard-{ep_idx:05d}.safetensors"))
    return int(sum(len(s) for s in steps))


def capture(ctx: Ctx, chosen: dict | None = None) -> dict[str, Any]:
    chosen = chosen or load_json(ctx.run / "probe" / "chosen.json")
    if not chosen:
        raise RuntimeError("no probe/chosen.json; run probe first")
    index, pid = int(chosen["index"]), int(chosen["id"])
    mpath = ctx.run / "manifest.json"
    man = load_json(mpath, {}) or {}
    if man.get("task_index") not in (None, index):
        raise RuntimeError(f"manifest task_index {man.get('task_index')} != chosen {index}; use a new RUN dir")
    _n, _b, hook_info = resolve_block(ctx.policy, ctx.layer)
    ah = ctx.policy._groot_model.action_head
    hook_info.update({"capture": "final_denoise_step_per_chunk",
                      "num_inference_timesteps": int(ah.num_inference_timesteps),
                      "denoise_step_index": int(ah.num_inference_timesteps) - 1})
    man.update({
        "format": "rollcap-v0", "v7": V7_VERSION, "run_id": ctx.run_id,
        "model": {"adapter": "groot", "name": ctx.ckpt, "family": "groot"},
        "layer": ctx.layer, "hook_target": hook_info["module"], "hook": hook_info,
        "task_index": index, "task_id": pid, "task_name": chosen.get("name"), "probe_chosen": chosen,
        "dtype": "fp16", "amp_mode": ctx.amp_mode, "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "capture_target_rollouts": ctx.capture_n, "tokens_per_step": ctx.tokens_per_step,
        "keep_mean_pool": True, "mean_pool_tok": -1,
        "step_semantics": ("acts rows exist only at chunk-start policy steps: `step` = env policy step at which "
                           "the action chunk was predicted; that chunk is executed for steps "
                           "[step, step + action_queue_steps). fail_t uses the same policy-step unit."),
        "action_queue_steps": int(ctx.policy._action_queue_steps), "action_horizon": int(ah.action_horizon),
        "token_layout": ("DiT sequence = [state token (tok 0)] + action_horizon action tokens "
                         "(tok 1..action_horizon = horizon positions 0..action_horizon-1); "
                         "only the first action_queue_steps positions are executed"),
        "language_rule": ("policy instruction = (:language) of the bddl the env actually loads "
                          "(LIBERO-plus Task.language is filename-derived junk for view/initstate variants)"
                          if CLEAN_LANGUAGE else "LIBERO-plus Task.language (unmodified)"),
        "episode_seed_rule": "seed = 20000 + episode_idx; init_state_id = episode_idx % n_init_states",
        "failure_detector": "physint_inproc_v8.derive_failure on object_z_rel (z minus step-0 z of the "
                            "active object) + contact-based grasped; active object = movable "
                            "obj_of_interest nearest the EEF at each step",
    })
    man.setdefault("episodes", [])
    man.setdefault("token_indices", None)
    atomic_json(mpath, man)
    done = {int(e["episode_idx"]) for e in man["episodes"]}
    suite = plus_suite()
    env = make_libero_env(suite, index)
    n_init = None
    n_new = 0
    try:
        with FinalDenoiseCapture(ctx.policy, ctx.layer) as cap:
            for ep_idx in range(ctx.capture_n):
                if ep_idx in done:
                    continue
                if ctx.hours_left() < 0.35:
                    ctx.log(f"capture stop at ep={ep_idx}: soft cap")
                    break
                if n_init is None:
                    n_init = len(env._init_states) if env._init_states is not None else 1
                    man["n_init_states"] = n_init
                seed, isid = 20000 + ep_idx, ep_idx % n_init
                try:
                    ep = _episode_with_amp_fallback(ctx, env, seed=seed, init_state_id=isid, capture=cap)
                except NonFiniteAction:
                    raise
                except Exception as e:  # one bad episode must not kill the run
                    ctx.log(f"ep={ep_idx} ERROR {e!r}\n{traceback.format_exc()[-1500:]}")
                    cap.chunks = []
                    try:
                        env.close()
                    except Exception:
                        pass
                    continue
                fr = derive_failure(success=ep.success, object_z=ep.sim["object_z_rel"], eef_pos=ep.sim["eef_pos"],
                                    gripper_width=ep.sim["gripper_width"], grasped=ep.sim["grasped"],
                                    obj_xy=ep.sim["object_xy"])
                n_rows = _save_episode(ctx, ep_idx, ep, man, fr)
                row = {"episode_idx": ep_idx, "episode_id": f"ep-{ep_idx:05d}", "task_id": str(pid), "seed": seed,
                       "init_state_id": isid, "success": bool(ep.success), "fail_t": fr["fail_t"],
                       "fail_kind": fr["fail_kind"], "n_steps": ep.n_steps, "n_chunks": ep.n_chunks,
                       "n_act_rows": n_rows, "seconds": ep.seconds, "status": "done",
                       "failure_details": fr["details"], "language": ep.language}
                man["episodes"] = sorted([e for e in man["episodes"] if int(e["episode_idx"]) != ep_idx] + [row],
                                         key=lambda e: int(e["episode_idx"]))
                man["seeds"] = [int(e["seed"]) for e in man["episodes"]]
                man["amp_mode"] = ctx.amp_mode
                atomic_json(mpath, man)
                with open(ctx.run / "episodes.jsonl", "a") as f:
                    f.write(json.dumps(row, default=str) + "\n")
                n_new += 1
                print(f"EPISODE ep={ep_idx} task_id={pid} seed={seed} success={ep.success} fail_t={fr['fail_t']} "
                      f"fail_kind={fr['fail_kind']} n_steps={ep.n_steps} seconds_per_episode={ep.seconds}", flush=True)
                if n_new % ctx.snapshot_every == 0:
                    write_snapshot(ctx, f"capture_n{len(man['episodes'])}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        env.close()
    eps = man["episodes"]
    summ = {"n_done": len(eps), "n_success": sum(e["success"] for e in eps),
            "fail_kinds": {k: sum(e["fail_kind"] == k for e in eps) for k in FAIL_KINDS},
            "mean_seconds_per_episode": round(float(np.mean([e["seconds"] for e in eps])), 2) if eps else None}
    man["summary"] = summ
    atomic_json(mpath, man)
    write_snapshot(ctx, f"capture_final_n{len(eps)}")
    ctx.log(f"capture summary {summ}")
    return summ
'''
_POT_EVENTS_SRC = r'''"""Per-pot manipulation events + failure label for 'put both moka pots on the stove' (LIBERO KITCHEN_SCENE8), numpy only.

ONE implementation shared by
  * the online detector in physint_inproc_v9.capture (Colab; written next to physint_inproc_v9.py as /content/pot_events.py), and
  * the offline relabel /workspace/runs/groot-1028-v9-n100/tools/derive_pot0_fail.py (rule 'pot0-v1'),
so the two cannot drift. tests/test_pot_events.py replays the 100 recorded traces of groot-1028-v9-n100 and requires the
labels in its episodes.jsonl exactly.

Why: v8.derive_failure judges a 'drop' on whichever movable object is nearest the EEF (the 'active' object). In task 1028 the
policy always moves pot 1 (obj_names[1] = moka_pot_2) first and pot 0 (obj_names[0] = moka_pot_1) second; in 37/40 v9 'drop'
episodes the active detector fired on pot 1 being set down on the stove (normal). Its label is kept as fail_kind_active.

Inputs (one row per env step; step == arange(T)): obj_pos [T,K,3], obj_grasped [T,K] (contact flag, kept only as a marker: it
misses most pot-0 handle grasps), gripper_width [T], eef_pos [T,3]. Units: env policy steps (same as acts 'step').

Thresholds (from the 100-episode capture; README 'pot-0 failure relabel'):
  Z_LIFT 0.020  lift = pot z more than 2 cm above its step-0 z (per-episode max is <= 0.011 or >= 0.028; nothing between)
  D_NEAR 0.100  ||eef - pot|| at every successful lift onset is 0.051-0.095
  G_OPEN 0.078 / G_CLOSE 0.075 (>= CLOSE_MIN 3 steps)  open gripper reads 0.078-0.081; pot handle grasps hold at 0.014-0.075
  SEP 0.015 / SEP_CONF 0.030  release by separation: d0 leaves the carry distance by > SEP and exceeds carry + SEP_CONF within
                              10 steps (else it is a re-grip and the carry distance is reset); waived in the last 10 steps.
                              Release by gripper: after the lift gw went < G_CLOSE and then back to >= G_OPEN; onset = start of
                              that monotone rise. The earlier of the two is the release step.
  REGION (box)  stove placement region, xy: both pots' final xy over the 56 successes, padded 1 cm (STOVE_REGIONS below).
                A release is OFF the stove if the pot rests outside REGION before its next lift (or at the episode end).
  Z_REST_HI 1.012  touchdown = pot z <= 1.012 inside REGION after the lift; place_t = first touchdown that stays in REGION
                   to the end (successful episodes terminate at touchdown with the pot still held).
  V_STALL 0.002 / N_STALL 10   held-at-timeout: first 10-step run moving < 2 mm/step outside REGION after the lift.
"""
from __future__ import annotations

from typing import Any

import numpy as np

RULE = "pot0-v1"
Z_LIFT, D_NEAR, G_OPEN, G_CLOSE, CLOSE_MIN, SEP, PAD = 0.020, 0.100, 0.078, 0.075, 3, 0.015, 0.010
Z_REST_HI, V_STALL, N_STALL = 1.012, 0.002, 10
SEP_CONF = 0.030
POT_NAMES = ("moka_pot_1", "moka_pot_2")  # obj_names order in the LIBERO KITCHEN_SCENE8 moka task; pot 0 = moka_pot_1
# xy box (lo, hi) in world coordinates; derived by stove_region_from_successes() on groot-1028-v9-n100 (rounded to 0.1 mm)
STOVE_REGIONS = {"KITCHEN_SCENE8": ((-0.1292, -0.2819), (0.0349, -0.1047))}
FAIL_KINDS = ("lifted_not_placed", "never_grasped", "pot0_placed_other_failure", "empty_grasp")


def region_for_task(task_name: str | None):
    """(lo, hi) arrays for a task name, or None if the scene has no measured stove region."""
    for scene, (lo, hi) in STOVE_REGIONS.items():
        if task_name and scene in task_name and "moka_pots_on_the_stove" in task_name:
            return np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    return None


def stove_region_from_successes(final_xy: np.ndarray, pad: float = PAD):
    """final_xy [N,2] = final xy of every pot in successful episodes -> padded bounding box (lo, hi)."""
    F = np.asarray(final_xy, np.float64).reshape(-1, 2)
    return F.min(0) - pad, F.max(0) + pad


def segs(mask, min_len=1):
    m = np.r_[0, np.asarray(mask).astype(int), 0]; d = np.diff(m)
    return [(a, b) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if b - a >= min_len]  # [a, b)


def _inreg(xy, lo, hi):
    return np.all((xy >= lo) & (xy <= hi), axis=-1)


def pot_events(obj_pos, obj_grasped, gripper_width, eef_pos, k: int, lo, hi) -> dict[str, Any]:
    """Events of pot k. Keys are unprefixed (lift_t, place_t, ...); see pot_event_fields()."""
    P_in = np.asarray(obj_pos)[:, k]; P = P_in.astype(np.float64); T = len(P)
    gw = np.asarray(gripper_width, np.float64).reshape(-1); eef = np.asarray(eef_pos, np.float64).reshape(T, 3)
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    z = P[:, 2] - P[0, 2]; d0 = np.linalg.norm(eef - P, axis=1)
    inreg = _inreg(P[:, :2], lo, hi)
    flag = np.flatnonzero(np.asarray(obj_grasped)[:, k] == 1); out = dict(flag_grasp_t=int(flag[0]) if len(flag) else None)
    near = np.flatnonzero(d0 < D_NEAR); out["first_near_t"] = int(near[0]) if len(near) else None
    closes = [a for a, b in segs(gw < G_CLOSE, CLOSE_MIN) if d0[a:b].min() < D_NEAR]  # close attempts near pot k
    out["close_attempts"] = [int(a) for a in closes]
    out["max_zrel"] = round(float(z.max()), 4)
    # ---- lift -> release events (a re-lift after a release starts a new event)
    lifts, releases = [], []
    t = 0
    while t < T:
        up = np.flatnonzero(z[t:] > Z_LIFT)
        if not len(up): break
        lt = t + int(up[0])
        carry = float(np.median(d0[lt:lt + 10]))
        ts, u0 = None, lt
        while True:  # separation must be confirmed (d0 > carry + SEP_CONF within 10 steps); otherwise it is a re-grip shift
            sep = np.flatnonzero(d0[u0:] > carry + SEP)
            if not len(sep): break
            c = u0 + int(sep[0])
            if d0[c:c + 10].max() > carry + SEP_CONF or c + 10 >= T: ts = c; break
            carry = float(np.median(d0[c:c + 10])); u0 = c + 1
        r_sep = None
        if ts is not None:
            back = np.flatnonzero(d0[lt:ts] <= carry + 0.005); r_sep = lt + int(back[-1]) if len(back) else lt
        # gripper-open release: after lift, gw had been < G_CLOSE and then rises to >= G_OPEN; onset = start of that monotone rise
        r_open = None; closed = np.flatnonzero(gw[lt:] < G_CLOSE)
        if len(closed):
            c0 = lt + int(closed[0]); op = np.flatnonzero(gw[c0:] >= G_OPEN)
            if len(op):
                r = c0 + int(op[0])
                while r - 1 > c0 and gw[r - 1] < gw[r] - 1e-4: r -= 1
                r_open = r
        cands = [x for x in (r_sep, r_open) if x is not None]
        if not cands:
            lifts.append(lt); releases.append(None); break
        rt = min(cands); ts = max(rt + 1, ts if ts is not None else rt + 1)
        carry = float(np.median(d0[lt:lt + 10]))
        lifts.append(lt); releases.append(rt)
        # next lift = re-grasped: d0 back to carry range AND pot rises > Z_LIFT above where it settled
        j = int(np.argmin(P[ts:ts + 15, 2])); rest_z = float(P[ts + j, 2]); nxt = None
        for u in range(ts + j + 1, T):
            if d0[u] <= carry + 0.005 and P[u, 2] - rest_z > Z_LIFT: nxt = u; break
        if nxt is None: break
        z = np.r_[z[:nxt], P[nxt:, 2] - rest_z]; t = nxt
    out["lift_t"] = lifts[0] if lifts else None
    out["lift_events"] = [int(x) for x in lifts]; out["release_events"] = [None if r is None else int(r) for r in releases]
    # OFF the stove = pot rests outside REGION before its next lift / at the end (released pots slide off or onto the stove edge)
    nxt_l = lifts[1:] + [T]
    off = [r is not None and not inreg[nl - 1] for r, nl in zip(releases, nxt_l)]
    rel_in = [r for r, o in zip(releases, off) if r is not None and not o]; rel_out = [r for r, o in zip(releases, off) if o]
    after = np.arange(T) > (lifts[0] if lifts else T)
    td = np.flatnonzero(inreg & (P[:, 2] <= Z_REST_HI) & after)
    kept = [t for t in td if inreg[t:].all()]
    out["place_t"] = int(kept[0]) if kept else None
    out["first_touchdown_in_region_t"] = int(td[0]) if len(td) else None
    out["release_in_region_t"] = int(rel_in[0]) if rel_in else None
    sp = np.r_[np.linalg.norm(np.diff(P, axis=0), axis=1), 0]
    st = [a for a, b in segs((sp < V_STALL) & ~inreg & after, N_STALL)]; out["stall_off_region_t"] = int(st[0]) if st else None
    out["n_drops"] = len(rel_out)
    out["first_drop_t"] = int(rel_out[0]) if rel_out else None
    out["drop_xyz"] = P_in[rel_out[0]].round(4).tolist() if rel_out else None
    out["final_in_region"] = bool(inreg[-1])
    out["held_at_end"] = bool(releases and releases[-1] is None)
    # grasp onset: last gripper-close onset (gw leaves fully open) within 40 steps before the first lift, eef near the pot
    g = None
    if lifts:
        lt = lifts[0]; a0 = max(0, lt - 40); op = np.flatnonzero(gw[a0:lt + 1] >= G_OPEN)
        if len(op) and a0 + int(op[-1]) + 1 <= lt:
            c = a0 + int(op[-1]) + 1
            if d0[c] < D_NEAR: g = c
    out["grasp_t"] = g
    return out


def all_pot_events(sim: dict, lo, hi) -> list[dict[str, Any]]:
    K = np.asarray(sim["obj_pos"]).shape[1]
    return [pot_events(sim["obj_pos"], sim["obj_grasped"], sim["gripper_width"], sim["eef_pos"], k, lo, hi) for k in range(K)]


def classify(success: bool, ev: list[dict], fail_kind_active: str | None, fail_t_active: int | None):
    """Episode label from pot events -> (fail_t, fail_kind, subkind, fail_pot). Rule pot0-v1: the failure is judged on pot 0
    (handled second); pot 1 (handled first) only decides empty_grasp (pot 1 never picked up -> keep the active detector's
    empty_grasp step) and the '_pot1_off_region' suffix. Successes -> (None, None, None, None)."""
    if success: return None, None, None, None
    o, o1 = ev[0], ev[1]
    if fail_kind_active == "empty_grasp":
        return fail_t_active, "empty_grasp", "pot1_empty_grasp_kept_v9", 1
    if o["lift_t"] is None:
        if o["close_attempts"]: return o["close_attempts"][0], "never_grasped", "close_attempt_failed", 0
        return o["first_near_t"], "never_grasped", "no_close_attempt_anchor_first_near", 0
    if o["first_drop_t"] is not None and (o["place_t"] is None or o["first_drop_t"] < o["place_t"]):
        return o["first_drop_t"], "lifted_not_placed", "released_off_stove", 0
    if o["place_t"] is not None:  # pot 0 reached the stove: not a pot-0 failure
        return None, "pot0_placed_other_failure", "pot0_on_stove_" + ("held_at_timeout" if o["held_at_end"] else "released") + \
            ("_pot1_off_region" if not o1["final_in_region"] else ""), None
    if o["stall_off_region_t"] is not None: return o["stall_off_region_t"], "lifted_not_placed", "held_until_timeout_stall_off_stove", 0
    return o["lift_t"], "lifted_not_placed", "held_until_timeout_anchor_lift", 0


def pot_event_fields(ev: list[dict], pots=(0,)) -> dict[str, Any]:
    """Flatten events to the manifest columns (pot0_lift_t, ...). Default pot 0 only, matching the relabel."""
    out = {}
    for k in pots:
        for key, v in ev[k].items():
            out[f"pot{k}_{key}"] = v
    if 0 in pots and len(ev) > 1:
        out["pot1_final_in_region"] = ev[1]["final_in_region"]
    return out


def detect(sim: dict, *, success: bool, task_name: str | None, names, fail_kind_active, fail_t_active,
           region=None) -> dict[str, Any] | None:
    """Online entry point. Returns {fail_t, fail_kind, fail_subkind, fail_pot, fail_rule, fields, pot1_events} or None when the
    task is not the two-moka-pot stove task, the pot names differ, or the trace has non-finite positions (caller keeps
    the active detector's label then)."""
    reg = region if region is not None else region_for_task(task_name)
    if reg is None or tuple(names) != POT_NAMES: return None
    if not np.isfinite(np.asarray(sim["obj_pos"], np.float64)).all(): return None
    ev = all_pot_events(sim, *reg)
    ft, fk, sub, pot = classify(success, ev, fail_kind_active, fail_t_active)
    return dict(fail_t=None if ft is None else int(ft), fail_kind=fk, fail_subkind=sub, fail_pot=pot,
                fail_rule=f"{RULE}: " + ("success" if success else sub), fields=pot_event_fields(ev), pot1_events=ev[1])
'''
_V9_SRC = r'''"""physint in-process GR00T rollouts, v9: N LIBERO envs in parallel CPU worker processes.

v9 vs v8 (see v9/V9_NOTES.md):
  * Env stepping runs in N spawned worker processes (default min(os.cpu_count()-1, 8), env
    GROOT_N_ENVS). Each worker owns its LiberoEnv (own MuJoCo/EGL context, same MUJOCO_GL as the
    notebook), resets it with v8's seeds and executes one whole 16-step action chunk per message,
    so the main process only talks to a worker once per chunk.
  * The main process keeps the policy on the GPU, runs the same preprocessing / predict / postprocess
    path as v8.run_episode, and keeps a pool: when an episode ends, that slot starts the next one.
  * Noise: v8 seeded the global RNG with torch.manual_seed(seed) per episode and GR00T drew one
    flow-matching noise sample per chunk from it. v9 gives every episode its own
    torch.Generator(seed) on the policy device and routes GR00T's sample_noise through it, so each
    episode sees the same noise stream it would see in the sequential v8 runner.
  * Predict: GROOT_BATCH_PREDICT=0 (default) = one predict call per env (bit-identical to v8's
    sequential runner); =1 = one batched predict per group of envs with identical input shapes,
    enabled only after verify_batched_predict passes (per-env max |diff| reported).
  * Probe: 20 episodes per probe task (GROOT_PROBE_N), pick nearest 50% with Wilson 95% CI.
Unchanged (imported from physint_inproc_v8): libero config, RoPE repair/verify (raises), processors,
build_policy_obs, postprocess_action, hook target transformer_blocks.{LAYER} final denoise step,
read_sim, _save_episode (PICKER.md / rollcap-v0 layout), seeds.
Failure label (capture): per-pot detector pot_events.detect (rule pot0-v1, shared with the offline relabel
tools/derive_pot0_fail.py); v8.derive_failure's active-object label is kept as fail_kind_active / fail_t_active.
"""
from __future__ import annotations

import gc
import json
import math
import multiprocessing as mp
import os
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from multiprocessing.connection import wait as mp_wait
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

import physint_inproc_v8 as v8
import pot_events
from physint_inproc_v8 import (  # noqa: F401  (re-exported so notebooks can use v9.* for everything)
    CLEAN_LANGUAGE, FAIL_KINDS, MOKA, Episode, NonFiniteAction, atomic_json, build_policy_obs,
    build_processors, check_obs_keys, derive_failure, load_json, make_libero_env, plus_suite,
    postprocess_action, read_sim, repair_rope, resolve_block, setup, smoke, vanilla_moka_suite,
    verify_rotary_inv_freq, write_snapshot,
)

V9_VERSION = "v9.0"
# A LiberoEnv reused across episodes (v8) is NOT a function of (seed, init_state_id): every hard reset
# moves the KITCHEN_SCENE8 stove fixture (model.body_pos of flat_stove_1_main) by ~1.2-1.4 cm even with
# identical seeds, so v8 episodes depended on which episodes ran before them on that env. v9 builds a
# fresh env per episode (+~1.5 s/episode), which makes every episode reproducible in any worker/order.
FRESH_ENV = os.environ.get("GROOT_FRESH_ENV", "1") == "1"
RUN_ID_V9 = "colab-l4-libero-plus-v9"


def default_n_envs() -> int:
    env = os.environ.get("GROOT_N_ENVS")
    if env:
        return max(1, int(env))
    n = max(1, min((os.cpu_count() or 2) - 1, 8))
    try:  # an OOM would kill the Colab kernel (and the loaded policy): leave ~2 GB headroom
        import psutil  # noqa: PLC0415

        avail_gb = psutil.virtual_memory().available / 1e9
        per = float(os.environ.get("GROOT_WORKER_GB", "2.0"))
        n = max(1, min(n, int((avail_gb - 2.0) / per)))
    except ImportError:
        pass
    return n


# --------------------------------------------------------------------------- noise routing
class _NoiseRoute(threading.local):
    gens: list | None = None
    calls: int = 0


_ROUTE = _NoiseRoute()
_ORIG_SAMPLE_NOISE = None


def _routed_sample_noise(shape, device, *, dtype=torch.float32, distribution="normal"):
    gens = _ROUTE.gens
    if gens is None:
        return _ORIG_SAMPLE_NOISE(shape, device, dtype=dtype, distribution=distribution)
    if distribution != "normal":
        raise RuntimeError(f"routed noise only supports 'normal', got {distribution!r}")
    shape = tuple(int(s) for s in shape)
    if shape[0] != len(gens):
        raise RuntimeError(f"noise batch {shape[0]} != number of episode generators {len(gens)}")
    _ROUTE.calls += 1
    # One (1, H, D) draw per env from that env's own generator = exactly what the sequential v8
    # runner drew (batch 1, global RNG seeded with the same episode seed).
    rows = [torch.normal(mean=0.0, std=1.0, size=(1, *shape[1:]), dtype=dtype, device=device, generator=g)
            for g in gens]
    return torch.cat(rows, 0)


def install_noise_router() -> None:
    """Patch lerobot.policies.groot.groot_n1_7.sample_noise (looked up as a module global at call time)."""
    global _ORIG_SAMPLE_NOISE
    import lerobot.policies.groot.groot_n1_7 as gm  # noqa: PLC0415

    # Keep the true original on the lerobot module itself, so a re-exec of this file in the same
    # runtime (new module instance, new _ROUTE) re-points the patch at *this* instance instead of
    # leaving a stale router from the previous exec (that produced "sample_noise called 0x").
    if not hasattr(gm, "_v9_orig_sample_noise"):
        cur = gm.sample_noise
        # Runtimes patched by the pre-fix code hold a stale router whose module kept the original.
        while getattr(cur, "_v9_routed", False):
            cur = cur.__globals__["_ORIG_SAMPLE_NOISE"]
        gm._v9_orig_sample_noise = cur
    _ORIG_SAMPLE_NOISE = gm._v9_orig_sample_noise
    _routed_sample_noise._v9_routed = True
    gm.sample_noise = _routed_sample_noise


def episode_generator(seed: int, device) -> torch.Generator:
    return torch.Generator(device=torch.device(device)).manual_seed(int(seed))


def check_generator_matches_global(device, shape=(1, 40, 132), dtype=torch.float16) -> float:
    """torch.manual_seed(s) + global draw vs Generator(s) draw on `device`; global RNG state is restored."""
    dev = torch.device(device)
    devices = [dev.index or 0] if dev.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(12345)
        a = torch.normal(mean=0.0, std=1.0, size=shape, dtype=dtype, device=dev)
    b = torch.normal(mean=0.0, std=1.0, size=shape, dtype=dtype, device=dev, generator=episode_generator(12345, dev))
    return float((a.float() - b.float()).abs().max())


# --------------------------------------------------------------------------- hook tap
class BlockTap:
    """Forward hook on transformer_blocks.{layer}; keeps the last call (= final denoise step) of a predict."""

    def __init__(self, policy, layer: int):
        self.name, self.block, self.info = resolve_block(policy, layer)
        self.n_denoise = int(policy._groot_model.action_head.num_inference_timesteps)
        self.last: torch.Tensor | None = None
        self.calls = 0
        self._h = None

    def _hook(self, _m, _i, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        self.last = t.detach()
        self.calls += 1

    def install(self):
        self.remove()
        self._h = self.block.register_forward_hook(self._hook)
        return self

    def remove(self):
        if self._h is not None:
            self._h.remove()
            self._h = None

    def arm(self):
        self.last, self.calls = None, 0

    def take(self, B: int) -> list[torch.Tensor]:
        if self.last is None:
            raise RuntimeError(f"hook {self.name} did not fire during predict_action_chunk")
        if self.calls != self.n_denoise:
            raise RuntimeError(f"hook {self.name} fired {self.calls}x, expected num_inference_timesteps={self.n_denoise}")
        x = self.last
        if x.ndim != 3 or x.shape[0] != B:
            raise RuntimeError(f"hook output shape {tuple(x.shape)} does not have batch {B}")
        self.last = None
        return [x[b].to(torch.float16).cpu() for b in range(B)]  # per env [n_tok, D], same as v8


# --------------------------------------------------------------------------- batching
def _sig(obs: dict) -> tuple:
    """Shapes that must match for obs to be concatenated along dim 0 (input_ids length, image patches...)."""
    return tuple(sorted((k, tuple(v.shape[1:]), str(v.dtype)) for k, v in obs.items() if isinstance(v, torch.Tensor)))


def collate_obs(obs_list: list[dict]) -> dict:
    """Concatenate per-env preprocessed batches (each batch=1) along dim 0.

    Batch-first tensors (input_ids, attention_mask, state, ...) and Qwen3-VL's per-image tensors
    (pixel_values = all image patches in order, image_grid_thw = one row per image) are both
    concatenated in env order, which is how the Qwen3-VL processor itself packs a batch.
    """
    if len(obs_list) == 1:
        return obs_list[0]
    s0 = _sig(obs_list[0])
    if any(_sig(o) != s0 for o in obs_list[1:]):
        raise ValueError("obs shapes differ; group by _sig before collating")
    out = {}
    for k, v in obs_list[0].items():
        if isinstance(v, torch.Tensor):
            out[k] = torch.cat([o[k] for o in obs_list], 0)
        elif isinstance(v, list):
            out[k] = [x for o in obs_list for x in o[k]]
        else:
            out[k] = v
    return out


def predict_group(policy, obs_list: list[dict], gens: list[torch.Generator], tap: BlockTap | None):
    """One predict_action_chunk call for len(obs_list) envs. Returns (actions [B,H,A], per-env acts or None)."""
    B = len(obs_list)
    batch = collate_obs(obs_list)
    if tap is not None:
        tap.arm()
    _ROUTE.gens, _ROUTE.calls = list(gens), 0
    try:
        with torch.inference_mode():
            out = policy.predict_action_chunk(batch)
        n_noise = _ROUTE.calls
    finally:
        _ROUTE.gens = None
    if n_noise != 1:
        raise RuntimeError(f"routed sample_noise called {n_noise}x in one predict (expected 1); "
                           "noise would not be reproducible")
    if out.ndim != 3 or out.shape[0] != B:
        raise RuntimeError(f"predict_action_chunk returned {tuple(out.shape)} for batch {B}")
    acts = tap.take(B) if tap is not None else None
    return out, acts


def verify_batched_predict(policy, obs_list: list[dict], layer: int, *, seed0: int = 777,
                           tol: float | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Batched vs one-at-a-time predict on the same obs + same per-env noise seeds; max |diff| per env."""
    tol = float(tol if tol is not None else os.environ.get("GROOT_BATCH_TOL", "2e-2"))
    dev = next(policy.parameters()).device
    rep: dict[str, Any] = {"B": len(obs_list), "tol_action": tol}
    tap = BlockTap(policy, layer).install()
    try:
        singles = [predict_group(policy, [o], [episode_generator(seed0 + i, dev)], tap) for i, o in enumerate(obs_list)]
        same = len({_sig(o) for o in obs_list}) == 1
        rep["same_shapes"] = same
        if not same:
            rep.update({"pass": False, "reason": "obs shapes differ across envs (different prompt length?)"})
            return rep
        gens = [episode_generator(seed0 + i, dev) for i in range(len(obs_list))]
        out, acts = predict_group(policy, obs_list, gens, tap)
    except Exception as e:
        rep.update({"pass": False, "reason": f"batched predict raised {e!r}", "traceback": traceback.format_exc()[-2000:]})
        return rep
    finally:
        tap.remove()
    da = [float((out[i:i + 1].float() - s[0].float()).abs().max()) for i, s in enumerate(singles)]
    dh = [float((acts[i].float() - s[1][0].float()).abs().max()) for i, s in enumerate(singles)]
    rep.update({"max_abs_diff_action_per_env": da, "max_abs_diff_hook_per_env": dh,
                "max_abs_diff_action": max(da), "max_abs_diff_hook": max(dh),
                "action_absmax": float(max(s[0].float().abs().max() for s in singles)),
                "pass": max(da) <= tol and all(math.isfinite(x) for x in da + dh)})
    log(f"[v9] batched-vs-single predict: {json.dumps({k: v for k, v in rep.items() if k != 'traceback'})}")
    return rep


# --------------------------------------------------------------------------- worker process
def _worker_main(conn, wid: int) -> None:
    """Spawned env worker. One LiberoEnv (rebuilt only when the task changes), v8 reset/step semantics."""
    import random  # noqa: F401,PLC0415

    torch.set_num_threads(1)
    from lerobot.sim.integrations.libero import canonical_observation  # noqa: PLC0415

    env = None
    env_key = None
    suites: dict = {}
    st: dict[str, Any] = {}

    def get_env(key, fresh: bool = False):
        nonlocal env, env_key
        if env is not None and env_key == tuple(key) and not fresh:
            return env
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
            env = None
        kind, index = key
        if kind not in suites:
            suites[kind] = vanilla_moka_suite() if kind == "moka" else plus_suite()
        env = make_libero_env(suites[kind], int(index))
        env_key = tuple(key)
        return env

    while True:
        try:
            msg = conn.recv()
        except (EOFError, KeyboardInterrupt):
            break
        cmd = msg[0]
        if cmd == "close":
            break
        try:
            if cmd == "n_init":
                e = get_env(msg[1])
                conn.send(("n_init", len(e._init_states) if e._init_states is not None else 1))
            elif cmd == "reset":
                _, key, seed, isid, max_steps = msg
                t0 = time.perf_counter()
                e = get_env(key, fresh=FRESH_ENV)
                # == v8.run_episode prologue (torch seed is irrelevant here: noise lives in the main process)
                torch.manual_seed(seed)
                np.random.seed(seed % (2**32))
                e.init_state_id = int(isid)
                raw, _ = e.reset(seed=int(seed))
                if not hasattr(e, "_v7_lang_suite"):
                    e._v7_lang_suite = e.task_description
                if CLEAN_LANGUAGE:
                    li = getattr(e._env, "language_instruction", None)
                    if isinstance(li, (list, tuple)):
                        li = " ".join(li)
                    if li:
                        e.task_description = str(li).strip().strip('"')
                names = v8._movable_objects(e._env.env)
                T = int(max_steps or os.environ.get("GROOT_MAX_STEPS") or e._max_episode_steps)
                st = {"raw": raw, "names": names, "T": T, "t": 0, "env": e}
                conn.send(("reset", {"obs": canonical_observation(raw), "names": names, "T": T, "task_name": e.task,
                                     "language": e.task_description, "language_suite": e._v7_lang_suite,
                                     "reset_s": time.perf_counter() - t0, "pid": os.getpid()}))
            elif cmd == "chunk":
                acts = msg[1]
                e, raw, names, T = st["env"], st["raw"], st["names"], st["T"]
                eef, gw, op, og = [], [], [], []
                env_s = sim_s = 0.0
                success = done = False
                nonfinite = None
                n = 0
                for a in acts:
                    if st["t"] >= T:
                        break
                    ta = time.perf_counter()
                    s = read_sim(e, raw, names)
                    sim_s += time.perf_counter() - ta
                    if not np.isfinite(a).all():  # v8 checks before env.step and records nothing for this step
                        nonfinite = st["t"]
                        break
                    eef.append(s["eef_pos"]); gw.append(s["gripper_width"]); op.append(s["obj_pos"]); og.append(s["obj_grasped"])
                    te = time.perf_counter()
                    raw, _r, term, trunc, info = e.step(np.asarray(a, np.float32))
                    env_s += time.perf_counter() - te
                    st["t"] += 1
                    n += 1
                    if bool(info.get("is_success", False)):
                        success = True
                    if term or trunc or success:
                        done = True
                        break
                if st["t"] >= T:
                    done = True
                st["raw"] = raw
                k = len(names)
                rep = {"n": n, "done": done, "success": success, "nonfinite_at": nonfinite, "env_s": env_s,
                       "sim_read_s": sim_s,
                       "eef_pos": np.asarray(eef, np.float32).reshape(n, 3),
                       "gripper_width": np.asarray(gw, np.float32).reshape(n),
                       "obj_pos": np.asarray(op, np.float32).reshape(n, k, 3),
                       "obj_grasped": np.asarray(og, np.int8).reshape(n, k),
                       "obs": None if (done or nonfinite is not None) else canonical_observation(raw)}
                conn.send(("chunk", rep))
            else:
                raise ValueError(f"unknown worker command {cmd!r}")
        except Exception as ex:
            tb = traceback.format_exc()[-3000:]
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass
            env, env_key = None, None
            conn.send(("error", {"error": repr(ex), "traceback": tb}))
    try:
        if env is not None:
            env.close()
    except Exception:
        pass


# --------------------------------------------------------------------------- pool
@dataclass
class Job:
    key: tuple            # ("plus", index) | ("moka", 0)
    seed: int
    init_state_id: int
    meta: dict = field(default_factory=dict)
    max_steps: int | None = None   # None -> GROOT_MAX_STEPS or env cap (520), like v8


class _Slot:
    def __init__(self, i):
        self.i = i
        self.proc = None
        self.conn = None
        self.state = "idle"
        self.job: Job | None = None
        self.restart = False

    def begin(self, job: Job, device):
        self.job = job
        self.restart = False
        self.gen = episode_generator(job.seed, device)
        self.t = 0
        self.T = None
        self.obs_raw = None
        self.acts, self.eef, self.gw, self.op, self.og = [], [], [], [], []
        self.chunks = []
        self.first = []
        self.n_predict = 0
        self.tm = {"obs_pre_s": 0.0, "policy_s": 0.0, "post_s": 0.0, "env_s": 0.0, "sim_read_s": 0.0, "reset_s": 0.0}
        self.sent = None
        self.t0 = time.time()
        self.info = {}
        self.success = False


def _spawn(slot: _Slot, ctxmp):
    parent, child = ctxmp.Pipe()
    old = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"  # workers: 1 BLAS thread each
    try:
        p = ctxmp.Process(target=_worker_main, args=(child, slot.i), daemon=True, name=f"groot-env-{slot.i}")
        p.start()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    child.close()
    slot.proc, slot.conn = p, parent


def _sim_dict(eefs, gws, objp, objg, names) -> dict:
    """== the sim block at the end of v8.run_episode."""
    n = len(eefs)
    objp_a = np.asarray(objp, np.float32).reshape(n, len(names), 3)
    objg_a = np.asarray(objg, np.int8).reshape(n, len(names))
    eef_a = np.asarray(eefs, np.float32).reshape(n, 3)
    if len(names):
        d = np.linalg.norm(np.nan_to_num(objp_a, nan=1e3) - eef_a[:, None, :], axis=-1)
        active = d.argmin(axis=1).astype(np.int16)
        rows = np.arange(n)
        az = objp_a[rows, active, 2]
        z0 = objp_a[0, :, 2]
        object_z_rel = az - z0[active]
        object_xy = objp_a[rows, active, :2]
        grasped = objg_a[rows, active]
    else:
        active = np.full((n,), -1, np.int16)
        az = object_z_rel = np.full((n,), np.nan, np.float32)
        object_xy = np.full((n, 2), np.nan, np.float32)
        grasped = np.full((n,), -1, np.int8)
    return {
        "object_z": az.astype(np.float32), "object_z_rel": object_z_rel.astype(np.float32), "eef_pos": eef_a,
        "gripper_width": np.asarray(gws, np.float32), "object_xy": object_xy.astype(np.float32),
        "grasped": grasped.astype(np.int8), "step": np.arange(n, dtype=np.int32), "active_obj": active,
        "obj_names": np.asarray(names, dtype=str), "obj_pos": objp_a, "obj_grasped": objg_a,
    }


@dataclass
class PoolStats:
    n_envs: int = 0
    batch_mode: str = "per_env"
    t_start: float = 0.0
    episodes: int = 0
    env_steps: int = 0
    predict_calls: int = 0
    predict_s: float = 0.0
    errors: int = 0
    verify: dict | None = None

    def line(self) -> str:
        wall = max(1e-6, time.time() - self.t_start)
        return (f"THROUGHPUT episodes={self.episodes} n_envs={self.n_envs} predict={self.batch_mode} "
                f"wall_s={wall:.1f} episodes_per_hour={3600 * self.episodes / wall:.1f} "
                f"env_steps_per_s={self.env_steps / wall:.2f} predict_calls={self.predict_calls} "
                f"gpu_predict_busy={100 * self.predict_s / wall:.1f}% errors={self.errors}")

    def as_dict(self) -> dict:
        wall = max(1e-6, time.time() - self.t_start)
        return {"n_envs": self.n_envs, "predict": self.batch_mode, "wall_s": round(wall, 1),
                "episodes": self.episodes, "env_steps": self.env_steps,
                "episodes_per_hour": round(3600 * self.episodes / wall, 2),
                "env_steps_per_s": round(self.env_steps / wall, 3), "predict_calls": self.predict_calls,
                "gpu_predict_busy_frac": round(self.predict_s / wall, 4), "errors": self.errors,
                "verify_batched": self.verify}


def run_pool(ctx, jobs: list[Job], *, on_done: Callable[[Job, Episode], None],
             on_error: Callable[[Job, str], None] | None = None, capture: bool = False,
             n_envs: int | None = None, keep_going: Callable[[], bool] = lambda: True,
             batch_predict: str | None = None, label: str = "") -> PoolStats:
    """Run `jobs` on a pool of env workers; on_done(job, Episode) is called in the main process."""
    from lerobot.processor import policy_output_to_transition, transition_to_policy_action  # noqa: PLC0415

    policy, procs = ctx.policy, ctx.procs
    if procs is None:
        raise RuntimeError("ctx.procs is None; run setup() first")
    if "predict_action_chunk" in policy.__dict__:
        raise RuntimeError("policy.predict_action_chunk is patched (a v8 FinalDenoiseCapture is still installed?); "
                           "remove it first: policy.__dict__.pop('predict_action_chunk')")
    install_noise_router()
    pre, post, _ep, _epost = procs
    dev = next(policy.parameters()).device
    q = int(policy._action_queue_steps)
    jobs_q = deque(jobs)
    N = max(1, min(int(n_envs or getattr(ctx, "n_envs", None) or default_n_envs()), len(jobs_q) or 1))
    mode = str(batch_predict if batch_predict is not None else getattr(ctx, "batch_predict", "0"))
    stats = PoolStats(n_envs=N, batch_mode="per_env", t_start=time.time())
    if not jobs_q:
        return stats
    log = ctx.log
    on_error = on_error or (lambda job, err: log(f"[v9] {label} job {job.meta} ERROR {err}"))
    ctxmp = mp.get_context("spawn")
    slots = [_Slot(i) for i in range(N)]
    tap = BlockTap(policy, ctx.layer).install() if capture else None
    keys_checked = [False]
    verify_pending = mode == "1" and N > 1

    def start_next(s: _Slot, job: Job | None = None):
        if job is None:
            if not jobs_q or not keep_going():
                s.state, s.job = "idle", None
                return
            job = jobs_q.popleft()
        s.begin(job, dev)
        policy.reset(); pre.reset(); post.reset()  # stateless for this checkpoint; kept for parity with v8
        s.conn.send(("reset", job.key, int(job.seed), int(job.init_state_id), job.max_steps))
        s.state = "resetting"

    def restart_all_active(reason: str):
        for s2 in slots:
            if s2.state in ("resetting", "stepping"):
                s2.restart = True
            elif s2.state == "need_predict":
                start_next(s2, s2.job)
        log(f"[v9] restarting in-flight episodes: {reason}")

    def finalize(s: _Slot):
        n = len(s.acts)
        expect = -(-n // q)
        if s.n_predict != expect:
            log(f"[v9] WARNING predict calls {s.n_predict} != ceil({n}/{q})={expect}: action queue not used as expected")
        timing = {k: round(v, 2) for k, v in s.tm.items()}
        timing.update({"n_predict": s.n_predict, "expected_predict": expect, "queue": q,
                       "policy_s_per_predict": round(s.tm["policy_s"] / max(1, s.n_predict), 3),
                       "env_s_per_step": round(s.tm["env_s"] / max(1, n), 4), "worker": s.i, "n_envs": N,
                       "predict_mode": stats.batch_mode})
        wall = round(time.time() - s.t0, 2)
        ep = Episode(success=s.success, n_steps=n, seconds=wall,
                     actions=np.asarray(s.acts, np.float32).reshape(n, -1),
                     sim=_sim_dict(s.eef, s.gw, s.op, s.og, s.info["names"]), chunks=list(s.chunks),
                     seed=int(s.job.seed), init_state_id=int(s.job.init_state_id), task_name=s.info["task_name"],
                     n_chunks=len(s.chunks), first_action=s.first, language=s.info["language"],
                     language_suite=s.info["language_suite"], timing=timing)
        stats.episodes += 1
        stats.env_steps += n
        print(f"TIMING {label} {s.job.meta} worker={s.i} success={ep.success} n_steps={n} wall_s={wall} "
              f"env_s={timing['env_s']} env_s_per_step={timing['env_s_per_step']} reset_s={timing['reset_s']} "
              f"obs_pre_s={timing['obs_pre_s']} policy_s={timing['policy_s']} post_s={timing['post_s']} "
              f"n_predict={s.n_predict}", flush=True)
        on_done(s.job, ep)
        print(stats.line(), flush=True)

    def handle(s: _Slot):
        try:
            kind, payload = s.conn.recv()
        except (EOFError, OSError) as e:
            stats.errors += 1
            job = s.job
            code = s.proc.exitcode if s.proc is not None else None
            try:
                s.proc.join(timeout=5)
            except Exception:
                pass
            _spawn(s, ctxmp)
            if job is not None:
                on_error(job, f"worker {s.i} died ({e!r}, exitcode={code})")
            start_next(s)
            return
        if s.restart:
            start_next(s, s.job)
            return
        if kind == "error":
            stats.errors += 1
            on_error(s.job, payload["error"] + "\n" + payload["traceback"])
            start_next(s)
            return
        if kind == "reset":
            s.info = payload
            s.T = payload["T"]
            s.tm["reset_s"] += payload["reset_s"]
            s.obs_raw = payload["obs"]
            s.state = "need_predict"
            return
        if kind == "chunk":
            r = payload
            n = r["n"]
            s.acts.extend(list(s.sent[:n]))
            s.eef.extend(list(r["eef_pos"])); s.gw.extend(list(r["gripper_width"]))
            s.op.extend(list(r["obj_pos"])); s.og.extend(list(r["obj_grasped"]))
            s.tm["env_s"] += r["env_s"]; s.tm["sim_read_s"] += r["sim_read_s"]
            s.t += n
            s.success = s.success or r["success"]
            if r["nonfinite_at"] is not None:
                bad = s.sent[n]
                if ctx.amp_mode != "fp16":
                    raise NonFiniteAction(f"non-finite action at step {r['nonfinite_at']}: {bad}")
                log(f"[v9] non-finite action at step {r['nonfinite_at']} ({s.job.meta}) -> switching to bf16 autocast "
                    "and restarting all in-flight episodes")
                ctx.amp_mode = "bf16_autocast"
                policy.config.use_bf16 = True
                restart_all_active("amp switch to bf16_autocast")
                start_next(s, s.job)
                return
            if r["done"]:
                finalize(s)
                start_next(s)
            else:
                s.obs_raw = r["obs"]
                s.state = "need_predict"
            return
        raise RuntimeError(f"unexpected worker reply {kind!r}")

    def do_predicts(need: list[_Slot]):
        nonlocal verify_pending
        prepped = []
        for s in need:
            ta = time.perf_counter()
            if not keys_checked[0]:
                check_obs_keys(policy, s.obs_raw.keys())
                keys_checked[0] = True
            obs = build_policy_obs(s.obs_raw, s.info["language"], procs)
            s.obs_raw = None
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            s.tm["obs_pre_s"] += time.perf_counter() - ta
            prepped.append((s, obs))
        if verify_pending and len(prepped) > 1:
            verify_pending = False
            vr = verify_batched_predict(policy, [o for _s, o in prepped[: min(len(prepped), 4)]], ctx.layer, log=log)
            stats.verify = vr
            stats.batch_mode = "batched" if vr.get("pass") else "per_env (batched verify failed)"
            if not vr.get("pass"):
                log(f"[v9] batched predict FAILED verification -> per-env predict calls: {vr.get('reason') or vr}")
        batched = stats.batch_mode == "batched"
        groups: list[list] = []
        if batched:
            by: dict = {}
            for s, o in prepped:
                by.setdefault(_sig(o), []).append((s, o))
            groups = list(by.values())
        else:
            groups = [[p] for p in prepped]
        for grp in groups:
            tc = time.perf_counter()
            out, acts = predict_group(policy, [o for _s, o in grp], [s.gen for s, _o in grp], tap)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - tc
            stats.predict_calls += 1
            stats.predict_s += dt
            for b, (s, _o) in enumerate(grp):
                s.tm["policy_s"] += dt / len(grp)
                s.n_predict += 1
                if acts is not None:
                    s.chunks.append((int(s.t), acts[b]))
                td = time.perf_counter()
                k = max(0, min(q, s.T - s.t))
                rows = []
                for j in range(k):
                    a = transition_to_policy_action(policy_output_to_transition(out[b:b + 1, j]))
                    a = postprocess_action(a, procs)
                    a_np = a.to("cpu").float().numpy()
                    if a_np.ndim != 2 or a_np.shape[0] != 1:
                        raise RuntimeError(f"unexpected action shape {a_np.shape}")
                    rows.append(a_np[0].astype(np.float32))
                s.sent = np.asarray(rows, np.float32).reshape(k, -1)
                if s.t == 0 and k:
                    s.first = s.sent[0].astype(np.float64).tolist()
                s.tm["post_s"] += time.perf_counter() - td
                s.conn.send(("chunk", s.sent))
                s.state = "stepping"

    log(f"[v9] {label} pool start: n_envs={N} jobs={len(jobs_q)} predict_mode_requested={mode} "
        f"device={dev} MUJOCO_GL={os.environ.get('MUJOCO_GL')} capture={capture}")
    try:
        for s in slots:
            _spawn(s, ctxmp)
        for s in slots:
            start_next(s)
        while any(s.state != "idle" for s in slots):
            waiting = {s.conn: s for s in slots if s.state in ("resetting", "stepping")}
            have_need = any(s.state == "need_predict" for s in slots)
            ready = mp_wait(list(waiting), timeout=0 if have_need else None) if waiting else []
            for c in ready:
                handle(waiting[c])
            need = [s for s in slots if s.state == "need_predict"]
            if need:
                do_predicts(need)
    finally:
        if tap is not None:
            tap.remove()
        for s in slots:
            try:
                if s.conn is not None:
                    s.conn.send(("close",))
            except Exception:
                pass
        for s in slots:
            if s.proc is not None:
                s.proc.join(timeout=20)
                if s.proc.is_alive():
                    s.proc.terminate()
                    s.proc.join(timeout=5)
    log(f"[v9] {label} pool done {json.dumps(stats.as_dict(), default=str)}")
    print(stats.line(), flush=True)
    return stats


def worker_query_n_init(key: tuple) -> int:
    """Build the env in a throwaway worker and return len(env._init_states) (main never builds an env)."""
    ctxmp = mp.get_context("spawn")
    s = _Slot(0)
    _spawn(s, ctxmp)
    try:
        s.conn.send(("n_init", key))
        kind, payload = s.conn.recv()
        if kind != "n_init":
            raise RuntimeError(f"n_init query failed: {payload}")
        return int(payload)
    finally:
        try:
            s.conn.send(("close",))
        except Exception:
            pass
        s.proc.join(timeout=20)
        if s.proc.is_alive():
            s.proc.terminate()


# --------------------------------------------------------------------------- ctx / stats
@dataclass
class Ctx(v8.Ctx):
    probe_n: int = int(os.environ.get("GROOT_PROBE_N", "20"))
    capture_n: int = int(os.environ.get("GROOT_CAPTURE_N", "100"))
    n_envs: int = field(default_factory=default_n_envs)
    batch_predict: str = os.environ.get("GROOT_BATCH_PREDICT", "0")  # 0 per-env (exact) | 1 batched if verified
    pool_stats: dict = field(default_factory=dict)


def ctx_from_v8(c8, *, run: Path | None = None, run_id: str | None = None, **over) -> "Ctx":
    """v9 Ctx sharing the live v8 policy / processors / precision."""
    kw = dict(run=Path(run or c8.run), run_id=run_id or c8.run_id, policy=c8.policy, ckpt=c8.ckpt, layer=c8.layer,
              plus_dir=c8.plus_dir, snapshot_tar=c8.snapshot_tar, tokens_per_step=c8.tokens_per_step,
              probe_ids=c8.probe_ids, started_at=c8.started_at, soft_cap_h=c8.soft_cap_h, procs=c8.procs,
              amp_mode=c8.amp_mode)
    kw.update(over)
    return Ctx(**kw)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


# --------------------------------------------------------------------------- probe
def probe(ctx: Ctx) -> dict[str, Any]:
    """probe_n eps per Plus id (init_state_id k, seed 10000+k, same as v8), run on the env pool;
    pick success rate nearest 0.5 (tie -> primary 614); Wilson 95% CI reported. Resumable by ep index."""
    path = ctx.run / "probe" / "probe.json"
    pr = load_json(path, {"instances": {}}) or {"instances": {}}
    suite = plus_suite()
    jobs = []
    for index, pid in ctx.probe_ids:
        inst = pr["instances"].setdefault(str(index), {"index": index, "id": pid, "name": suite.tasks[index].name,
                                                       "results": []})
        done = {int(r["ep"]) for r in inst["results"]}
        jobs += [Job(key=("plus", index), seed=10000 + k, init_state_id=k, meta={"probe_id": pid, "index": index, "ep": k})
                 for k in range(ctx.probe_n) if k not in done]
    since_snap = [0]

    def refresh(inst):
        inst["results"] = sorted({int(r["ep"]): r for r in inst["results"]}.values(), key=lambda r: int(r["ep"]))
        inst["successes"] = sum(bool(r["success"]) for r in inst["results"])
        inst["episodes_done"] = len(inst["results"])
        inst["sr"] = inst["successes"] / max(1, inst["episodes_done"])
        lo, hi = wilson(inst["successes"], inst["episodes_done"])
        inst["wilson95"] = [round(lo, 4), round(hi, 4)]

    def on_done(job: Job, ep: Episode):
        inst = pr["instances"][str(job.meta["index"])]
        k = int(job.meta["ep"])
        inst["results"].append({"ep": k, "seed": job.seed, "init_state_id": job.init_state_id, "success": ep.success,
                                "n_steps": ep.n_steps, "s": ep.seconds, "timing": ep.timing, "runner": V9_VERSION})
        inst["language"], inst["language_suite"] = ep.language, ep.language_suite
        refresh(inst)
        atomic_json(path, pr)
        print(f"EPISODE probe id={job.meta['probe_id']} ep={k} success={ep.success} n_steps={ep.n_steps} "
              f"seconds_per_episode={ep.seconds} sr={inst['sr']:.2f} ({inst['successes']}/{inst['episodes_done']}) "
              f"wilson95=[{inst['wilson95'][0]:.2f},{inst['wilson95'][1]:.2f}]", flush=True)
        since_snap[0] += 1
        if since_snap[0] % ctx.snapshot_every == 0:
            write_snapshot(ctx, f"probe_n{since_snap[0]}")

    def keep_going():
        if ctx.hours_left() < 0.5:
            ctx.log("probe stop: soft cap")
            return False
        return True

    if jobs:
        st = run_pool(ctx, jobs, on_done=on_done, capture=False, keep_going=keep_going, label="probe")
        ctx.pool_stats["probe"] = st.as_dict()
        pr.setdefault("pool_stats", []).append(st.as_dict())
    for inst in pr["instances"].values():
        refresh(inst)
    atomic_json(path, pr)
    write_snapshot(ctx, "probe_pool")
    cands = [v for v in pr["instances"].values() if v.get("episodes_done")]
    order = {idx: i for i, (idx, _) in enumerate(ctx.probe_ids)}
    best = min(cands, key=lambda v: (abs(v["sr"] - 0.5), order.get(int(v["index"]), 99)))
    chosen = {"index": int(best["index"]), "id": int(best["id"]), "name": best["name"], "sr": best["sr"],
              "successes": best["successes"], "episodes_done": best["episodes_done"], "wilson95": best["wilson95"],
              "in_30_70": 0.3 <= best["sr"] <= 0.7, "reason": "closest_to_0.5 (tie -> primary 614)",
              "all": {str(v["id"]): {"sr": v["sr"], "k": v["successes"], "n": v["episodes_done"],
                                     "wilson95": v["wilson95"]} for v in cands}}
    pr["chosen"] = chosen
    atomic_json(path, pr)
    atomic_json(ctx.run / "probe" / "chosen.json", chosen)
    for v in cands:
        print(f"PROBE id={v['id']} sr={v['sr']:.3f} ({v['successes']}/{v['episodes_done']}) "
              f"wilson95=[{v['wilson95'][0]:.3f},{v['wilson95'][1]:.3f}]", flush=True)
    ctx.log(f"CHOSEN {chosen}")
    write_snapshot(ctx, "probe")
    return chosen


# --------------------------------------------------------------------------- capture
def label_episode(ep: Episode, task_name: str | None) -> dict[str, Any]:
    """fail_t / fail_kind for one episode: pot_events.detect (per-pot, rule pot0-v1) when the task has a stove region, else
    v8.derive_failure. The active-object label is always kept as fail_kind_active / fail_t_active; failure_details stays
    v8's details (peak_i etc.) so readers of the v9.0 layout keep working."""
    act = derive_failure(success=ep.success, object_z=ep.sim["object_z_rel"], eef_pos=ep.sim["eef_pos"],
                         gripper_width=ep.sim["gripper_width"], grasped=ep.sim["grasped"], obj_xy=ep.sim["object_xy"])
    extra = {"fail_t_active": act["fail_t"], "fail_kind_active": act["fail_kind"]}
    pe = pot_events.detect(ep.sim, success=ep.success, task_name=task_name, names=[str(n) for n in ep.sim["obj_names"]],
                           fail_kind_active=act["fail_kind"], fail_t_active=act["fail_t"])
    if pe is None:
        extra.update(fail_rule="v8.derive_failure (active object; no stove region / pot names for this task)", fail_subkind=None, fail_pot=None)
        return {"fail_t": act["fail_t"], "fail_kind": act["fail_kind"], "details": act["details"], "extra": extra}
    extra.update(fail_subkind=pe["fail_subkind"], fail_pot=pe["fail_pot"], fail_rule=pe["fail_rule"], **pe["fields"],
                 pot1_events=pe["pot1_events"])
    return {"fail_t": pe["fail_t"], "fail_kind": pe["fail_kind"], "details": act["details"], "extra": extra}


def capture(ctx: Ctx, chosen: dict | None = None) -> dict[str, Any]:
    """Same manifest / seeds / files as v8.capture; episodes run on the env pool (hooked)."""
    chosen = chosen or load_json(ctx.run / "probe" / "chosen.json")
    if not chosen:
        raise RuntimeError("no probe/chosen.json; run probe first")
    index, pid = int(chosen["index"]), int(chosen["id"])
    mpath = ctx.run / "manifest.json"
    man = load_json(mpath, {}) or {}
    if man.get("task_index") not in (None, index):
        raise RuntimeError(f"manifest task_index {man.get('task_index')} != chosen {index}; use a new RUN dir")
    _n, _b, hook_info = resolve_block(ctx.policy, ctx.layer)
    ah = ctx.policy._groot_model.action_head
    hook_info.update({"capture": "final_denoise_step_per_chunk",
                      "num_inference_timesteps": int(ah.num_inference_timesteps),
                      "denoise_step_index": int(ah.num_inference_timesteps) - 1})
    man.update({
        "format": "rollcap-v0", "v7": v8.V7_VERSION, "v9": V9_VERSION, "run_id": ctx.run_id,
        "model": {"adapter": "groot", "name": ctx.ckpt, "family": "groot"},
        "layer": ctx.layer, "hook_target": hook_info["module"], "hook": hook_info,
        "task_index": index, "task_id": pid, "task_name": chosen.get("name"), "probe_chosen": chosen,
        "dtype": "fp16", "amp_mode": ctx.amp_mode, "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "capture_target_rollouts": ctx.capture_n, "tokens_per_step": ctx.tokens_per_step,
        "keep_mean_pool": True, "mean_pool_tok": -1,
        "step_semantics": ("acts rows exist only at chunk-start policy steps: `step` = env policy step at which "
                           "the action chunk was predicted; that chunk is executed for steps "
                           "[step, step + action_queue_steps). fail_t uses the same policy-step unit."),
        "action_queue_steps": int(ctx.policy._action_queue_steps), "action_horizon": int(ah.action_horizon),
        "token_layout": ("DiT sequence = [state token (tok 0)] + action_horizon action tokens "
                         "(tok 1..action_horizon = horizon positions 0..action_horizon-1); "
                         "only the first action_queue_steps positions are executed"),
        "language_rule": ("policy instruction = (:language) of the bddl the env actually loads "
                          "(LIBERO-plus Task.language is filename-derived junk for view/initstate variants)"
                          if CLEAN_LANGUAGE else "LIBERO-plus Task.language (unmodified)"),
        "episode_seed_rule": "seed = 20000 + episode_idx; init_state_id = episode_idx % n_init_states",
        "failure_detector": (f"pot_events.detect rule {pot_events.RULE}: per-pot lift / release / placement from obj_pos, "
                             "gripper_width and eef distance; stove region box "
                             f"{pot_events.region_for_task(chosen.get('name'))}; failure judged on pot 0 (moka_pot_1). "
                             "fail_kind_active / fail_t_active = physint_inproc_v8.derive_failure on the active object "
                             "(movable obj_of_interest nearest the EEF). Tasks without a stove region keep the active label."),
        "failure_rule": pot_events.RULE,
        "runner": {"version": V9_VERSION, "parallel_envs": int(ctx.n_envs), "predict_mode_requested": ctx.batch_predict,
                   "noise": "per-episode torch.Generator(seed) on the policy device (== v8 torch.manual_seed(seed))"},
    })
    man.setdefault("episodes", [])
    man.setdefault("token_indices", None)
    atomic_json(mpath, man)
    done = {int(e["episode_idx"]) for e in man["episodes"]}
    if man.get("n_init_states") is None:
        man["n_init_states"] = worker_query_n_init(("plus", index))
        atomic_json(mpath, man)
    n_init = int(man["n_init_states"])
    jobs = [Job(key=("plus", index), seed=20000 + i, init_state_id=i % n_init, meta={"task_id": pid, "ep": i})
            for i in range(ctx.capture_n) if i not in done]
    n_new = [0]

    def on_done(job: Job, ep: Episode):
        ep_idx = int(job.meta["ep"])
        fr = label_episode(ep, chosen.get("name"))
        n_rows = v8._save_episode(ctx, ep_idx, ep, man, fr)
        row = {"episode_idx": ep_idx, "episode_id": f"ep-{ep_idx:05d}", "task_id": str(pid), "seed": job.seed,
               "init_state_id": job.init_state_id, "success": bool(ep.success), "fail_t": fr["fail_t"],
               "fail_kind": fr["fail_kind"], "n_steps": ep.n_steps, "n_chunks": ep.n_chunks,
               "n_act_rows": n_rows, "seconds": ep.seconds, "status": "done",
               "failure_details": fr["details"], "language": ep.language, **fr["extra"]}
        man["episodes"] = sorted([e for e in man["episodes"] if int(e["episode_idx"]) != ep_idx] + [row],
                                 key=lambda e: int(e["episode_idx"]))
        man["seeds"] = [int(e["seed"]) for e in man["episodes"]]
        man["amp_mode"] = ctx.amp_mode
        atomic_json(mpath, man)
        with open(ctx.run / "episodes.jsonl", "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        n_new[0] += 1
        print(f"EPISODE ep={ep_idx} task_id={pid} seed={job.seed} success={ep.success} fail_t={fr['fail_t']} "
              f"fail_kind={fr['fail_kind']} fail_kind_active={fr['extra']['fail_kind_active']} n_steps={ep.n_steps} "
              f"seconds_per_episode={ep.seconds}", flush=True)
        if n_new[0] % ctx.snapshot_every == 0:
            write_snapshot(ctx, f"capture_n{len(man['episodes'])}")
        gc.collect()

    def keep_going():
        if ctx.hours_left() < 0.35:
            ctx.log("capture stop: soft cap")
            return False
        return True

    if jobs:
        st = run_pool(ctx, jobs, on_done=on_done, capture=True, keep_going=keep_going, label="capture")
        man["runner"]["pool"] = st.as_dict()
        ctx.pool_stats["capture"] = st.as_dict()
    eps = man["episodes"]
    summ = {"n_done": len(eps), "n_success": sum(e["success"] for e in eps),
            "fail_kinds": {k: sum(e["fail_kind"] == k for e in eps) for k in pot_events.FAIL_KINDS + FAIL_KINDS},
            "fail_kinds_active": {k: sum(e.get("fail_kind_active", e["fail_kind"]) == k for e in eps) for k in FAIL_KINDS},
            "mean_seconds_per_episode": round(float(np.mean([e["seconds"] for e in eps])), 2) if eps else None}
    man["summary"] = summ
    atomic_json(mpath, man)
    write_snapshot(ctx, f"capture_final_n{len(eps)}")
    ctx.log(f"capture summary {summ}")
    return summ
'''

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
CHOSEN = v9.probe(V9)
print(json.dumps(CHOSEN, indent=2))
if os.environ.get("GROOT_STOP_AFTER_PROBE") != "1":
    CAPTURE_SUMMARY = v9.capture(V9, CHOSEN)
    print(json.dumps({"chosen": CHOSEN, "capture": CAPTURE_SUMMARY, "pool": V9.pool_stats}, indent=2))
