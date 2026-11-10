"""physint in-process GR00T rollouts for Colab T4 (v8).

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
