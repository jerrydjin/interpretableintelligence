"""physint in-process GR00T rollouts, v9: N LIBERO envs in parallel CPU worker processes.

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
