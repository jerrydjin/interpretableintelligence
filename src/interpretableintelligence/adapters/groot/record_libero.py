"""Record LIBERO-plus episodes as GR00T clips (``ii record-libero``).

The GR00T adapter reads a clip as the main-camera frames plus a sidecar with the
wrist camera, the robot state and the instruction (``adapter.py``, "Inputs").
The v9 capture stores only the hooked block's output, not those inputs, so the
steering scorecard needs new episodes recorded with every observation kept.

This drives the real policy closed loop on one LIBERO-plus task, the way the v8/v9
runners do (``v8/inproc_v8.py``: canonical observation, ``prepare_canonical_observation``,
the LIBERO env processor, the checkpoint's own pre- and postprocessor, a 16-step
action queue, the bddl's own instruction), and writes, per episode::

    <out>/ep-00000.npy           main camera, [steps, 256, 256, 3] uint8, one frame per env step
    <out>/ep-00000.groot.json    {"task", "state": [[8 floats] per step], "wrist": "wrist/ep-00000.npy", "libero": {...}}
    <out>/wrist/ep-00000.npy     wrist camera, aligned with the main frames
    <out>/clips.json             clip index (fps 20, the LIBERO control rate)
    <out>/record.json            settings, per-episode outcomes and the input parity check

Frame ``t`` is the observation the policy saw before step ``t``, so the adapter's
window ``k`` (frames ``[16k, 16k + 16)``, observed at its first frame) is exactly the
input of the policy's ``k``-th action-chunk prediction.

Input parity: on the first prediction of the first new episode, the recorder runs
the GR00T backbone twice, once on the batch the policy actually got and once on
the adapter's batch rebuilt from the recorded arrays, and compares the backbone
and state features. A mismatch stops the recording, because the adapter would be
steering a model that sees different inputs from the one that acted.

Seeds default to ``60000 + i`` (``init_state_id = i % n_init_states``), away from the
v9 capture (``20000 + i``, which trained the SAE) and the ablation runs (``40000 + i``).
Recording is resumable: episodes already in ``clips.json`` are kept.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

RECORDER_VERSION = "record-libero-v1"
CONTROL_HZ = 20.0
DEFAULT_TASK_ID = 1028
DEFAULT_SEED0 = 60000
TASK_NAMES = {1028: "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"}  # data/groot-1028.md
PARITY_RTOL = 1e-3
WRIST_DIR = "wrist"
INDEX_FILE = "clips.json"
RECORD_FILE = "record.json"


@dataclass
class Step:
    """One observation, as the policy got it: main and wrist camera (HWC uint8) and state."""

    image: np.ndarray
    wrist: np.ndarray
    state: np.ndarray


class Runtime(Protocol):
    """The env plus policy. ``LeRobotLibero`` is the real one; tests pass a stand-in."""

    task_name: str
    checkpoint: str

    def n_init_states(self) -> int: ...

    def reset(self, seed: int, init_state_id: int) -> tuple[str, int]:
        """Reset the env and the policy. Returns ``(instruction, max_steps)``."""

    def observe(self) -> Step: ...

    def act(self) -> np.ndarray:
        """The env action for the current observation (runs the policy only when its queue is empty)."""

    def step(self, action: np.ndarray) -> tuple[bool, bool]:
        """Apply ``action``. Returns ``(success, done)``."""

    def parity(self, step: Step, task: str) -> dict:
        """Compare the policy's own inputs with the adapter's rebuild from ``step``."""


def episode_id(index: int) -> str:
    return f"ep-{int(index):05d}"


def record(
    runtime: Runtime,
    out: Path,
    *,
    task_id: int = DEFAULT_TASK_ID,
    episodes: int = 24,
    seed0: int = DEFAULT_SEED0,
    max_steps: int | None = None,
    check_parity: bool = True,
    log: Callable[[str], None] = print,
) -> dict:
    """Record ``episodes`` episodes into ``out`` (resumable) and return ``record.json``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / WRIST_DIR).mkdir(exist_ok=True)
    expected = TASK_NAMES.get(int(task_id))
    if expected is not None and not str(runtime.task_name).startswith(expected):
        raise RuntimeError(f"LIBERO-plus task {task_id} loaded {runtime.task_name!r}, expected a {expected} variant")
    report = _read_report(out)
    settings = {
        "recorder": RECORDER_VERSION,
        "task_id": int(task_id),
        "task_name": str(runtime.task_name),
        "checkpoint": str(runtime.checkpoint),
        "seed0": int(seed0),
        "max_steps": None if max_steps is None else int(max_steps),
        "fps": CONTROL_HZ,
    }
    if report.get("settings") not in (None, settings):
        raise RuntimeError(f"{out} was recorded with other settings {report['settings']}; record into a new folder")
    report["settings"] = settings
    report.setdefault("episodes", [])
    report.setdefault("parity", None)
    n_init = int(runtime.n_init_states())
    done = {row["clip_id"] for row in report["episodes"]}
    for index in range(int(episodes)):
        clip_id = episode_id(index)
        if clip_id in done and (out / f"{clip_id}.npy").is_file():
            continue
        seed, init_state = int(seed0) + index, index % n_init
        started = time.time()
        task, cap = runtime.reset(seed, init_state)
        cap = int(max_steps) if max_steps is not None else int(cap)
        images, wrists, states = [], [], []
        success = False
        for t in range(cap):
            obs = runtime.observe()
            images.append(obs.image)
            wrists.append(obs.wrist)
            states.append(np.asarray(obs.state, dtype=np.float32).reshape(-1))
            if t == 0 and check_parity and report["parity"] is None:
                report["parity"] = runtime.parity(obs, task)
                _write_json(out / RECORD_FILE, report)
                if not report["parity"].get("pass"):
                    raise RuntimeError(f"input parity check failed: {json.dumps(report['parity'])}")
                log(f"input parity ok: {json.dumps(report['parity'])}")
            success, finished = runtime.step(runtime.act())
            if finished:
                break
        _write_episode(out, clip_id, np.stack(images), np.stack(wrists), np.stack(states), task)
        row = {
            "clip_id": clip_id,
            "seed": seed,
            "init_state_id": init_state,
            "success": bool(success),
            "n_steps": len(images),
            "seconds": round(time.time() - started, 2),
        }
        _write_sidecar_meta(out, clip_id, {**settings, **row})
        report["episodes"] = sorted([r for r in report["episodes"] if r["clip_id"] != clip_id] + [row], key=lambda r: r["clip_id"])
        report["n_success"] = sum(1 for r in report["episodes"] if r["success"])
        _write_index(out, report["episodes"])
        _write_json(out / RECORD_FILE, report)
        log(f"{clip_id} seed={seed} init_state={init_state} success={success} steps={len(images)} ({row['seconds']} s)")
    return report


def _write_episode(out: Path, clip_id: str, images: np.ndarray, wrists: np.ndarray, states: np.ndarray, task: str) -> None:
    for name, array in ((f"{clip_id}.npy", images), (f"{WRIST_DIR}/{clip_id}.npy", wrists)):
        if array.dtype != np.uint8 or array.ndim != 4 or array.shape[-1] != 3:
            raise ValueError(f"{name}: frames must be [steps, H, W, 3] uint8, got {array.dtype} {array.shape}")
        tmp = out / f"{name}.tmp.npy"
        np.save(tmp, array)
        tmp.replace(out / name)
    sidecar = {"task": task, "state": states.astype(np.float32).tolist(), "wrist": f"{WRIST_DIR}/{clip_id}.npy"}
    _write_json(out / f"{clip_id}.groot.json", sidecar)


def _write_sidecar_meta(out: Path, clip_id: str, meta: dict) -> None:
    path = out / f"{clip_id}.groot.json"
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    sidecar["libero"] = meta  # the adapter reads only task / state / wrist
    _write_json(path, sidecar)


def _write_index(out: Path, rows: list[dict]) -> None:
    clips = [{"clip_id": r["clip_id"], "file": f"{r['clip_id']}.npy", "fps": CONTROL_HZ} for r in rows]
    _write_json(out / INDEX_FILE, {"fps": CONTROL_HZ, "clips": clips})


def _read_report(out: Path) -> dict:
    path = out / RECORD_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


# ------------------------------------------------------------------ the real env and policy


def hwc_uint8(value) -> np.ndarray:
    """One image from a prepared observation (``[1, C, H, W]`` or ``[1, H, W, C]``, float in [0, 1] or uint8) as HWC uint8."""
    import torch  # noqa: PLC0415

    array = value.detach().cpu() if isinstance(value, torch.Tensor) else value
    array = np.asarray(array)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"expected one image, got batch {array.shape[0]}")
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"image must be 3-d, got {array.shape}")
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = np.clip(np.rint(array.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return np.ascontiguousarray(array)


def _load_v8():
    """The v8 runner's env helpers (``v8/`` is a script directory, not a package)."""
    name = "physint_inproc_v8"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parent / "v8" / "inproc_v8.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class LeRobotLibero:
    """LIBERO-plus through LeRobot 0.7 and GR00T through the adapter's own loader. Needs a GPU box with
    ``lerobot[groot]==0.7.0`` and LIBERO-plus installed (``scripts/groot_gpu_setup.sh``)."""

    def __init__(self, checkpoint: str, task_id: int, plus_dir: Path | None = None, device: str | None = None) -> None:
        import torch  # noqa: PLC0415

        self.v8 = _load_v8()
        if plus_dir is not None:  # before anything imports libero, which asks for its paths on stdin without a config
            self.v8.write_libero_config(plus_dir, log=lambda _m: None)
        from lerobot.envs.configs import LiberoEnvConfig  # noqa: PLC0415

        from interpretableintelligence.adapters.groot.adapter import LeRobotGroot  # noqa: PLC0415
        self.checkpoint = checkpoint
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.backend = LeRobotGroot(checkpoint)
        self.backend.load(random_init=False, device=self.device)
        self.policy = self.backend.policy
        self.env_pre, self.env_post = LiberoEnvConfig(camera_name_mapping=dict(self.v8.CAMERA_MAP)).get_env_processors()
        self.env = self.v8.make_libero_env(self.v8.plus_suite(), int(task_id) - 1)  # LIBERO-plus ids are 1-based
        self.task_name = str(self.env.task)
        keys = list(self.backend.camera_keys)
        if len(keys) != 2:
            raise RuntimeError(f"expected two cameras (main, wrist), the checkpoint reads {keys}")
        self.camera_keys = keys
        self._raw = None
        self._obs = None
        self._batch = None
        self.task = ""

    def n_init_states(self) -> int:
        states = getattr(self.env, "_init_states", None)
        return len(states) if states is not None else 1

    def reset(self, seed: int, init_state_id: int) -> tuple[str, int]:
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))
        self.policy.reset()
        self.backend.pre.reset()
        self.backend.post.reset()
        self._batch = None
        self.env.init_state_id = int(init_state_id)
        self._raw, _ = self.env.reset(seed=int(seed))
        task = getattr(self.env._env, "language_instruction", None)  # the bddl's own instruction (v8 CLEAN_LANGUAGE)
        if isinstance(task, (list, tuple)):
            task = " ".join(task)
        self.task = str(task).strip().strip('"') if task and self.v8.CLEAN_LANGUAGE else str(self.env.task_description)
        self.env.task_description = self.task
        cap = int(os.environ.get("GROOT_MAX_STEPS") or self.env._max_episode_steps)
        return self.task, cap

    def _prepared(self) -> dict:
        from lerobot.envs.utils import prepare_canonical_observation  # noqa: PLC0415
        from lerobot.sim.integrations.libero import canonical_observation  # noqa: PLC0415

        canon = canonical_observation(self._raw)
        obs = prepare_canonical_observation({k: np.asarray(v)[None] for k, v in canon.items()})
        obs["task"] = [self.task]
        return self.env_pre(obs)

    def observe(self) -> Step:
        self._obs = self._prepared()
        main, wrist = (hwc_uint8(self._obs[key]) for key in self.camera_keys)
        state = np.asarray(self._obs["observation.state"].detach().cpu().float().numpy()).reshape(-1)
        return Step(image=main, wrist=wrist, state=state)

    def act(self) -> np.ndarray:
        import torch  # noqa: PLC0415
        from lerobot.processor import policy_output_to_transition, transition_to_policy_action  # noqa: PLC0415
        from lerobot.utils.constants import ACTION  # noqa: PLC0415

        if self._batch is None or len(self.policy._action_queue) == 0:
            self._batch = self.backend.pre(self._obs)  # Qwen preprocessing only when a new chunk is predicted (v8)
        with torch.inference_mode():
            action = transition_to_policy_action(policy_output_to_transition(self.policy.select_action(self._batch)))
        action = self.env_post({ACTION: self.backend.post(action)})[ACTION]
        out = action.to("cpu").float().numpy()
        if out.ndim != 2 or out.shape[0] != 1 or not np.isfinite(out).all():
            raise RuntimeError(f"bad action {out!r}")
        return out[0].astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[bool, bool]:
        self._raw, _reward, term, trunc, info = self.env.step(action)
        success = bool(info.get("is_success", False))
        return success, bool(term or trunc or success)

    def parity(self, step: Step, task: str) -> dict:
        import torch  # noqa: PLC0415

        from interpretableintelligence.adapters.groot.adapter import Observation  # noqa: PLC0415

        backend = self.backend
        with torch.no_grad():
            batch = backend.pre(self._prepared())
            inputs = backend.policy._filter_groot_inputs(batch, include_action=False)
            backbone_in, action_in = backend.model.prepare_input(inputs)
            real = backend.head._encode_features(backend.model.backbone(backbone_in), action_in)
            rebuilt = backend.context(Observation(image=step.image, wrist=step.wrist, state=step.state.astype(np.float32), task=task))
        report: dict = {"rtol": PARITY_RTOL, "camera_keys": self.camera_keys}
        ok = True
        for name in ("backbone_features", "state_features"):
            a = getattr(real, name).float()
            b = rebuilt[name].float()
            if tuple(a.shape) != tuple(b.shape):
                report[name] = {"shape_real": list(a.shape), "shape_adapter": list(b.shape)}
                ok = False
                continue
            diff = float((a - b).abs().max())
            scale = float(a.abs().max().clamp(min=1e-6))
            report[name] = {"max_abs_diff": diff, "max_abs": scale, "rel": diff / scale}
            ok = ok and diff / scale <= PARITY_RTOL
        report["pass"] = bool(ok)
        return report
