"""Derive fail_t (policy steps) and fail_kind from per-step sim traces.

fail_kind in {drop, empty_grasp, stall, timeout}. Successful episodes: fail_t=None, fail_kind=None.
Inputs are 1-D / 2-D numpy arrays aligned to policy steps (same unit as activation `step`).
"""
from __future__ import annotations
from typing import Any
import numpy as np

FAIL_KINDS = ("drop", "empty_grasp", "stall", "timeout")


def derive_failure(
    *,
    success: bool,
    object_z: np.ndarray,
    eef_pos: np.ndarray,
    gripper_width: np.ndarray,
    grasped: np.ndarray | None = None,
    n_steps: int | None = None,
    # drop: was lifted above lift_z, then fell by fall_delta within fall_window
    lift_z: float = 0.05,
    fall_delta: float = 0.03,
    # empty_grasp: gripper closes below close_w while object far / not grasped
    close_w: float = 0.02,
    grasp_near: float = 0.05,
    # stall: EEF and object move < stall_eps for stall_n consecutive steps
    stall_n: int = 25,
    stall_eps: float = 0.002,
    obj_xy: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return {fail_t, fail_kind, details}. fail_t is int policy step or None if success."""
    object_z = np.asarray(object_z, dtype=np.float64).reshape(-1)
    eef_pos = np.asarray(eef_pos, dtype=np.float64)
    if eef_pos.ndim == 1:
        eef_pos = eef_pos.reshape(-1, 1)
    gripper_width = np.asarray(gripper_width, dtype=np.float64).reshape(-1)
    T = int(n_steps if n_steps is not None else len(object_z))
    T = min(T, len(object_z), len(gripper_width), len(eef_pos))
    if T <= 0:
        return {"fail_t": None if success else 0, "fail_kind": None if success else "timeout", "details": {"reason": "empty"}}
    object_z = object_z[:T]
    eef_pos = eef_pos[:T]
    gripper_width = gripper_width[:T]
    if grasped is not None:
        grasped = np.asarray(grasped, dtype=bool).reshape(-1)[:T]
    if obj_xy is not None:
        obj_xy = np.asarray(obj_xy, dtype=np.float64).reshape(-1, 2)[:T]

    if success:
        return {"fail_t": None, "fail_kind": None, "details": {}}

    # --- drop: lifted then fell ---
    lifted = object_z >= lift_z
    if lifted.any():
        first_lift = int(np.argmax(lifted))
        peak = float(object_z[first_lift:].max())
        peak_i = first_lift + int(np.argmax(object_z[first_lift:]))
        # after peak, look for a fall of fall_delta
        after = object_z[peak_i:]
        if len(after) > 1:
            drop_rel = peak - after
            if (drop_rel >= fall_delta).any():
                fail_t = peak_i + int(np.argmax(drop_rel >= fall_delta))
                return {
                    "fail_t": int(fail_t),
                    "fail_kind": "drop",
                    "details": {"peak_z": peak, "peak_i": peak_i, "z_at_fail": float(object_z[fail_t])},
                }

    # --- empty_grasp: gripper closes without holding the object ---
    closed = gripper_width <= close_w
    # closing edge
    was_open = np.concatenate([[True], gripper_width[:-1] > close_w])
    close_events = np.flatnonzero(closed & was_open)
    for t in close_events.tolist():
        holding = False
        if grasped is not None:
            holding = bool(grasped[t])
        else:
            # proxy: object XY near EEF XY and object z near EEF z
            eef = eef_pos[t]
            if obj_xy is not None and eef.shape[-1] >= 2:
                dist = float(np.linalg.norm(obj_xy[t, :2] - eef[:2]))
                holding = dist <= grasp_near and abs(float(object_z[t]) - float(eef[2] if eef.shape[-1] > 2 else object_z[t])) <= grasp_near
            elif eef.shape[-1] >= 3:
                # only z available for object — weak proxy: object within grasp_near of eef z and was not lifted before
                holding = abs(float(object_z[t]) - float(eef[2])) <= grasp_near and float(object_z[t]) >= lift_z * 0.5
        if not holding:
            return {
                "fail_t": int(t),
                "fail_kind": "empty_grasp",
                "details": {"gripper_width": float(gripper_width[t]), "object_z": float(object_z[t])},
            }

    # --- stall: little motion for stall_n steps ---
    if T >= stall_n:
        eef_delta = np.linalg.norm(np.diff(eef_pos, axis=0), axis=1)
        obj_delta = np.abs(np.diff(object_z))
        if obj_xy is not None:
            obj_delta = np.linalg.norm(np.diff(obj_xy, axis=0), axis=1)
        # pad to length T (delta[i] = motion from i -> i+1); stall window ends at step i+stall_n-1
        eef_d = np.concatenate([[0.0], eef_delta])
        obj_d = np.concatenate([[0.0], obj_delta])
        for start in range(0, T - stall_n + 1):
            sl = slice(start + 1, start + stall_n)  # motions inside the window
            if float(eef_d[sl].max()) < stall_eps and float(obj_d[sl].max()) < stall_eps:
                # only count as failure if we're not already basically done successfully mid-episode
                fail_t = start + stall_n - 1
                return {
                    "fail_t": int(fail_t),
                    "fail_kind": "stall",
                    "details": {"start": start, "stall_n": stall_n, "eef_max": float(eef_d[sl].max()), "obj_max": float(obj_d[sl].max())},
                }

    # --- timeout fallback ---
    return {"fail_t": int(T - 1), "fail_kind": "timeout", "details": {"n_steps": T}}
