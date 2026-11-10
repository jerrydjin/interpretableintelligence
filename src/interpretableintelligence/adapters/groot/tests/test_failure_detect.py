"""Unit tests for failure_detect.derive_failure on synthetic traces."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from failure_detect import derive_failure

def _base(T=80):
    t = np.arange(T)
    eef = np.stack([0.01 * t, np.zeros(T), 0.1 + 0.001 * t], axis=1)
    obj_z = np.full(T, 0.01)
    gw = np.full(T, 0.05)
    return obj_z, eef, gw

def test_success():
    z, eef, gw = _base()
    r = derive_failure(success=True, object_z=z, eef_pos=eef, gripper_width=gw)
    assert r["fail_t"] is None and r["fail_kind"] is None

def test_drop():
    T = 80
    z, eef, gw = _base(T)
    # lift then drop
    z[:] = 0.01
    z[20:40] = np.linspace(0.01, 0.12, 20)
    z[40:] = 0.12
    z[50] = 0.12
    z[51:] = 0.04  # fall
    r = derive_failure(success=False, object_z=z, eef_pos=eef, gripper_width=gw, lift_z=0.05, fall_delta=0.03)
    assert r["fail_kind"] == "drop", r
    assert r["fail_t"] is not None and r["fail_t"] >= 50

def test_empty_grasp():
    T = 60
    z, eef, gw = _base(T)
    # object stays on table far away; gripper closes at t=30
    z[:] = 0.01
    eef[:, :] = 0.5  # far
    gw[:] = 0.05
    gw[30:] = 0.01
    grasped = np.zeros(T, dtype=bool)
    r = derive_failure(success=False, object_z=z, eef_pos=eef, gripper_width=gw, grasped=grasped, close_w=0.02)
    assert r["fail_kind"] == "empty_grasp", r
    assert r["fail_t"] == 30

def test_stall():
    T = 80
    z = np.full(T, 0.02)
    eef = np.zeros((T, 3))
    eef[:10, 0] = np.linspace(0, 0.05, 10)  # move then freeze
    eef[10:, 0] = 0.05
    gw = np.full(T, 0.04)
    r = derive_failure(success=False, object_z=z, eef_pos=eef, gripper_width=gw, stall_n=25, stall_eps=0.002)
    assert r["fail_kind"] == "stall", r
    assert r["fail_t"] is not None

def test_timeout():
    T = 40
    # moving enough to avoid stall, never lifts, gripper never closes
    z = np.full(T, 0.01)
    eef = np.stack([0.01 * np.arange(T), np.zeros(T), np.zeros(T)], axis=1)
    gw = np.full(T, 0.05)
    r = derive_failure(success=False, object_z=z, eef_pos=eef, gripper_width=gw, stall_n=100)
    assert r["fail_kind"] == "timeout", r
    assert r["fail_t"] == T - 1

def test_priority_drop_over_timeout():
    T = 50
    z, eef, gw = _base(T)
    z[:] = 0.01; z[10:20] = 0.1; z[20:] = 0.02
    r = derive_failure(success=False, object_z=z, eef_pos=eef, gripper_width=gw)
    assert r["fail_kind"] == "drop"

if __name__ == "__main__":
    test_success(); print("ok success")
    test_drop(); print("ok drop")
    test_empty_grasp(); print("ok empty_grasp")
    test_stall(); print("ok stall")
    test_timeout(); print("ok timeout")
    test_priority_drop_over_timeout(); print("ok priority")
    print("ALL_FAILURE_TESTS_PASSED")
