"""Gate A simulator-backed checks of the upright placement task: the
schedule binding, the 147-D observation, the physical premise of the
stability predicate, the monitor's final-state fields, and per-env
speedup-factor independence. Run in their own process,

  python -m pytest tests/test_upright_physics.py --isaac-upright -q
"""

import math
import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.isaac_upright

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "manipulation"))
sys.path.insert(0, REPO)


@pytest.fixture(scope="module")
def sim(request, upright_scene):
    return upright_scene


def _reset(ns, rate=1.0):
    ns["task_clock"].RATE_SPEC.clear()
    ns["task_clock"].RATE_SPEC.update({"mode": "fixed", "value": rate})
    obs, _ = ns["env"].reset()
    return obs["policy"]


def _zero_action(ns):
    return ns["torch"].zeros(ns["base"].num_envs, 16, device=ns["base"].device)


def _teleport(ns, pos, R, settle_steps=0):
    base, torch, U = ns["base"], ns["torch"], ns["U"]
    obj = base.scene["object"]
    p = torch.tensor(pos, dtype=torch.float32, device=base.device).unsqueeze(0).expand(base.num_envs, -1)
    p = p + base.scene.env_origins
    q = torch.tensor(U.quat_from_mat(R), dtype=torch.float32, device=base.device)
    q = q.unsqueeze(0).expand(base.num_envs, -1)
    obj.write_root_pose_to_sim(torch.cat([p, q], dim=-1))
    obj.write_root_velocity_to_sim(torch.zeros(base.num_envs, 6, device=base.device))
    for _ in range(settle_steps):
        ns["env"].step(_zero_action(ns))


def _tilt_deg(ns):
    base, torch = ns["base"], ns["torch"]
    from isaaclab.utils.math import quat_apply
    q = base.scene["object"].data.root_quat_w
    ez = torch.zeros(base.num_envs, 3, device=base.device)
    ez[:, 2] = 1.0
    z_w = quat_apply(q, ez)
    return torch.rad2deg(torch.acos(z_w[:, 2].clamp(-1.0, 1.0)))


def test_schedule_bound_and_observation(sim):
    ns = sim
    U, task_clock = ns["U"], ns["task_clock"]
    sched = task_clock.SCHEDULE
    assert sched is not None and sched.names == [p[0] for p in U.PHASES]
    assert "LOWER" in sched.names and "PLACE_DWELL" in sched.names
    obs = _reset(ns, rate=1.7)
    assert obs.shape[1] == 147
    assert bool((abs(obs[:, -1] - 1.7) < 1e-5).all()), obs[:, -1].tolist()
    assert abs(sched.cycle_time(2.0) - (5.7 + 14.8 / 2.0 + 1.0)) < 1e-9
    # No idle robot body may hang inside the placement workspace.
    torch = ns["torch"]
    robot = ns["base"].scene["robot"]
    place = torch.tensor([U.TARGET_CENTER[0], U.TARGET_CENTER[1], U.PLACE_Z], device=ns["base"].device)
    place = place.unsqueeze(0) + ns["base"].scene.env_origins
    d = (robot.data.body_pos_w - place.unsqueeze(1)).norm(dim=-1)
    j = int(d.amin(dim=1).argmin())
    assert float(d.amin()) > 0.12, (float(d.amin()), robot.body_names[int(d[j].argmin())])


def test_monitor_final_state_fields(sim):
    ns = sim
    base, U, umon = ns["base"], ns["U"], ns["umon"]
    _reset(ns, rate=1.0)
    mon = umon.UprightMonitor(base)
    mon.reset(range(base.num_envs))
    rest = [U.TARGET_CENTER[0], U.TARGET_CENTER[1], U.TABLE_TOP + U.OBJECT_HALF_HEIGHT + 0.001]
    _teleport(ns, rest, U.R_UPRIGHT, settle_steps=10)
    mon.step()
    assert float(mon.last_tilt.max()) < math.radians(3.0)
    assert bool(mon.last_inside.all())
    # base center offset beyond the target radius
    off = [rest[0] + 0.05, rest[1], rest[2]]
    _teleport(ns, off, U.R_UPRIGHT, settle_steps=2)
    mon.step()
    assert not bool(mon.last_inside.any())
    # a tilt above the final tolerance is measured as such
    tilted = U.rotz(math.radians(U.START_YAW_DEG)) @ U.roty(math.radians(8.0))
    _teleport(ns, [rest[0], rest[1], rest[2] + 0.02], tilted)
    mon.step()
    td = np.degrees(float(mon.last_tilt[0]))
    assert 6.0 < td < 10.0, td
    assert td > U.FINAL_TILT_TOL_DEG


def test_standing_object_is_stable_and_tilted_object_tips(sim):
    """Runs last: a teleport after a mid-episode reset of displaced objects
    inherits a residual drift, an artifact of writing root poses outside
    the event manager, so the destructive tipping check follows the
    monitor checks. The physical premise of the stability predicate: an upright object
    at the place pose stays standing, a tilt beyond the tipping angle
    (atan(27.5/90) = 17 deg) falls over."""
    ns = sim
    U = ns["U"]
    _reset(ns, rate=1.0)
    rest = [U.TARGET_CENTER[0], U.TARGET_CENTER[1], U.TABLE_TOP + U.OBJECT_HALF_HEIGHT + 0.001]
    _teleport(ns, rest, U.R_UPRIGHT, settle_steps=60)
    tilt = _tilt_deg(ns)
    assert bool((tilt < 2.0).all()), f"upright object moved, tilt {tilt.tolist()}"
    tipped = U.rotz(math.radians(U.START_YAW_DEG)) @ U.roty(math.radians(25.0))
    high = [U.TARGET_CENTER[0], U.TARGET_CENTER[1], U.TABLE_TOP + U.OBJECT_HALF_HEIGHT + 0.01]
    _teleport(ns, high, tipped, settle_steps=250)
    tilt = _tilt_deg(ns)
    assert bool((tilt > 45.0).all()), f"tilted object did not tip over, tilt {tilt.tolist()}"


def test_per_env_speedup_factor_independence(sim):
    ns = sim
    base, torch, task_clock = ns["base"], ns["torch"], ns["task_clock"]
    vals = torch.tensor([0.5, 3.0, 1.0, 1.0], device=base.device)
    task_clock.RATE_SPEC.clear()
    task_clock.RATE_SPEC.update({"mode": "per_env", "values": vals})
    ns["env"].reset()
    base.episode_length_buf[:] = int(8.0 / base.step_dt)
    k, f, t, cycle = task_clock.phase_state(base)
    assert int(k[0]) != int(k[1]), (k.tolist(), "same phase at different speeds")
    assert float(cycle[0]) > float(cycle[1])
