"""Gate A simulator-backed checks of the keyed-peg insertion task: the
schedule binding, the 147-D observation, the pocket collision physics,
and the monitor's containment fields. Run in their own process,

  python -m pytest tests/test_peg_physics.py --isaac-peg -q
"""

import math
import os
import sys

import pytest

pytestmark = pytest.mark.isaac_peg

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "manipulation"))
sys.path.insert(0, REPO)


@pytest.fixture(scope="module")
def sim(request, peg_scene):
    return peg_scene


def _reset(ns, rate=1.0):
    ns["task_clock"].RATE_SPEC.clear()
    ns["task_clock"].RATE_SPEC.update({"mode": "fixed", "value": rate})
    obs, _ = ns["env"].reset()
    return obs["policy"]


def _zero_action(ns):
    return ns["torch"].zeros(ns["base"].num_envs, 16, device=ns["base"].device)


def _teleport(ns, pos, R, settle_steps=0):
    base, torch, P = ns["base"], ns["torch"], ns["P"]
    obj = base.scene["object"]
    p = torch.tensor(pos, dtype=torch.float32, device=base.device).unsqueeze(0).expand(base.num_envs, -1)
    p = p + base.scene.env_origins
    q = torch.tensor(P.quat_from_mat(R), dtype=torch.float32, device=base.device)
    q = q.unsqueeze(0).expand(base.num_envs, -1)
    obj.write_root_pose_to_sim(torch.cat([p, q], dim=-1))
    obj.write_root_velocity_to_sim(torch.zeros(base.num_envs, 6, device=base.device))
    for _ in range(settle_steps):
        ns["env"].step(_zero_action(ns))


def test_schedule_bound_and_observation(sim):
    ns = sim
    P, task_clock, torch = ns["P"], ns["task_clock"], ns["torch"]
    sched = task_clock.SCHEDULE
    assert sched is not None and sched.names == [p[0] for p in P.PHASES]
    assert "INSERT" in sched.names and "INSERT_DWELL" in sched.names
    obs = _reset(ns, rate=1.3)
    assert obs.shape[1] == 147
    assert bool((abs(obs[:, -1] - 1.3) < 1e-5).all())
    # No idle robot body may hang inside the insertion approach corridor.
    robot = ns["base"].scene["robot"]
    above = torch.tensor([P.POCKET_CENTER[0], P.POCKET_CENTER[1], P.BLOCK_TOP + 0.06],
                         device=ns["base"].device)
    above = above.unsqueeze(0) + ns["base"].scene.env_origins
    d = (robot.data.body_pos_w - above.unsqueeze(1)).norm(dim=-1)
    assert float(d.amin()) > 0.12, float(d.amin())


def test_monitor_containment_fields(sim):
    ns = sim
    base, P, pmon = ns["base"], ns["P"], ns["pmon"]
    _reset(ns, rate=1.0)
    mon = pmon.PegMonitor(base)
    mon.reset(range(base.num_envs))
    seat = [P.POCKET_CENTER[0], P.POCKET_CENTER[1], P.SEAT_Z + 0.001]
    upright = P.rotz(math.radians(P.START_YAW_DEG))
    _teleport(ns, seat, upright, settle_steps=6)
    mon.step()
    assert bool(mon.last_inside.all()), mon.last_depth.tolist()
    assert float(mon.last_depth.min()) >= P.INSERTED_MIN_DEPTH
    # An offset beyond the clearance leaves the cross-section (held above
    # the pocket so the walls do not correct it).
    off = P.R_POCKET @ [P.CLEARANCE + 0.010, 0.0, 0.0]
    high = [seat[0] + off[0], seat[1] + off[1], P.BLOCK_TOP + P.OBJECT_HALF_HEIGHT + 0.005]
    _teleport(ns, high, upright)
    mon.step()
    assert not bool(mon.last_inside.any())


def test_pocket_collides(sim):
    """A peg dropped centered above the pocket rests on the pocket floor;
    a peg above the wall ring rests on the block top, not inside."""
    ns = sim
    base, P, torch = ns["base"], ns["P"], ns["torch"]
    _reset(ns, rate=1.0)
    upright = P.rotz(math.radians(P.START_YAW_DEG))
    above = [P.POCKET_CENTER[0], P.POCKET_CENTER[1], P.BLOCK_TOP + P.OBJECT_HALF_HEIGHT + 0.01]
    _teleport(ns, above, upright, settle_steps=120)
    z = base.scene["object"].data.root_pos_w[:, 2] - base.scene.env_origins[:, 2]
    assert bool((abs(z - P.SEAT_Z) < 0.006).all()), z.tolist()
    # offset onto the wall ring
    off = P.R_POCKET @ [P.POCKET_W / 2 + P.WALL_T / 2, 0.0, 0.0]
    wall_top = [above[0] + off[0], above[1] + off[1], P.BLOCK_TOP + P.OBJECT_HALF_HEIGHT + 0.01]
    _teleport(ns, wall_top, upright, settle_steps=120)
    z = base.scene["object"].data.root_pos_w[:, 2] - base.scene.env_origins[:, 2]
    assert bool((z > P.POCKET_FLOOR_Z + P.OBJECT_HALF_HEIGHT + 0.02).all()), z.tolist()
    forces = ns["pmon"].PegMonitor(base).pocket_forces_w()
    assert float(forces.max()) > 0.0
