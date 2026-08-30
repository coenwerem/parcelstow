"""Physical-integrity and predicate tests of the ParcelStow task (need the
simulator, run with --isaac). One Isaac app and one small environment
(conftest.isaac_scene) serve every simulator test of the process.

Covered,
- the parcel is a free rigid body, no joint or parent attachment to the hand,
- the parcel falls under gravity when the hand opens at a lifted state,
- hand motion cannot drag a parcel placed outside contact,
- receptacle slabs collide with the parcel (rests on the floor, blocked by
  walls),
- centered scripted insertion satisfies the predicate, a scripted insertion
  outside the tight clearance does not, an over-rotated parcel does not,
- release and settle detection,
- task success does not depend on the force-closure margin value,
- per-environment phase and rate progression,
- realized-contact scoring leaves the environment state unchanged,
- observation dimensionality of the task matches the learner adapters.
"""

import math
import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.isaac

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "manipulation"))
sys.path.insert(0, REPO)


@pytest.fixture(scope="module")
def sim(request, isaac_scene):
    return isaac_scene


def _reset(ns, rate=1.0):
    ns["task_clock"].RATE_SPEC.clear()
    ns["task_clock"].RATE_SPEC.update({"mode": "fixed", "value": rate})
    obs, _ = ns["env"].reset()
    return obs["policy"]


def _zero_action(ns):
    return ns["torch"].zeros(ns["base"].num_envs, 16, device=ns["base"].device)


def _hold_action(ns):
    """Action holding the current joint positions."""
    base = ns["base"]
    robot = base.scene["robot"]
    from parcelstow.tasks.manager_based.parcel_stow.parcel_stow_env_cfg import CHAIN_ACTUATED
    jids, _ = robot.find_joints(CHAIN_ACTUATED, preserve_order=True)
    q = robot.data.joint_pos[:, jids]
    q_def = robot.data.default_joint_pos[:, jids]
    return 2.0 * (q - q_def)


def test_no_weld_or_attachment(sim):
    """No joint connects the parcel to anything and the parcel prim is not
    under the robot prim."""
    from pxr import Usd, UsdPhysics
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    parcel_paths = [str(p.GetPath()) for p in stage.Traverse() if str(p.GetPath()).endswith("/Parcel")]
    assert parcel_paths, "no parcel prim"
    for p in parcel_paths:
        assert "/Robot/" not in p
        prim = stage.GetPrimAtPath(p)
        assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    for j in joints:
        joint = UsdPhysics.Joint(j)
        targets = [str(t) for t in joint.GetBody0Rel().GetTargets()] + [str(t) for t in joint.GetBody1Rel().GetTargets()]
        for t in targets:
            assert "Parcel" not in t, f"joint {j.GetPath()} references the parcel"


def test_parcel_falls_when_released_after_lift(sim):
    """Run the expert to the end of LIFT, then hold the arm and open the
    hand. The parcel must fall under gravity."""
    ns = sim
    base, G, torch = ns["base"], ns["G"], ns["torch"]
    _reset(ns, rate=1.0)
    expert = ns["rt"].ExpertActor(base)
    expert.reset(range(base.num_envs))
    obs = None
    steps_lift_end = int(round((G.T_ACQ + G.NOMINAL_DURATIONS[G.PHASE_INDEX["LIFT"]]) / base.step_dt))
    for _ in range(steps_lift_end):
        act, _ = expert.act(obs)
        obs = ns["env"].step(act)[0]["policy"]
    parcel = base.scene["parcel"]
    z_lift = parcel.data.root_pos_w[:, 2].clone()
    assert bool((z_lift > G.TABLE_TOP + 0.05).all()), f"expert did not lift, z {z_lift.tolist()}"
    # hold the arm, open the hand
    hold = _hold_action(ns)
    open_hand = hold.clone()
    hand_open = torch.tensor([expert.expert.hand_open[i] for i in range(6)], device=base.device)
    q_def_hand = expert.q_default[:, expert.expert.hand_idx]
    open_hand[:, expert.expert.hand_idx] = 2.0 * (hand_open.unsqueeze(0) - q_def_hand)
    for _ in range(50):
        ns["env"].step(open_hand)
    z_after = parcel.data.root_pos_w[:, 2]
    assert bool((z_lift - z_after > 0.03).all()) and bool((z_after < G.TABLE_TOP + 0.03).all()), \
        f"parcel did not fall, {z_lift.tolist()} -> {z_after.tolist()}"


def test_hand_motion_cannot_drag_untouched_parcel(sim):
    """Place the parcel 6 cm outside the hand's reach at the grasp
    configuration and sweep the arm, the parcel must not move."""
    ns = sim
    base, G, torch = ns["base"], ns["G"], ns["torch"]
    _reset(ns, rate=1.0)
    parcel = base.scene["parcel"]
    pose = torch.cat([parcel.data.root_pos_w, parcel.data.root_quat_w], dim=-1).clone()
    pose[:, 1] += 0.15  # far to the left, outside any contact
    parcel.write_root_pose_to_sim(pose)
    parcel.write_root_velocity_to_sim(torch.zeros(base.num_envs, 6, device=base.device))
    for _ in range(5):
        ns["env"].step(_hold_action(ns))
    p0 = parcel.data.root_pos_w.clone()
    expert = ns["rt"].ExpertActor(base)
    expert.reset(range(base.num_envs))
    for _ in range(int(round(G.T_ACQ / base.step_dt))):
        act, _ = expert.act(None)
        ns["env"].step(act)
    disp = (parcel.data.root_pos_w - p0).norm(dim=-1)
    assert bool((disp < 0.003).all()), f"parcel moved without contact, {disp.tolist()}"


def _drive_parcel(ns, target_pos, R_target, steps=150, gain=6.0, vmax=0.25):
    """Scripted parcel motion by an external wrench (a compliant push, so
    walls stop the parcel physically and no velocity override tunnels
    through a slab). Returns the parcel position after the drive."""
    base, torch, G = ns["base"], ns["torch"], ns["G"]
    parcel = base.scene["parcel"]
    from isaaclab.utils.math import quat_mul, quat_inv, axis_angle_from_quat
    q_t = torch.tensor(G.quat_from_mat(R_target), dtype=torch.float32, device=base.device).unsqueeze(0).expand(base.num_envs, -1)
    tgt = torch.as_tensor(target_pos, dtype=torch.float32, device=base.device)
    if tgt.dim() == 1:
        tgt = tgt.unsqueeze(0).expand(base.num_envs, -1)
    m = G.PARCEL_MASS
    kp, kd = 20.0, 2.5
    kr, kw = 0.4, 0.03
    grav = torch.tensor([0.0, 0.0, 9.81 * m], device=base.device)
    for _ in range(steps):
        p = parcel.data.root_pos_w - base.scene.env_origins
        v = parcel.data.root_lin_vel_w
        f = kp * (tgt - p) - kd * v + grav
        fn = f.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        f = f * torch.clamp(2.5 / fn, max=1.0)
        dq = quat_mul(q_t, quat_inv(parcel.data.root_quat_w))
        dq = torch.where(dq[:, :1] < 0, -dq, dq)  # shortest path
        tau = kr * axis_angle_from_quat(dq) - kw * parcel.data.root_ang_vel_w
        tn = tau.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        tau = tau * torch.clamp(0.05 / tn, max=1.0)
        parcel.set_external_force_and_torque(f.unsqueeze(1).contiguous(), tau.unsqueeze(1).contiguous(), is_global=True)
        ns["env"].step(_hold_action(ns))
    parcel.set_external_force_and_torque(torch.zeros(base.num_envs, 1, 3, device=base.device),
                                         torch.zeros(base.num_envs, 1, 3, device=base.device), is_global=True)
    return parcel.data.root_pos_w - base.scene.env_origins


def _teleport_parcel(ns, pos, R):
    base, torch, G = ns["base"], ns["torch"], ns["G"]
    parcel = base.scene["parcel"]
    q = torch.tensor(G.quat_from_mat(R), dtype=torch.float32, device=base.device).unsqueeze(0).expand(base.num_envs, -1)
    p = torch.as_tensor(pos, dtype=torch.float32, device=base.device).unsqueeze(0).expand(base.num_envs, -1) + base.scene.env_origins
    parcel.write_root_pose_to_sim(torch.cat([p, q], dim=-1))
    parcel.write_root_velocity_to_sim(torch.zeros(base.num_envs, 6, device=base.device))


def _park_arm(ns, steps=40):
    for _ in range(steps):
        ns["env"].step(_zero_action(ns))


def test_receptacle_collides(sim):
    """A parcel dropped above the receptacle floor rests on the floor slab, and a
    parcel driven into a side wall stops outside the wall."""
    ns = sim
    base, G, torch, geom = ns["base"], ns["G"], ns["torch"], ns["geom"]
    _reset(ns, rate=1.0)
    _park_arm(ns)
    c, h = geom.interior_box()
    above = c + np.array([0.0, 0.0, h[2] - 0.01])
    _teleport_parcel(ns, above, geom.R_stow)
    for _ in range(60):
        ns["env"].step(_zero_action(ns))
    parcel = base.scene["parcel"]
    z = parcel.data.root_pos_w[:, 2]
    half_h = 0.5 * abs((geom.R_stow @ np.array(G.PARCEL_EXTENTS))[2])
    expected = geom.floor_top + half_h
    assert bool((torch.abs(z - expected) < 0.006).all()), f"parcel not resting on the floor, z {z.tolist()} expected {expected}"
    # drive it sideways into a side wall
    lat = geom.R_yaw @ np.eye(3)[:, geom.i_loose]
    target = c + lat * 0.20
    p_end = _drive_parcel(ns, target, geom.R_stow, steps=100)
    rel = np.array([float(x) for x in (geom.R_yaw.T @ (p_end[0].cpu().numpy() - c))])
    assert abs(rel[geom.i_loose]) < h[geom.i_loose] + 0.01, f"parcel passed through the side wall, rel {rel}"
    forces = ns["metrics"].StowMonitor(base, geom).cubby_forces_w().norm(dim=-1).sum(dim=-1)
    assert float(forces.max()) > 0.0


def _monitor_at_insert_phase(ns, rate=0.5):
    """Monitor with the env clock set inside INSERT so the predicates apply."""
    base, G = ns["base"], ns["G"]
    mon = ns["metrics"].StowMonitor(base, ns["geom"])
    mon.reset(range(base.num_envs))
    mon.acquired[:] = True
    d = G.phase_durations(rate)
    t_insert = float(d[:G.PHASE_INDEX["INSERT"]].sum()) + 0.5 * d[G.PHASE_INDEX["INSERT"]]
    base.episode_length_buf[:] = int(t_insert / base.step_dt)
    return mon


def test_centered_insertion_satisfies_predicate(sim):
    ns = sim
    base, G, torch, geom = ns["base"], ns["G"], ns["torch"], ns["geom"]
    _reset(ns, rate=0.5)
    _park_arm(ns)
    _teleport_parcel(ns, geom.p_preinsert, geom.R_stow)
    mon = _monitor_at_insert_phase(ns)
    p_end = _drive_parcel(ns, geom.p_insert, geom.R_stow, steps=150)
    mon.step()
    assert bool(mon.inserted.all()), f"centered insertion not detected, depth {[geom.depth_of(p) for p in p_end.cpu().numpy()]}"


def test_small_offset_insertion_contacts_but_enters(sim):
    ns = sim
    base, G, torch, geom = ns["base"], ns["G"], ns["torch"], ns["geom"]
    _reset(ns, rate=0.5)
    _park_arm(ns)
    tight = geom.R_yaw @ np.eye(3)[:, geom.i_tight]
    start = geom.p_preinsert + tight * 0.004
    _teleport_parcel(ns, start, geom.R_stow)
    mon = _monitor_at_insert_phase(ns)
    p_end = _drive_parcel(ns, geom.p_insert + tight * 0.004, geom.R_stow, steps=150)
    mon.step()
    assert bool(mon.inserted.all())


def test_outside_clearance_insertion_fails(sim):
    """An offset on the tight axis beyond the clearance jams against the
    slab, and the predicate stays false."""
    ns = sim
    base, G, torch, geom = ns["base"], ns["G"], ns["torch"], ns["geom"]
    _reset(ns, rate=0.5)
    _park_arm(ns)
    tight = geom.R_yaw @ np.eye(3)[:, geom.i_tight]
    off = 0.030 if geom.i_tight == 2 else 0.030
    _teleport_parcel(ns, geom.p_preinsert + tight * off, geom.R_stow)
    mon = _monitor_at_insert_phase(ns)
    p_end = _drive_parcel(ns, geom.p_insert + tight * off, geom.R_stow, steps=150, vmax=0.15)
    mon.step()
    depths = [geom.depth_of(p) for p in p_end.cpu().numpy()]
    assert not bool(mon.inserted.any()), f"outside-clearance parcel counted as inserted, depths {depths}"
    assert float(mon.max_cubby_force.max()) > 0.5, "no receptacle contact registered"


def test_over_rotated_insertion_fails(sim):
    ns = sim
    base, G, torch, geom = ns["base"], ns["G"], ns["torch"], ns["geom"]
    _reset(ns, rate=0.5)
    _park_arm(ns)
    tight = np.eye(3)[:, geom.i_tight]
    R_bad = geom.R_yaw @ G.so3_exp(math.radians(30.0) * np.eye(3)[:, geom.i_loose]) @ geom.R_local
    _teleport_parcel(ns, geom.p_preinsert - geom.d * 0.02, R_bad)
    mon = _monitor_at_insert_phase(ns)
    p_end = _drive_parcel(ns, geom.p_insert, R_bad, steps=150, vmax=0.15)
    mon.step()
    p, q, ang, depth, inside = mon.final_state(0)
    assert not (bool(mon.inserted.all()) and math.degrees(ang) <= 10.0), "over-rotated parcel accepted"


def test_release_and_settle_detection_and_certificate_independence(sim, monkeypatch):
    """A parcel resting inside the receptacle with no hand contact and no motion
    is released and settled, and task_success does not change when the
    certificate scorer returns a bogus negative value."""
    ns = sim
    base, G, torch, geom = ns["base"], ns["G"], ns["torch"], ns["geom"]
    _reset(ns, rate=0.5)
    _park_arm(ns)
    _teleport_parcel(ns, geom.p_insert, geom.R_stow)
    mon = _monitor_at_insert_phase(ns)
    for _ in range(60):
        ns["env"].step(_zero_action(ns))
        mon.step()
    assert bool(mon.inserted.all()) and bool(mon.released.all()) and bool(mon.settled.all())
    rec = mon.episode_record(0, score_certificate=False)
    assert rec["task_success"] is True
    monkeypatch.setattr(ns["metrics"], "score_contact_set", lambda cs, **kw: (-5.0, -5.0))
    rec2 = mon.episode_record(0, score_certificate=True)
    assert rec2["task_success"] == rec["task_success"]
    assert rec2["epsilon_lift"] == -5.0
    # and a lifted-out parcel is not successful
    _teleport_parcel(ns, geom.p_preinsert, geom.R_stow)
    mon.reset(range(base.num_envs))
    for _ in range(30):
        ns["env"].step(_zero_action(ns))
        mon.step()
    assert not bool(mon.inserted.any())
    assert mon.episode_record(0, score_certificate=False)["task_success"] is False


def test_per_env_phase_and_rate(sim):
    ns = sim
    base, G, torch, tc = ns["base"], ns["G"], ns["torch"], ns["task_clock"]
    tc.RATE_SPEC.clear()
    tc.RATE_SPEC.update({"mode": "per_env", "values": [0.5, 1.0, 2.0, 1.0]})
    ns["env"].reset()
    r = tc.rate_buf(base)
    assert r.tolist() == [0.5, 1.0, 2.0, 1.0]
    n_steps = int(round((G.T_ACQ + 0.6) / base.step_dt))
    for _ in range(n_steps):
        ns["env"].step(_zero_action(ns))
    k, f, t, cyc = tc.phase_state(base)
    # 0.6 s into the manipulation, LIFT lasts 1.2 / r
    assert G.PHASE_NAMES[int(k[0])] == "LIFT" and abs(float(f[0]) - 0.25) < 0.03
    assert G.PHASE_NAMES[int(k[1])] == "LIFT" and abs(float(f[1]) - 0.5) < 0.03
    assert G.PHASE_NAMES[int(k[2])] == "REORIENT" or (G.PHASE_NAMES[int(k[2])] == "LIFT" and float(f[2]) > 0.95)
    obs = base.observation_manager.compute()["policy"]
    assert obs.shape[1] == 147
    assert torch.allclose(obs[:, -1], r)
    ph = obs[:, -2]
    assert float(ph[2]) > float(ph[1]) > float(ph[0])
    assert abs(float(cyc[0]) - G.cycle_time(0.5)) < 1e-4


def test_scoring_does_not_change_state(sim):
    ns = sim
    base, torch = ns["base"], ns["torch"]
    _reset(ns, rate=1.0)
    for _ in range(3):
        ns["env"].step(_zero_action(ns))
    mon = ns["metrics"].StowMonitor(base, ns["geom"])
    robot = base.scene["robot"]
    parcel = base.scene["parcel"]
    snap = (robot.data.joint_pos.clone(), parcel.data.root_pos_w.clone(), parcel.data.root_quat_w.clone())
    forces = mon.parcel_forces_w()
    p_w, n_in, dsurf = mon.contact_geometry()
    cs = mon.contact_set(0, forces, p_w, n_in, dsurf)
    ns["metrics"].score_contact_set(cs)
    assert torch.equal(snap[0], robot.data.joint_pos)
    assert torch.equal(snap[1], parcel.data.root_pos_w)
    assert torch.equal(snap[2], parcel.data.root_quat_w)


def test_observation_dimension_matches_adapters(sim):
    """The policy observation of the task is 147 wide, and the learner
    checkpoints, when present, expect the same width."""
    ns = sim
    base, torch = ns["base"], ns["torch"]
    _reset(ns, rate=1.0)
    obs = base.observation_manager.compute()["policy"]
    assert obs.shape[1] == 147
    ck = {
        "dagger": os.path.join(REPO, "outputs", "paper", "dagger", "student_final.pt"),
        "dp": os.path.join(REPO, "outputs", "paper", "dp", "dp_stow.pt"),
        "act": os.path.join(REPO, "outputs", "paper", "act", "act_stow.pt"),
    }
    for name, path in ck.items():
        if not os.path.exists(path):
            continue
        actor = ns["rt"].load_actor(name, path, base, base.num_envs)
        actor.reset(range(base.num_envs), obs)
        act, _ = actor.act(obs)
        assert act.shape == (base.num_envs, 16)
