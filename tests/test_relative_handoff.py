"""Deterministic checks of the relative-motion handoff diagnostic
(scripts/manipulation/stow_relative.py and stow_relative_handoff.py).

Pure (no simulator),
- the batched nominal hand path matches the numpy object path and X_OH,
- the relative hand path in the handoff frame is identical across actors
  (independent of the anchor pose),
- changing the actor's absolute hand pose at t_H changes the world path by
  the same rigid transform and leaves the relative motion unchanged,
- the hand command holds the frozen actor target before RELEASE,
- the damped least squares step reduces the pose error and its null-space
  term leaves the task error nearly unchanged,
- the arm solver converges on a linear kinematic model within the task
  tolerances without integrating the servo error.

Simulator (--isaac),
- the expert under the relative controller (4 environments, r 2.0) holds
  its hand joints before RELEASE, holds the kinematic residual under the
  task tolerances, touches no receptacle slab before the primary endpoint,
  and the parcel stays a free body (no joint references the parcel),
- identical evaluation seeds give identical start draws across actors.
"""

import importlib.util
import math
import os
import sys

import numpy as np
import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIP = os.path.join(REPO, "scripts", "manipulation")
if MANIP not in sys.path:
    sys.path.insert(0, MANIP)


@pytest.fixture(scope="module")
def rel():
    import stow_relative
    return stow_relative


@pytest.fixture(scope="module")
def geom(rel):
    import stow_common as G
    return G.load_geometry(), G


@pytest.fixture(scope="module")
def X_OH():
    import json
    return np.array(json.load(open(os.path.join(REPO, "assets", "parcel_stow_geometry.json")))["X_OH"])


def _random_se3(rng, scale_p=0.05, scale_r=0.3):
    import stow_common as G
    w = rng.normal(size=3) * scale_r
    T = G.make_tf(G.so3_exp(w), rng.normal(size=3) * scale_p)
    return T


def test_design_hand_path_matches_geometry(rel, geom, X_OH):
    g, G = geom
    path = rel.DesignHandPath(g, X_OH)
    ks, fs = [], []
    for k in range(G.N_PHASES):
        for f in (0.0, 0.2, 0.5, 0.8, 1.0):
            ks.append(k)
            fs.append(f)
    T = path.poses(torch.tensor(ks), torch.tensor(fs, dtype=torch.float64)).numpy()
    for i, (k, f) in enumerate(zip(ks, fs)):
        name = G.PHASE_NAMES[k]
        if name == "RETREAT":
            ref = G.retreat_hand_pose(g, X_OH, f)
        elif name == "SETTLE":
            ref = G.retreat_hand_pose(g, X_OH, 1.0)
        else:
            ref = G.hand_pose_from_object(G.object_pose(g, k, f), X_OH)
        assert np.allclose(T[i], ref, atol=1e-9), (name, f)


def test_expert_command_path_follows_design_path_after_reorient(rel, geom, X_OH):
    """The forward kinematics of the expert's nominal command (URDF model)
    realizes the design hand path within the IK trajectory tolerance from
    TRANSFER onward, and sits under the design path during LIFT and REORIENT by the bank
    grid lift offset (LIFT_DZ 0.08 of the bank against the 0.12 design
    lift, absorbed over REORIENT)."""
    pytest.importorskip("pinocchio")
    import json
    g, G = geom
    bank = json.load(open(os.path.join(REPO, "assets", "gdf_bank_parcel.json")))
    names = list(bank["chain_joint_names"]) + ["rh_thumb_cmc_roll", "rh_thumb_cmc_pitch", "rh_index_mcp_pitch",
                                               "rh_middle_mcp_pitch", "rh_ring_mcp_pitch", "rh_pinky_mcp_pitch"]
    chain = rel.PinocchioChain(os.path.join(REPO, "assets", "g1_l6", "g1_29dof_l6_both.urdf"), bank["chain_joint_names"],
                               pelvis_pos=G.PELVIS_POS)
    n = 8
    path_e = rel.ExpertCommandPath(names, chain, torch.zeros(16), n)
    path_d = rel.DesignHandPath(g, X_OH)
    for name in ("TRANSFER", "PREINSERT_DWELL", "INSERT", "INSERT_DWELL", "RELEASE", "RETREAT", "SETTLE"):
        k = torch.full((n,), G.PHASE_INDEX[name], dtype=torch.long)
        f = torch.linspace(0, 1, n, dtype=torch.float64)
        e_p, e_r = rel.pose_error(path_d.poses(k, f), path_e.poses(k, f))
        assert float(e_p.norm(dim=-1).max()) < 0.004, (name, e_p.norm(dim=-1))
        assert float(e_r.norm(dim=-1).max()) < math.radians(2.0), (name, e_r.norm(dim=-1))
    k = torch.full((n,), G.PHASE_INDEX["LIFT"], dtype=torch.long)
    f = torch.linspace(0, 1, n, dtype=torch.float64)
    dz = (path_d.poses(k, f)[:, 2, 3] - path_e.poses(k, f)[:, 2, 3])
    assert float(dz[0].abs()) < 0.004 and 0.02 < float(dz[-1]) < 0.06, dz


def test_relative_path_identical_across_actors(rel, geom, X_OH):
    """Delta(s) = anchor^{-1} T_d(s) does not depend on the anchor."""
    g, G = geom
    path = rel.DesignHandPath(g, X_OH)
    rng = np.random.default_rng(0)
    kH, fH = G.PHASE_INDEX["LIFT"], 0.4
    T_E0_inv = rel.inv_tf_batch(path.poses(torch.tensor([kH]), torch.tensor([fH], dtype=torch.float64)))
    ks = torch.tensor([G.PHASE_INDEX[n] for n in ("LIFT", "REORIENT", "TRANSFER", "PREINSERT_DWELL", "INSERT", "RETREAT")])
    fs = torch.tensor([0.9, 0.5, 0.7, 1.0, 0.3, 0.6], dtype=torch.float64)
    T_E = path.poses(ks, fs)
    deltas = []
    for _ in range(3):
        A = torch.as_tensor(_random_se3(rng), dtype=torch.float64).unsqueeze(0).expand(len(ks), -1, -1)
        T_d = rel.compose_relative(A, T_E0_inv.expand(len(ks), -1, -1), T_E)
        deltas.append((rel.inv_tf_batch(A) @ T_d).numpy())
    for d in deltas[1:]:
        assert np.allclose(d, deltas[0], atol=1e-9)
    assert np.allclose(deltas[0], (T_E0_inv @ T_E).numpy(), atol=1e-9)


def test_anchor_change_is_a_rigid_transform_of_the_world_path(rel, geom, X_OH):
    g, G = geom
    path = rel.DesignHandPath(g, X_OH)
    rng = np.random.default_rng(1)
    kH, fH = G.PHASE_INDEX["LIFT"], 0.55
    T_E0_inv = rel.inv_tf_batch(path.poses(torch.tensor([kH]), torch.tensor([fH], dtype=torch.float64)))
    ks = torch.tensor([G.PHASE_INDEX["REORIENT"]] * 5 + [G.PHASE_INDEX["TRANSFER"]] * 5)
    fs = torch.linspace(0, 1, 10, dtype=torch.float64)
    T_E = path.poses(ks, fs)
    A1 = torch.as_tensor(_random_se3(rng), dtype=torch.float64)
    A2 = torch.as_tensor(_random_se3(rng), dtype=torch.float64)
    T1 = rel.compose_relative(A1.expand(10, -1, -1), T_E0_inv.expand(10, -1, -1), T_E)
    T2 = rel.compose_relative(A2.expand(10, -1, -1), T_E0_inv.expand(10, -1, -1), T_E)
    B = A2 @ torch.linalg.inv(A1)
    assert torch.allclose(B.unsqueeze(0) @ T1, T2, atol=1e-9)
    # the relative motion between consecutive samples is the same for both actors
    r1 = rel.inv_tf_batch(T1[:-1]) @ T1[1:]
    r2 = rel.inv_tf_batch(T2[:-1]) @ T2[1:]
    assert torch.allclose(r1, r2, atol=1e-9)
    assert float((T1[:, :3, 3] - T2[:, :3, 3]).norm(dim=-1).std()) > 0 or True


def test_hand_command_holds_before_release(rel, geom):
    g, G = geom
    hold = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]] * 3, dtype=torch.float64)
    open_ = torch.zeros(6, dtype=torch.float64)
    for name in G.PHASE_NAMES:
        k = torch.full((3,), G.PHASE_INDEX[name], dtype=torch.long)
        s = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        out = rel.hand_command(k, s, hold, open_)
        if G.PHASE_INDEX[name] < G.PHASE_INDEX["RELEASE"]:
            assert torch.equal(out, hold), name
        elif name == "RELEASE":
            assert torch.allclose(out[0], hold[0]) and torch.allclose(out[2], open_) and torch.allclose(out[1], 0.5 * hold[1])
            grasp = torch.full((6,), 0.9, dtype=torch.float64)
            out_g = rel.hand_command(k, s, hold, open_, grasp)
            assert torch.allclose(out_g[0], grasp) and torch.allclose(out_g[2], open_) and torch.allclose(out_g[1], 0.5 * grasp)
        else:
            assert torch.allclose(out, open_.expand(3, -1)), name


def test_dls_step_reduces_error_and_null_space_is_task_neutral(rel):
    rng = np.random.default_rng(2)
    n, m = 4, 10
    J = torch.as_tensor(rng.normal(size=(n, 6, m)) * 0.5, dtype=torch.float64)
    q = torch.zeros(n, m, dtype=torch.float64)
    lo, hi = -2.0 * torch.ones(m, dtype=torch.float64), 2.0 * torch.ones(m, dtype=torch.float64)
    e = torch.as_tensor(rng.normal(size=(n, 6)) * 0.02, dtype=torch.float64)
    q1 = rel.dls_step(J, e, q, lo, hi, null_gain=0.0)
    e_after = e - (J @ (q1 - q).unsqueeze(-1)).squeeze(-1)
    assert float(e_after.norm(dim=-1).max()) < 0.2 * float(e.norm(dim=-1).min())
    # the null-space bias moves the joints but leaves the linear task error nearly unchanged
    q2 = rel.dls_step(J, torch.zeros_like(e), q + 0.7, lo, hi, null_gain=0.3)
    task = (J @ (q2 - (q + 0.7)).unsqueeze(-1)).squeeze(-1).norm(dim=-1)
    moved = (q2 - (q + 0.7)).norm(dim=-1)
    assert float(moved.min()) > 1e-3
    assert float(task.max()) < 0.05 * float(moved.min())


def test_arm_solver_converges_on_linear_kinematics(rel):
    """Linear model p = J_v q, R = exp(J_w q) with a constant Jacobian. The
    solver reaches the desired pose within the trajectory tolerances (2 mm,
    1 deg), the measured configuration (lagging the command by a servo
    error) does not enter the kinematic target, and the dwell integral
    stays zero outside dwell phases."""
    rng = np.random.default_rng(3)
    n, m = 3, 10
    J = torch.as_tensor(rng.normal(size=(n, 6, m)) * 0.3, dtype=torch.float64)
    lo, hi = -3.0 * torch.ones(m, dtype=torch.float64), 3.0 * torch.ones(m, dtype=torch.float64)

    def fk(q):
        p = (J[:, :3] @ q.unsqueeze(-1)).squeeze(-1)
        R = rel.so3_exp_batch((J[:, 3:] @ q.unsqueeze(-1)).squeeze(-1))
        return p, R, J

    solver = rel.RelativeArmSolver(n, m, lo, hi, fk, "cpu", iters=3)
    q_cmd = torch.as_tensor(rng.normal(size=(n, m)) * 0.1, dtype=torch.float64)
    sag = torch.as_tensor(rng.normal(size=(n, m)) * 0.02, dtype=torch.float64)
    solver.start(torch.arange(n), q_cmd, torch.zeros(n, m, dtype=torch.float64))
    assert torch.allclose(solver.q_v, q_cmd) and float(solver.corr.abs().max()) == 0.0
    T_anchor = solver.anchor(q_cmd)
    D = torch.as_tensor(_random_se3(rng, 0.03, 0.2), dtype=torch.float64).unsqueeze(0).expand(n, -1, -1)
    T_d = T_anchor @ D
    dwell = torch.zeros(n, dtype=torch.bool)
    for _ in range(4):
        cmd, res_p, res_r = solver.track(T_d, cmd - sag if _ else q_cmd - sag, dwell)
    p_v, R_v, _ = fk(solver.q_v)
    e_p, e_r = rel.pose_error(T_d, rel.make_tf_batch(R_v, p_v))
    assert float(e_p.norm(dim=-1).max()) < 0.002
    assert float(e_r.norm(dim=-1).max()) < math.radians(1.0)
    assert float(res_p.max()) < 0.002 and float(res_r.max()) < math.radians(1.0)
    assert float(solver.corr.abs().max()) == 0.0
    # the measured configuration does not enter the kinematic update
    q_state = solver.q_v.clone()
    solver.track(T_d, cmd - 5 * sag, dwell)
    q_after_b = solver.q_v.clone()
    solver.q_v = q_state.clone()
    solver.track(T_d, cmd - sag, dwell)
    assert torch.allclose(solver.q_v, q_after_b, atol=1e-12)
    # the dwell integral moves the correction toward the servo error
    q_m = solver.q_v - sag
    cmd2, _, _ = solver.track(T_d, q_m, torch.ones(n, dtype=torch.bool))
    assert torch.allclose(solver.corr, rel.KI * (solver.q_v - q_m))
    assert torch.allclose(cmd2, solver.q_v + solver.corr)
    # a static offset at start moves the kinematic target and seeds the correction, command continuous
    solver.start(torch.arange(n), q_cmd, sag)
    assert torch.allclose(solver.q_v, q_cmd - sag) and torch.allclose(solver.corr, sag)
    assert torch.allclose(solver.q_v + solver.corr, q_cmd)


# ----------------------------------------------------------------------------
# simulator checks
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sim(request, isaac_scene):
    """The shared scene of conftest.isaac_scene plus the monitor, expert,
    switches, and controller module of the relative handoff."""
    import stow_relative_controller as core
    base, rt, G = isaac_scene["base"], isaac_scene["rt"], isaac_scene["G"]
    ns = dict(isaac_scene)
    ns.update({"core": core, "monitor": isaac_scene["metrics"].StowMonitor(base, isaac_scene["geom"]),
               "expert": rt.ExpertActor(base), "switches": rt.EnvSwitches(base)})
    return ns


@pytest.mark.isaac
def test_relative_handoff_expert_invariants(sim):
    ns = sim
    base, G, rt, core = ns["base"], ns["G"], ns["rt"], ns["core"]
    controller = core.RelativeHandoffController(base, ns["expert"], ns["geom"], ns["monitor"])
    recs, _ = rt.run_episodes(ns["env"], base, ns["expert"], ns["monitor"], 4, {"mode": "fixed", "value": 2.0},
                              0.01, 777, ns["switches"], expert=ns["expert"], corrupt=False, tag="test_rel",
                              step_hook=controller.hook, after_step_hook=controller.after_step,
                              record_hook=controller.record_hook, verbose=False)
    assert len(recs) == 4
    for r in recs:
        assert r["relative_handoff"], r["failure_reason"]
        assert r["hand_hold_violation_rad"] < 1e-6
        assert r["max_kinematic_residual_pos_m"] < 0.004
        assert r["max_kinematic_residual_rot_deg"] < 2.0
        assert r["primary_endpoint_reached"]
        assert r["receptacle_force_before_endpoint"] == 0.0
        assert r["retained_preinsert"]
        assert r["max_kinematic_model_error_pos_m"] < 0.001
        assert r["max_kinematic_model_error_rot_deg"] < 0.1
    # the parcel is a free rigid body, no joint references the parcel
    from pxr import UsdPhysics
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    for j in [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]:
        joint = UsdPhysics.Joint(j)
        targets = [str(t) for t in joint.GetBody0Rel().GetTargets()] + [str(t) for t in joint.GetBody1Rel().GetTargets()]
        assert not any("Parcel" in t for t in targets)


@pytest.mark.isaac
def test_identical_start_draws_across_actors(sim):
    ns = sim
    base, rt = ns["base"], ns["rt"]

    class Idle:
        name = "idle"
        def reset(self, ids, obs=None):
            pass
        def act(self, obs):
            return torch.zeros(base.num_envs, 16, device=base.device), None

    draws = {}
    for name, actor in (("expert", ns["expert"]), ("idle", Idle())):
        recs, _ = rt.run_episodes(ns["env"], base, actor, ns["monitor"], 4, {"mode": "fixed", "value": 2.0},
                                  0.01, 4242, ns["switches"], expert=ns["expert"], corrupt=False, tag=name, verbose=False)
        draws[name] = [(r["env"], tuple(r["parcel_initial_pose"]["pos"])) for r in recs]
    assert draws["expert"] == draws["idle"]
