"""Pure pieces of the relative-motion handoff diagnostic of ParcelStow
(scripts/manipulation/stow_relative_handoff.py), torch and numpy only, so
the checks import them without the simulator.

The diagnostic freezes, at the stable handoff time t_H, the actor's actual
world hand pose T_WH^pi(t_H) and applies the expert's nominal downstream
relative hand motion in the handoff frame,

    Delta T_H^E(s) = (T_WH^E(t_H))^{-1} T_WH^E(s),
    T_WH,d^pi(s)   = T_WH^pi(t_H) Delta T_H^E(s),

where T_WH^E(k, f) is the hand pose of the expert's nominal command
(ExpertCommandPath, the URDF forward kinematics of the expert's joint target
at the nominal start, within the IK tolerance of the frozen object path
T_WO^d(k, f) X_OH of geometry.object_pose from the end of REORIENT onward,
DesignHandPath). compose_relative forms the actor-specific desired pose,
and RelativeArmSolver tracks the desired pose in the joint-target domain
with the damped least squares step of stow_ik.ChainIK (same damping,
null-space bias toward the joint mid-ranges, step clamp, and limit clip) on
the URDF kinematic model of the chain (PinocchioChain, checked against the
PhysX hand pose in every episode). The solver never reads the parcel, and
the parcel stays a free rigid body.

The anchor is the forward kinematics of the actor's kinematic arm target at
t_H, the arm command minus the actor's static servo offset (command minus
measured configuration, sampled at rest at the end of GRASP_DWELL), not the
measured hand pose and not the raw command. The measured pose sits behind
the command by the servo transient, and anchoring the relative motion at
the measured pose turns the transient into a permanent offset of the whole
downstream path. The raw command contains the actor's sag compensation (the
expert's dwell correction), and anchoring at the raw command and then
integrating the dwell correction again lifts the insertion by the
compensation (a debug run of the expert at r 2 with the raw-command anchor
jammed 32 of 32 insertions). With the static
offset removed from the target and retained as the initial dwell correction,
the expert under the relative controller commands its own path, and every
actor's servo transient plays out as in its own run. The measured pose, the
anchor, and their difference are recorded per episode.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import stow_common as G  # the geometry module by file path, or the package module when loaded
import parcel_stow_expert as pse

DLS_LAMBDA = 0.05
NULL_GAIN = 0.3
MAX_STEP = 0.15
KI = 0.08
CORR_CLAMP = 0.35
DWELL_PHASES = ("PREGRASP_DWELL", "GRASP_DWELL", "PREINSERT_DWELL", "INSERT_DWELL", "SETTLE")


# ----------------------------------------------------------------------------
# batched SO(3) and SE(3) helpers (torch)
# ----------------------------------------------------------------------------
def so3_exp_batch(w):
    """(n, 3) rotation vectors -> (n, 3, 3) rotation matrices."""
    a = w.norm(dim=-1, keepdim=True)
    small = a < 1e-8
    k = torch.where(small, torch.zeros_like(w), w / a.clamp(min=1e-12))
    K = torch.zeros(w.shape[0], 3, 3, dtype=w.dtype, device=w.device)
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    eye = torch.eye(3, dtype=w.dtype, device=w.device).unsqueeze(0)
    s = torch.sin(a).unsqueeze(-1)
    c = (1.0 - torch.cos(a)).unsqueeze(-1)
    R = eye + s * K + c * (K @ K)
    return torch.where(small.unsqueeze(-1), eye.expand_as(R), R)


def so3_log_batch(R):
    """(n, 3, 3) rotation matrices -> (n, 3) rotation vectors, the general
    branch with the small-angle limit (errors of the tracking loop stay far
    from pi)."""
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    c = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    a = torch.acos(c)
    w = torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], dim=-1)
    small = a < 1e-6
    scale = torch.where(small, torch.full_like(a, 0.5), a / (2.0 * torch.sin(a).clamp(min=1e-12)))
    return w * scale.unsqueeze(-1)


def mat_from_quat_batch(q):
    """(n, 4) wxyz quaternions -> (n, 3, 3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=-1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1),
    ], dim=1)
    return R


def make_tf_batch(R, p):
    n = R.shape[0]
    T = torch.zeros(n, 4, 4, dtype=R.dtype, device=R.device)
    T[:, :3, :3] = R
    T[:, :3, 3] = p
    T[:, 3, 3] = 1.0
    return T


def inv_tf_batch(T):
    R = T[:, :3, :3]
    p = T[:, :3, 3]
    Rt = R.transpose(1, 2)
    return make_tf_batch(Rt, -(Rt @ p.unsqueeze(-1)).squeeze(-1))


def pose_error(T_d, T):
    """World-frame position error and rotation-vector error of T toward T_d,
    e_p = p_d - p, e_r = log(R_d R^T), the convention of stow_ik.ChainIK."""
    e_p = T_d[:, :3, 3] - T[:, :3, 3]
    e_r = so3_log_batch(T_d[:, :3, :3] @ T[:, :3, :3].transpose(1, 2))
    return e_p, e_r


# ----------------------------------------------------------------------------
# nominal hand path of the expert, T_WH^E(k, f)
# ----------------------------------------------------------------------------
class DesignHandPath:
    """Batched evaluation of the design hand pose on the frozen object path
    from the nominal parcel start, T_WO^d(k, f) X_OH for LIFT through
    RELEASE, the retreat hand pose during RETREAT, and the retreat end during
    SETTLE. Before LIFT the object sits at its start pose (the hand pose of
    the grasp). The IK trajectory knots sit within 2 mm and 1 deg of the
    design path from the end of REORIENT onward. The expert's LIFT command
    ends at the bank grid lift knot (LIFT_DZ 0.08 m of geometry.py at bank
    build time) while the frozen geometry lifts 0.12 m, and the difference
    decays over REORIENT, so during LIFT and REORIENT the expert's commanded
    path sits up to 4 cm under the design path. The relative controller
    therefore uses ExpertCommandPath, the forward kinematics of the expert's
    own command, and DesignHandPath serves the checks."""

    def __init__(self, geom, X_OH, device="cpu", dtype=torch.float64):
        self.geom = geom
        d = device
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=dtype, device=d)  # noqa: E731
        self.dtype = dtype
        self.device = d
        self.X_OH = t(X_OH)
        self.R0 = t(geom.R_start)
        self.R_stow = t(geom.R_stow)
        self.w_reorient = t(G.so3_log(np.asarray(geom.R_start).T @ np.asarray(geom.R_stow)))
        p0 = np.array(G.PARCEL_START)
        p_lift = p0 + geom.R_yaw @ np.array([geom.lift_dx, 0.0, geom.lift_dz])
        p_mid = p_lift + geom.reorient_travel * (geom.p_preinsert - p_lift)
        self.p0, self.p_lift, self.p_mid = t(p0), t(p_lift), t(p_mid)
        self.p_pre, self.p_ins = t(geom.p_preinsert), t(geom.p_insert)
        self.d = t(geom.d)
        self.retreat = float(G.RETREAT_DISTANCE)
        self.k = {name: G.PHASE_INDEX[name] for name in G.PHASE_NAMES}

    def poses(self, k, f):
        """k (n,) long phase indices, f (n,) in-phase fractions -> (n, 4, 4)
        nominal world hand poses."""
        n = k.shape[0]
        f = f.to(self.dtype).clamp(0.0, 1.0)
        s = 0.5 * (1.0 - torch.cos(math.pi * f))
        s3 = s.unsqueeze(-1)
        K = self.k
        eyeR = self.R0.unsqueeze(0).expand(n, -1, -1)
        R = eyeR.clone()
        p = self.p0.unsqueeze(0).expand(n, -1).clone()
        m = (k == K["LIFT"]).unsqueeze(-1)
        p = torch.where(m, self.p0 + s3 * (self.p_lift - self.p0), p)
        m = k == K["REORIENT"]
        R = torch.where(m.view(n, 1, 1), self.R0.unsqueeze(0) @ so3_exp_batch(s3 * self.w_reorient), R)
        p = torch.where(m.unsqueeze(-1), self.p_lift + s3 * (self.p_mid - self.p_lift), p)
        stow = k >= K["TRANSFER"]
        R = torch.where(stow.view(n, 1, 1), self.R_stow.unsqueeze(0).expand(n, -1, -1), R)
        m = (k == K["TRANSFER"]).unsqueeze(-1)
        p = torch.where(m, self.p_mid + s3 * (self.p_pre - self.p_mid), p)
        m = (k == K["PREINSERT_DWELL"]).unsqueeze(-1)
        p = torch.where(m, self.p_pre.unsqueeze(0).expand(n, -1), p)
        m = (k == K["INSERT"]).unsqueeze(-1)
        p = torch.where(m, self.p_pre + s3 * (self.p_ins - self.p_pre), p)
        m = (k >= K["INSERT_DWELL"]).unsqueeze(-1)
        p = torch.where(m, self.p_ins.unsqueeze(0).expand(n, -1), p)
        T_WO = make_tf_batch(R, p)
        T_WH = T_WO @ self.X_OH.unsqueeze(0)
        # retreat, the insert hand pose backed out along -d by the eased fraction
        back = torch.where(k == K["RETREAT"], s, torch.zeros_like(s))
        back = torch.where(k == K["SETTLE"], torch.ones_like(s), back)
        T_WH = T_WH.clone()
        T_WH[:, :3, 3] = T_WH[:, :3, 3] - self.retreat * back.unsqueeze(-1) * self.d
        return T_WH


def compose_relative(T_anchor, T_E0_inv, T_E):
    """Actor-specific desired pose T_WH,d = T_anchor (T_E(t_H))^{-1} T_E(s),
    all (n, 4, 4)."""
    return T_anchor @ T_E0_inv @ T_E


def hand_command(k, s, hold, hand_open, hand_grasp=None):
    """Hand joint target of the diagnostic, the frozen actor target before
    RELEASE, the expert's opening over RELEASE (the cosine blend from the
    bank grasp shape to the open shape, the same hand path stow_handoff.py
    applies from RELEASE, the expert zeroes its own hand correction there),
    and the open shape afterward. With hand_grasp None the blend starts from
    the frozen target. k (n,), s (n,) eased RELEASE fraction, hold (n, 6),
    hand_open (6,), hand_grasp (6,)."""
    k_rel = G.PHASE_INDEX["RELEASE"]
    start = hold if hand_grasp is None else hand_grasp.unsqueeze(0).expand_as(hold)
    opened = start + s.unsqueeze(-1) * (hand_open.unsqueeze(0) - start)
    out = torch.where((k == k_rel).unsqueeze(-1), opened, hold)
    out = torch.where((k > k_rel).unsqueeze(-1), hand_open.unsqueeze(0).expand_as(hold), out)
    return out


# ----------------------------------------------------------------------------
# damped least squares step and the arm solver
# ----------------------------------------------------------------------------
def dls_step(J, e, q, lo, hi, lam=DLS_LAMBDA, null_gain=NULL_GAIN, max_step=MAX_STEP, q_ref=None):
    """One damped least squares update of stow_ik.ChainIK.solve, batched.
    J (n, 6, m), e (n, 6), q (n, m), lo and hi (n, m) or (m,). q_ref is the
    null-space attractor (the joint mid-ranges when None). Returns the
    clipped new configuration."""
    n, _, m = J.shape
    dtype, device = J.dtype, J.device
    lo = torch.as_tensor(lo, dtype=dtype, device=device).expand(n, m)
    hi = torch.as_tensor(hi, dtype=dtype, device=device).expand(n, m)
    JJt = J @ J.transpose(1, 2) + (lam * lam) * torch.eye(6, dtype=dtype, device=device).unsqueeze(0)
    y = torch.linalg.solve(JJt, e.unsqueeze(-1))
    dq_task = (J.transpose(1, 2) @ y).squeeze(-1)
    N = torch.eye(m, dtype=dtype, device=device).unsqueeze(0) - J.transpose(1, 2) @ torch.linalg.solve(JJt, J)
    ref = 0.5 * (lo + hi) if q_ref is None else q_ref
    rng = (hi - lo).clamp(min=1e-6)
    bias = null_gain * (ref - q) / rng
    dq = dq_task + (N @ bias.unsqueeze(-1)).squeeze(-1)
    step = dq.norm(dim=-1, keepdim=True)
    dq = dq * torch.where(step > max_step, max_step / step.clamp(min=1e-12), torch.ones_like(step))
    return torch.minimum(torch.maximum(q + dq, lo), hi)


class PinocchioChain:
    """Forward kinematics and world-aligned frame Jacobian of the waist-arm
    chain from the robot URDF (pinocchio), the kinematic model of the online
    solver. The pelvis is the fixed root at pelvis_pos with identity
    rotation, so poses come out in the environment-local frame of ParcelStow.
    Joints outside the chain sit at the given defaults (they do not move the
    right hand). The driver checks the URDF model against the PhysX hand
    pose of the live articulation at every step of the diagnostic."""

    def __init__(self, urdf_path, chain_names, frame_name="rh_hand_base_link", pelvis_pos=None,
                 defaults=None):
        import pinocchio as pin
        self.pin = pin
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.fid = self.model.getFrameId(frame_name)
        assert self.fid < self.model.nframes, frame_name
        self.chain_names = list(chain_names)
        jids = [self.model.getJointId(n) for n in self.chain_names]
        assert all(j < self.model.njoints for j in jids), self.chain_names
        self.idx_q = np.array([self.model.joints[j].idx_q for j in jids])
        self.idx_v = np.array([self.model.joints[j].idx_v for j in jids])
        self.q_full = pin.neutral(self.model)
        if defaults:
            for name, val in defaults.items():
                if self.model.existJointName(name):
                    self.q_full[self.model.joints[self.model.getJointId(name)].idx_q] = float(val)
        self.pelvis_pos = np.zeros(3) if pelvis_pos is None else np.asarray(pelvis_pos, dtype=np.float64)

    def fk_jac(self, q_arm):
        """q_arm (n, m) -> p (n, 3), R (n, 3, 3), J (n, 6, m), the frame
        origin position, orientation, and LOCAL_WORLD_ALIGNED Jacobian."""
        pin = self.pin
        qa = np.asarray(q_arm.detach().cpu().numpy() if torch.is_tensor(q_arm) else q_arm, dtype=np.float64)
        n, m = qa.shape
        p = np.zeros((n, 3))
        R = np.zeros((n, 3, 3))
        J = np.zeros((n, 6, m))
        q = self.q_full.copy()
        for i in range(n):
            q[self.idx_q] = qa[i]
            pin.forwardKinematics(self.model, self.data, q)
            oMf = pin.updateFramePlacement(self.model, self.data, self.fid)
            p[i] = oMf.translation + self.pelvis_pos
            R[i] = oMf.rotation
            J[i] = pin.computeFrameJacobian(self.model, self.data, q, self.fid, pin.LOCAL_WORLD_ALIGNED)[:, self.idx_v]
        dev = q_arm.device if torch.is_tensor(q_arm) else "cpu"
        t = lambda a: torch.as_tensor(a, dtype=torch.float64, device=dev)  # noqa: E731
        return t(p), t(R), t(J)


class ExpertCommandPath:
    """Hand pose of the expert's nominal command, T_WH^E(k, f) = FK(target^E(k,
    f)) with target^E parcel_stow_expert.StowExpert.target at the nominal
    start (grid entry dx = dy = 0, no dwell correction) and FK the URDF
    kinematics of PinocchioChain. This is the path the expert commands in its
    own episodes (LIFT ends at the bank grid lift knot and REORIENT absorbs
    the difference to the design lift, see IMPLEMENTATION_LOG.md), so an
    actor under the relative controller inherits the expert's own commanded
    downstream motion. The path is a function of (phase, fraction) only, the
    same for every actor and episode."""

    def __init__(self, actuated_names, chain, q_default, n, device="cpu"):
        self.expert = pse.StowExpert(actuated_names, device=device)
        self.expert.allocate(n)
        self.expert.reset(range(n), torch.zeros(n, 2, device=device))
        self.chain = chain
        self.q_default = torch.as_tensor(q_default, dtype=torch.float32, device=device)
        if self.q_default.dim() == 1:
            self.q_default = self.q_default.unsqueeze(0).expand(n, -1).contiguous()
        self.arm_idx = list(self.expert.arm_idx)

    def targets(self, k, f):
        """(n, 16) joint targets of the nominal expert command."""
        return self.expert.target(k, f.to(torch.float32), self.q_default)

    def poses(self, k, f):
        """k (n,) long, f (n,) -> (n, 4, 4) hand poses of the nominal expert
        command in the environment-local frame."""
        q_t = self.targets(k, f)
        p, R, _ = self.chain.fk_jac(q_t[:, self.arm_idx].to(torch.float64))
        return make_tf_batch(R, p)


class RelativeArmSolver:
    """Kinematic tracking of a desired hand pose in the joint-target domain,
    the damped least squares iteration of stow_ik.ChainIK.solve run online
    on a kinematic model. The solver holds a target configuration q_v per
    environment (initialized to the actor's arm command at t_H, whose
    forward kinematics is the anchor pose) and moves it by DLS steps toward
    the desired pose. The measured state never enters the kinematic update,
    so the position servo's lag and sag do not feed back into the target
    (no windup, no double lag). The dwell integral of the expert (KI on the
    target minus the measured configuration during dwell phases, clamped)
    sits on top as in parcel_stow_expert.StowExpert.act, so the arm settles
    before release."""

    def __init__(self, n, m, lo, hi, fk, device, dtype=torch.float64, lam=DLS_LAMBDA, null_gain=NULL_GAIN,
                 max_step=MAX_STEP, iters=3, ki=KI, corr_clamp=CORR_CLAMP):
        self.n, self.m = n, m
        self.device, self.dtype = device, dtype
        self.fk = fk
        self.lo = torch.as_tensor(lo, dtype=dtype, device=device).expand(n, m).clone()
        self.hi = torch.as_tensor(hi, dtype=dtype, device=device).expand(n, m).clone()
        self.lam, self.null_gain, self.max_step, self.iters = lam, null_gain, max_step, iters
        self.ki, self.corr_clamp = ki, corr_clamp
        self.q_v = torch.zeros(n, m, dtype=dtype, device=device)
        self.corr = torch.zeros(n, m, dtype=dtype, device=device)

    def start(self, ids, q_cmd, offset):
        """Begin tracking for environments ids. The kinematic target is the
        actor's arm command minus its static servo offset (the command minus
        the measured configuration sampled at rest at the end of
        GRASP_DWELL), and the correction starts at that offset, so the
        command q_v + corr is continuous at t_H and the offset plays the
        role of the expert's dwell correction from then on."""
        off = offset[ids].to(self.dtype).clamp(-self.corr_clamp, self.corr_clamp)
        self.q_v[ids] = q_cmd[ids].to(self.dtype) - off
        self.corr[ids] = off

    def anchor(self, q_cmd):
        """Forward kinematics of a command batch, (n, 4, 4)."""
        p, R, _ = self.fk(q_cmd.to(self.dtype))
        return make_tf_batch(R, p)

    def track(self, T_d, q_meas, dwell, q_ref=None):
        """One control step. T_d (n, 4, 4) desired poses, q_meas (n, m)
        measured configuration (dwell integral only), dwell (n,) bool.
        Returns the arm command q_v + corr (n, m) and the kinematic residual
        (position m, rotation rad) of q_v after the update."""
        q_meas = q_meas.to(self.dtype)
        for _ in range(self.iters):
            p, R, J = self.fk(self.q_v)
            e_p, e_r = pose_error(T_d, make_tf_batch(R, p))
            self.q_v = dls_step(J, torch.cat([e_p, e_r], dim=-1), self.q_v, self.lo, self.hi,
                                self.lam, self.null_gain, self.max_step, q_ref)
        p, R, _ = self.fk(self.q_v)
        e_p, e_r = pose_error(T_d, make_tf_batch(R, p))
        upd = self.ki * (self.q_v - q_meas)
        self.corr = torch.where(dwell.unsqueeze(-1), self.corr + upd, self.corr).clamp(-self.corr_clamp, self.corr_clamp)
        return self.q_v + self.corr, e_p.norm(dim=-1), e_r.norm(dim=-1)


def dwell_mask(k):
    m = torch.zeros_like(k, dtype=torch.bool)
    for name in DWELL_PHASES:
        m |= k == G.PHASE_INDEX[name]
    return m
