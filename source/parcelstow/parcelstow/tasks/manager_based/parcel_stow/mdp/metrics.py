"""Physical monitor of the ParcelStow task, stage markers, failure
attribution, in-hand slip, realized contact sets, receptacle contact, and
actuator utilization, all read from the simulator state after every control
step. The drivers own the object, call step() after env.step(), and take
episode_record(i) when environment i finishes.

Every predicate here reads physical state only. The certificate values
(epsilon, epsilon^(beta)) are computed from the recorded contact sets as
diagnostics and never feed a marker. TASK_SPEC.md sections 8 and 9 hold the
thresholds, repeated as module constants below.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_inv, quat_mul, subtract_frame_transforms

from .. import geometry as G

DISTAL_BODIES = ["rh_thumb_distal", "rh_index_distal", "rh_middle_distal", "rh_ring_distal", "rh_pinky_distal"]
TIP_FRAMES = ["rh_thumb_tip", "rh_index_tip", "rh_middle_tip", "rh_ring_tip", "rh_pinky_tip"]
HAND_ROOT = "rh_hand_base_link"
SEGMENT_SAMPLES = 12

# thresholds (TASK_SPEC.md section 8)
ACQUIRE_DZ = 0.020
LIFT_CLEAR_Z = G.TABLE_TOP + 0.060
REORIENT_TOL_DEG = 15.0
PREINSERT_POS_TOL = 0.030
PREINSERT_ANG_TOL_DEG = 15.0
FINAL_ANG_TOL_DEG = 10.0
CONTACT_THRESHOLD = 1.0
RELEASE_FORCE = 0.5
RELEASE_STEPS = 5
SETTLE_LIN = 0.02
SETTLE_ANG = 0.2
SETTLE_STEPS = 20
DROP_FORCE = 0.5
DROP_STEPS = 10
JAM_FORCE = 2.0
JAM_STEPS = 10
SLIP_TRANS_LIMIT = 0.008
SLIP_ROT_LIMIT_DEG = 10.0

FAILURE_REASONS = [
    "acquisition_failure", "dropped_during_lift", "dropped_during_reorientation",
    "excessive_inhand_slip", "transfer_failure", "insertion_misalignment", "insertion_jam",
    "release_failure", "timeout", "other",
]


def _quat_angle(q):
    """Rotation angle of a wxyz quaternion batch, radians, (E,)."""
    w = q[:, 0].abs().clamp(max=1.0)
    return 2.0 * torch.acos(w)


def score_contact_set(cs, mu=0.5, beta=0.95, mu_std=0.15, n_particles=50):
    """Ferrari-Canny margin at friction mu about the parcel center (computed
    in-repo, ferrari_canny.py) and the risk-adjusted epsilon^(beta) of a
    recorded contact set (optional, needs the firmgrasp package). Fewer than
    two contacts gives (-1, -1), a missing firmgrasp gives epsilon_beta -1."""
    if cs is None or len(cs["contacts"]) < 2:
        return -1.0, -1.0
    from .ferrari_canny import ferrari_canny_from_contacts

    p = np.array([c["point_w"] for c in cs["contacts"]])
    nrm = np.array([c["normal_in_w"] for c in cs["contacts"]])
    com = np.array(cs["object_pos_w"])
    try:
        eps = float(ferrari_canny_from_contacts(p, nrm, mu, points_about_com=p - com))
    except Exception:
        eps = -1.0
    try:
        fg_path = os.environ.get("PARCELSTOW_FIRMGRASP")
        if fg_path and fg_path not in sys.path:
            sys.path.insert(0, fg_path)
        import firmgrasp

        g = firmgrasp.Grasp(points=p, normals=nrm, mu_nominal=mu, object_com=com)
        r = firmgrasp.evaluate(g, firmgrasp.FrictionPrior.gaussian(mu, mu_std), beta=beta, n_particles=n_particles)
        epsb = float(r.epsilon_cvar)
    except Exception:
        epsb = -1.0
    return eps, epsb


class StowMonitor:
    def __init__(self, env, geom: G.StowGeometry, trace_envs=(), threshold: float = CONTACT_THRESHOLD):
        self.env = env
        self.geom = geom
        self.n = env.num_envs
        self.device = env.device
        self.threshold = threshold
        self.robot = env.scene["robot"]
        self.parcel = env.scene["parcel"]
        self.distal_ids, _ = self.robot.find_bodies(DISTAL_BODIES, preserve_order=True)
        self.tip_ids, _ = self.robot.find_bodies(TIP_FRAMES, preserve_order=True)
        self.hand_id = self.robot.find_bodies(HAND_ROOT)[0][0]
        from ..parcel_stow_env_cfg import CHAIN_ACTUATED
        self.chain_ids, _ = self.robot.find_joints(CHAIN_ACTUATED, preserve_order=True)
        self.vel_limits = self.robot.data.joint_vel_limits[:, self.chain_ids].clamp(min=1e-3)
        self.half = torch.tensor(G.PARCEL_HALF, dtype=torch.float32, device=self.device)
        d = self.device
        self.R_stow = torch.tensor(geom.R_stow, dtype=torch.float32, device=d)
        self.q_stow = torch.tensor(G.quat_from_mat(geom.R_stow), dtype=torch.float32, device=d).unsqueeze(0).expand(self.n, -1).contiguous()
        self.d_axis = torch.tensor(geom.d, dtype=torch.float32, device=d)
        self.entrance = torch.tensor(geom.entrance, dtype=torch.float32, device=d)
        c, h = geom.interior_box()
        self.interior_c = torch.tensor(c, dtype=torch.float32, device=d)
        self.interior_h = torch.tensor(h, dtype=torch.float32, device=d)
        self.R_yaw_T = torch.tensor(geom.R_yaw.T, dtype=torch.float32, device=d)
        self.p_preinsert = torch.tensor(geom.p_preinsert, dtype=torch.float32, device=d)
        self.trace_envs = set(int(i) for i in trace_envs)
        self.k_lift = G.PHASE_INDEX["LIFT"]
        self.k_reorient = G.PHASE_INDEX["REORIENT"]
        self.k_transfer = G.PHASE_INDEX["TRANSFER"]
        self.k_preinsert = G.PHASE_INDEX["PREINSERT_DWELL"]
        self.k_insert = G.PHASE_INDEX["INSERT"]
        self.k_release = G.PHASE_INDEX["RELEASE"]
        self._alloc()

    # ------------------------------------------------------------------
    def _alloc(self):
        n, d = self.n, self.device
        z = lambda: torch.zeros(n, dtype=torch.bool, device=d)  # noqa: E731
        f = lambda v=0.0: torch.full((n,), float(v), device=d)  # noqa: E731
        self.acquired, self.lifted_clear, self.reoriented = z(), z(), z()
        self.preinsert_reached, self.inserted, self.released, self.settled = z(), z(), z(), z()
        self.dropped = z()
        self.drop_phase = torch.full((n,), -1, dtype=torch.long, device=d)
        self.acquire_step = torch.full((n,), -1, dtype=torch.long, device=d)
        self.low_force_steps = torch.zeros(n, dtype=torch.long, device=d)
        self.release_steps = torch.zeros(n, dtype=torch.long, device=d)
        self.settle_steps = torch.zeros(n, dtype=torch.long, device=d)
        self.jam_steps = torch.zeros(n, dtype=torch.long, device=d)
        self.jammed = z()
        self.max_slip_t, self.max_slip_r = f(), f()
        self.slip_t_at_cubby, self.slip_r_at_cubby = f(float("nan")), f(float("nan"))
        self.cubby_contact_step = torch.full((n,), -1, dtype=torch.long, device=d)
        self.slip_reorient_t, self.slip_reorient_r = f(float("nan")), f(float("nan"))
        self.slip_insert_t, self.slip_insert_r = f(float("nan")), f(float("nan"))
        self.p_ho0 = torch.zeros(n, 3, device=d)
        self.q_ho0 = torch.zeros(n, 4, device=d)
        self.q_ho0[:, 0] = 1.0
        self.max_cubby_force = f()
        self.insert_contact_impulse = f()
        self.max_hand_lin, self.max_hand_ang = f(), f()
        self.max_vel_util, self.max_target_err, self.max_action = f(), f(), f()
        self.max_arm_vel_util = f()
        self.min_ang_err = f(1e9)
        self.min_dist_preinsert = f(1e9)
        self.max_depth = f(-1e9)
        self.last_k = torch.zeros(n, dtype=torch.long, device=d)
        self.contact_lift = [None] * n
        self.contact_reorient = [None] * n
        self.contact_preinsert = [None] * n
        self.n_contacts_lift = torch.zeros(n, dtype=torch.long, device=d)
        self.n_contacts_reorient = torch.zeros(n, dtype=torch.long, device=d)
        self.n_contacts_preinsert = torch.zeros(n, dtype=torch.long, device=d)
        self.steps = torch.zeros(n, dtype=torch.long, device=d)
        self.last_p = torch.zeros(n, 3, device=d)
        self.last_q = torch.zeros(n, 4, device=d)
        self.last_q[:, 0] = 1.0
        self.last_ang = torch.zeros(n, device=d)
        self.last_depth = torch.zeros(n, device=d)
        self.last_inside = torch.zeros(n, dtype=torch.bool, device=d)
        self.last_start = torch.zeros(n, 3, device=d)
        self.last_start_quat = torch.zeros(n, 4, device=d)
        self.last_start_quat[:, 0] = 1.0
        self.traces = {i: [] for i in self.trace_envs}

    def reset(self, ids):
        ids = torch.as_tensor(list(ids) if not torch.is_tensor(ids) else ids, dtype=torch.long, device=self.device)
        if len(ids) == 0:
            return
        for name in ("acquired", "lifted_clear", "reoriented", "preinsert_reached", "inserted", "released",
                     "settled", "dropped", "jammed"):
            getattr(self, name)[ids] = False
        for name in ("low_force_steps", "release_steps", "settle_steps", "jam_steps", "steps", "last_k"):
            getattr(self, name)[ids] = 0
        self.drop_phase[ids] = -1
        self.acquire_step[ids] = -1
        for name in ("max_slip_t", "max_slip_r", "max_cubby_force", "insert_contact_impulse", "max_hand_lin",
                     "max_hand_ang", "max_vel_util", "max_target_err", "max_action", "max_arm_vel_util"):
            getattr(self, name)[ids] = 0.0
        for name in ("slip_reorient_t", "slip_reorient_r", "slip_insert_t", "slip_insert_r",
                     "slip_t_at_cubby", "slip_r_at_cubby"):
            getattr(self, name)[ids] = float("nan")
        self.cubby_contact_step[ids] = -1
        self.min_ang_err[ids] = 1e9
        self.min_dist_preinsert[ids] = 1e9
        self.max_depth[ids] = -1e9
        self.p_ho0[ids] = 0.0
        self.q_ho0[ids] = 0.0
        self.q_ho0[ids, 0] = 1.0
        for i in ids.tolist():
            self.contact_lift[i] = None
            self.contact_reorient[i] = None
            self.contact_preinsert[i] = None
            self.n_contacts_lift[i] = 0
            self.n_contacts_reorient[i] = 0
            self.n_contacts_preinsert[i] = 0
            if i in self.traces:
                self.traces[i] = []

    # ------------------------------------------------------------------
    def parcel_forces_w(self):
        """(E, 5, 3) parcel-filtered contact force per distal phalanx."""
        cols = []
        for b in DISTAL_BODIES:
            fm = self.env.scene.sensors[f"{b}_parcel_s"].data.force_matrix_w
            cols.append(fm.view(self.n, -1, 3).sum(dim=1))
        return torch.stack(cols, dim=1)

    def cubby_forces_w(self):
        """(E, 5, 3) parcel force from each receptacle slab."""
        fm = self.env.scene.sensors["parcel_cubby_s"].data.force_matrix_w
        return fm.view(self.n, -1, 3)

    def contact_geometry(self):
        """Realized contact points on the parcel surface, inward normals, and
        the segment-to-surface distances, (E, 5, 3), (E, 5, 3), (E, 5)."""
        a = self.robot.data.body_pos_w[:, self.distal_ids]
        b = self.robot.data.body_pos_w[:, self.tip_ids]
        frac = torch.linspace(0.0, 1.0, SEGMENT_SAMPLES, device=self.device).view(1, 1, -1, 1)
        pts = a.unsqueeze(2) + (b - a).unsqueeze(2) * frac
        cpos = self.parcel.data.root_pos_w
        cquat = self.parcel.data.root_quat_w
        e = pts.shape[0]
        flat = (pts - cpos.view(e, 1, 1, 3)).reshape(e, -1, 3)
        q = cquat.unsqueeze(1).expand(-1, flat.shape[1], -1).reshape(-1, 4)
        local = quat_apply_inverse(q, flat.reshape(-1, 3)).reshape(e, 5, SEGMENT_SAMPLES, 3)
        outside = torch.clamp(local.abs() - self.half, min=0.0)
        dist = outside.norm(dim=-1)
        best = dist.argmin(dim=-1)
        idx = best.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)
        local_best = torch.gather(local, 2, idx).squeeze(2)
        clamped = torch.maximum(torch.minimum(local_best, self.half), -self.half)
        ratio = local_best.abs() / self.half
        face = ratio.argmax(dim=-1)
        proj = clamped.clone()
        sign = torch.sign(torch.gather(local_best, 2, face.unsqueeze(-1))).squeeze(-1)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        half_face = self.half[face]
        proj.scatter_(2, face.unsqueeze(-1), (sign * half_face).unsqueeze(-1))
        normal_local = torch.zeros_like(proj)
        normal_local.scatter_(2, face.unsqueeze(-1), sign.unsqueeze(-1))
        qb = cquat.unsqueeze(1).expand(-1, 5, -1).reshape(-1, 4)
        p_w = quat_apply(qb, proj.reshape(-1, 3)).reshape(e, 5, 3) + cpos.unsqueeze(1)
        n_out_w = quat_apply(qb, normal_local.reshape(-1, 3)).reshape(e, 5, 3)
        return p_w, -n_out_w, dist.amin(dim=-1)

    def contact_set(self, i, forces, p_w, n_in, dsurf):
        rows = []
        for j, body in enumerate(DISTAL_BODIES):
            fmag = float(forces[i, j].norm())
            if fmag > self.threshold:
                rows.append({"body": body, "force_w": forces[i, j].tolist(), "force_mag": fmag,
                             "point_w": p_w[i, j].tolist(), "normal_in_w": n_in[i, j].tolist(),
                             "surface_distance": float(dsurf[i, j])})
        return {"contacts": rows, "object_pos_w": self.parcel.data.root_pos_w[i].tolist(),
                "object_quat_w": self.parcel.data.root_quat_w[i].tolist()}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, done=None, action=None, q_target=None):
        """Update every marker from the state after env.step(). done marks
        the environments the step finished (the manager env has already
        reset them, so their state belongs to the next episode and the
        cached final state of the previous step stands). action is the
        (E, 16) raw action, q_target the (E, 16) commanded joint target, both
        optional (utilization diagnostics)."""
        from .task_clock import phase_state
        env = self.env
        if done is None:
            done = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        live = ~done
        k, f, t, _ = phase_state(env)
        origins = env.scene.env_origins
        p_w = self.parcel.data.root_pos_w
        p = p_w - origins  # environment-local position, the geometry frame
        q = self.parcel.data.root_quat_w
        v = self.parcel.data.root_lin_vel_w.norm(dim=-1)
        w = self.parcel.data.root_ang_vel_w.norm(dim=-1)
        start = env._stow_start_pos
        forces = self.parcel_forces_w()
        fmag = forces.norm(dim=-1)
        f_sum = fmag.sum(dim=-1)
        opposed = (fmag[:, 0] > self.threshold) & (fmag[:, 1:].amax(dim=-1) > self.threshold)
        any_contact = f_sum > DROP_FORCE
        hand_pos = self.robot.data.body_pos_w[:, self.hand_id]
        hand_quat = self.robot.data.body_quat_w[:, self.hand_id]
        p_ho, q_ho = subtract_frame_transforms(hand_pos, hand_quat, p_w, q)
        ang_err = _quat_angle(quat_mul(quat_inv(self.q_stow), q))
        depth = ((p - self.entrance) * self.d_axis).sum(dim=-1)
        inside = (torch.abs((p - self.interior_c) @ self.R_yaw_T.T) <= self.interior_h).all(dim=-1)
        inserted_now = (depth >= self.geom.inserted_min_depth) & inside
        cubby = self.cubby_forces_w().norm(dim=-1).sum(dim=-1)
        # cache the final-state candidates of live environments
        self.last_p[live] = p[live]
        self.last_q[live] = q[live]
        self.last_ang[live] = ang_err[live]
        self.last_depth[live] = depth[live]
        self.last_inside[live] = inside[live]
        self.last_start[live] = env._stow_start_pos[live]
        self.last_start_quat[live] = env._stow_start_quat[live]

        # acquired, first step with the parcel 2 cm up and an opposed grip
        newly = live & (~self.acquired) & (p[:, 2] >= start[:, 2] + ACQUIRE_DZ) & opposed
        if newly.any():
            self.p_ho0[newly] = p_ho[newly]
            self.q_ho0[newly] = q_ho[newly]
            self.acquire_step[newly] = self.steps[newly]
            cp_w, n_in, dsurf = self.contact_geometry()
            for i in newly.nonzero(as_tuple=False).flatten().tolist():
                cs = self.contact_set(i, forces, cp_w, n_in, dsurf)
                self.contact_lift[i] = cs
                self.n_contacts_lift[i] = len(cs["contacts"])
        self.acquired |= newly
        held = live & self.acquired & ~self.dropped
        # slip against the acquisition transform
        slip_t = (p_ho - self.p_ho0).norm(dim=-1)
        slip_r = _quat_angle(quat_mul(quat_inv(self.q_ho0), q_ho))
        active = held & (k < self.k_release)
        self.max_slip_t = torch.where(active, torch.maximum(self.max_slip_t, slip_t), self.max_slip_t)
        self.max_slip_r = torch.where(active, torch.maximum(self.max_slip_r, slip_r), self.max_slip_r)
        # phase transitions, snapshot at end of REORIENT and at INSERT start
        entered = live & (k != self.last_k)
        end_reorient = entered & (k == self.k_transfer)
        start_insert = entered & (k == self.k_insert)
        if (end_reorient | start_insert).any():
            cp_w, n_in, dsurf = self.contact_geometry()
            for i in end_reorient.nonzero(as_tuple=False).flatten().tolist():
                self.slip_reorient_t[i] = slip_t[i] if held[i] else float("nan")
                self.slip_reorient_r[i] = slip_r[i] if held[i] else float("nan")
                cs = self.contact_set(i, forces, cp_w, n_in, dsurf)
                self.contact_reorient[i] = cs
                self.n_contacts_reorient[i] = len(cs["contacts"])
            for i in start_insert.nonzero(as_tuple=False).flatten().tolist():
                self.slip_insert_t[i] = slip_t[i] if held[i] else float("nan")
                self.slip_insert_r[i] = slip_r[i] if held[i] else float("nan")
                cs = self.contact_set(i, forces, cp_w, n_in, dsurf)
                self.contact_preinsert[i] = cs
                self.n_contacts_preinsert[i] = len(cs["contacts"])
        # drop, contact lost for DROP_STEPS before RELEASE while not inserted
        self.low_force_steps = torch.where(held & ~any_contact, self.low_force_steps + 1,
                                           torch.where(live, torch.zeros_like(self.low_force_steps), self.low_force_steps))
        drop_now = held & (self.low_force_steps >= DROP_STEPS) & (k < self.k_release) & ~inserted_now
        self.drop_phase = torch.where(drop_now & ~self.dropped, k, self.drop_phase)
        self.dropped |= drop_now
        # stage markers
        self.lifted_clear |= held & (p[:, 2] >= LIFT_CLEAR_Z)
        self.reoriented |= held & (k < self.k_insert) & (ang_err < math.radians(REORIENT_TOL_DEG))
        dist_pre = (p - self.p_preinsert).norm(dim=-1)
        self.preinsert_reached |= held & (dist_pre < PREINSERT_POS_TOL) & (ang_err < math.radians(PREINSERT_ANG_TOL_DEG))
        self.inserted |= live & inserted_now & (k >= self.k_insert)
        self.release_steps = torch.where(live & self.inserted & (f_sum < RELEASE_FORCE), self.release_steps + 1,
                                         torch.where(live, torch.zeros_like(self.release_steps), self.release_steps))
        self.released |= self.release_steps >= RELEASE_STEPS
        still = live & self.released & inserted_now & (v < SETTLE_LIN) & (w < SETTLE_ANG)
        self.settle_steps = torch.where(still, self.settle_steps + 1,
                                        torch.where(live, torch.zeros_like(self.settle_steps), self.settle_steps))
        self.settled |= self.settle_steps >= SETTLE_STEPS
        # first receptacle contact while held, the slip at that moment separates
        # in-hand slip before contact from slip the receptacle caused
        first_cubby = live & held & (cubby > 0.5) & (self.cubby_contact_step < 0)
        self.cubby_contact_step = torch.where(first_cubby, self.steps, self.cubby_contact_step)
        self.slip_t_at_cubby = torch.where(first_cubby, slip_t, self.slip_t_at_cubby)
        self.slip_r_at_cubby = torch.where(first_cubby, slip_r, self.slip_r_at_cubby)
        # jam, sustained parcel-receptacle contact during INSERT without insertion
        jam_now = live & (k == self.k_insert) & (cubby > JAM_FORCE) & ~inserted_now
        self.jam_steps = torch.where(jam_now, self.jam_steps + 1,
                                     torch.where(live, torch.zeros_like(self.jam_steps), self.jam_steps))
        self.jammed |= self.jam_steps >= JAM_STEPS
        self.max_cubby_force = torch.where(live, torch.maximum(self.max_cubby_force, cubby), self.max_cubby_force)
        self.insert_contact_impulse += torch.where(live & (k >= self.k_insert), cubby * float(env.step_dt), torch.zeros_like(cubby))
        # diagnostics
        self.min_ang_err = torch.where(held, torch.minimum(self.min_ang_err, ang_err), self.min_ang_err)
        self.min_dist_preinsert = torch.where(held, torch.minimum(self.min_dist_preinsert, dist_pre), self.min_dist_preinsert)
        self.max_depth = torch.where(live, torch.maximum(self.max_depth, depth), self.max_depth)
        # peak velocities and utilization over the manipulation segment (LIFT onward)
        manip = live & (k >= self.k_lift)
        hl = self.robot.data.body_lin_vel_w[:, self.hand_id].norm(dim=-1)
        ha = self.robot.data.body_ang_vel_w[:, self.hand_id].norm(dim=-1)
        self.max_hand_lin = torch.where(manip, torch.maximum(self.max_hand_lin, hl), self.max_hand_lin)
        self.max_hand_ang = torch.where(manip, torch.maximum(self.max_hand_ang, ha), self.max_hand_ang)
        util_all = self.robot.data.joint_vel[:, self.chain_ids].abs() / self.vel_limits
        util = util_all.amax(dim=-1)
        util_arm = util_all[:, :10].amax(dim=-1)
        self.max_vel_util = torch.where(live, torch.maximum(self.max_vel_util, util), self.max_vel_util)
        self.max_arm_vel_util = torch.where(manip, torch.maximum(self.max_arm_vel_util, util_arm), self.max_arm_vel_util)
        if q_target is not None:
            terr = (q_target - self.robot.data.joint_pos[:, self.chain_ids]).abs().amax(dim=-1)
            self.max_target_err = torch.where(live, torch.maximum(self.max_target_err, terr), self.max_target_err)
        if action is not None:
            self.max_action = torch.where(live, torch.maximum(self.max_action, action.abs().amax(dim=-1)), self.max_action)
        # traces
        if self.traces:
            for i in self.traces:
                if not live[i]:
                    continue
                self.traces[i].append(np.concatenate([
                    [float(t[i]), float(k[i]), float(f[i])], p_w[i].cpu().numpy(), q[i].cpu().numpy(),
                    hand_pos[i].cpu().numpy(), hand_quat[i].cpu().numpy(), fmag[i].cpu().numpy(),
                    [float(cubby[i]), float(slip_t[i]), float(slip_r[i]), float(ang_err[i]), float(depth[i])],
                    self.robot.data.joint_pos[i, self.chain_ids].cpu().numpy(),
                    (q_target[i].cpu().numpy() if q_target is not None else np.zeros(16)),
                ]).astype(np.float32))
        self.last_k = torch.where(live, k, self.last_k)
        self.steps += live.long()

    # ------------------------------------------------------------------
    def final_state(self, i):
        """Cached state of the last live step of environment i."""
        return self.last_p[i], self.last_q[i], float(self.last_ang[i]), float(self.last_depth[i]), bool(self.last_inside[i])

    def failure_reason(self, i, task_success, final_ang_deg, inside):
        """First applicable category of TASK_SPEC.md section 8."""
        if task_success:
            return "none", "none"
        if not bool(self.acquired[i]):
            return "acquisition_failure", "never_acquired"
        if bool(self.dropped[i]):
            ph = int(self.drop_phase[i])
            name = G.PHASE_NAMES[ph] if ph >= 0 else "?"
            if ph <= self.k_lift:
                return "dropped_during_lift", f"drop_in_{name}"
            if ph == self.k_reorient:
                return "dropped_during_reorientation", f"drop_in_{name}"
            if ph in (self.k_transfer, self.k_preinsert):
                return "transfer_failure", f"drop_in_{name}"
            if ph == self.k_insert:
                if bool(self.jammed[i]):
                    return "insertion_jam", f"drop_in_{name}"
                return "insertion_misalignment", f"drop_in_{name}"
        slip_big = float(self.max_slip_t[i]) > SLIP_TRANS_LIMIT or math.degrees(float(self.max_slip_r[i])) > SLIP_ROT_LIMIT_DEG
        touched = int(self.cubby_contact_step[i]) >= 0
        st_c, sr_c = float(self.slip_t_at_cubby[i]), float(self.slip_r_at_cubby[i])
        slip_big_before = slip_big and (not touched or st_c > SLIP_TRANS_LIMIT or math.degrees(sr_c) > SLIP_ROT_LIMIT_DEG)
        if not bool(self.inserted[i]):
            if slip_big_before and bool(self.preinsert_reached[i]):
                return "excessive_inhand_slip", "slip_before_receptacle_contact"
            if not bool(self.preinsert_reached[i]):
                return "transfer_failure", "preinsert_not_reached"
            if bool(self.jammed[i]) or (touched and slip_big):
                return "insertion_jam", "slip_after_receptacle_contact" if slip_big else "sustained_receptacle_contact"
            return "insertion_misalignment", "not_inserted"
        if not bool(self.released[i]):
            return "release_failure", "hand_contact_persisted"
        if not bool(self.settled[i]):
            return "timeout", "not_settled"
        if not inside:
            return "other", "left_receptacle_after_settle"
        if final_ang_deg > FINAL_ANG_TOL_DEG:
            return "insertion_misalignment", "final_orientation"
        return "other", "unclassified"

    def episode_record(self, i, score_certificate=True, mu=G.PARCEL_FRICTION):
        p, q, ang, depth, inside = self.final_state(i)
        final_ang_deg = math.degrees(ang)
        task_success = bool(self.inserted[i]) and bool(self.released[i]) and bool(self.settled[i]) \
            and inside and final_ang_deg <= FINAL_ANG_TOL_DEG
        reason, detail = self.failure_reason(i, task_success, final_ang_deg, inside)
        start_pos = self.last_start[i].tolist()
        start_quat = self.last_start_quat[i].tolist()
        rec = {
            "task_success": task_success,
            "acquired": bool(self.acquired[i]), "lifted_clear": bool(self.lifted_clear[i]),
            "reoriented": bool(self.reoriented[i]), "preinsert_reached": bool(self.preinsert_reached[i]),
            "inserted": bool(self.inserted[i]), "released": bool(self.released[i]), "settled": bool(self.settled[i]),
            "dropped": bool(self.dropped[i]), "drop_phase": G.PHASE_NAMES[int(self.drop_phase[i])] if int(self.drop_phase[i]) >= 0 else None,
            "jammed": bool(self.jammed[i]),
            "failure_reason": reason, "failure_detail": detail,
            "parcel_initial_pose": {"pos": start_pos, "quat_wxyz": start_quat},
            "parcel_final_pose": {"pos": p.tolist(), "quat_wxyz": q.tolist()},
            "insertion_depth": depth, "max_depth": float(self.max_depth[i]),
            "final_position_error": float((p - torch.tensor(self.geom.p_insert, device=self.device)).norm()),
            "final_orientation_error_deg": final_ang_deg,
            "min_orientation_error_deg": math.degrees(float(self.min_ang_err[i])) if float(self.min_ang_err[i]) < 1e8 else None,
            "min_dist_preinsert": float(self.min_dist_preinsert[i]) if float(self.min_dist_preinsert[i]) < 1e8 else None,
            "max_hand_object_translation_m": float(self.max_slip_t[i]),
            "max_hand_object_rotation_deg": math.degrees(float(self.max_slip_r[i])),
            "slip_reorient_translation_m": float(self.slip_reorient_t[i]),
            "slip_reorient_rotation_deg": math.degrees(float(self.slip_reorient_r[i])),
            "slip_insert_translation_m": float(self.slip_insert_t[i]),
            "slip_insert_rotation_deg": math.degrees(float(self.slip_insert_r[i])),
            "slip_translation_at_receptacle_contact_m": float(self.slip_t_at_cubby[i]),
            "slip_rotation_at_receptacle_contact_deg": math.degrees(float(self.slip_r_at_cubby[i])),
            "receptacle_contact_step": int(self.cubby_contact_step[i]),
            "acquire_step": int(self.acquire_step[i]),
            "contact_count_lift": int(self.n_contacts_lift[i]),
            "contact_count_reorient": int(self.n_contacts_reorient[i]),
            "contact_count_preinsert": int(self.n_contacts_preinsert[i]),
            "max_receptacle_force": float(self.max_cubby_force[i]),
            "insert_contact_impulse": float(self.insert_contact_impulse[i]),
            "peak_hand_linear_velocity": float(self.max_hand_lin[i]),
            "peak_hand_angular_velocity": float(self.max_hand_ang[i]),
            "max_joint_velocity_utilization": float(self.max_vel_util[i]),
            "max_arm_velocity_utilization": float(self.max_arm_vel_util[i]),
            "max_target_tracking_error_rad": float(self.max_target_err[i]),
            "max_action_magnitude": float(self.max_action[i]),
            "steps": int(self.steps[i]),
            "contact_set_lift": self.contact_lift[i],
            "contact_set_reorient": self.contact_reorient[i],
            "contact_set_preinsert": self.contact_preinsert[i],
        }
        for tag, cs in (("lift", self.contact_lift[i]), ("reorient", self.contact_reorient[i]),
                        ("preinsert", self.contact_preinsert[i])):
            if score_certificate:
                eps, epsb = score_contact_set(cs, mu=mu)
            else:
                eps, epsb = None, None
            rec[f"epsilon_{tag}"] = eps
            rec[f"epsilon_beta_{tag}"] = epsb
        return rec

    def take_trace(self, i):
        rows = self.traces.get(i)
        if not rows:
            return None
        arr = np.stack(rows)
        self.traces[i] = []
        return arr


TRACE_COLUMNS = (
    ["t", "k", "f"] + [f"parcel_pos_{a}" for a in "xyz"] + [f"parcel_quat_{a}" for a in "wxyz"]
    + [f"hand_pos_{a}" for a in "xyz"] + [f"hand_quat_{a}" for a in "wxyz"]
    + [f"force_{b}" for b in ("thumb", "index", "middle", "ring", "pinky")]
    + ["cubby_force", "slip_t", "slip_r", "ang_err", "depth"]
    + [f"q_{j}" for j in range(16)] + [f"qt_{j}" for j in range(16)]
)
