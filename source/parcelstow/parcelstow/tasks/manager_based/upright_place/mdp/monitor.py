"""Physical monitor of the upright placement task, stage markers,
failure attribution, in-hand slip, realized contact sets, and actuator
utilization, all read from the simulator state after every control
step. The drivers own the object, call step() after env.step(), and
take episode_record(i) when environment i finishes, the StowMonitor
pattern of the ParcelStow task.

Every predicate reads physical state only. The force-closure margins
are computed from the recorded contact sets as diagnostics and never
feed a marker. The thresholds shared with ParcelStow keep the v1
values; the placement thresholds come from the task geometry.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, subtract_frame_transforms

from ...parcel_stow.mdp.metrics import score_contact_set
from .. import geometry as U

DISTAL_BODIES = ["rh_thumb_distal", "rh_index_distal", "rh_middle_distal", "rh_ring_distal", "rh_pinky_distal"]
TIP_FRAMES = ["rh_thumb_tip", "rh_index_tip", "rh_middle_tip", "rh_ring_tip", "rh_pinky_tip"]
HAND_ROOT = "rh_hand_base_link"
SEGMENT_SAMPLES = 12

# thresholds shared with the v1 monitor
ACQUIRE_DZ = 0.020
LIFT_CLEAR_DZ = 0.060
CONTACT_THRESHOLD = 1.0
RELEASE_FORCE = 0.5
RELEASE_STEPS = 5
SETTLE_LIN = 0.02
SETTLE_ANG = 0.2
SETTLE_STEPS = 20
DROP_FORCE = 0.5
DROP_STEPS = 10
# Placement thresholds are frozen in docs/TASK_SPEC_UPRIGHT.md.
STANDING_TOL_DEG = 15.0
PLACE_POS_TOL = 0.030

STAGE_KEYS = ["acquired", "lifted_clear", "reoriented_upright", "placed", "released", "settled"]


def _quat_angle(q):
    w = q[:, 0].abs().clamp(max=1.0)
    return 2.0 * torch.acos(w)


class UprightMonitor:
    def __init__(self, env, trace_envs=(), threshold: float = CONTACT_THRESHOLD):
        self.env = env
        self.n = env.num_envs
        self.device = env.device
        self.threshold = threshold
        self.robot = env.scene["robot"]
        self.obj = env.scene["object"]
        self.distal_ids, _ = self.robot.find_bodies(DISTAL_BODIES, preserve_order=True)
        self.hand_id = self.robot.find_bodies(HAND_ROOT)[0][0]
        from ..upright_place_env_cfg import CHAIN_ACTUATED
        self.chain_ids, _ = self.robot.find_joints(CHAIN_ACTUATED, preserve_order=True)
        self.vel_limits = self.robot.data.joint_vel_limits[:, self.chain_ids].clamp(min=1e-3)
        d = self.device
        self.half = torch.tensor([e / 2 for e in U.OBJECT_EXTENTS], dtype=torch.float32, device=d)
        self.ez_half = torch.tensor([0.0, 0.0, U.OBJECT_HALF_HEIGHT], dtype=torch.float32, device=d)
        self.target = torch.tensor(U.TARGET_CENTER, dtype=torch.float32, device=d)
        self.p_place = torch.tensor([*U.TARGET_CENTER, U.PLACE_Z], dtype=torch.float32, device=d)
        self.trace_envs = set(int(i) for i in trace_envs)
        self.k_lift = U.PHASE_INDEX["LIFT"]
        self.k_reorient = U.PHASE_INDEX["REORIENT"]
        self.k_transfer = U.PHASE_INDEX["TRANSFER"]
        self.k_lower = U.PHASE_INDEX["LOWER"]
        self.k_release = U.PHASE_INDEX["RELEASE"]
        self._alloc()

    # ------------------------------------------------------------------
    def _alloc(self):
        n, d = self.n, self.device
        z = lambda: torch.zeros(n, dtype=torch.bool, device=d)  # noqa: E731
        f = lambda v=0.0: torch.full((n,), float(v), device=d)  # noqa: E731
        self.acquired, self.lifted_clear, self.reoriented_upright = z(), z(), z()
        self.placed, self.released, self.settled = z(), z(), z()
        self.dropped = z()
        self.drop_phase = torch.full((n,), -1, dtype=torch.long, device=d)
        self.acquire_step = torch.full((n,), -1, dtype=torch.long, device=d)
        self.low_force_steps = torch.zeros(n, dtype=torch.long, device=d)
        self.release_steps = torch.zeros(n, dtype=torch.long, device=d)
        self.settle_steps = torch.zeros(n, dtype=torch.long, device=d)
        self.max_slip_t, self.max_slip_r = f(), f()
        self.slip_reorient_t, self.slip_reorient_r = f(float("nan")), f(float("nan"))
        self.slip_lower_t, self.slip_lower_r = f(float("nan")), f(float("nan"))
        self.p_ho0 = torch.zeros(n, 3, device=d)
        self.q_ho0 = torch.zeros(n, 4, device=d)
        self.q_ho0[:, 0] = 1.0
        self.max_hand_lin, self.max_hand_ang = f(), f()
        self.max_vel_util, self.max_target_err, self.max_action = f(), f(), f()
        self.max_arm_vel_util = f()
        self.min_tilt = f(1e9)
        self.min_dist_place = f(1e9)
        self.last_k = torch.zeros(n, dtype=torch.long, device=d)
        self.contact_lift = [None] * n
        self.contact_reorient = [None] * n
        self.contact_lower = [None] * n
        self.n_contacts_lift = torch.zeros(n, dtype=torch.long, device=d)
        self.n_contacts_reorient = torch.zeros(n, dtype=torch.long, device=d)
        self.n_contacts_lower = torch.zeros(n, dtype=torch.long, device=d)
        self.steps = torch.zeros(n, dtype=torch.long, device=d)
        self.last_p = torch.zeros(n, 3, device=d)
        self.last_q = torch.zeros(n, 4, device=d)
        self.last_q[:, 0] = 1.0
        self.last_tilt = torch.zeros(n, device=d)
        self.last_base_off = torch.zeros(n, device=d)
        self.last_inside = torch.zeros(n, dtype=torch.bool, device=d)
        self.last_start = torch.zeros(n, 3, device=d)
        self.last_start_quat = torch.zeros(n, 4, device=d)
        self.last_start_quat[:, 0] = 1.0
        self.traces = {i: [] for i in self.trace_envs}

    def reset(self, ids):
        ids = torch.as_tensor(list(ids) if not torch.is_tensor(ids) else ids, dtype=torch.long, device=self.device)
        if len(ids) == 0:
            return
        for name in ("acquired", "lifted_clear", "reoriented_upright", "placed", "released", "settled", "dropped"):
            getattr(self, name)[ids] = False
        for name in ("low_force_steps", "release_steps", "settle_steps", "steps", "last_k"):
            getattr(self, name)[ids] = 0
        self.drop_phase[ids] = -1
        self.acquire_step[ids] = -1
        for name in ("max_slip_t", "max_slip_r", "max_hand_lin", "max_hand_ang",
                     "max_vel_util", "max_target_err", "max_action", "max_arm_vel_util"):
            getattr(self, name)[ids] = 0.0
        for name in ("slip_reorient_t", "slip_reorient_r", "slip_lower_t", "slip_lower_r"):
            getattr(self, name)[ids] = float("nan")
        self.min_tilt[ids] = 1e9
        self.min_dist_place[ids] = 1e9
        self.p_ho0[ids] = 0.0
        self.q_ho0[ids] = 0.0
        self.q_ho0[ids, 0] = 1.0
        for i in ids.tolist():
            self.contact_lift[i] = None
            self.contact_reorient[i] = None
            self.contact_lower[i] = None
            self.n_contacts_lift[i] = 0
            self.n_contacts_reorient[i] = 0
            self.n_contacts_lower[i] = 0
            if i in self.traces:
                self.traces[i] = []

    # ------------------------------------------------------------------
    def object_forces_w(self):
        """(E, 5, 3) object-filtered contact force per distal phalanx."""
        cols = []
        for b in DISTAL_BODIES:
            fm = self.env.scene.sensors[f"{b}_object_s"].data.force_matrix_w
            cols.append(fm.view(self.n, -1, 3).sum(dim=1))
        return torch.stack(cols, dim=1)

    def contact_geometry(self):
        """Realized contact points on the object surface, inward normals,
        and segment-to-surface distances, the v1 sampling."""
        from isaaclab.utils.math import quat_apply_inverse
        a = self.robot.data.body_pos_w[:, self.distal_ids]
        tip_ids, _ = self.robot.find_bodies(TIP_FRAMES, preserve_order=True)
        b = self.robot.data.body_pos_w[:, tip_ids]
        frac = torch.linspace(0.0, 1.0, SEGMENT_SAMPLES, device=self.device).view(1, 1, -1, 1)
        pts = a.unsqueeze(2) + (b - a).unsqueeze(2) * frac
        cpos = self.obj.data.root_pos_w
        cquat = self.obj.data.root_quat_w
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
        return {"contacts": rows, "object_pos_w": self.obj.data.root_pos_w[i].tolist(),
                "object_quat_w": self.obj.data.root_quat_w[i].tolist()}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, done=None, action=None, q_target=None):
        """Update every marker from the state after env.step(), the v1
        contract (done environments keep their cached final state)."""
        from ...parcel_stow.mdp.task_clock import phase_state
        env = self.env
        if done is None:
            done = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        live = ~done
        k, f, t, _ = phase_state(env)
        origins = env.scene.env_origins
        p_w = self.obj.data.root_pos_w
        p = p_w - origins
        q = self.obj.data.root_quat_w
        v = self.obj.data.root_lin_vel_w.norm(dim=-1)
        w = self.obj.data.root_ang_vel_w.norm(dim=-1)
        start = env._stow_start_pos
        forces = self.object_forces_w()
        fmag = forces.norm(dim=-1)
        f_sum = fmag.sum(dim=-1)
        opposed = (fmag[:, 0] > self.threshold) & (fmag[:, 1:].amax(dim=-1) > self.threshold)
        any_contact = f_sum > DROP_FORCE
        hand_pos = self.robot.data.body_pos_w[:, self.hand_id]
        hand_quat = self.robot.data.body_quat_w[:, self.hand_id]
        p_ho, q_ho = subtract_frame_transforms(hand_pos, hand_quat, p_w, q)
        # tilt of the object z axis from the vertical, and the base center
        z_w = quat_apply(q, torch.zeros_like(p) + torch.tensor([0.0, 0.0, 1.0], device=self.device))
        tilt = torch.acos(z_w[:, 2].clamp(-1.0, 1.0))
        base = p - quat_apply(q, self.ez_half.unsqueeze(0).expand(self.n, -1))
        base_off = (base[:, :2] - self.target).norm(dim=-1)
        inside = base_off <= U.TARGET_RADIUS
        standing_now = (tilt < math.radians(STANDING_TOL_DEG)) & inside
        dist_place = (p - self.p_place).norm(dim=-1)
        placed_now = (dist_place < PLACE_POS_TOL) & (tilt < math.radians(STANDING_TOL_DEG))
        # cache final-state candidates of live environments
        self.last_p[live] = p[live]
        self.last_q[live] = q[live]
        self.last_tilt[live] = tilt[live]
        self.last_base_off[live] = base_off[live]
        self.last_inside[live] = inside[live]
        self.last_start[live] = env._stow_start_pos[live]
        self.last_start_quat[live] = env._stow_start_quat[live]

        # acquired, first step with the object 2 cm up and an opposed grip
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
        slip_t = (p_ho - self.p_ho0).norm(dim=-1)
        slip_r = _quat_angle(quat_mul(quat_inv(self.q_ho0), q_ho))
        active = held & (k < self.k_release)
        self.max_slip_t = torch.where(active, torch.maximum(self.max_slip_t, slip_t), self.max_slip_t)
        self.max_slip_r = torch.where(active, torch.maximum(self.max_slip_r, slip_r), self.max_slip_r)
        # phase transitions, snapshots at REORIENT end and at LOWER start
        entered = live & (k != self.last_k)
        end_reorient = entered & (k == self.k_transfer)
        start_lower = entered & (k == self.k_lower)
        if (end_reorient | start_lower).any():
            cp_w, n_in, dsurf = self.contact_geometry()
            for i in end_reorient.nonzero(as_tuple=False).flatten().tolist():
                self.slip_reorient_t[i] = slip_t[i] if held[i] else float("nan")
                self.slip_reorient_r[i] = slip_r[i] if held[i] else float("nan")
                cs = self.contact_set(i, forces, cp_w, n_in, dsurf)
                self.contact_reorient[i] = cs
                self.n_contacts_reorient[i] = len(cs["contacts"])
            for i in start_lower.nonzero(as_tuple=False).flatten().tolist():
                self.slip_lower_t[i] = slip_t[i] if held[i] else float("nan")
                self.slip_lower_r[i] = slip_r[i] if held[i] else float("nan")
                cs = self.contact_set(i, forces, cp_w, n_in, dsurf)
                self.contact_lower[i] = cs
                self.n_contacts_lower[i] = len(cs["contacts"])
        # drop, contact lost before RELEASE while the object is not placed
        self.low_force_steps = torch.where(
            held & ~any_contact, self.low_force_steps + 1,
            torch.where(live, torch.zeros_like(self.low_force_steps), self.low_force_steps))
        drop_now = held & (self.low_force_steps >= DROP_STEPS) & (k < self.k_release) & ~placed_now
        self.drop_phase = torch.where(drop_now & ~self.dropped, k, self.drop_phase)
        self.dropped |= drop_now
        # stage markers
        self.lifted_clear |= held & (p[:, 2] >= start[:, 2] + LIFT_CLEAR_DZ)
        self.reoriented_upright |= held & (k < self.k_release) & (tilt < math.radians(STANDING_TOL_DEG))
        self.placed |= held & placed_now
        self.release_steps = torch.where(live & self.placed & (f_sum < RELEASE_FORCE), self.release_steps + 1,
                                         torch.where(live, torch.zeros_like(self.release_steps), self.release_steps))
        self.released |= self.release_steps >= RELEASE_STEPS
        still = live & self.released & standing_now & (v < SETTLE_LIN) & (w < SETTLE_ANG)
        self.settle_steps = torch.where(still, self.settle_steps + 1,
                                        torch.where(live, torch.zeros_like(self.settle_steps), self.settle_steps))
        self.settled |= self.settle_steps >= SETTLE_STEPS
        # diagnostics
        self.min_tilt = torch.where(held, torch.minimum(self.min_tilt, tilt), self.min_tilt)
        self.min_dist_place = torch.where(held, torch.minimum(self.min_dist_place, dist_place), self.min_dist_place)
        manip = live & (k >= self.k_lift)
        hl = self.robot.data.body_lin_vel_w[:, self.hand_id].norm(dim=-1)
        ha = self.robot.data.body_ang_vel_w[:, self.hand_id].norm(dim=-1)
        self.max_hand_lin = torch.where(manip, torch.maximum(self.max_hand_lin, hl), self.max_hand_lin)
        self.max_hand_ang = torch.where(manip, torch.maximum(self.max_hand_ang, ha), self.max_hand_ang)
        util_all = self.robot.data.joint_vel[:, self.chain_ids].abs() / self.vel_limits
        util = util_all.amax(dim=-1)
        util_arm = util_all[:, :10].amax(dim=-1)
        self.max_vel_util = torch.where(live, torch.maximum(self.max_vel_util, util), self.max_vel_util)
        self.max_arm_vel_util = torch.where(manip, torch.maximum(self.max_arm_vel_util, util_arm),
                                            self.max_arm_vel_util)
        if q_target is not None:
            terr = (q_target - self.robot.data.joint_pos[:, self.chain_ids]).abs().amax(dim=-1)
            self.max_target_err = torch.where(live, torch.maximum(self.max_target_err, terr), self.max_target_err)
        if action is not None:
            self.max_action = torch.where(live, torch.maximum(self.max_action, action.abs().amax(dim=-1)),
                                          self.max_action)
        if self.traces:
            for i in self.traces:
                if not live[i]:
                    continue
                self.traces[i].append(np.concatenate([
                    [float(t[i]), float(k[i]), float(f[i])], p_w[i].cpu().numpy(), q[i].cpu().numpy(),
                    hand_pos[i].cpu().numpy(), hand_quat[i].cpu().numpy(), fmag[i].cpu().numpy(),
                    [float(slip_t[i]), float(slip_r[i]), float(tilt[i]), float(base_off[i])],
                    self.robot.data.joint_pos[i, self.chain_ids].cpu().numpy(),
                    (q_target[i].cpu().numpy() if q_target is not None else np.zeros(16)),
                ]).astype(np.float32))
        self.last_k = torch.where(live, k, self.last_k)
        self.steps += live.long()

    # ------------------------------------------------------------------
    def failure_reason(self, i, task_success, final_tilt_deg, final_inside):
        """Return the first applicable task-specific failure category."""
        if task_success:
            return "none", "none"
        if not bool(self.acquired[i]):
            return "acquisition_failure", "never_acquired"
        if bool(self.dropped[i]):
            ph = int(self.drop_phase[i])
            name = U.PHASES[ph][0] if ph >= 0 else "?"
            return "dropped_during_transport", f"drop_in_{name}"
        if not bool(self.placed[i]):
            return "placement_miss", "place_not_reached"
        if not bool(self.released[i]):
            return "timeout", "hand_contact_persisted"
        if final_tilt_deg > U.FINAL_TILT_TOL_DEG:
            return "tipped_after_release", "final_tilt"
        if not final_inside:
            return "tipped_after_release", "left_target_after_release"
        if not bool(self.settled[i]):
            return "timeout", "not_settled"
        return "other", "unclassified"

    def episode_record(self, i, score_certificate=True, mu=U.OBJECT_FRICTION):
        p, q = self.last_p[i], self.last_q[i]
        final_tilt_deg = math.degrees(float(self.last_tilt[i]))
        final_inside = bool(self.last_inside[i])
        task_success = bool(self.placed[i]) and bool(self.released[i]) and bool(self.settled[i]) \
            and final_inside and final_tilt_deg <= U.FINAL_TILT_TOL_DEG
        reason, detail = self.failure_reason(i, task_success, final_tilt_deg, final_inside)
        rec = {
            "task_success": task_success,
            "acquired": bool(self.acquired[i]), "lifted_clear": bool(self.lifted_clear[i]),
            "reoriented_upright": bool(self.reoriented_upright[i]), "placed": bool(self.placed[i]),
            "released": bool(self.released[i]), "settled": bool(self.settled[i]),
            "dropped": bool(self.dropped[i]),
            "drop_phase": U.PHASES[int(self.drop_phase[i])][0] if int(self.drop_phase[i]) >= 0 else None,
            "failure_reason": reason, "failure_detail": detail,
            "object_initial_pose": {"pos": self.last_start[i].tolist(), "quat_wxyz": self.last_start_quat[i].tolist()},
            "object_final_pose": {"pos": p.tolist(), "quat_wxyz": q.tolist()},
            "final_tilt_deg": final_tilt_deg,
            "final_base_offset_m": float(self.last_base_off[i]),
            "min_tilt_deg": math.degrees(float(self.min_tilt[i])) if float(self.min_tilt[i]) < 1e8 else None,
            "min_dist_place": float(self.min_dist_place[i]) if float(self.min_dist_place[i]) < 1e8 else None,
            "max_hand_object_translation_m": float(self.max_slip_t[i]),
            "max_hand_object_rotation_deg": math.degrees(float(self.max_slip_r[i])),
            "slip_reorient_translation_m": float(self.slip_reorient_t[i]),
            "slip_reorient_rotation_deg": math.degrees(float(self.slip_reorient_r[i])),
            "slip_lower_translation_m": float(self.slip_lower_t[i]),
            "slip_lower_rotation_deg": math.degrees(float(self.slip_lower_r[i])),
            "acquire_step": int(self.acquire_step[i]),
            "contact_count_lift": int(self.n_contacts_lift[i]),
            "contact_count_reorient": int(self.n_contacts_reorient[i]),
            "contact_count_lower": int(self.n_contacts_lower[i]),
            "peak_hand_linear_velocity": float(self.max_hand_lin[i]),
            "peak_hand_angular_velocity": float(self.max_hand_ang[i]),
            "max_joint_velocity_utilization": float(self.max_vel_util[i]),
            "max_arm_velocity_utilization": float(self.max_arm_vel_util[i]),
            "max_target_tracking_error_rad": float(self.max_target_err[i]),
            "max_action_magnitude": float(self.max_action[i]),
            "steps": int(self.steps[i]),
            "contact_set_lift": self.contact_lift[i],
            "contact_set_reorient": self.contact_reorient[i],
            "contact_set_lower": self.contact_lower[i],
        }
        for tag, cs in (("lift", self.contact_lift[i]), ("reorient", self.contact_reorient[i]),
                        ("lower", self.contact_lower[i])):
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
    ["t", "k", "f"] + [f"object_pos_{a}" for a in "xyz"] + [f"object_quat_{a}" for a in "wxyz"]
    + [f"hand_pos_{a}" for a in "xyz"] + [f"hand_quat_{a}" for a in "wxyz"]
    + [f"force_{b}" for b in ("thumb", "index", "middle", "ring", "pinky")]
    + ["slip_t", "slip_r", "tilt", "base_offset"]
    + [f"q_{j}" for j in range(16)] + [f"qt_{j}" for j in range(16)]
)
