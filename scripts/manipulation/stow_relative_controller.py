"""Relative-motion handoff controller of ParcelStow, the step, after-step,
and record hooks used by scripts/manipulation/stow_relative_handoff.py and
by the checks. Import after the Isaac app has launched (the controller
reads the live articulation and the monitor).

At the stable handoff time t_H (the monitor's acquired marker plus the
parcel handoff_dz above its start height, before RELEASE), the controller
records the actor's actual world hand pose T_WH^pi(t_H) and the forward
kinematics of the actor's arm command, freezes the actor's realized hand
joint target, and from then on commands the waist and arm so the commanded
hand pose follows T_anchor Delta T_H^E(s), the expert's nominal downstream
relative hand motion in the handoff frame (stow_relative) applied to the
anchor. The anchor is the commanded hand pose at t_H (see stow_relative for
why the measured pose is not the anchor, the servo transient at high rate).
The arm command comes from the damped least squares iteration of the
ParcelStow IK on the URDF kinematic model (stow_relative.RelativeArmSolver,
PinocchioChain) with the expert's dwell integral on top. From RELEASE the
hand follows the expert's opening (bank grasp shape to open shape, cosine
blend, as in stow_handoff.py), and the arm retreats along the relative
retreat path.

The controller never reads, welds, or resets the parcel, and no
object-pose feedback exists. The primary endpoint of the diagnostic
is the first control step in INSERT (the state at the end of
PREINSERT_DWELL) or the first receptacle contact if a preserved absolute
offset brings the parcel against a slab earlier, whichever comes first, so
the segment ends before any receptacle interaction. There the after-step
hook records retention, the hand-object transform change since handoff, the
contact count, and the endpoint reason. The episode continues through
insertion and release under the same relative controller as the secondary
endpoint.
"""

from __future__ import annotations

import math
import os

import torch

from isaaclab.utils.math import quat_inv, quat_mul, subtract_frame_transforms

from parcelstow.tasks.manager_based.parcel_stow import geometry as G
from parcelstow.tasks.manager_based.parcel_stow.mdp import task_clock
from parcelstow.tasks.manager_based.parcel_stow.mdp.metrics import DROP_FORCE, HAND_ROOT, score_contact_set

RECEPTACLE_CONTACT_FORCE = 0.5  # N, the monitor's first-contact threshold

import parcel_stow_expert as pse
import stow_relative as rel

URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "g1_l6", "g1_29dof_l6_both.urdf")


def _quat_angle(q):
    return 2.0 * torch.acos(q[:, 0].abs().clamp(max=1.0))


class RelativeHandoffController:
    def __init__(self, base, expert, geom, monitor, dz=0.04, iters=3, dls_lambda=rel.DLS_LAMBDA,
                 null_gain=rel.NULL_GAIN, max_step=rel.MAX_STEP, ki=rel.KI, corr_clamp=rel.CORR_CLAMP,
                 null_reference="expert_nominal_joint_path"):
        self.base, self.expert, self.monitor, self.geom = base, expert, monitor, geom
        self.dz = float(dz)
        n, d = base.num_envs, base.device
        self.n, self.device = n, d
        self.robot = base.scene["robot"]
        self.parcel = base.scene["parcel"]
        self.arm_idx = list(expert.expert.arm_idx)
        self.hand_idx = list(expert.expert.hand_idx)
        self.chain_ids = [int(expert.jids[i]) for i in self.arm_idx]
        self.hand_id = self.robot.find_bodies(HAND_ROOT)[0][0]
        limits = self.robot.data.joint_pos_limits[0, self.chain_ids]
        chain_names = [self.robot.joint_names[j] for j in self.chain_ids]
        defaults = {nm: float(v) for nm, v in zip(self.robot.joint_names, self.robot.data.default_joint_pos[0].tolist())}
        self.chain = rel.PinocchioChain(os.path.abspath(URDF_PATH), chain_names, HAND_ROOT, pelvis_pos=G.PELVIS_POS,
                                        defaults=defaults)
        self.solver = rel.RelativeArmSolver(n, len(self.chain_ids), limits[:, 0], limits[:, 1], self.chain.fk_jac, d,
                                            iters=iters, lam=dls_lambda, null_gain=null_gain, max_step=max_step, ki=ki,
                                            corr_clamp=corr_clamp)
        self.params = {"handoff_dz": self.dz, "dls_lambda": dls_lambda, "null_gain": null_gain, "max_step": max_step,
                       "iters": iters, "ki": ki, "corr_clamp": corr_clamp, "null_reference": null_reference,
                       "anchor": "forward_kinematics_of_actor_arm_command_minus_static_servo_offset_at_t_H",
                       "kinematic_model": "urdf_pinocchio",
                       "reference_path": "forward_kinematics_of_nominal_expert_command",
                       "urdf": os.path.relpath(os.path.abspath(URDF_PATH), os.getcwd())}
        self.path = rel.ExpertCommandPath(expert.expert.names, self.chain, expert.q_default[0], n, device=d)
        assert self.path.arm_idx == self.arm_idx
        assert null_reference in ("expert_nominal_joint_path", "joint_mid_range")
        self.null_reference = null_reference
        self.hand_open = expert.expert.hand_open.to(torch.float64)
        self.hand_grasp = expert.expert.hand_grasp.to(torch.float64)
        self.k_release = G.PHASE_INDEX["RELEASE"]
        self.k_insert = G.PHASE_INDEX["INSERT"]
        eye = torch.eye(4, dtype=torch.float64, device=d)
        z = lambda *shape, dtype=torch.float64: torch.zeros(*shape, dtype=dtype, device=d)  # noqa: E731
        self.active = z(n, dtype=torch.bool)
        self.hand_hold = z(n, len(self.hand_idx))
        self.T_anchor = eye.unsqueeze(0).repeat(n, 1, 1)
        self.T_E0_inv = eye.unsqueeze(0).repeat(n, 1, 1)
        self.handoff_step = torch.full((n,), -1, dtype=torch.long, device=d)
        self.kH = torch.zeros(n, dtype=torch.long, device=d)
        self.fH = z(n)
        self.hand_pose_H = z(n, 7)
        self.T_cmd_H = eye.unsqueeze(0).repeat(n, 1, 1)
        self.d0 = z(n, len(self.chain_ids))
        self.d0_norm_H = z(n)
        self.k_lift = G.PHASE_INDEX["LIFT"]
        self.max_model_p, self.max_model_r = z(n), z(n)
        self.p_ho_H = z(n, 3)
        self.q_ho_H = z(n, 4)
        self.contact_H = [None] * n
        self.max_res_p, self.max_res_r = z(n), z(n)
        self.max_track_p, self.max_track_r = z(n), z(n)
        self.hold_viol = z(n)
        self.endpoint_done = z(n, dtype=torch.bool)
        self.max_slipH_t, self.max_slipH_r = z(n), z(n)
        self.max_cubby_seg = z(n)
        self.ep_retained = z(n, dtype=torch.bool)
        self.ep_slip_t, self.ep_slip_r = z(n), z(n)
        self.ep_slip_acq_t, self.ep_slip_acq_r = z(n), z(n)
        self.ep_contacts = torch.zeros(n, dtype=torch.long, device=d)
        self.ep_p_ho, self.ep_q_ho = z(n, 3), z(n, 4)
        self.ep_step = torch.full((n,), -1, dtype=torch.long, device=d)
        self.ep_by_contact = z(n, dtype=torch.bool)
        self.ep_cubby_force = z(n)
        self.last_steps = torch.zeros(n, dtype=torch.long, device=d)

    # ------------------------------------------------------------------
    def _clear(self, mask):
        if not mask.any():
            return
        self.active[mask] = False
        self.handoff_step[mask] = -1
        self.endpoint_done[mask] = False
        for name in ("max_res_p", "max_res_r", "max_track_p", "max_track_r", "max_model_p", "max_model_r",
                     "hold_viol", "max_slipH_t",
                     "max_slipH_r", "max_cubby_seg", "ep_slip_t", "ep_slip_r", "ep_slip_acq_t", "ep_slip_acq_r"):
            getattr(self, name)[mask] = 0.0
        self.ep_retained[mask] = False
        self.ep_by_contact[mask] = False
        self.ep_cubby_force[mask] = 0.0
        self.ep_contacts[mask] = 0
        self.ep_step[mask] = -1
        for i in mask.nonzero(as_tuple=False).flatten().tolist():
            self.contact_H[i] = None

    def _hand_state(self):
        """World hand pose (for the hand-object transform) and the
        environment-local hand pose (the frame of the geometry and the
        kinematic model)."""
        pos = self.robot.data.body_pos_w[:, self.hand_id]
        quat = self.robot.data.body_quat_w[:, self.hand_id]
        return pos, quat, pos - self.base.scene.env_origins

    def _hand_object(self, pos, quat):
        return subtract_frame_transforms(pos, quat, self.parcel.data.root_pos_w, self.parcel.data.root_quat_w)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def hook(self, base, obs, act, monitor):
        reset = monitor.steps < self.last_steps
        self._clear(reset)
        self.last_steps = monitor.steps.clone()
        k, f, _, _ = task_clock.phase_state(base)
        q_default = self.expert.q_default
        q_target_actor = 0.5 * act + q_default
        z_rel = self.parcel.data.root_pos_w[:, 2] - base.scene.env_origins[:, 2] - base._stow_start_pos[:, 2]
        trigger = monitor.acquired & (z_rel >= self.dz) & ~self.active & (k < self.k_release)
        hand_pos, hand_quat, hand_pos_l = self._hand_state()
        q_meas = self.robot.data.joint_pos[:, self.chain_ids]
        q_cmd = q_target_actor[:, self.arm_idx].to(torch.float64)
        # static servo offset of the actor, sampled while the arm rests at the
        # grasp (through the end of GRASP_DWELL)
        pre_lift = (k < self.k_lift).unsqueeze(-1)
        self.d0 = torch.where(pre_lift, q_cmd - q_meas.to(torch.float64), self.d0)
        if trigger.any():
            ids = trigger.nonzero(as_tuple=False).flatten()
            self.solver.start(ids, q_cmd, self.d0)
            T_anchor = self.solver.anchor(self.solver.q_v)
            self.T_anchor[trigger] = T_anchor[trigger]
            self.T_cmd_H[trigger] = self.solver.anchor(q_cmd)[trigger]
            self.d0_norm_H[trigger] = self.d0[trigger].norm(dim=-1)
            self.T_E0_inv[trigger] = rel.inv_tf_batch(self.path.poses(k, f))[trigger]
            self.hand_hold[trigger] = q_target_actor[trigger][:, self.hand_idx].to(torch.float64)
            self.handoff_step[trigger] = monitor.steps[trigger]
            self.kH[trigger] = k[trigger]
            self.fH[trigger] = f[trigger].to(torch.float64)
            self.hand_pose_H[trigger] = torch.cat([hand_pos_l, hand_quat], dim=-1)[trigger].to(torch.float64)
            p_ho, q_ho = self._hand_object(hand_pos, hand_quat)
            self.p_ho_H[trigger] = p_ho[trigger].to(torch.float64)
            self.q_ho_H[trigger] = q_ho[trigger].to(torch.float64)
            forces = monitor.parcel_forces_w()
            cp_w, n_in, dsurf = monitor.contact_geometry()
            for i in ids.tolist():
                self.contact_H[i] = monitor.contact_set(i, forces, cp_w, n_in, dsurf)
            self.active |= trigger
        if not self.active.any():
            return act
        T_d = rel.compose_relative(self.T_anchor, self.T_E0_inv, self.path.poses(k, f))
        dwell = rel.dwell_mask(k) & self.active
        # null-space attractor, the expert's nominal joint configuration at the
        # same phase (the same DLS step, the redundant degrees of freedom stay
        # near the expert's own joint path), or the joint mid-ranges of ChainIK
        q_ref = None
        if self.null_reference == "expert_nominal_joint_path":
            q_ref = self.path.targets(k, f)[:, self.arm_idx].to(torch.float64)
        cmd, res_p, res_r = self.solver.track(T_d, q_meas, dwell, q_ref=q_ref)
        s = 0.5 * (1.0 - torch.cos(math.pi * f.to(torch.float64)))
        hand_cmd = rel.hand_command(k, s, self.hand_hold, self.hand_open, self.hand_grasp)
        act_mask = self.active.unsqueeze(-1)
        q_new = q_target_actor.clone()
        q_new[:, self.arm_idx] = torch.where(act_mask, cmd.to(q_new.dtype), q_new[:, self.arm_idx])
        q_new[:, self.hand_idx] = torch.where(act_mask, hand_cmd.to(q_new.dtype), q_new[:, self.hand_idx])
        # diagnostics, kinematic residual of the target, actual pose error of the
        # measured hand toward the desired pose, and the kinematic model against
        # the PhysX hand pose at the measured configuration
        a = self.active
        R_meas = rel.mat_from_quat_batch(hand_quat.to(torch.float64))
        T_meas = rel.make_tf_batch(R_meas, hand_pos_l.to(torch.float64))
        tr_p, tr_r = rel.pose_error(T_d, T_meas)
        p_m, R_m, _ = self.chain.fk_jac(q_meas.to(torch.float64))
        md_p, md_r = rel.pose_error(rel.make_tf_batch(R_m, p_m), T_meas)
        self.max_res_p = torch.where(a, torch.maximum(self.max_res_p, res_p), self.max_res_p)
        self.max_res_r = torch.where(a, torch.maximum(self.max_res_r, res_r), self.max_res_r)
        self.max_track_p = torch.where(a, torch.maximum(self.max_track_p, tr_p.norm(dim=-1)), self.max_track_p)
        self.max_track_r = torch.where(a, torch.maximum(self.max_track_r, tr_r.norm(dim=-1)), self.max_track_r)
        self.max_model_p = torch.where(a, torch.maximum(self.max_model_p, md_p.norm(dim=-1)), self.max_model_p)
        self.max_model_r = torch.where(a, torch.maximum(self.max_model_r, md_r.norm(dim=-1)), self.max_model_r)
        viol = (hand_cmd - self.hand_hold).abs().amax(dim=-1)
        pre = a & (k < self.k_release)
        self.hold_viol = torch.where(pre, torch.maximum(self.hold_viol, viol), self.hold_viol)
        return pse.to_action(q_new, q_default)

    @torch.no_grad()
    def after_step(self, base, monitor, done):
        live = ~done
        k, f, _, _ = task_clock.phase_state(base)
        seg = self.active & live & ~self.endpoint_done
        if not seg.any():
            return
        hand_pos, hand_quat, _ = self._hand_state()
        p_ho, q_ho = self._hand_object(hand_pos, hand_quat)
        p_ho, q_ho = p_ho.to(torch.float64), q_ho.to(torch.float64)
        slip_t = (p_ho - self.p_ho_H).norm(dim=-1)
        slip_r = _quat_angle(quat_mul(quat_inv(self.q_ho_H), q_ho))
        self.max_slipH_t = torch.where(seg, torch.maximum(self.max_slipH_t, slip_t), self.max_slipH_t)
        self.max_slipH_r = torch.where(seg, torch.maximum(self.max_slipH_r, slip_r), self.max_slipH_r)
        forces = monitor.parcel_forces_w()
        fmag = forces.norm(dim=-1)
        f_sum = fmag.sum(dim=-1)
        n_contacts = (fmag > monitor.threshold).sum(dim=-1)
        cubby = monitor.cubby_forces_w().norm(dim=-1).sum(dim=-1).to(torch.float64)
        # the primary endpoint, the first step in INSERT or the first receptacle
        # contact (force above RECEPTACLE_CONTACT_FORCE), whichever comes first,
        # so the free-space segment ends before any receptacle interaction
        touch = cubby > RECEPTACLE_CONTACT_FORCE
        endpoint = seg & ((k >= self.k_insert) | touch)
        before = seg & ~endpoint
        self.max_cubby_seg = torch.where(before, torch.maximum(self.max_cubby_seg, cubby), self.max_cubby_seg)
        if endpoint.any():
            self.ep_by_contact[endpoint] = (touch & (k < self.k_insert))[endpoint]
            self.ep_cubby_force[endpoint] = cubby[endpoint]
            retained = monitor.acquired & ~monitor.dropped & (f_sum > DROP_FORCE)
            self.ep_retained[endpoint] = retained[endpoint]
            self.ep_slip_t[endpoint] = slip_t[endpoint]
            self.ep_slip_r[endpoint] = slip_r[endpoint]
            slip_acq_t = (p_ho - monitor.p_ho0.to(torch.float64)).norm(dim=-1)
            slip_acq_r = _quat_angle(quat_mul(quat_inv(monitor.q_ho0.to(torch.float64)), q_ho))
            self.ep_slip_acq_t[endpoint] = slip_acq_t[endpoint]
            self.ep_slip_acq_r[endpoint] = slip_acq_r[endpoint]
            self.ep_contacts[endpoint] = n_contacts[endpoint]
            self.ep_p_ho[endpoint] = p_ho[endpoint]
            self.ep_q_ho[endpoint] = q_ho[endpoint]
            self.ep_step[endpoint] = monitor.steps[endpoint]
            self.endpoint_done |= endpoint

    def record_hook(self, i, rec, mu=G.PARCEL_FRICTION):
        on = bool(self.active[i])
        rec["relative_handoff"] = on
        rec["relative_handoff_params"] = self.params
        rec["handoff_step"] = int(self.handoff_step[i]) if on else -1
        if on:
            rec["handoff_phase"] = G.PHASE_NAMES[int(self.kH[i])]
            rec["handoff_fraction"] = float(self.fH[i])
            hp = self.hand_pose_H[i].tolist()
            rec["hand_pose_handoff"] = {"pos": hp[:3], "quat_wxyz": hp[3:], "frame": "env_local"}
            Tc = self.T_cmd_H[i]
            Ta = self.T_anchor[i]
            rec["hand_command_pose_handoff"] = {"pos": Tc[:3, 3].tolist(), "R": Tc[:3, :3].tolist(), "frame": "env_local"}
            rec["anchor_pose_handoff"] = {"pos": Ta[:3, 3].tolist(), "R": Ta[:3, :3].tolist(), "frame": "env_local"}
            R_meas = rel.mat_from_quat_batch(self.hand_pose_H[i:i + 1, 3:7])
            T_meas = rel.make_tf_batch(R_meas, self.hand_pose_H[i:i + 1, :3])
            lead_p, lead_r = rel.pose_error(Tc.unsqueeze(0), T_meas)
            rec["handoff_command_lead_m"] = float(lead_p.norm())
            rec["handoff_command_lead_deg"] = math.degrees(float(lead_r.norm()))
            an_p, an_r = rel.pose_error(Ta.unsqueeze(0), T_meas)
            rec["handoff_anchor_minus_measured_m"] = float(an_p.norm())
            rec["handoff_anchor_minus_measured_deg"] = math.degrees(float(an_r.norm()))
            rec["handoff_static_offset_rad"] = float(self.d0_norm_H[i])
            rec["hand_object_handoff"] = {"pos": self.p_ho_H[i].tolist(), "quat_wxyz": self.q_ho_H[i].tolist()}
            cs = self.contact_H[i]
            rec["contact_set_handoff"] = cs
            rec["contact_count_handoff"] = len(cs["contacts"]) if cs else 0
            eps, epsb = score_contact_set(cs, mu=mu)
            rec["epsilon_handoff"] = eps
            rec["epsilon_beta_handoff"] = epsb
            reached = bool(self.endpoint_done[i])
            rec["primary_endpoint_reached"] = reached
            rec["primary_endpoint_step"] = int(self.ep_step[i])
            rec["primary_endpoint_reason"] = (("receptacle_contact" if bool(self.ep_by_contact[i]) else "insert_start")
                                              if reached else "not_reached")
            rec["receptacle_force_at_endpoint"] = float(self.ep_cubby_force[i]) if reached else None
            rec["retained_preinsert"] = bool(self.ep_retained[i]) if reached else False
            rec["dp_preinsert_m"] = float(self.ep_slip_t[i]) if reached else None
            rec["dR_preinsert_deg"] = math.degrees(float(self.ep_slip_r[i])) if reached else None
            rec["dp_preinsert_from_acquisition_m"] = float(self.ep_slip_acq_t[i]) if reached else None
            rec["dR_preinsert_from_acquisition_deg"] = math.degrees(float(self.ep_slip_acq_r[i])) if reached else None
            rec["dp_max_segment_m"] = float(self.max_slipH_t[i])
            rec["dR_max_segment_deg"] = math.degrees(float(self.max_slipH_r[i]))
            rec["contact_count_preinsert_endpoint"] = int(self.ep_contacts[i]) if reached else None
            rec["hand_object_preinsert"] = ({"pos": self.ep_p_ho[i].tolist(), "quat_wxyz": self.ep_q_ho[i].tolist()}
                                            if reached else None)
            rec["receptacle_force_before_endpoint"] = float(self.max_cubby_seg[i])
            rec["max_kinematic_residual_pos_m"] = float(self.max_res_p[i])
            rec["max_kinematic_residual_rot_deg"] = math.degrees(float(self.max_res_r[i]))
            rec["max_pose_tracking_error_pos_m"] = float(self.max_track_p[i])
            rec["max_pose_tracking_error_rot_deg"] = math.degrees(float(self.max_track_r[i]))
            rec["max_kinematic_model_error_pos_m"] = float(self.max_model_p[i])
            rec["max_kinematic_model_error_rot_deg"] = math.degrees(float(self.max_model_r[i]))
            rec["hand_hold_violation_rad"] = float(self.hold_viol[i])
        else:
            rec["primary_endpoint_reached"] = False
            rec["retained_preinsert"] = False
        mask = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        mask[i] = True
        self._clear(mask)
