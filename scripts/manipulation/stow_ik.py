"""Damped least squares IK on the G1 chain for the ParcelStow waypoints,
the pattern of scripts/stress/build_bank_grid.py wrapped as a class. Import
after the Isaac app has launched. The solver runs a standalone
SimulationContext without gravity holding one articulation, writes joint
states directly, and reads the hand-root pose and the PhysX Jacobian.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix, subtract_frame_transforms

from parcelstow.robots import G1_L6_CFG

TARGET_BODY = "rh_hand_base_link"
CHAIN_NAMES = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
HAND_BODIES_PREFIX = "rh_"
# frogger record hand joint names -> L6 URDF joint names
HAND_NAME_MAP = {
    "thumb_cmc_yaw": "rh_thumb_cmc_roll", "thumb_cmc_pitch": "rh_thumb_cmc_pitch",
    "thumb_ip": "rh_thumb_dip", "index_mcp_pitch": "rh_index_mcp_pitch", "index_dip": "rh_index_dip",
    "middle_mcp_pitch": "rh_middle_mcp_pitch", "middle_dip": "rh_middle_dip",
    "ring_mcp_pitch": "rh_ring_mcp_pitch", "ring_dip": "rh_ring_dip",
    "pinky_mcp_pitch": "rh_pinky_mcp_pitch", "pinky_dip": "rh_pinky_dip",
}


class ChainIK:
    def __init__(self, pelvis_pos=(0.0, 0.0, 0.75), device="cuda:0", pos_tol=0.004, ori_tol_deg=2.0,
                 iters=300, null_gain=0.3, dls_lambda=0.05, max_step=0.15):
        self.null_gain = null_gain
        self.dls_lambda = dls_lambda
        self.max_step = max_step
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=1 / 200, device=device, gravity=(0.0, 0.0, 0.0))
        )
        cfg = G1_L6_CFG.copy()
        cfg.prim_path = "/World/Robot"
        cfg.spawn.articulation_props.fix_root_link = True
        cfg.init_state.pos = tuple(float(v) for v in pelvis_pos)
        cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.robot = Articulation(cfg)
        self.sim.reset()
        self.device = self.robot.device
        self.chain_ids, resolved = self.robot.find_joints(CHAIN_NAMES, preserve_order=True)
        assert resolved == CHAIN_NAMES, resolved
        self.body_id = self.robot.find_bodies(TARGET_BODY)[0][0]
        self.jacobi_id = self.body_id - 1
        self.hand_body_ids = [i for i, n in enumerate(self.robot.body_names) if n.startswith(HAND_BODIES_PREFIX)]
        self.hand_body_names = [self.robot.body_names[i] for i in self.hand_body_ids]
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=1, device=self.device,
        )
        self.pos_tol = pos_tol
        self.ori_tol = math.radians(ori_tol_deg)
        self.iters = iters
        self.limits = self.robot.data.joint_pos_limits[0, self.chain_ids].clone()
        self.hand_joint_ids = None

    # ------------------------------------------------------------------
    def set_hand(self, hand_named: dict):
        """Hold the hand joints at the given named values in every settle."""
        names = list(hand_named.keys())
        ids, resolved = self.robot.find_joints(names, preserve_order=True)
        assert resolved == names, resolved
        self.hand_joint_ids = ids
        self.hand_values = torch.tensor([float(hand_named[n]) for n in names], device=self.device)

    def settle(self, q_chain):
        q = self.robot.data.default_joint_pos.clone()
        q[:, self.chain_ids] = torch.as_tensor(q_chain, dtype=torch.float32, device=self.device)
        if self.hand_joint_ids is not None:
            q[:, self.hand_joint_ids] = self.hand_values
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        self.robot.set_joint_position_target(q)
        self.robot.write_data_to_sim()
        self.sim.step(render=False)
        self.robot.update(self.sim.get_physics_dt())

    def hand_pose(self):
        """4x4 world pose of the hand root at the current state."""
        T = np.eye(4)
        T[:3, :3] = matrix_from_quat(self.robot.data.body_quat_w[0:1, self.body_id])[0].cpu().numpy()
        T[:3, 3] = self.robot.data.body_pos_w[0, self.body_id].cpu().numpy()
        return T

    def body_positions(self, names=None):
        """World positions of hand bodies (dict name -> (3,))."""
        pos = self.robot.data.body_pos_w[0].cpu().numpy()
        out = {}
        for i, n in zip(self.hand_body_ids, self.hand_body_names):
            out[n] = pos[i]
        if names is not None:
            for n in names:
                out[n] = pos[self.robot.body_names.index(n)]
        return out

    def limit_margin(self, q_chain):
        q = torch.as_tensor(q_chain, device=self.device)
        lo = self.limits[:, 0]
        hi = self.limits[:, 1]
        m = torch.minimum(q - lo, hi - q) / (hi - lo).clamp(min=1e-6)
        return float(m.min()), m.cpu().numpy()

    def solve_multi(self, T_target, seeds, n_random=8, scale=0.6, rng_seed=0, return_all=False,
                    stop_early=True):
        """solve() from several seeds (the given ones plus random
        perturbations of the first inside the limits), returns the best
        result by (ok, pos_err + 0.05 * ori_err), or with return_all every
        result sorted by that score."""
        rng = np.random.default_rng(rng_seed)
        lo = self.limits[:, 0].cpu().numpy()
        hi = self.limits[:, 1].cpu().numpy()
        cands = [np.asarray(s, dtype=np.float64) for s in seeds]
        base = cands[0]
        for _ in range(n_random):
            q = np.clip(base + rng.uniform(-scale, scale, size=base.shape), lo, hi)
            cands.append(q)
        best = None
        results = []
        for q0 in cands:
            r = self.solve(T_target, q0)
            score = (0 if r["ok"] else 1, r["pos_err"] + 0.05 * math.radians(r["ori_err_deg"]))
            results.append((score, r))
            if best is None or score < best[0]:
                best = (score, r)
            if stop_early and not return_all and r["ok"] and r["pos_err"] < 0.5 * self.pos_tol:
                break
        if return_all:
            results.sort(key=lambda t: t[0])
            return [r for _, r in results]
        return best[1]

    def solve(self, T_target, q_seed):
        """Damped least squares IK toward the world pose T_target from the
        seed chain configuration, with a null-space bias toward the joint
        mid-ranges (limit avoidance on the four redundant degrees of freedom
        of the 10-joint chain). Returns dict(ok, pos_err, ori_err_deg, q,
        in_limits)."""
        self.settle(q_seed)
        R_t = np.asarray(T_target[:3, :3], dtype=np.float64)
        p_t = np.asarray(T_target[:3, 3], dtype=np.float64)
        lo = self.limits[:, 0].cpu().numpy().astype(np.float64)
        hi = self.limits[:, 1].cpu().numpy().astype(np.float64)
        mid = 0.5 * (lo + hi)
        rng = np.maximum(hi - lo, 1e-6)
        n = len(self.chain_ids)
        lam2 = self.dls_lambda ** 2
        pos_err = ori_err = float("inf")
        for _ in range(self.iters):
            T_now = self.hand_pose()
            e_p = p_t - T_now[:3, 3]
            e_r = _so3_log(R_t @ T_now[:3, :3].T)
            pos_err = float(np.linalg.norm(e_p))
            ori_err = float(np.linalg.norm(e_r))
            if pos_err < self.pos_tol and ori_err < self.ori_tol:
                break
            J = self.robot.root_physx_view.get_jacobians()[0, self.jacobi_id][:, self.chain_ids].cpu().numpy().astype(np.float64)
            e = np.concatenate([e_p, e_r])
            JJt = J @ J.T + lam2 * np.eye(6)
            dq_task = J.T @ np.linalg.solve(JJt, e)
            q = self.robot.data.joint_pos[0, self.chain_ids].cpu().numpy().astype(np.float64)
            N = np.eye(n) - J.T @ np.linalg.solve(JJt, J)
            bias = self.null_gain * (mid - q) / rng
            dq = dq_task + N @ bias
            step = np.linalg.norm(dq)
            if step > self.max_step:
                dq = dq * (self.max_step / step)
            q_new = np.clip(q + dq, lo, hi)
            self.settle(q_new)
        q_final = self.robot.data.joint_pos[0, self.chain_ids].clone()
        in_limits = bool(((q_final >= self.limits[:, 0] - 1e-6) & (q_final <= self.limits[:, 1] + 1e-6)).all().item())
        ok = pos_err < self.pos_tol and ori_err < self.ori_tol and in_limits
        return {"ok": ok, "pos_err": pos_err, "ori_err_deg": math.degrees(ori_err),
                "q": q_final.cpu().numpy(), "in_limits": in_limits}


def _so3_log(R):
    c = (np.trace(R) - 1.0) * 0.5
    c = min(1.0, max(-1.0, c))
    a = math.acos(c)
    if a < 1e-9:
        return np.zeros(3)
    if abs(math.pi - a) < 1e-6:
        A = (R + np.eye(3)) * 0.5
        axis = np.sqrt(np.clip(np.diag(A), 0.0, 1.0))
        if A[0, 1] < 0:
            axis[1] = -axis[1]
        if A[0, 2] < 0:
            axis[2] = -axis[2]
        return axis / (np.linalg.norm(axis) + 1e-12) * a
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (a / (2.0 * math.sin(a)))
