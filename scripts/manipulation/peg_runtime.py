"""Shared runtime pieces of the keyed-peg insertion drivers: the phase
schedule, the expert actor, and the configuration stamp. Import after
the Isaac app has launched, the stow_runtime convention."""

from __future__ import annotations

import time

import peg_insert_expert as upe
import stow_runtime as rt
import torch
from parcelstow.phase_schedule import PhaseSchedule
from parcelstow.tasks.manager_based.parcel_stow.mdp import task_clock
from parcelstow.tasks.manager_based.peg_insert import geometry as U

SCHED = PhaseSchedule(U.PHASES)


class PegExpertActor:
    name = "expert"

    def __init__(self, base, bank=None, trajectory=None, candidate=0):
        self.base = base
        self.robot = base.scene["robot"]
        from parcelstow.tasks.manager_based.peg_insert.peg_insert_env_cfg import CHAIN_ACTUATED
        self.jids, self.jnames = self.robot.find_joints(CHAIN_ACTUATED, preserve_order=True)
        self.q_default = self.robot.data.default_joint_pos[:, self.jids]
        self.expert = upe.PegExpert(self.jnames, bank=bank, trajectory=trajectory,
                                        device=base.device, candidate=candidate)
        self.expert.allocate(base.num_envs)
        self.start_xy = torch.tensor(U.START_POS[:2], device=base.device)

    def reset(self, ids, obs=None):
        ids = torch.as_tensor(list(ids), dtype=torch.long, device=self.base.device)
        if len(ids) == 0:
            return
        off = self.base._stow_start_pos[ids, :2] - self.start_xy
        self.expert.reset(ids, off)

    @torch.no_grad()
    def act(self, obs):
        k, f, _, _ = task_clock.phase_state(self.base)
        q_meas = self.robot.data.joint_pos[:, self.jids]
        return self.expert.act(k, f, self.q_default, q_meas)


def config_stamp(base, task_id="PegInsert-L6-Play-v0"):
    return {
        "git_sha": rt.git_sha(),
        "task": task_id,
        "object_extents": list(U.OBJECT_EXTENTS), "object_mass": U.OBJECT_MASS,
        "object_friction": U.OBJECT_FRICTION, "object_start": list(U.START_POS),
        "pocket_center": list(U.POCKET_CENTER), "clearance": U.CLEARANCE,
        "pocket_depth": U.POCKET_DEPTH, "seat_z": U.SEAT_Z,
        "inserted_min_depth": U.INSERTED_MIN_DEPTH,
        "final_tilt_tol_deg": U.FINAL_TILT_TOL_DEG,
        "grasp_shift": U.GRASP_SHIFT,
        "phases": [[n, d, s] for n, d, s in U.PHASES],
        "control_dt": float(base.step_dt),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
