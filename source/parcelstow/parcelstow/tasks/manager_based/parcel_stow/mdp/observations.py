"""Observation terms of the ParcelStow task beyond the shared ones. The
distal phalanx positions and contact forces come from the shared contact terms
(fingertip_pos_b, tip_contact_forces), the phase and rate come from
task_clock."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, quat_mul, quat_inv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def parcel_pose_b(
    env: ManagerBasedRLEnv,
    parcel_cfg: SceneEntityCfg = SceneEntityCfg("parcel"),
) -> torch.Tensor:
    """Parcel position and orientation in the pelvis frame, (E, 7), position
    then quaternion wxyz."""
    robot = env.scene["robot"]
    parcel = env.scene[parcel_cfg.name]
    rel = parcel.data.root_pos_w - robot.data.root_pos_w
    pos_b = quat_apply_inverse(robot.data.root_quat_w, rel)
    quat_b = quat_mul(quat_inv(robot.data.root_quat_w), parcel.data.root_quat_w)
    return torch.cat([pos_b, quat_b], dim=-1)
