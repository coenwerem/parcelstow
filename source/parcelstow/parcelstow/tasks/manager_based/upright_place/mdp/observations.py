"""Observation terms of the upright placement task beyond the shared
ones. The distal phalanx positions and contact forces come from the
shared contact terms, the phase and rate from the shared task clock."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_pose_b(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object position and orientation in the pelvis frame, (E, 7),
    position then quaternion wxyz."""
    robot = env.scene["robot"]
    obj = env.scene[object_cfg.name]
    rel = obj.data.root_pos_w - robot.data.root_pos_w
    pos_b = quat_apply_inverse(robot.data.root_quat_w, rel)
    quat_b = quat_mul(quat_inv(robot.data.root_quat_w), obj.data.root_quat_w)
    return torch.cat([pos_b, quat_b], dim=-1)
