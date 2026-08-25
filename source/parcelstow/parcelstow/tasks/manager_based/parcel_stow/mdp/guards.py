"""Guard terminations that reset divergent envs before they poison a batch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def exploding_state(
    env: ManagerBasedRLEnv,
    max_joint_vel: float = 100.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True for envs whose state is diverging or already non-finite.

    Physics divergence winds up over several steps before reaching inf, so
    resetting at a finite velocity threshold keeps NaN out of the rollout,
    where observation clipping cannot help (clipping NaN yields NaN).
    """
    robot = env.scene[robot_cfg.name]
    qv = robot.data.joint_vel
    bad = (qv.abs() > max_joint_vel).any(dim=1)
    bad |= ~torch.isfinite(qv).all(dim=1)
    bad |= ~torch.isfinite(robot.data.root_pos_w).all(dim=1)
    bad |= ~torch.isfinite(robot.data.root_quat_w).all(dim=1)
    return bad
