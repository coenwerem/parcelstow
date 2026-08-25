"""Reset events of the ParcelStow task. The parcel start pose reset reuses
the shared reset_root_state_uniform, the task-rate sampling lives in
task_clock, and the parcel material and mass are fixed by the scene
configuration (no per-episode randomization of friction or mass in the
main experiment)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def record_parcel_start(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    parcel_cfg: SceneEntityCfg = SceneEntityCfg("parcel"),
):
    """Reset event (run after the pose reset), stores the parcel start pose
    per environment so the expert and the monitor read the actual start."""
    parcel = env.scene[parcel_cfg.name]
    if not hasattr(env, "_stow_start_pos"):
        env._stow_start_pos = torch.zeros(env.num_envs, 3, device=env.device)
        env._stow_start_quat = torch.zeros(env.num_envs, 4, device=env.device)
        env._stow_start_quat[:, 0] = 1.0
    env._stow_start_pos[env_ids] = parcel.data.root_pos_w[env_ids] - env.scene.env_origins[env_ids]
    env._stow_start_quat[env_ids] = parcel.data.root_quat_w[env_ids]
