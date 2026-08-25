"""Fingertip contact terms of the ParcelStow observation, per-tip contact
force magnitudes and fingertip positions in the robot base frame."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Contact bodies are the DISTAL phalanges, not the tip point frames. On
# the L6 URDF the rh_*_tip frames sit past the distal collision geometry
# and read 0 N through a full certified pick, while rh_*_distal (and the
# thumb metacarpal) carry 3 to 20 N at hold and lift, measured with the
# scripted expert through an all-links ContactSensor on 2026-08-16. A
# tip-only sensor therefore rewards a region no grasp can visit.
TIP_BODIES = [
    "rh_thumb_distal",
    "rh_index_distal",
    "rh_middle_distal",
    "rh_ring_distal",
    "rh_pinky_distal",
]
PALM_BODY = "rh_hand_base_link"


def _tip_ids(env: ManagerBasedRLEnv):
    if not hasattr(env, "_contact_tip_ids"):
        robot = env.scene["robot"]
        ids, names = robot.find_bodies(TIP_BODIES, preserve_order=True)
        palm_id = robot.find_bodies(PALM_BODY)[0][0]
        env._contact_tip_ids = (ids, palm_id)
    return env._contact_tip_ids


def _tip_forces(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-tip net contact force magnitudes, (E, 5) in TIP_BODIES order."""
    sensor = env.scene.sensors["tip_contacts"]
    if not hasattr(env, "_contact_sensor_order"):
        order = [sensor.body_names.index(n) for n in TIP_BODIES]
        env._contact_sensor_order = torch.tensor(order, device=env.device)
    forces = sensor.data.net_forces_w.norm(dim=-1)
    return forces[:, env._contact_sensor_order]


def fingertip_pos_b(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Fingertip positions in the robot base frame, (E, 15)."""
    from isaaclab.utils.math import quat_apply_inverse

    robot = env.scene["robot"]
    tip_ids, _ = _tip_ids(env)
    rel = robot.data.body_pos_w[:, tip_ids] - robot.data.root_pos_w.unsqueeze(1)
    quat = robot.data.root_quat_w.unsqueeze(1).expand(-1, len(TIP_BODIES), -1)
    return quat_apply_inverse(quat.reshape(-1, 4), rel.reshape(-1, 3)).reshape(
        env.num_envs, -1
    )


def tip_contact_forces(env: ManagerBasedRLEnv, scale: float = 10.0) -> torch.Tensor:
    """Clipped per-tip contact force magnitudes, (E, 5)."""
    return torch.clamp(_tip_forces(env), max=5.0 * scale) / scale
