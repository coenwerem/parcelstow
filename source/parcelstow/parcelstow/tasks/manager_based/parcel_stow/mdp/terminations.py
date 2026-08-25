"""Termination terms of the ParcelStow task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .. import geometry as G

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def parcel_fell(
    env: ManagerBasedRLEnv,
    parcel_cfg: SceneEntityCfg = SceneEntityCfg("parcel"),
    margin: float = 0.10,
) -> torch.Tensor:
    """The parcel left the table (its center dropped below the table top by
    more than margin), the episode cannot succeed any more."""
    parcel = env.scene[parcel_cfg.name]
    return parcel.data.root_pos_w[:, 2] < G.TABLE_TOP - margin
