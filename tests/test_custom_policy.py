import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.custom_policy import HoldPosturePolicy  # noqa: E402


class FakeEnvironment:
    device = "cpu"
    num_envs = 3


def test_same_custom_policy_loads_for_every_task_contract():
    observation = torch.zeros(3, 147)
    observation[:, 146] = 1.5
    for task in ("parcel", "upright", "peg"):
        policy = HoldPosturePolicy(FakeEnvironment(), checkpoint=f"{task}.pt")
        action, target = policy.act(observation)
        assert action.shape == (3, 16)
        assert target is None
        assert policy.checkpoint == f"{task}.pt"
