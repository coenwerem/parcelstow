import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "manipulation"))

from act_diagnostics import (  # noqa: E402
    FixedPhaseRateActor,
    chunk_crosses_phase_boundary,
    initial_condition_bank_sha256,
    override_rate_during_phases,
)


def _observation(phase, rate):
    observation = torch.zeros(147)
    observation[145] = (phase + 0.5) / 13
    observation[146] = rate
    return observation


def test_rate_override_changes_only_selected_phases_and_does_not_mutate_input():
    observations = torch.stack([_observation(0, 1.5), _observation(5, 1.5)])
    original = observations.clone()
    output = override_rate_during_phases(observations, 1.0, phases=(0, 1, 2, 3, 4))
    assert output[0, 146].item() == pytest.approx(1.0)
    assert output[1, 146].item() == pytest.approx(1.5)
    assert torch.equal(observations, original)
    assert torch.equal(output[:, :145], observations[:, :145])


def test_rate_override_rejects_invalid_phase_indices():
    with pytest.raises(ValueError):
        override_rate_during_phases(_observation(0, 1.5), 1.0, phases=(13,))


def test_chunk_boundary_detection():
    phases = torch.tensor([0, 0, 0, 1, 1])
    assert not chunk_crosses_phase_boundary(phases, 0, 3)
    assert chunk_crosses_phase_boundary(phases, 1, 3)


def test_initial_condition_bank_hash_is_stable_and_value_sensitive():
    joints = torch.tensor([[0.1, -0.2], [0.3, 0.4]])
    poses = torch.tensor([[0.35, 0.0, 0.7, 1.0, 0.0, 0.0, 0.0]])
    first = initial_condition_bank_sha256("object", joints, poses)
    second = initial_condition_bank_sha256("object", joints.clone(), poses.clone())
    changed = initial_condition_bank_sha256("object", joints + 0.01, poses)
    assert first == second
    assert first != changed


def test_diagnostic_actor_transforms_reset_and_action_observations():
    class Actor:
        name = "act"

        def __init__(self):
            self.reset_observation = None
            self.action_observation = None

        def reset(self, ids, obs=None):
            self.reset_observation = obs

        def act(self, obs):
            self.action_observation = obs
            return obs[:, :1], None

    base = Actor()
    actor = FixedPhaseRateActor(base, 1.0)
    observations = torch.stack([_observation(2, 1.5), _observation(7, 1.5)])
    actor.reset([0, 1], observations)
    actor.act(observations)
    assert base.reset_observation[:, 146].tolist() == pytest.approx([1.0, 1.5])
    assert base.action_observation[:, 146].tolist() == pytest.approx([1.0, 1.5])
