import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "manipulation"))

from act_training import (  # noqa: E402
    Normalization,
    build_action_batch,
    demonstration_collection_summary,
    fixed_validation_starts,
    masked_l1_loss,
    split_episode_indices,
)


def _load_train_script():
    spec = importlib.util.spec_from_file_location("train_act", ROOT / "scripts" / "train_act.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _episodes(count=10):
    return [
        (torch.full((index + 2, 3), float(index)), torch.full((index + 2, 2), float(index)))
        for index in range(count)
    ]


def test_episode_split_is_deterministic_and_disjoint():
    first = split_episode_indices(10, 0.2, 7)
    second = split_episode_indices(10, 0.2, 7)
    assert first == second
    assert len(first.train) == 8
    assert len(first.validation) == 2
    assert set(first.train).isdisjoint(first.validation)
    assert set(first.train) | set(first.validation) == set(range(10))


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1])
def test_episode_split_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        split_episode_indices(10, fraction, 7)


def test_fixed_validation_starts_are_reproducible():
    episodes = _episodes()
    first = fixed_validation_starts(episodes, [2, 5], 3, 11)
    second = fixed_validation_starts(episodes, [2, 5], 3, 11)
    assert first == second
    assert len(first) == 6
    assert all(0 <= start < episodes[index][0].shape[0] for index, start in first)


def test_demonstration_collection_summary_records_admission():
    source = {
        "all_records": [
            {"task_success": True, "failure_reason": "none"},
            {"task_success": True, "failure_reason": "none"},
            {"task_success": False, "failure_reason": "insertion_jam"},
        ],
        "rate_spec": {"mode": "uniform", "lo": 0.5, "hi": 1.0},
        "jitter": 0.01,
        "seed": 1,
    }
    assert demonstration_collection_summary(source, 2) == {
        "attempted_episodes": 3,
        "admitted_episodes": 2,
        "rejected_failure_reasons": {"insertion_jam": 1},
        "rate_spec": {"mode": "uniform", "lo": 0.5, "hi": 1.0},
        "jitter": 0.01,
        "seed": 1,
    }


def test_demonstration_collection_summary_rejects_count_mismatch():
    source = {"records": [{"task_success": True, "failure_reason": "none"}]}
    with pytest.raises(ValueError, match="do not match"):
        demonstration_collection_summary(source, 2)


def test_action_batch_pads_without_changing_valid_values():
    episodes = [(torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0], [5.0]]))]
    normalization = Normalization(
        obs_mean=torch.tensor([1.0]),
        obs_std=torch.tensor([2.0]),
        act_mean=torch.tensor([1.0]),
        act_std=torch.tensor([2.0]),
    )
    observation, actions, is_pad = build_action_batch(
        episodes, [(0, 1)], normalization, chunk_size=3, device="cpu"
    )
    assert torch.equal(observation, torch.tensor([[0.5]]))
    assert torch.equal(actions, torch.tensor([[[2.0], [0.0], [0.0]]]))
    assert torch.equal(is_pad, torch.tensor([[False, True, True]]))


def test_masked_l1_averages_only_valid_action_elements():
    prediction = torch.tensor([[[2.0, 4.0], [100.0, 100.0]]])
    target = torch.zeros_like(prediction)
    is_pad = torch.tensor([[False, True]])
    assert masked_l1_loss(prediction, target, is_pad).item() == pytest.approx(3.0)


def test_masked_l1_rejects_an_all_padding_batch():
    prediction = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError):
        masked_l1_loss(prediction, prediction, torch.ones(1, 2, dtype=torch.bool))


def test_candidate_retention_keeps_lowest_validation_losses_and_final_epoch():
    train_act = _load_train_script()
    candidates = [
        {"epoch": 100, "validation_chunk_l1": 0.3},
        {"epoch": 200, "validation_chunk_l1": 0.1},
        {"epoch": 300, "validation_chunk_l1": 0.2},
    ]
    assert train_act.retained_epochs(candidates, keep_top=2, final_epoch=300) == {
        200,
        300,
    }
