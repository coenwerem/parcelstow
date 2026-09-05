"""Pure data and loss utilities for state-only ACT training."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EpisodeSplit:
    """Disjoint episode indices used for optimization and validation."""

    train: tuple[int, ...]
    validation: tuple[int, ...]


@dataclass(frozen=True)
class Normalization:
    """Per-feature statistics computed from training episodes only."""

    obs_mean: torch.Tensor
    obs_std: torch.Tensor
    act_mean: torch.Tensor
    act_std: torch.Tensor


def demonstration_collection_summary(
    source: Mapping[str, Any], admitted_episode_count: int
) -> dict[str, Any]:
    """Extract and validate demonstration admission provenance."""
    records = source.get("all_records", source.get("records", ()))
    if records:
        admitted_records = [record for record in records if record["task_success"]]
        if len(admitted_records) != admitted_episode_count:
            raise ValueError("successful demonstration records do not match stored episodes")
        rejected = [record for record in records if not record["task_success"]]
        attempted_episode_count = len(records)
    else:
        rejected = []
        attempted_episode_count = admitted_episode_count
    return {
        "attempted_episodes": attempted_episode_count,
        "admitted_episodes": admitted_episode_count,
        "rejected_failure_reasons": dict(
            Counter(record["failure_reason"] for record in rejected)
        ),
        "rate_spec": source.get("rate_spec"),
        "jitter": source.get("jitter"),
        "seed": source.get("seed"),
    }


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(chunk_size):
            digest.update(data)
    return digest.hexdigest()


def split_episode_indices(
    episode_count: int,
    validation_fraction: float,
    seed: int,
) -> EpisodeSplit:
    """Return a deterministic, episode-level train-validation partition."""
    if episode_count < 2:
        raise ValueError("at least two episodes are required")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    validation_count = max(1, int(round(episode_count * validation_fraction)))
    validation_count = min(validation_count, episode_count - 1)
    permutation = np.random.default_rng(seed).permutation(episode_count)
    validation = tuple(sorted(int(index) for index in permutation[:validation_count]))
    train = tuple(sorted(int(index) for index in permutation[validation_count:]))
    return EpisodeSplit(train=train, validation=validation)


def compute_normalization(
    episodes: Sequence[tuple[torch.Tensor, torch.Tensor]],
    indices: Sequence[int],
    minimum_std: float = 1e-2,
) -> Normalization:
    """Compute feature statistics without reading validation episodes."""
    if not indices:
        raise ValueError("normalization requires at least one training episode")
    observations = torch.cat([episodes[index][0] for index in indices])
    actions = torch.cat([episodes[index][1] for index in indices])
    return Normalization(
        obs_mean=observations.mean(0),
        obs_std=observations.std(0).clamp(min=minimum_std),
        act_mean=actions.mean(0),
        act_std=actions.std(0).clamp(min=minimum_std),
    )


def sample_episode_starts(
    episodes: Sequence[tuple[torch.Tensor, torch.Tensor]],
    indices: Sequence[int],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Sample one valid start step from each listed episode."""
    return [
        (int(index), int(rng.integers(episodes[index][0].shape[0])))
        for index in indices
    ]


def fixed_validation_starts(
    episodes: Sequence[tuple[torch.Tensor, torch.Tensor]],
    indices: Sequence[int],
    samples_per_episode: int,
    seed: int,
) -> tuple[tuple[int, int], ...]:
    """Create validation starts once so every checkpoint sees the same chunks."""
    if samples_per_episode <= 0:
        raise ValueError("samples_per_episode must be positive")
    rng = np.random.default_rng(seed)
    starts = []
    for index in indices:
        length = episodes[index][0].shape[0]
        for start in rng.integers(length, size=samples_per_episode):
            starts.append((int(index), int(start)))
    return tuple(starts)


def build_action_batch(
    episodes: Sequence[tuple[torch.Tensor, torch.Tensor]],
    starts: Sequence[tuple[int, int]],
    normalization: Normalization,
    chunk_size: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build normalized observations, padded action chunks, and padding masks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not starts:
        raise ValueError("a batch requires at least one episode start")
    obs_mean = normalization.obs_mean.to(device)
    obs_std = normalization.obs_std.to(device)
    act_mean = normalization.act_mean.to(device)
    act_std = normalization.act_std.to(device)
    action_dim = normalization.act_mean.shape[0]
    observations, actions, padding = [], [], []
    for episode_index, start in starts:
        episode_obs, episode_actions = episodes[episode_index]
        if not 0 <= start < episode_obs.shape[0]:
            raise IndexError(f"start {start} is outside episode {episode_index}")
        observations.append((episode_obs[start].to(device) - obs_mean) / obs_std)
        chunk = torch.zeros(chunk_size, action_dim, device=device)
        segment = episode_actions[start : start + chunk_size].to(device)
        chunk[: segment.shape[0]] = (segment - act_mean) / act_std
        mask = torch.ones(chunk_size, dtype=torch.bool, device=device)
        mask[: segment.shape[0]] = False
        actions.append(chunk)
        padding.append(mask)
    return torch.stack(observations), torch.stack(actions), torch.stack(padding)


def masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    is_pad: torch.Tensor,
) -> torch.Tensor:
    """Average absolute error over valid action elements only."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    if is_pad.shape != prediction.shape[:-1]:
        raise ValueError("padding mask does not match the action chunk")
    valid = (~is_pad).unsqueeze(-1).expand_as(prediction)
    valid_count = valid.sum()
    if valid_count == 0:
        raise ValueError("the batch contains no valid action elements")
    error = F.l1_loss(prediction, target, reduction="none")
    return error.masked_select(valid).sum() / valid_count
