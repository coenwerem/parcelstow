"""Pure observation interventions and phase diagnostics for state-only ACT."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

import torch

PHASE_FEATURE_INDEX = 145
RATE_FEATURE_INDEX = 146
PHASE_COUNT = 13
FIXED_ACQUISITION_PHASES = (0, 1, 2, 3, 4)


def initial_condition_bank_sha256(
    object_name: str,
    joint_offsets: torch.Tensor,
    object_poses: torch.Tensor,
) -> str:
    """Hash the indexed reset values that define a simulator episode bank."""
    digest = hashlib.sha256()
    digest.update(object_name.encode("utf-8"))
    for tensor in (joint_offsets, object_poses):
        cpu_tensor = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(cpu_tensor.shape)).encode("ascii"))
        digest.update(str(cpu_tensor.dtype).encode("ascii"))
        digest.update(cpu_tensor.numpy().tobytes())
    return digest.hexdigest()


def phase_indices(observations: torch.Tensor) -> torch.Tensor:
    """Decode task-specific phase indices from the shared numeric feature."""
    phase = torch.floor(observations[..., PHASE_FEATURE_INDEX] * PHASE_COUNT + 1e-5)
    return phase.clamp(0, PHASE_COUNT - 1).to(torch.long)


def override_rate_during_phases(
    observations: torch.Tensor,
    replacement_rate: float,
    phases: Iterable[int] = FIXED_ACQUISITION_PHASES,
) -> torch.Tensor:
    """Copy observations and replace the raw rate only in selected phases."""
    phase_set = tuple(int(phase) for phase in phases)
    if not phase_set:
        return observations.clone()
    if min(phase_set) < 0 or max(phase_set) >= PHASE_COUNT:
        raise ValueError(f"phase indices must be between 0 and {PHASE_COUNT - 1}")
    output = observations.clone()
    decoded = phase_indices(observations)
    selected = torch.zeros_like(decoded, dtype=torch.bool)
    for phase in phase_set:
        selected |= decoded == phase
    output[..., RATE_FEATURE_INDEX] = torch.where(
        selected,
        torch.as_tensor(replacement_rate, device=output.device, dtype=output.dtype),
        output[..., RATE_FEATURE_INDEX],
    )
    return output


def chunk_crosses_phase_boundary(
    phase: torch.Tensor,
    start: int,
    chunk_size: int,
) -> bool:
    """Report whether a stored target chunk enters a later task phase."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= start < phase.shape[0]:
        raise IndexError("start is outside the phase sequence")
    chunk = phase[start : start + chunk_size]
    return bool(torch.any(chunk != chunk[0]))


class FixedPhaseRateActor:
    """Diagnostic actor that changes only the policy's raw rate feature."""

    def __init__(
        self,
        actor,
        replacement_rate: float,
        phases: Sequence[int] = FIXED_ACQUISITION_PHASES,
    ):
        self.actor = actor
        self.name = actor.name
        self.replacement_rate = float(replacement_rate)
        self.phases = tuple(int(phase) for phase in phases)

    def reset(self, ids, obs=None):
        transformed = None
        if obs is not None:
            transformed = override_rate_during_phases(obs, self.replacement_rate, self.phases)
        return self.actor.reset(ids, transformed)

    def act(self, observations):
        transformed = override_rate_during_phases(
            observations, self.replacement_rate, self.phases
        )
        return self.actor.act(transformed)
