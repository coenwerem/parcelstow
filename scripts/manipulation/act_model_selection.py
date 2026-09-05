"""Pure validation and ranking utilities for ACT development checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .file_integrity import sha256_file
except ImportError:  # Loaded directly by the pure tests.
    from file_integrity import sha256_file


@dataclass(frozen=True)
class CandidateResult:
    """Validation and matched-bank rollout evidence for one checkpoint."""

    epoch: int
    validation_chunk_l1: float
    task_success: int
    stage_counts: tuple[int, ...]
    bank_sha256: str

    @property
    def rank(self) -> tuple[float | int, ...]:
        """Return the predeclared ascending selection rank."""
        return (
            -self.task_success,
            *(-count for count in self.stage_counts),
            self.validation_chunk_l1,
            self.epoch,
        )


def select_candidate(candidates: Sequence[CandidateResult]) -> CandidateResult:
    """Select a checkpoint using the predeclared development rule."""
    if not candidates:
        raise ValueError("at least one checkpoint candidate is required")
    bank_hashes = {candidate.bank_sha256 for candidate in candidates}
    if len(bank_hashes) != 1:
        raise ValueError("checkpoint candidates do not share one initial-condition bank")
    return min(candidates, key=lambda candidate: candidate.rank)


def validate_checkpoint_binding(
    records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    policy: str,
    checkpoint_path: str | Path | None,
) -> str | None:
    """Verify that records, summary, and current checkpoint bytes agree."""
    if not records:
        raise ValueError("checkpoint validation requires episode records")
    try:
        record_paths = {record["checkpoint"] for record in records}
        record_hashes = {record["checkpoint_sha256"] for record in records}
        summary_path = summary["checkpoint"]
        summary_hash = summary["checkpoint_sha256"]
    except KeyError as exc:
        raise ValueError(f"missing checkpoint provenance field {exc.args[0]!r}") from exc

    if len(record_paths) != 1:
        raise ValueError("episode records contain multiple checkpoint paths")
    if len(record_hashes) != 1:
        raise ValueError("episode records contain multiple checkpoint hashes")
    recorded_path = next(iter(record_paths))
    recorded_hash = next(iter(record_hashes))
    if summary_path != recorded_path:
        raise ValueError("summary checkpoint path differs from the episode records")
    if summary_hash != recorded_hash:
        raise ValueError("summary checkpoint hash differs from the episode records")

    if policy == "expert":
        if checkpoint_path is not None or recorded_path is not None or recorded_hash is not None:
            raise ValueError("expert records must use null checkpoint provenance")
        return None

    if checkpoint_path is None:
        raise ValueError("a learned policy requires an expected checkpoint path")
    if recorded_path is None or recorded_hash is None:
        raise ValueError("learned-policy records require checkpoint path and hash")
    expected_path = Path(checkpoint_path)
    if Path(recorded_path) != expected_path:
        raise ValueError(f"recorded checkpoint does not match {expected_path}")
    if not expected_path.is_file():
        raise ValueError(f"checkpoint file does not exist: {expected_path}")
    current_hash = sha256_file(expected_path)
    if recorded_hash != current_hash:
        raise ValueError("checkpoint contents changed after evaluation")
    return str(recorded_hash)


def validate_development_records(
    records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    gym_id: str,
    policy: str,
    stage_keys: Sequence[str],
    expected_episodes: int,
    expected_seed: int,
    expected_num_envs: int,
    expected_jitter: float,
    checkpoint_path: str | Path | None,
) -> str:
    """Validate one nominal development evaluation and return its bank hash."""
    if len(records) != expected_episodes:
        raise ValueError(
            f"expected {expected_episodes} episode records, found {len(records)}"
        )
    validate_checkpoint_binding(
        records,
        summary,
        policy=policy,
        checkpoint_path=checkpoint_path,
    )
    expected_ids = set(range(expected_episodes))
    identifiers = [int(record["initial_condition_id"]) for record in records]
    if len(set(identifiers)) != len(identifiers) or set(identifiers) != expected_ids:
        raise ValueError("initial-condition identifiers are not unique and complete")

    required = {
        "task": gym_id,
        "policy": policy,
        "actor_spec": policy,
        "bank_role": "development",
        "seed": expected_seed,
        "num_envs": expected_num_envs,
    }
    for field, expected in required.items():
        values = {record[field] for record in records}
        if values != {expected}:
            raise ValueError(f"episode field {field!r} does not equal {expected!r}")
        if summary[field] != expected:
            raise ValueError(f"summary field {field!r} does not equal {expected!r}")
    if {float(record["task_rate"]) for record in records} != {1.0}:
        raise ValueError("development records must use nominal rate r=1")
    if float(summary["rate"]) != 1.0:
        raise ValueError("development summary must use nominal rate r=1")
    if {float(record["jitter"]) for record in records} != {expected_jitter}:
        raise ValueError("episode jitter does not match the development protocol")
    if float(summary["jitter"]) != expected_jitter:
        raise ValueError("summary jitter does not match the development protocol")
    if any(bool(record["corrupt"]) for record in records):
        raise ValueError("development records must not use observation corruption")
    if any(float(record["action_noise"]) != 0.0 for record in records):
        raise ValueError("development records must not use action noise")
    if any(record["fixed_phase_rate_override"] is not None for record in records):
        raise ValueError("development records must not use a rate-input intervention")

    bank_hashes = {str(record["initial_condition_bank_sha256"]) for record in records}
    if len(bank_hashes) != 1:
        raise ValueError("episode records contain multiple initial-condition bank hashes")

    success_count = sum(bool(record["task_success"]) for record in records)
    if int(summary["task_success"]["k"]) != success_count:
        raise ValueError("summary task-success count does not match episode records")
    if int(summary["episodes"]) != expected_episodes:
        raise ValueError("summary episode count does not match the protocol")
    if int(summary["episodes_requested"]) != expected_episodes:
        raise ValueError("summary requested-episode count does not match the protocol")
    if int(summary["task_success"]["n"]) != expected_episodes:
        raise ValueError("summary task-success denominator does not match the protocol")
    for stage in stage_keys:
        count = sum(bool(record[stage]) for record in records)
        if int(summary[stage]["k"]) != count:
            raise ValueError(f"summary stage count {stage!r} does not match episode records")
        if int(summary[stage]["n"]) != expected_episodes:
            raise ValueError(f"summary stage denominator {stage!r} does not match the protocol")
    return next(iter(bank_hashes))


def paired_success_counts(
    expert_records: Sequence[Mapping[str, Any]],
    learner_records: Sequence[Mapping[str, Any]],
) -> dict[str, int | float]:
    """Return paired success counts after validating the initial-condition join."""
    def by_id(records: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
        indexed = {int(record["initial_condition_id"]): record for record in records}
        if len(indexed) != len(records):
            raise ValueError("initial-condition identifiers are not unique")
        return indexed

    expert = by_id(expert_records)
    learner = by_id(learner_records)
    if set(expert) != set(learner):
        raise ValueError("expert and learner initial-condition identifiers differ")
    for identifier in expert:
        expert_record = expert[identifier]
        learner_record = learner[identifier]
        if (
            expert_record["initial_condition_bank_sha256"]
            != learner_record["initial_condition_bank_sha256"]
        ):
            raise ValueError("expert and learner initial-condition banks differ")
        if expert_record["object_initial_pose"] != learner_record["object_initial_pose"]:
            raise ValueError(f"initial object pose differs for condition {identifier}")

    expert_only = sum(
        bool(expert[index]["task_success"])
        and not bool(learner[index]["task_success"])
        for index in expert
    )
    learner_only = sum(
        bool(learner[index]["task_success"])
        and not bool(expert[index]["task_success"])
        for index in expert
    )
    both = sum(
        bool(expert[index]["task_success"])
        and bool(learner[index]["task_success"])
        for index in expert
    )
    neither = len(expert) - expert_only - learner_only - both
    return {
        "episodes": len(expert),
        "both_succeed": both,
        "expert_only_succeeds": expert_only,
        "learner_only_succeeds": learner_only,
        "neither_succeeds": neither,
        "learner_minus_expert": (learner_only - expert_only) / len(expert),
    }
