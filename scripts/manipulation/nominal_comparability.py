"""Pure validation and paired-bootstrap calculation for nominal evaluations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 0
CONFIDENCE_LEVEL = 0.95
ROLE_CONDITIONS = {
    "development": {"episodes": 100, "eval_seed": 42001},
    # The final grid places r=1 at index 2; drivers add 1000 per rate index.
    "final": {"episodes": 200, "eval_seed": 75001},
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class MatchedNominalRecords:
    """Validated matched outcomes for one expert-learner condition."""

    task: str
    bank_role: str
    bank_sha256: str
    learner_policy: str
    outcomes: np.ndarray
    expert_successes: int
    learner_successes: int
    both_success_count: int
    expert_only_count: int
    learner_only_count: int
    neither_success_count: int

    @property
    def episode_count(self) -> int:
        return int(self.outcomes.size)

    @property
    def paired_difference(self) -> float:
        return float(self.outcomes.mean())


def _one_value(
    records: Sequence[Mapping[str, Any]], field: str, source_name: str
) -> Any:
    try:
        values = {record[field] for record in records}
    except KeyError as exc:
        raise ValueError(f"{source_name} record is missing field {field!r}") from exc
    if len(values) != 1:
        raise ValueError(f"{source_name} records contain multiple {field!r} values")
    return next(iter(values))


def _records_by_identifier(
    records: Sequence[Mapping[str, Any]], source_name: str
) -> dict[int, Mapping[str, Any]]:
    try:
        indexed = {int(record["initial_condition_id"]): record for record in records}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source_name} records contain an invalid condition identifier") from exc
    if len(indexed) != len(records):
        raise ValueError(f"{source_name} condition identifiers are not unique")
    if set(indexed) != set(range(len(records))):
        raise ValueError(f"{source_name} condition identifiers are not complete")
    return indexed


def _initial_pose(record: Mapping[str, Any], source_name: str) -> Any:
    for field in ("object_initial_pose", "parcel_initial_pose"):
        if field in record:
            return record[field]
    raise ValueError(f"{source_name} record is missing the initial object pose")


def validate_nominal_pair(
    expert_records: Sequence[Mapping[str, Any]],
    learner_records: Sequence[Mapping[str, Any]],
) -> MatchedNominalRecords:
    """Validate two nominal record sets and return their paired outcomes."""
    if not expert_records or not learner_records:
        raise ValueError("expert and learner record files must be nonempty")
    if len(expert_records) != len(learner_records):
        raise ValueError("expert and learner episode counts differ")

    expert_task = _one_value(expert_records, "task", "expert")
    learner_task = _one_value(learner_records, "task", "learner")
    if expert_task != learner_task:
        raise ValueError("expert and learner tasks differ")

    expert_role = _one_value(expert_records, "bank_role", "expert")
    learner_role = _one_value(learner_records, "bank_role", "learner")
    if expert_role != learner_role:
        raise ValueError("expert and learner bank roles differ")
    if expert_role not in ROLE_CONDITIONS:
        raise ValueError("nominal comparison requires development or final records")
    conditions = ROLE_CONDITIONS[expert_role]
    if len(expert_records) != conditions["episodes"]:
        raise ValueError(
            f"{expert_role} comparison requires {conditions['episodes']} episodes"
        )

    expert_bank = _one_value(
        expert_records, "initial_condition_bank_sha256", "expert"
    )
    learner_bank = _one_value(
        learner_records, "initial_condition_bank_sha256", "learner"
    )
    if not _is_sha256(expert_bank) or not _is_sha256(learner_bank):
        raise ValueError("initial-condition bank checksum must be a SHA-256 digest")
    if expert_bank != learner_bank:
        raise ValueError("expert and learner initial-condition banks differ")

    for source_name, records in (
        ("expert", expert_records),
        ("learner", learner_records),
    ):
        if float(_one_value(records, "task_rate", source_name)) != 1.0:
            raise ValueError("nominal comparison requires task rate r=1")
        if int(_one_value(records, "seed", source_name)) != conditions["eval_seed"]:
            raise ValueError(
                f"{expert_role} comparison requires evaluation seed "
                f"{conditions['eval_seed']}"
            )
        if float(_one_value(records, "jitter", source_name)) != 0.01:
            raise ValueError("nominal comparison requires jitter 0.01")
        if bool(_one_value(records, "corrupt", source_name)):
            raise ValueError("nominal comparison rejects observation corruption")
        if float(_one_value(records, "action_noise", source_name)) != 0.0:
            raise ValueError("nominal comparison rejects action noise")
        if _one_value(records, "fixed_phase_rate_override", source_name) is not None:
            raise ValueError("nominal comparison rejects rate-feature interventions")

    if _one_value(expert_records, "policy", "expert") != "expert":
        raise ValueError("the expert record file must contain the expert policy")
    learner_policy = str(_one_value(learner_records, "policy", "learner"))
    if learner_policy == "expert":
        raise ValueError("the learner record file must contain a learned policy")

    expert = _records_by_identifier(expert_records, "expert")
    learner = _records_by_identifier(learner_records, "learner")
    if set(expert) != set(learner):
        raise ValueError("expert and learner condition identifiers differ")

    outcomes = []
    both_success = expert_only = learner_only = neither_success = 0
    for identifier in sorted(expert):
        expert_record = expert[identifier]
        learner_record = learner[identifier]
        if _initial_pose(expert_record, "expert") != _initial_pose(
            learner_record, "learner"
        ):
            raise ValueError(f"initial object pose differs for condition {identifier}")
        expert_value = expert_record.get("task_success")
        learner_value = learner_record.get("task_success")
        if not isinstance(expert_value, bool) or not isinstance(learner_value, bool):
            raise ValueError("task_success must be Boolean in every record")
        if expert_value and learner_value:
            both_success += 1
            outcomes.append(0)
        elif expert_value:
            expert_only += 1
            outcomes.append(-1)
        elif learner_value:
            learner_only += 1
            outcomes.append(1)
        else:
            neither_success += 1
            outcomes.append(0)

    return MatchedNominalRecords(
        task=str(expert_task),
        bank_role=str(expert_role),
        bank_sha256=str(expert_bank),
        learner_policy=learner_policy,
        outcomes=np.asarray(outcomes, dtype=np.int8),
        expert_successes=both_success + expert_only,
        learner_successes=both_success + learner_only,
        both_success_count=both_success,
        expert_only_count=expert_only,
        learner_only_count=learner_only,
        neither_success_count=neither_success,
    )


def validate_margin(margin: float | None) -> float:
    """Return a finite margin in (0, 1], rejecting missing values."""
    if margin is None:
        raise ValueError("a noninferiority margin is required")
    value = float(margin)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("the noninferiority margin must be in (0, 1]")
    return value


def describe_nominal_pair(
    expert_records: Sequence[Mapping[str, Any]],
    learner_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report paired outcomes and a pointwise interval without a decision rule."""
    matched = validate_nominal_pair(expert_records, learner_records)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.choice(
        matched.outcomes,
        size=(BOOTSTRAP_RESAMPLES, matched.episode_count),
        replace=True,
    )
    lower, upper = np.percentile(sampled.mean(axis=1), [2.5, 97.5], method="linear")
    return {
        "schema_version": 2,
        "reporting_rule": "descriptive_paired_difference",
        "task": matched.task,
        "rate": 1.0,
        "bank_role": matched.bank_role,
        "evidence_status": (
            "development evidence; not a final conclusion"
            if matched.bank_role == "development" else "final-bank descriptive evidence"
        ),
        "initial_condition_bank_sha256": matched.bank_sha256,
        "learner_policy": matched.learner_policy,
        "expert_successes": matched.expert_successes,
        "expert_denominator": matched.episode_count,
        "learner_successes": matched.learner_successes,
        "learner_denominator": matched.episode_count,
        "both_success_count": matched.both_success_count,
        "expert_only_count": matched.expert_only_count,
        "learner_only_count": matched.learner_only_count,
        "neither_success_count": matched.neither_success_count,
        "paired_difference": matched.paired_difference,
        "paired_difference_percentage_points": 100.0 * matched.paired_difference,
        "difference_direction": "learner_minus_expert",
        "confidence_interval": [float(lower), float(upper)],
        "confidence_interval_percentage_points": [100.0 * float(lower), 100.0 * float(upper)],
        "confidence_level": CONFIDENCE_LEVEL,
        "interval_method": "two-sided paired percentile bootstrap; linear percentiles",
        "interval_scope": "pointwise; not simultaneous across comparisons",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "degenerate_interval": bool(lower == upper),
        "interpretation_limit": (
            "An interval containing zero does not establish equivalence. Identical "
            "observed paired outcomes yield a degenerate bootstrap interval, not "
            "certainty about the population difference."
        ),
    }


def paired_bootstrap_lower_bound(
    outcomes: Sequence[int] | np.ndarray,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> float:
    """Return the one-sided lower percentile bound for the paired mean."""
    values = np.asarray(outcomes, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("paired outcomes must be a nonempty one-dimensional sequence")
    if not np.isin(values, (-1.0, 0.0, 1.0)).all():
        raise ValueError("paired outcomes must contain only -1, 0, and 1")
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must be between zero and one")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(resamples, values.size), replace=True)
    differences = sampled.mean(axis=1)
    percentile = 100.0 - 100.0 * confidence_level
    return float(np.percentile(differences, percentile, method="linear"))


def passes_noninferiority(lower_bound: float, margin: float) -> bool:
    """Apply the predeclared strict paired noninferiority comparison."""
    return bool(lower_bound > -validate_margin(margin))


def assess_nominal_pair(
    expert_records: Sequence[Mapping[str, Any]],
    learner_records: Sequence[Mapping[str, Any]],
    *,
    margin: float,
) -> dict[str, Any]:
    """Validate records and return the fixed paired-bootstrap assessment."""
    checked_margin = validate_margin(margin)
    matched = validate_nominal_pair(expert_records, learner_records)
    lower_bound = paired_bootstrap_lower_bound(matched.outcomes)
    status = (
        "development evidence; not a final conclusion"
        if matched.bank_role == "development"
        else "final-bank evidence evaluated at the supplied margin"
    )
    return {
        "task": matched.task,
        "rate": 1.0,
        "bank_role": matched.bank_role,
        "evidence_status": status,
        "initial_condition_bank_sha256": matched.bank_sha256,
        "learner_policy": matched.learner_policy,
        "expert_successes": matched.expert_successes,
        "expert_denominator": matched.episode_count,
        "learner_successes": matched.learner_successes,
        "learner_denominator": matched.episode_count,
        "both_success_count": matched.both_success_count,
        "expert_only_count": matched.expert_only_count,
        "learner_only_count": matched.learner_only_count,
        "neither_success_count": matched.neither_success_count,
        "paired_difference": matched.paired_difference,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confidence_level": CONFIDENCE_LEVEL,
        "lower_one_sided_confidence_bound": lower_bound,
        "noninferiority_margin": checked_margin,
        "passes_noninferiority": passes_noninferiority(lower_bound, checked_margin),
    }
