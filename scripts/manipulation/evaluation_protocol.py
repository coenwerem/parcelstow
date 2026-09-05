"""Pure validation helpers for task-policy evaluation commands."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

try:
    from .file_integrity import sha256_file
except ImportError:  # Loaded directly by the pure tests and task drivers.
    from file_integrity import sha256_file

BUILTIN_LEARNERS = frozenset({"act", "dp", "dagger"})
BANK_ROLES = frozenset({"unclassified", "diagnostic", "development", "final"})
PROTECTED_ROLES = frozenset({"development", "final"})


@dataclass(frozen=True)
class CheckpointBinding:
    """Checkpoint path and execution-time digest recorded for one actor."""

    checkpoint: str | None
    checkpoint_sha256: str | None


def resolve_checkpoint_bindings(
    actors: Sequence[str],
    builtin_checkpoints: Mapping[str, str | None],
    custom_checkpoint: str | None,
) -> dict[str, CheckpointBinding]:
    """Resolve and hash every actor checkpoint before simulator startup."""
    bindings = {}
    for actor in actors:
        checkpoint = None
        if actor != "expert":
            checkpoint = builtin_checkpoints.get(actor, custom_checkpoint)
        if checkpoint is None:
            if actor in BUILTIN_LEARNERS:
                raise ValueError(f"actor {actor!r} requires a checkpoint")
            bindings[actor] = CheckpointBinding(None, None)
            continue
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint file does not exist: {path}")
        bindings[actor] = CheckpointBinding(str(checkpoint), sha256_file(path))
    return bindings


def is_learned_actor(actor: str) -> bool:
    """Return whether an actor specification denotes a supported learned policy."""
    return actor in BUILTIN_LEARNERS or ":" in actor


def validate_evaluation_role(
    *,
    bank_role: str,
    rates: Sequence[float],
    registered_rates: Sequence[float],
    episodes: int,
    eval_seed: int,
    jitter: float,
    corrupt: bool,
    action_noise: float,
    fixed_phase_rate_override: float | None,
    actors: Sequence[str],
) -> None:
    """Reject commands that violate a protected evaluation role."""
    if bank_role not in BANK_ROLES:
        raise ValueError(f"unknown evaluation bank role: {bank_role!r}")
    if fixed_phase_rate_override is not None and bank_role != "diagnostic":
        raise ValueError(
            "a fixed-phase rate override requires bank_role='diagnostic'"
        )
    if bank_role == "diagnostic":
        if fixed_phase_rate_override is not None and not all(
            is_learned_actor(actor) for actor in actors
        ):
            raise ValueError(
                "a fixed-phase rate override requires every actor to be a learned policy"
            )
        return
    if bank_role == "unclassified":
        return

    expected_rates = (1.0,) if bank_role == "development" else tuple(registered_rates)
    expected_episodes = 100 if bank_role == "development" else 200
    expected_seed = 42001 if bank_role == "development" else 73001
    supplied_rates = tuple(float(rate) for rate in rates)
    if supplied_rates != expected_rates:
        raise ValueError(
            f"{bank_role} evaluation requires rates {list(expected_rates)}, "
            f"received {list(supplied_rates)}"
        )
    if episodes != expected_episodes:
        raise ValueError(
            f"{bank_role} evaluation requires {expected_episodes} episodes"
        )
    if eval_seed != expected_seed:
        raise ValueError(
            f"{bank_role} evaluation requires evaluation seed {expected_seed}"
        )
    if not math.isclose(jitter, 0.01, rel_tol=0.0, abs_tol=0.0):
        raise ValueError(f"{bank_role} evaluation requires jitter 0.01")
    if corrupt:
        raise ValueError(
            f"{bank_role} evaluation requires observation corruption to be disabled"
        )
    if not math.isclose(action_noise, 0.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError(f"{bank_role} evaluation requires zero action noise")


def actor_record_filename(actor: str, tag: str) -> str:
    """Return the episode-record filename used by the evaluation drivers."""
    safe_actor = actor.replace(":", "_").replace(".", "_")
    return f"{safe_actor}{tag}.jsonl"


def evaluation_output_paths(
    out_dir: str | Path,
    actors: Sequence[str],
    tag: str,
) -> tuple[Path, ...]:
    """Return every episode and summary path written by one command."""
    root = Path(out_dir)
    episode_paths = tuple(root / actor_record_filename(actor, tag) for actor in actors)
    return (*episode_paths, root / f"summary{tag}.jsonl")


def reject_existing_protected_outputs(
    bank_role: str,
    paths: Sequence[str | Path],
) -> None:
    """Prevent development and final evaluations from appending to evidence."""
    if bank_role not in PROTECTED_ROLES:
        return
    existing = [Path(path) for path in paths if Path(path).exists()]
    if existing:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"{bank_role} evaluation refuses existing output target(s): {rendered}"
        )
