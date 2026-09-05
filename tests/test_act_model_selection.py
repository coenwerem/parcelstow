import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "manipulation"))

from act_model_selection import (  # noqa: E402
    CandidateResult,
    paired_success_counts,
    select_candidate,
    validate_checkpoint_binding,
    validate_development_records,
)
from file_integrity import sha256_file  # noqa: E402


def _records(policy="act", checkpoint="checkpoint.pt", checkpoint_sha256="hash"):
    return [
        {
            "task": "PegInsert-L6-Play-v0",
            "policy": policy,
            "actor_spec": policy,
            "bank_role": "development",
            "seed": 42001,
            "num_envs": 32,
            "task_rate": 1.0,
            "jitter": 0.01,
            "corrupt": False,
            "action_noise": 0.0,
            "fixed_phase_rate_override": None,
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
            "initial_condition_id": index,
            "initial_condition_bank_sha256": "bank",
            "object_initial_pose": {"pos": [float(index), 0.0, 0.0]},
            "task_success": success,
            "acquired": True,
            "inserted": success,
        }
        for index, success in enumerate((True, False, True))
    ]


def _summary(policy="act", checkpoint="checkpoint.pt", checkpoint_sha256="hash"):
    return {
        "task": "PegInsert-L6-Play-v0",
        "policy": policy,
        "actor_spec": policy,
        "bank_role": "development",
        "seed": 42001,
        "num_envs": 32,
        "rate": 1.0,
        "jitter": 0.01,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "episodes": 3,
        "episodes_requested": 3,
        "task_success": {"k": 2, "n": 3},
        "acquired": {"k": 3, "n": 3},
        "inserted": {"k": 2, "n": 3},
    }


def _candidate(epoch, success, stages, validation, bank="bank"):
    return CandidateResult(epoch, validation, success, stages, bank)


def test_checkpoint_selection_applies_predeclared_order():
    candidates = [
        _candidate(100, 8, (10, 9), 0.01),
        _candidate(200, 9, (9, 9), 0.03),
        _candidate(300, 9, (10, 8), 0.04),
        _candidate(400, 9, (10, 8), 0.02),
    ]
    assert select_candidate(candidates).epoch == 400


def test_checkpoint_selection_rejects_different_banks():
    with pytest.raises(ValueError, match="one initial-condition bank"):
        select_candidate([
            _candidate(100, 8, (9,), 0.1, "a"),
            _candidate(200, 9, (9,), 0.1, "b"),
        ])


def test_development_record_validation_checks_summary(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_hash = sha256_file(checkpoint)
    records = _records(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash)
    summary = _summary(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash)
    assert validate_development_records(
        records,
        summary,
        gym_id="PegInsert-L6-Play-v0",
        policy="act",
        stage_keys=("acquired", "inserted"),
        expected_episodes=3,
        expected_seed=42001,
        expected_num_envs=32,
        expected_jitter=0.01,
        checkpoint_path=checkpoint,
    ) == "bank"
    bad_summary = dict(summary)
    bad_summary["inserted"] = {"k": 1}
    with pytest.raises(ValueError, match="inserted"):
        validate_development_records(
            records,
            bad_summary,
            gym_id="PegInsert-L6-Play-v0",
            policy="act",
            stage_keys=("acquired", "inserted"),
            expected_episodes=3,
            expected_seed=42001,
            expected_num_envs=32,
            expected_jitter=0.01,
            checkpoint_path=checkpoint,
        )


def test_checkpoint_binding_rejects_a_summary_hash_mismatch(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_hash = sha256_file(checkpoint)
    records = _records(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash)
    summary = _summary(checkpoint=str(checkpoint), checkpoint_sha256="different")
    with pytest.raises(ValueError, match="summary checkpoint hash"):
        validate_checkpoint_binding(
            records, summary, policy="act", checkpoint_path=checkpoint
        )


def test_checkpoint_binding_rejects_multiple_episode_hashes(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_hash = sha256_file(checkpoint)
    records = _records(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash)
    records[1]["checkpoint_sha256"] = "different"
    summary = _summary(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash)
    with pytest.raises(ValueError, match="multiple checkpoint hashes"):
        validate_checkpoint_binding(
            records, summary, policy="act", checkpoint_path=checkpoint
        )


def test_checkpoint_binding_rejects_a_replaced_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"evaluated checkpoint")
    evaluated_hash = sha256_file(checkpoint)
    records = _records(checkpoint=str(checkpoint), checkpoint_sha256=evaluated_hash)
    summary = _summary(checkpoint=str(checkpoint), checkpoint_sha256=evaluated_hash)
    checkpoint.write_bytes(b"replacement checkpoint")
    with pytest.raises(ValueError, match="contents changed"):
        validate_checkpoint_binding(
            records, summary, policy="act", checkpoint_path=checkpoint
        )


def test_checkpoint_binding_requires_null_expert_fields():
    records = _records("expert", checkpoint=None, checkpoint_sha256=None)
    summary = _summary("expert", checkpoint=None, checkpoint_sha256=None)
    assert validate_checkpoint_binding(
        records, summary, policy="expert", checkpoint_path=None
    ) is None
    for record in records:
        record["checkpoint_sha256"] = "unexpected"
    summary["checkpoint_sha256"] = "unexpected"
    with pytest.raises(ValueError, match="null checkpoint provenance"):
        validate_checkpoint_binding(
            records, summary, policy="expert", checkpoint_path=None
        )


def test_paired_counts_require_exact_initial_poses():
    expert = _records("expert")
    learner = _records()
    learner[1]["task_success"] = True
    counts = paired_success_counts(expert, learner)
    assert counts == {
        "episodes": 3,
        "both_succeed": 2,
        "expert_only_succeeds": 0,
        "learner_only_succeeds": 1,
        "neither_succeeds": 0,
        "learner_minus_expert": pytest.approx(1 / 3),
    }
    learner[0]["object_initial_pose"] = {"pos": [99.0, 0.0, 0.0]}
    with pytest.raises(ValueError, match="initial object pose"):
        paired_success_counts(expert, learner)
