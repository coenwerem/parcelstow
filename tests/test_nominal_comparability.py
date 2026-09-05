import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "manipulation"))

from assess_nominal_comparability import main, parse_args  # noqa: E402
from file_integrity import sha256_file  # noqa: E402
from nominal_comparability import (  # noqa: E402
    assess_nominal_pair,
    paired_bootstrap_lower_bound,
    passes_noninferiority,
    validate_margin,
    validate_nominal_pair,
)


def _record_sets(
    *,
    both=90,
    expert_only=4,
    learner_only=3,
    neither=3,
):
    outcomes = (
        [(True, True)] * both
        + [(True, False)] * expert_only
        + [(False, True)] * learner_only
        + [(False, False)] * neither
    )

    def records(policy, successes):
        return [
            {
                "task": "PegInsert-L6-Play-v0",
                "policy": policy,
                "actor_spec": policy,
                "bank_role": "development",
                "task_rate": 1.0,
                "seed": 42001,
                "jitter": 0.01,
                "corrupt": False,
                "action_noise": 0.0,
                "fixed_phase_rate_override": None,
                "initial_condition_bank_sha256": "b" * 64,
                "initial_condition_id": index,
                "object_initial_pose": {
                    "pos": [float(index), 0.0, 0.0],
                    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "task_success": success,
            }
            for index, success in enumerate(successes)
        ]

    return records("expert", [pair[0] for pair in outcomes]), records(
        "act", [pair[1] for pair in outcomes]
    )


def test_nominal_pair_reports_exact_paired_counts():
    expert, learner = _record_sets()
    result = assess_nominal_pair(expert, learner, margin=0.10)
    assert result["expert_successes"] == 94
    assert result["expert_denominator"] == 100
    assert result["learner_successes"] == 93
    assert result["learner_denominator"] == 100
    assert result["both_success_count"] == 90
    assert result["expert_only_count"] == 4
    assert result["learner_only_count"] == 3
    assert result["neither_success_count"] == 3
    assert result["paired_difference"] == pytest.approx(-0.01)
    assert result["evidence_status"] == "development evidence; not a final conclusion"


def test_paired_bootstrap_is_reproducible():
    expert, learner = _record_sets()
    outcomes = validate_nominal_pair(expert, learner).outcomes
    first = paired_bootstrap_lower_bound(outcomes)
    second = paired_bootstrap_lower_bound(outcomes)
    assert first == second
    assert first == pytest.approx(-0.05)


def test_paired_bootstrap_handles_perfect_agreement():
    expert, learner = _record_sets(both=100, expert_only=0, learner_only=0, neither=0)
    result = assess_nominal_pair(expert, learner, margin=0.10)
    assert result["paired_difference"] == 0.0
    assert result["lower_one_sided_confidence_bound"] == 0.0
    assert result["passes_noninferiority"] is True


def test_paired_bootstrap_handles_a_uniformly_better_learner():
    expert, learner = _record_sets(both=0, expert_only=0, learner_only=100, neither=0)
    result = assess_nominal_pair(expert, learner, margin=0.10)
    assert result["paired_difference"] == 1.0
    assert result["lower_one_sided_confidence_bound"] == 1.0
    assert result["passes_noninferiority"] is True


def test_paired_bootstrap_handles_a_uniformly_better_expert():
    expert, learner = _record_sets(both=0, expert_only=100, learner_only=0, neither=0)
    result = assess_nominal_pair(expert, learner, margin=0.10)
    assert result["paired_difference"] == -1.0
    assert result["lower_one_sided_confidence_bound"] == -1.0
    assert result["passes_noninferiority"] is False


def test_nominal_pair_rejects_a_mismatched_bank():
    expert, learner = _record_sets()
    for record in learner:
        record["initial_condition_bank_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="banks differ"):
        validate_nominal_pair(expert, learner)


def test_nominal_pair_rejects_mismatched_condition_identifiers():
    expert, learner = _record_sets()
    learner[-1]["initial_condition_id"] = 100
    with pytest.raises(ValueError, match="condition identifiers"):
        validate_nominal_pair(expert, learner)


def test_nominal_pair_rejects_a_mismatched_initial_pose():
    expert, learner = _record_sets()
    learner[10]["object_initial_pose"]["pos"][0] = 99.0
    with pytest.raises(ValueError, match="initial object pose differs"):
        validate_nominal_pair(expert, learner)


@pytest.mark.parametrize("margin", [None, 0.0, -0.1, 1.1, float("nan")])
def test_margin_is_required_and_bounded(margin):
    with pytest.raises(ValueError, match="margin"):
        validate_margin(margin)


def test_noninferiority_comparison_is_strict_at_the_margin():
    assert passes_noninferiority(-0.10, 0.10) is False
    assert passes_noninferiority(-0.099, 0.10) is True


def test_cli_requires_an_explicit_margin(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--expert-record",
                str(tmp_path / "expert.jsonl"),
                "--learner-record",
                str(tmp_path / "learner.jsonl"),
                "--output",
                str(tmp_path / "result.json"),
            ]
        )


def test_cli_writes_input_file_hashes(tmp_path):
    expert, learner = _record_sets()
    expert_path = tmp_path / "expert.jsonl"
    learner_path = tmp_path / "learner.jsonl"
    output_path = tmp_path / "result.json"
    expert_path.write_text("".join(json.dumps(record) + "\n" for record in expert))
    learner_path.write_text("".join(json.dumps(record) + "\n" for record in learner))
    assert main(
        [
            "--expert-record",
            str(expert_path),
            "--learner-record",
            str(learner_path),
            "--margin",
            "0.10",
            "--output",
            str(output_path),
        ]
    ) == 0
    result = json.loads(output_path.read_text())
    assert result["input_files"]["expert_record_sha256"] == sha256_file(expert_path)
    assert result["input_files"]["learner_record_sha256"] == sha256_file(learner_path)
    assert result["bootstrap_resamples"] == 20_000
    assert result["bootstrap_seed"] == 0
    assert result["confidence_level"] == 0.95
