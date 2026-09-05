import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "manipulation"))

from evaluation_protocol import (  # noqa: E402
    evaluation_output_paths,
    reject_existing_protected_outputs,
    resolve_checkpoint_bindings,
    validate_evaluation_role,
)
from file_integrity import sha256_file  # noqa: E402


def test_checkpoint_bindings_hash_learners_and_leave_expert_null(tmp_path):
    checkpoint = tmp_path / "act.pt"
    checkpoint.write_bytes(b"checkpoint")
    bindings = resolve_checkpoint_bindings(
        ["expert", "act"],
        {"act": str(checkpoint)},
        custom_checkpoint=None,
    )
    assert bindings["expert"].checkpoint is None
    assert bindings["expert"].checkpoint_sha256 is None
    assert bindings["act"].checkpoint == str(checkpoint)
    assert bindings["act"].checkpoint_sha256 == sha256_file(checkpoint)


def test_checkpoint_bindings_hash_a_supplied_custom_checkpoint(tmp_path):
    checkpoint = tmp_path / "custom.pt"
    checkpoint.write_bytes(b"custom checkpoint")
    binding = resolve_checkpoint_bindings(
        ["examples.policy:Policy"], {}, str(checkpoint)
    )["examples.policy:Policy"]
    assert binding.checkpoint == str(checkpoint)
    assert binding.checkpoint_sha256 == sha256_file(checkpoint)


def test_checkpoint_bindings_allow_a_checkpoint_free_custom_policy():
    binding = resolve_checkpoint_bindings(
        ["examples.policy:Policy"], {}, custom_checkpoint=None
    )["examples.policy:Policy"]
    assert binding.checkpoint is None
    assert binding.checkpoint_sha256 is None


def test_checkpoint_bindings_reject_a_missing_builtin_checkpoint():
    with pytest.raises(ValueError, match="requires a checkpoint"):
        resolve_checkpoint_bindings(["act"], {"act": None}, custom_checkpoint=None)


def test_checkpoint_bindings_reject_a_missing_file(tmp_path):
    missing = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_checkpoint_bindings(
            ["act"], {"act": str(missing)}, custom_checkpoint=None
        )


def _role_arguments(**overrides):
    arguments = {
        "bank_role": "development",
        "rates": [1.0],
        "registered_rates": [0.5, 1.0, 1.5],
        "episodes": 100,
        "eval_seed": 42001,
        "jitter": 0.01,
        "corrupt": False,
        "action_noise": 0.0,
        "fixed_phase_rate_override": None,
        "actors": ["act"],
    }
    arguments.update(overrides)
    return arguments


def test_development_role_accepts_the_frozen_conditions():
    validate_evaluation_role(**_role_arguments())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rates", [1.25], "requires rates"),
        ("eval_seed", 42002, "evaluation seed"),
        ("episodes", 99, "100 episodes"),
        ("jitter", 0.02, "jitter 0.01"),
        ("corrupt", True, "corruption to be disabled"),
        ("action_noise", 0.1, "zero action noise"),
        ("fixed_phase_rate_override", 1.0, "requires bank_role='diagnostic'"),
    ],
)
def test_development_role_rejects_changed_conditions(field, value, message):
    with pytest.raises(ValueError, match=message):
        validate_evaluation_role(**_role_arguments(**{field: value}))


def test_final_role_accepts_the_complete_registered_grid():
    validate_evaluation_role(
        **_role_arguments(
            bank_role="final",
            rates=[0.5, 1.0, 1.5],
            episodes=200,
            eval_seed=73001,
        )
    )


@pytest.mark.parametrize("rates", [[0.5, 1.0], [1.0, 0.5, 1.5]])
def test_final_role_rejects_incomplete_or_reordered_grids(rates):
    with pytest.raises(ValueError, match="requires rates"):
        validate_evaluation_role(
            **_role_arguments(
                bank_role="final",
                rates=rates,
                episodes=200,
                eval_seed=73001,
            )
        )


def test_final_role_rejects_a_rate_intervention():
    with pytest.raises(ValueError, match="requires bank_role='diagnostic'"):
        validate_evaluation_role(
            **_role_arguments(
                bank_role="final",
                rates=[0.5, 1.0, 1.5],
                episodes=200,
                eval_seed=73001,
                fixed_phase_rate_override=1.0,
            )
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("corrupt", True, "corruption to be disabled"),
        ("action_noise", 0.1, "zero action noise"),
    ],
)
def test_final_role_rejects_corruption_and_action_noise(field, value, message):
    with pytest.raises(ValueError, match=message):
        validate_evaluation_role(
            **_role_arguments(
                bank_role="final",
                rates=[0.5, 1.0, 1.5],
                episodes=200,
                eval_seed=73001,
                **{field: value},
            )
        )


def test_protected_output_paths_reject_an_existing_episode_file(tmp_path):
    paths = evaluation_output_paths(tmp_path, ["expert", "act"], "_protocol-1")
    paths[0].write_text("existing\n")
    with pytest.raises(FileExistsError, match="expert_protocol-1.jsonl"):
        reject_existing_protected_outputs("final", paths)


def test_protected_output_paths_reject_an_existing_summary_file(tmp_path):
    paths = evaluation_output_paths(tmp_path, ["act"], "_epoch-100")
    paths[-1].write_text("existing\n")
    with pytest.raises(FileExistsError, match="summary_epoch-100.jsonl"):
        reject_existing_protected_outputs("development", paths)


def test_unclassified_evaluation_remains_flexible(tmp_path):
    validate_evaluation_role(
        **_role_arguments(
            bank_role="unclassified",
            rates=[0.7, 3.1],
            episodes=7,
            eval_seed=9,
            jitter=0.03,
            corrupt=True,
            action_noise=0.4,
        )
    )
    existing = tmp_path / "records.jsonl"
    existing.write_text("existing\n")
    reject_existing_protected_outputs("unclassified", [existing])


def test_diagnostic_role_accepts_a_learned_policy_intervention():
    validate_evaluation_role(
        **_role_arguments(
            bank_role="diagnostic",
            rates=[1.5],
            episodes=40,
            eval_seed=52001,
            fixed_phase_rate_override=1.0,
        )
    )


def test_diagnostic_intervention_rejects_an_expert_actor():
    with pytest.raises(ValueError, match="every actor to be a learned policy"):
        validate_evaluation_role(
            **_role_arguments(
                bank_role="diagnostic",
                rates=[1.5],
                episodes=40,
                eval_seed=52001,
                fixed_phase_rate_override=1.0,
                actors=["act", "expert"],
            )
        )
