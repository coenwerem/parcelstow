import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from task_registry import ALIASES, TASKS, get_task, task_output_dir  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_is_complete_and_contract_is_verified():
    assert ALIASES == ("parcel", "upright", "peg")
    assert {task.observation_dim for task in TASKS.values()} == {147}
    assert {task.action_dim for task in TASKS.values()} == {16}
    assert {task.observation_pose_key for task in TASKS.values()} == {"parcel_pose", "object_pose"}
    for task in TASKS.values():
        assert task.driver_path.is_file()
        assert task.stage_keys
        assert task.cycle_time(1.0) > task.cycle_time(2.0)


def test_unknown_alias_lists_valid_aliases():
    try:
        get_task("unknown")
    except ValueError as exc:
        assert "parcel, upright, peg" in str(exc)
    else:
        raise AssertionError("unknown task accepted")


def test_output_paths_are_separate_and_parcel_legacy_is_preserved():
    assert task_output_dir(TASKS["parcel"], None, legacy_default=True) == "outputs/eval"
    assert {task_output_dir(task, None) for task in TASKS.values()} == {
        "outputs/eval/parcel", "outputs/eval/upright", "outputs/eval/peg"}
    assert len({task_output_dir(task, None) for task in TASKS.values()}) == 3


def test_evaluate_command_dispatches_and_preserves_legacy_defaults():
    evaluate = _load_script("evaluate")
    common = dict(actor=["expert"], rates=None, episodes=1, num_envs=2,
                  out_dir=None, custom_ckpt=None, eval_seed=12345)
    legacy = evaluate.build_command(argparse.Namespace(task="parcel", **common), [], task_explicit=False)
    assert "eval_stow_policies.py" in legacy[1]
    assert legacy[legacy.index("--out_dir") + 1] == "outputs/eval"
    for alias in ALIASES:
        command = evaluate.build_command(argparse.Namespace(task=alias, **common), [], task_explicit=True)
        assert TASKS[alias].gym_id in command
        assert command[command.index("--out_dir") + 1] == TASKS[alias].default_output_dir


def test_run_task_command_separates_explicit_task_outputs():
    run_task = _load_script("run_task")
    outputs = set()
    for alias in ALIASES:
        args = argparse.Namespace(task=alias, actor="expert", rate=1.0, episodes=1,
                                  num_envs=2, out_dir=None)
        command = run_task.build_command(args, [], task_explicit=True)
        outputs.add(command[command.index("--out_dir") + 1])
    assert outputs == {"outputs/quickstart/parcel", "outputs/quickstart/upright", "outputs/quickstart/peg"}
