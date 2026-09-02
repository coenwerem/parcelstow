import argparse
import ast
import gzip
import importlib.util
import json
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


def _load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
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
    rate_start = legacy.index("--rates") + 1
    rate_end = legacy.index("--episodes")
    assert tuple(float(value) for value in legacy[rate_start:rate_end]) == (
        0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0
    )
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


def test_registry_rate_grids_equal_released_record_grids():
    for alias, task in TASKS.items():
        record_dir = ROOT / "data" / "records"
        if alias != "parcel":
            record_dir /= alias
        rates = set()
        for actor in ("expert", "act"):
            with gzip.open(record_dir / f"{actor}_episodes.jsonl.gz", "rt", encoding="utf-8") as stream:
                rates.update(float(json.loads(line)["task_rate"]) for line in stream)
        assert task.default_rates == tuple(sorted(rates))


def test_registry_scientific_metadata_matches_canonical_task_sources():
    task_sources = {
        "parcel": "parcel_stow",
        "upright": "upright_place",
        "peg": "peg_insert",
    }
    for alias, source_name in task_sources.items():
        task = TASKS[alias]
        source_dir = (ROOT / "source" / "parcelstow" / "parcelstow" / "tasks"
                      / "manager_based" / source_name)
        geometry = _load_path(f"{alias}_registry_geometry", source_dir / "geometry.py")
        assert task.phases == tuple(tuple(phase) for phase in geometry.PHASES)

        stage_source = (ROOT / "scripts" / "manipulation" / "stow_runtime.py"
                        if alias == "parcel" else source_dir / "mdp" / "monitor.py")
        stage_name = "V1_STAGE_KEYS" if alias == "parcel" else "STAGE_KEYS"
        monitor_tree = ast.parse(stage_source.read_text())
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in monitor_tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == stage_name
        }
        assert task.stage_keys == tuple(assignments[stage_name])


def test_registry_routing_symbols_exist_in_source():
    for task in TASKS.values():
        for symbol_spec in (task.monitor, task.expert):
            module_name, symbol_name = symbol_spec.split(":", 1)
            if module_name.startswith("scripts."):
                source_path = ROOT / (module_name.replace(".", "/") + ".py")
            else:
                source_path = (ROOT / "source" / "parcelstow" / "parcelstow"
                               / (module_name.replace("parcelstow.", "").replace(".", "/") + ".py"))
            tree = ast.parse(source_path.read_text())
            definitions = {
                node.name for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            }
            assert symbol_name in definitions
