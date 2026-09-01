"""Pytest configuration for the ParcelStow checks. Pure tests load the
geometry and expert modules by file path (no simulator), physics tests are
marked isaac and skipped unless --isaac is given."""

import importlib.util
import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GEOMETRY_PY = os.path.join(REPO, "source", "parcelstow", "parcelstow", "tasks", "manager_based",
                           "parcel_stow", "geometry.py")
EXPERT_PY = os.path.join(REPO, "scripts", "manipulation", "parcel_stow_expert.py")
PHASE_SCHEDULE_PY = os.path.join(REPO, "source", "parcelstow", "parcelstow", "phase_schedule.py")
UPRIGHT_GEOMETRY_PY = os.path.join(REPO, "source", "parcelstow", "parcelstow", "tasks", "manager_based",
                                   "upright_place", "geometry.py")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def geometry():
    return load_module("stow_geometry_under_test", GEOMETRY_PY)


@pytest.fixture(scope="session")
def expert_mod():
    return load_module("stow_expert_under_test", EXPERT_PY)


@pytest.fixture(scope="session")
def phase_schedule_mod():
    return load_module("phase_schedule_under_test", PHASE_SCHEDULE_PY)


@pytest.fixture(scope="session")
def upright_geometry():
    return load_module("upright_geometry_under_test", UPRIGHT_GEOMETRY_PY)


def pytest_addoption(parser):
    parser.addoption("--isaac", action="store_true", default=False, help="run the simulator tests")
    parser.addoption("--isaac-upright", action="store_true", default=False,
                     help="run the upright placement simulator tests (separate process from --isaac, "
                          "one Isaac environment per process)")


def pytest_configure(config):
    config.addinivalue_line("markers", "isaac: needs the Isaac simulator")
    config.addinivalue_line("markers", "isaac_upright: needs the Isaac simulator with the upright scene")


def pytest_collection_modifyitems(config, items):
    skip_p = pytest.mark.skip(reason="needs --isaac")
    skip_u = pytest.mark.skip(reason="needs --isaac-upright")
    for item in items:
        if "isaac" in item.keywords and not config.getoption("--isaac"):
            item.add_marker(skip_p)
        if "isaac_upright" in item.keywords and not config.getoption("--isaac-upright"):
            item.add_marker(skip_u)


@pytest.fixture(scope="session")
def isaac_scene(request):
    """One Isaac app and one four-environment ParcelStow scene per pytest
    process, shared by every simulator module (a second SimulationApp in
    the same process never returns, and a second environment after
    env.close() hangs in the CUDA device switch of the simulation context).
    Modules reset the environment at the start of every test."""
    if not request.config.getoption("--isaac"):
        pytest.skip("needs --isaac")
    import argparse

    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    a = parser.parse_args([])
    a.headless = True
    app = AppLauncher(a).app
    sys.path.insert(0, os.path.join(REPO, "scripts", "manipulation"))
    sys.path.insert(0, REPO)
    import gymnasium as gym
    import parcelstow.tasks  # noqa: F401
    import stow_runtime as rt
    import torch
    from parcelstow.tasks.manager_based.parcel_stow import geometry as G
    from parcelstow.tasks.manager_based.parcel_stow.mdp import metrics, task_clock
    from parcelstow.tasks.manager_based.parcel_stow.parcel_stow_env_cfg import ParcelStowEnvCfg_PLAY
    cfg = ParcelStowEnvCfg_PLAY()
    cfg.scene.num_envs = 4
    cfg.observations.policy.enable_corruption = False
    env = gym.make("ParcelStow-L6-Distill-Play-v0", cfg=cfg)
    base = env.unwrapped
    ns = {"app": app, "env": env, "base": base, "torch": torch, "G": G, "task_clock": task_clock,
          "metrics": metrics, "rt": rt, "geom": G.load_geometry()}
    yield ns
    env.close()


@pytest.fixture(scope="session")
def upright_scene(request):
    """One Isaac app and one four-environment UprightPlace scene per pytest
    process. Exclusive with isaac_scene (one environment per process), so
    the upright simulator tests run in their own process via
    --isaac-upright."""
    if not request.config.getoption("--isaac-upright"):
        pytest.skip("needs --isaac-upright")
    assert not request.config.getoption("--isaac"), \
        "--isaac and --isaac-upright are exclusive, one Isaac environment per process"
    import argparse

    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    a = parser.parse_args([])
    a.headless = True
    app = AppLauncher(a).app
    sys.path.insert(0, os.path.join(REPO, "scripts", "manipulation"))
    sys.path.insert(0, REPO)
    import gymnasium as gym
    import parcelstow.tasks  # noqa: F401
    import torch
    from parcelstow.tasks.manager_based.parcel_stow.mdp import task_clock
    from parcelstow.tasks.manager_based.upright_place import geometry as U
    from parcelstow.tasks.manager_based.upright_place.mdp import monitor as umon
    from parcelstow.tasks.manager_based.upright_place.upright_place_env_cfg import UprightPlaceEnvCfg_PLAY
    cfg = UprightPlaceEnvCfg_PLAY()
    cfg.scene.num_envs = 4
    cfg.observations.policy.enable_corruption = False
    env = gym.make("UprightPlace-L6-Play-v0", cfg=cfg)
    base = env.unwrapped
    ns = {"app": app, "env": env, "base": base, "torch": torch, "U": U, "task_clock": task_clock,
          "umon": umon}
    yield ns
    env.close()
