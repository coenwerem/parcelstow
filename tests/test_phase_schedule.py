"""The task clock reads its phase schedule from a task-level
PhaseSchedule object instead of importing the ParcelStow geometry
module. The ParcelStow binding must reproduce the frozen schedule of
geometry.PHASES exactly."""

import os

import numpy as np
from conftest import REPO

TASK_CLOCK_PY = os.path.join(REPO, "source", "parcelstow", "parcelstow", "tasks", "manager_based",
                             "parcel_stow", "mdp", "task_clock.py")
MDP_INIT_PY = os.path.join(REPO, "source", "parcelstow", "parcelstow", "tasks", "manager_based",
                           "parcel_stow", "mdp", "__init__.py")

RATES = [0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0]


def test_schedule_matches_frozen_geometry(phase_schedule_mod, geometry):
    sched = phase_schedule_mod.PhaseSchedule(geometry.PHASES)
    assert sched.names == geometry.PHASE_NAMES
    assert sched.index == geometry.PHASE_INDEX
    assert sched.n_phases == geometry.N_PHASES
    assert np.array_equal(sched.nominal_durations, geometry.NOMINAL_DURATIONS)
    assert np.array_equal(sched.rate_scaled, geometry.RATE_SCALED)
    for r in RATES:
        assert np.array_equal(sched.phase_durations(r), geometry.phase_durations(r))
        assert sched.cycle_time(r) == geometry.cycle_time(r)


def test_task_clock_no_longer_imports_geometry():
    with open(TASK_CLOCK_PY) as fh:
        src = fh.read()
    assert "import geometry" not in src
    assert "SCHEDULE" in src
    with open(MDP_INIT_PY) as fh:
        init_src = fh.read()
    # The ParcelStow package binds its schedule when the mdp package loads.
    assert "task_clock.SCHEDULE" in init_src and "PHASES" in init_src
