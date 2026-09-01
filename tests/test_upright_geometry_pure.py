"""Gate A pure checks of the upright placement task skeleton: the
phase schedule, the deterministic object path, and the success
predicates. No simulator."""

import math

import numpy as np

PHASE_NAMES = ["PARK", "APPROACH", "PREGRASP_DWELL", "CLOSE", "GRASP_DWELL",
               "LIFT", "REORIENT", "TRANSFER", "LOWER", "PLACE_DWELL",
               "RELEASE", "RETREAT", "SETTLE"]
RATES = [0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0]


def test_phase_schedule(upright_geometry, phase_schedule_mod):
    U = upright_geometry
    sched = phase_schedule_mod.PhaseSchedule(U.PHASES)
    assert sched.names == PHASE_NAMES
    # The acquisition phases keep the v1 timings and stay fixed; the
    # settle window is extended to 1.0 s so tipping resolves in-episode.
    assert not sched.rate_scaled[:5].any()
    assert np.isclose(sched.nominal_durations[:5].sum(), 5.7)
    assert sched.rate_scaled[5:-1].all()
    assert not sched.rate_scaled[-1]
    assert sched.nominal_durations[-1] == 1.0
    scaled_sum = sched.nominal_durations[sched.rate_scaled].sum()
    for r in RATES:
        assert np.isclose(sched.cycle_time(r), 5.7 + scaled_sum / r + 1.0)


def test_path_continuity_and_determinism(upright_geometry):
    U = upright_geometry
    n = len(U.PHASES)
    for k in range(n - 1):
        p_end, r_end = U.object_pose(k, 1.0)
        p_next, r_next = U.object_pose(k + 1, 0.0)
        assert np.allclose(p_end, p_next, atol=1e-12), U.PHASES[k][0]
        assert np.allclose(r_end, r_next, atol=1e-12), U.PHASES[k][0]
    p_a, r_a = U.object_pose(6, 0.37)
    p_b, r_b = U.object_pose(6, 0.37)
    assert np.array_equal(p_a, p_b) and np.array_equal(r_a, r_b)


def test_path_endpoints(upright_geometry):
    U = upright_geometry
    idx = {name: i for i, (name, _, _) in enumerate(U.PHASES)}
    # Acquisition holds the lying start pose.
    p, r = U.object_pose(idx["GRASP_DWELL"], 1.0)
    assert np.allclose(p, U.START_POS)
    assert U.tilt_deg(r) == 90.0
    # LIFT raises the lying object straight up.
    p, r = U.object_pose(idx["LIFT"], 1.0)
    assert np.allclose(p, np.asarray(U.START_POS) + [0, 0, U.LIFT_DZ])
    assert U.tilt_deg(r) == 90.0
    # REORIENT ends upright at the lift point.
    p, r = U.object_pose(idx["REORIENT"], 1.0)
    assert np.allclose(p, np.asarray(U.START_POS) + [0, 0, U.LIFT_DZ])
    assert U.tilt_deg(r) < 1e-9
    # TRANSFER ends above the target center.
    p, r = U.object_pose(idx["TRANSFER"], 1.0)
    assert np.allclose(p[:2], U.TARGET_CENTER)
    # LOWER ends at the place pose, and the pose holds through SETTLE.
    p, r = U.object_pose(idx["LOWER"], 1.0)
    assert np.allclose(p, [*U.TARGET_CENTER, U.PLACE_Z])
    assert U.tilt_deg(r) < 1e-9
    p_s, r_s = U.object_pose(idx["SETTLE"], 1.0)
    assert np.allclose(p_s, p) and np.allclose(r_s, r)
    # The upright base center sits at the target center.
    assert np.allclose(U.base_center(p, r)[:2], U.TARGET_CENTER)


def test_success_predicates(upright_geometry):
    U = upright_geometry
    place = np.array([*U.TARGET_CENTER, U.PLACE_Z])
    upright = U.rotz(math.radians(30.0))
    assert U.tilt_deg(upright) < 1e-9
    assert U.inside_target(place, upright)
    # Planar offsets of the base center against the 30 mm target radius.
    assert U.inside_target(place + [0.029, 0, 0], upright)
    assert not U.inside_target(place + [0.031, 0, 0], upright)
    # Tilt tolerance, 4 degrees passes and 6 degrees fails.
    tilted = lambda deg: U.roty(math.radians(deg)) @ upright  # noqa: E731
    assert U.tilt_deg(tilted(4.0)) <= U.FINAL_TILT_TOL_DEG
    assert U.tilt_deg(tilted(6.0)) > U.FINAL_TILT_TOL_DEG
    # The tolerance is stricter than the tipping angle of the resting
    # object, atan(half width / half height).
    tip_deg = math.degrees(math.atan2(U.OBJECT_EXTENTS[0] / 2, U.OBJECT_EXTENTS[2] / 2))
    assert tip_deg > U.FINAL_TILT_TOL_DEG


def test_stage_and_failure_vocabulary(upright_geometry):
    U = upright_geometry
    assert U.STAGES == ["acquired", "lifted_clear", "reoriented_upright",
                        "placed", "released", "settled"]
    assert U.FAILURE_REASONS == ["acquisition_failure", "dropped_during_transport",
                                 "placement_miss", "tipped_after_release",
                                 "timeout", "other"]
