"""Gate A pure checks of the keyed-peg insertion task skeleton: the
phase schedule, the deterministic object path, the pocket containment
predicates, and the derived tolerance. No simulator."""

import math

import numpy as np

PHASE_NAMES = ["PARK", "APPROACH", "PREGRASP_DWELL", "CLOSE", "GRASP_DWELL",
               "LIFT", "REORIENT", "TRANSFER", "INSERT", "INSERT_DWELL",
               "RELEASE", "RETREAT", "SETTLE"]
RATES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]


def test_phase_schedule(peg_geometry, phase_schedule_mod):
    P = peg_geometry
    sched = phase_schedule_mod.PhaseSchedule(P.PHASES)
    assert sched.names == PHASE_NAMES
    assert not sched.rate_scaled[:5].any()
    assert np.isclose(sched.nominal_durations[:5].sum(), 5.7)
    assert sched.rate_scaled[5:-1].all()
    assert not sched.rate_scaled[-1]
    scaled_sum = sched.nominal_durations[sched.rate_scaled].sum()
    for r in RATES:
        assert np.isclose(sched.cycle_time(r), 5.7 + scaled_sum / r + 2.0)


def test_path_continuity_and_endpoints(peg_geometry):
    P = peg_geometry
    for k in range(len(P.PHASES) - 1):
        p_end, r_end = P.object_pose(k, 1.0)
        p_next, r_next = P.object_pose(k + 1, 0.0)
        assert np.allclose(p_end, p_next, atol=1e-12), P.PHASES[k][0]
        assert np.allclose(r_end, r_next, atol=1e-12), P.PHASES[k][0]
    idx = P.PHASE_INDEX
    p, r = P.object_pose(idx["GRASP_DWELL"], 1.0)
    assert np.allclose(p, P.START_POS)
    assert P.tilt_deg(r) == 90.0
    p, r = P.object_pose(idx["REORIENT"], 1.0)
    assert P.tilt_deg(r) < 1e-9
    assert np.allclose(p, np.asarray(P.START_POS) + [0, 0, P.LIFT_DZ])
    # TRANSFER ends above the pocket center, the peg base above the block top.
    p, r = P.object_pose(idx["TRANSFER"], 1.0)
    assert np.allclose(p[:2], P.POCKET_CENTER)
    assert p[2] - P.OBJECT_HALF_HEIGHT > P.BLOCK_TOP
    # INSERT ends seated on the pocket floor.
    p, r = P.object_pose(idx["INSERT"], 1.0)
    assert np.allclose(p, [*P.POCKET_CENTER, P.SEAT_Z])
    assert abs((P.SEAT_Z - P.OBJECT_HALF_HEIGHT) - (P.POCKET_FLOOR_Z + P.RELEASE_DROP)) < 1e-12


def test_containment_predicates(peg_geometry):
    P = peg_geometry
    seat = np.array([*P.POCKET_CENTER, P.SEAT_Z])
    upright = P.rotz(math.radians(P.START_YAW_DEG))
    # Seated peg: full depth, inside the cross-section.
    assert P.base_depth(seat) >= 0.040
    assert P.inside_pocket(seat, upright)
    # Above the block: no depth.
    assert P.base_depth(seat + [0, 0, 0.10]) < 0.0
    # Offset beyond the clearance leaves the cross-section.
    off = P.rotz(math.radians(P.START_YAW_DEG)) @ np.array([P.CLEARANCE + 0.055 / 2, 0.0, 0.0])
    assert not P.inside_pocket(seat + off, upright)
    # A yaw at the derived tolerance fails, a small yaw passes.
    yaw_tol = 2.0 * P.CLEARANCE / P.OBJECT_EXTENTS[0]
    twisted = upright @ P.rotz(1.2 * yaw_tol)
    assert not P.inside_pocket(seat, twisted)
    assert P.inside_pocket(seat, upright @ P.rotz(0.3 * yaw_tol))
    assert math.degrees(yaw_tol) > P.FINAL_TILT_TOL_DEG


def test_pocket_slabs(peg_geometry):
    P = peg_geometry
    slabs = P.pocket_slabs()
    assert set(slabs) == {"floor", "wall_a", "wall_b", "wall_c", "wall_d",
                          "lead_a", "lead_b", "lead_c", "lead_d"}
    for s in slabs.values():
        assert len(s["size"]) == 3 and len(s["center"]) == 3 and len(s["quat_wxyz"]) == 4
    # The floor top sits at the pocket floor height.
    f = slabs["floor"]
    assert abs((f["center"][2] + f["size"][2] / 2) - P.POCKET_FLOOR_Z) < 1e-9


def test_stage_and_failure_vocabulary(peg_geometry):
    P = peg_geometry
    assert P.STAGES == ["acquired", "lifted_clear", "reoriented_upright",
                        "aligned", "inserted", "released", "settled"]
    assert P.FAILURE_REASONS == ["acquisition_failure", "dropped_during_transport",
                                 "alignment_failure", "insertion_jam",
                                 "timeout", "other"]
