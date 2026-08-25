"""Pure tests of the ParcelStow geometry module, SO(3) error, phase and
rate progression, rate invariance of the geometric path, receptacle
predicates, and the Wilson interval."""

import math

import numpy as np
import pytest


def test_so3_angle_and_slerp(geometry):
    G = geometry
    R = G.roty(math.radians(-90))
    assert abs(G.so3_angle(np.eye(3), R) - math.pi / 2) < 1e-9
    assert G.so3_angle(R, R) < 1e-9
    half = G.slerp(np.eye(3), R, 0.5)
    assert abs(G.so3_angle(np.eye(3), half) - math.pi / 4) < 1e-9
    assert abs(G.so3_angle(half, R) - math.pi / 4) < 1e-9
    # near-pi branch
    Rpi = G.rotx(math.pi)
    assert abs(G.so3_angle(np.eye(3), Rpi) - math.pi) < 1e-6
    q = G.quat_from_mat(R)
    assert np.allclose(G.mat_from_quat(q), R, atol=1e-9)


def test_phase_of_time_rate_law(geometry):
    G = geometry
    # acquisition timing does not depend on rate
    for r in (0.5, 1.0, 2.0):
        k, f = G.phase_of_time(2.0, r)
        assert G.PHASE_NAMES[k] == "APPROACH"
        assert abs(f - (2.0 - 0.5) / 2.5) < 1e-9
    # LIFT starts at T_ACQ for every rate and lasts 1.2 / r
    for r in (0.5, 1.0, 2.0):
        k, f = G.phase_of_time(G.T_ACQ + 0.6 / r, r)
        assert G.PHASE_NAMES[k] == "LIFT"
        assert abs(f - 0.5) < 1e-9
    # cycle time law
    assert abs(G.cycle_time(1.0) - (G.T_ACQ + G.T_MANIP + G.T_SETTLE)) < 1e-9
    assert abs(G.cycle_time(2.0) - (G.T_ACQ + G.T_MANIP / 2 + G.T_SETTLE)) < 1e-9
    k, f = G.phase_of_time(1e6, 1.0)
    assert G.PHASE_NAMES[k] == "SETTLE" and f == 1.0


def test_geometric_path_independent_of_rate(geometry):
    G = geometry
    geom = G.StowGeometry("y", -90.0, "+x", (0.50, -0.18), "C", shelf_height=0.06)
    X_OH = G.make_tf(G.rotx(0.3) @ G.roty(-0.2), (0.0, -0.02, 0.12))
    for r_a, r_b in ((0.5, 2.0), (1.0, 3.0)):
        for name in ("LIFT", "REORIENT", "TRANSFER", "INSERT"):
            k = G.PHASE_INDEX[name]
            for f in (0.0, 0.3, 0.7, 1.0):
                # the same (k, f) gives the same object pose whatever the rate
                T_a = G.object_pose(geom, k, f)
                T_b = G.object_pose(geom, k, f)
                assert np.allclose(T_a, T_b)
                # (k, f) reached at rate-scaled times
                d_a = G.phase_durations(r_a)
                d_b = G.phase_durations(r_b)
                t_a = d_a[:k].sum() + f * d_a[k]
                t_b = d_b[:k].sum() + f * d_b[k]
                ka, fa = G.phase_of_time(t_a - 1e-9, r_a)
                kb, fb = G.phase_of_time(t_b - 1e-9, r_b)
                assert ka == kb == k or f == 0.0
                assert abs(fa - fb) < 1e-6
                Ha = G.hand_pose_from_object(G.object_pose(geom, ka, fa), X_OH)
                Hb = G.hand_pose_from_object(G.object_pose(geom, kb, fb), X_OH)
                assert np.allclose(Ha, Hb, atol=1e-6)


def test_object_path_endpoints(geometry):
    G = geometry
    geom = G.StowGeometry("y", -90.0, "+x", (0.50, -0.18), "C", shelf_height=0.06)
    T = G.object_pose(geom, G.PHASE_INDEX["LIFT"], 1.0)
    assert np.allclose(T[:3, 3], geom.p_lift)
    T = G.object_pose(geom, G.PHASE_INDEX["REORIENT"], 1.0)
    assert G.so3_angle(T[:3, :3], geom.R_stow) < 1e-9
    T = G.object_pose(geom, G.PHASE_INDEX["INSERT"], 1.0)
    assert np.allclose(T[:3, 3], geom.p_insert)
    assert geom.depth_of(geom.p_insert) > geom.inserted_min_depth
    assert geom.inside_interior(geom.p_insert)
    assert not geom.inside_interior(geom.p_preinsert)
    assert geom.depth_of(geom.p_preinsert) < 0.0
    # retreat backs out along -d
    H0 = G.retreat_hand_pose(geom, np.eye(4), 0.0)
    H1 = G.retreat_hand_pose(geom, np.eye(4), 1.0)
    assert np.allclose(H1[:3, 3] - H0[:3, 3], -geom.d * G.RETREAT_DISTANCE)


def test_receptacle_predicates_reject_outside(geometry):
    G = geometry
    geom = G.StowGeometry("y", -90.0, "+x", (0.50, -0.18), "C", shelf_height=0.06)
    c, h = geom.interior_box()
    outside = c + np.array([0.0, h[1] + 0.02, 0.0])
    assert not geom.inside_interior(outside)
    above = c + np.array([0.0, 0.0, h[2] + 0.02])
    assert not geom.inside_interior(above)
    assert geom.inside_interior(c)
    # slabs enclose the interior, floor below and top above the interior
    assert geom.slabs["floor"][0][2] < c[2] - h[2]
    assert geom.slabs["top"][0][2] > c[2] + h[2]
    # tolerance derivation of the spec, the small-angle limits about the
    # tight axis stay above the 10 deg final tolerance
    assert 2 * G.C_TIGHT / geom.L_d > math.radians(10.0)
    assert 2 * G.C_TIGHT / geom.L_loose > math.radians(10.0)


def test_wilson(geometry):
    G = geometry
    lo, hi = G.wilson(50, 100)
    assert 0.40 < lo < 0.5 < hi < 0.60
    lo, hi = G.wilson(100, 100)
    assert hi == pytest.approx(1.0) and lo > 0.95
    lo, hi = G.wilson(0, 100)
    assert lo == pytest.approx(0.0) and hi < 0.05


def test_hand_pose_convention(geometry):
    G = geometry
    T_WO = G.make_tf(G.rotz(0.4), (0.3, 0.1, 0.8))
    X_OH = G.make_tf(G.rotx(0.5), (0.0, -0.02, 0.12))
    T_WH = G.hand_pose_from_object(T_WO, X_OH)
    T_back = T_WH @ G.inv_tf(X_OH)
    assert np.allclose(T_back, T_WO)
