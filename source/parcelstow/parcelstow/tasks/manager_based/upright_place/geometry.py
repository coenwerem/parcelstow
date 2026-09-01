"""Upright placement task geometry (skeleton, provisional values).

A tall rigid cuboid starts lying on its side at a known pose; the task
stands it upright on a marked circular target region on the table. The
terminal predicate is quasi-static stability, final tilt from vertical
at most FINAL_TILT_TOL_DEG with the base center inside the target
region, rather than the geometric containment of ParcelStow
(docs/EXTENSION_PLAN.md).

The numeric constants below are nominal engineering choices pending
the kinematic probe, following the v1 freeze protocol of
docs/TASK_SPEC.md: probes and expert-only calibration fix the values,
and no learner outcome may inform them. The module is self-contained
(numpy only) so the pure tests load it by file path, the convention of
the ParcelStow geometry module.

The tilt tolerance derivation mirrors the v1 derived tolerances: the
resting cuboid tips when the center of mass leaves the base, at
atan(half width / half height) = atan(20 / 70) = 15.9 degrees, so the
5 degree tolerance is stricter than the tipping angle and was fixed
before any expert or learner ran.
"""

import math

import numpy as np

# ----------------------------------------------------------------------------
# object, table, and target (provisional pending the kinematic probe)
# ----------------------------------------------------------------------------
OBJECT_EXTENTS = (0.040, 0.040, 0.140)  # x, y, z in the object frame
OBJECT_HALF_HEIGHT = OBJECT_EXTENTS[2] / 2
OBJECT_MASS = 0.150
OBJECT_FRICTION = 0.5  # the v1 physics material
TABLE_TOP = 0.70  # the v1 table
START_YAW_DEG = 45.0
# Lying on a 40 x 140 face, center 1 mm above rest height, the v1 start
# convention.
START_POS = (0.35, 0.0, TABLE_TOP + OBJECT_EXTENTS[0] / 2 + 0.001)
LIFT_DZ = 0.12  # the reorientation clears the table by half height plus margin
TARGET_CENTER = (0.49, 0.14)
TARGET_RADIUS = 0.030
PLACE_DROP = 0.005  # release height of the base above the table
PLACE_Z = TABLE_TOP + OBJECT_HALF_HEIGHT + PLACE_DROP

# ----------------------------------------------------------------------------
# success thresholds (final tilt frozen by derivation, speeds from v1)
# ----------------------------------------------------------------------------
FINAL_TILT_TOL_DEG = 5.0
SETTLE_LIN = 0.02  # m/s, the v1 settling threshold
SETTLE_ANG = 0.2  # rad/s

# ----------------------------------------------------------------------------
# phase schedule, (name, nominal seconds, rate scaled)
# ----------------------------------------------------------------------------
# Acquisition keeps the v1 timings and stays fixed; the settle window is
# extended to 1.0 s so tipping resolves inside the episode.
PHASES = [
    ("PARK", 0.5, False),
    ("APPROACH", 2.5, False),
    ("PREGRASP_DWELL", 0.6, False),
    ("CLOSE", 1.5, False),
    ("GRASP_DWELL", 0.6, False),
    ("LIFT", 1.2, True),
    ("REORIENT", 1.6, True),
    ("TRANSFER", 1.6, True),
    ("LOWER", 1.0, True),
    ("PLACE_DWELL", 0.4, True),
    ("RELEASE", 0.6, True),
    ("RETREAT", 1.0, True),
    ("SETTLE", 1.0, False),
]
PHASE_INDEX = {name: i for i, (name, _, _) in enumerate(PHASES)}

# ----------------------------------------------------------------------------
# stage markers and failure reasons (docs/EXTENSION_PLAN.md)
# ----------------------------------------------------------------------------
STAGES = ["acquired", "lifted_clear", "reoriented_upright",
          "placed", "released", "settled"]
FAILURE_REASONS = ["acquisition_failure", "dropped_during_transport",
                   "placement_miss", "tipped_after_release",
                   "timeout", "other"]


# ----------------------------------------------------------------------------
# SO(3) helpers (numpy, self-contained as in the ParcelStow geometry)
# ----------------------------------------------------------------------------
def roty(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def so3_log(R):
    c = (np.trace(R) - 1.0) * 0.5
    a = math.acos(min(1.0, max(-1.0, c)))
    if a < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (a / (2.0 * math.sin(a)))


def so3_exp(w):
    a = np.linalg.norm(w)
    if a < 1e-12:
        return np.eye(3)
    k = w / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


def slerp(R_a, R_b, s):
    return np.asarray(R_a) @ so3_exp(s * so3_log(np.asarray(R_a).T @ np.asarray(R_b)))


def smoothstep(f):
    f = np.clip(f, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(math.pi * f))


# Lying start orientation (long axis horizontal along the yawed x axis)
# and the upright goal orientation. The 90 degree reorientation between
# them stays under the pi branch of so3_log.
R_START = rotz(math.radians(START_YAW_DEG)) @ roty(math.pi / 2)
R_UPRIGHT = rotz(math.radians(START_YAW_DEG))


def tilt_deg(R):
    """Angle between the object z axis and the world vertical, degrees."""
    return math.degrees(math.acos(min(1.0, max(-1.0, float(R[2, 2])))))


def base_center(p, R):
    """World position of the center of the object's -z face."""
    return np.asarray(p, dtype=np.float64) - np.asarray(R) @ [0.0, 0.0, OBJECT_HALF_HEIGHT]


def inside_target(p, R):
    """Base center inside the target region, the planar half of the
    success predicate; the tilt half is tilt_deg <= FINAL_TILT_TOL_DEG."""
    d = base_center(p, R)[:2] - np.asarray(TARGET_CENTER)
    return float(np.linalg.norm(d)) <= TARGET_RADIUS


def object_pose(k, f):
    """Desired object pose (p, R) at phase index k, in-phase fraction f.

    The path is a function of (k, f) alone; the speedup factor enters
    only through the phase clock, the v1 invariant. Moving phases blend
    with the cosine ease of the v1 expert.
    """
    s = float(smoothstep(f))
    p_start = np.asarray(START_POS, dtype=np.float64)
    p_lift = p_start + [0.0, 0.0, LIFT_DZ]
    p_above = np.array([TARGET_CENTER[0], TARGET_CENTER[1], p_lift[2]])
    p_place = np.array([TARGET_CENTER[0], TARGET_CENTER[1], PLACE_Z])
    if k <= PHASE_INDEX["GRASP_DWELL"]:
        return p_start.copy(), R_START.copy()
    if k == PHASE_INDEX["LIFT"]:
        return p_start + s * (p_lift - p_start), R_START.copy()
    if k == PHASE_INDEX["REORIENT"]:
        return p_lift.copy(), slerp(R_START, R_UPRIGHT, s)
    if k == PHASE_INDEX["TRANSFER"]:
        return p_lift + s * (p_above - p_lift), R_UPRIGHT.copy()
    if k == PHASE_INDEX["LOWER"]:
        return p_above + s * (p_place - p_above), R_UPRIGHT.copy()
    return p_place.copy(), R_UPRIGHT.copy()
