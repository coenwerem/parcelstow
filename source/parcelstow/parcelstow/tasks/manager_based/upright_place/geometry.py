"""Upright placement task geometry.

A tall rigid cuboid starts lying on its side at a known pose; the task
stands it upright on a marked circular target region on the table. The
terminal predicate is quasi-static stability, final tilt from vertical
at most FINAL_TILT_TOL_DEG with the base center inside the target
region, rather than the geometric containment of ParcelStow
(docs/EXTENSION_PLAN.md).

Geometry frozen by the kinematic probes and expert-only validation of
2026-09-01 (scripts/manipulation/probe_upright_geometry.py, reports
under outputs/probe/), following the v1 freeze protocol of
docs/TASK_SPEC.md: kinematic and expert-only criteria, no learner
outcome. The hypothesis probes over 141 candidates found that a
shaft-centered grasp saturates the waist roll while lowering (one
feasible candidate, minimum margin 0.001) and that goal-yaw offsets
drive the wrist yaw to its limit during reorientation; FRoGGeR
synthesis on the 55 x 55 x 180 mm cuboid placed the five-contact
grasp at +46 to +91 mm along the shaft (centroid +72 mm) on its own.
Simulation validation then measured three further mechanisms and set
the remaining values. The idle left hand at the arm-zero default
occupies the left-side placement zone (and any static re-park of the
left arm shifts the torso sag enough to break the millimeter-margin
open-loop acquisition), so the target sits on the robot's right of
the transport axis, 0.207 m clear of the idle fingers. An object that
pivots in the grasp hangs 142 mm below the grasp point, so the lift
rises to 0.18 m and the hanging end clears the table by 46 mm. The
synthesized grasp's pinky contact at +91 mm sits on the shaft's end
edge and ejects the object axially under squeeze, so the bank slides
the grasp 20 mm toward the center of mass along the constant
cross-section (contact centroid +52 mm, every fingertip on the
shaft). At the frozen configuration the probe solves all 38 knots
within 3.6 mm and 0.9 deg at minimum joint-limit margin 0.085 (worst
at the end of LOWER, waist pitch), the trajectory's 63 knots solve
within 2.0 mm at margin 0.110, and the scripted expert validates
20 of 20 at r = 0.5 with final tilt 0.0 deg and base offsets of 7 to
19 mm. Phase durations await the Gate B expert-only calibration. The
module is self-contained (numpy only) so the pure tests load it by
file path, the convention of the ParcelStow geometry module.

The tilt tolerance derivation mirrors the v1 derived tolerances: the
resting cuboid tips when the center of mass leaves the base, at
atan(half width / half height) = atan(27.5 / 90) = 17.0 degrees, so
the 5 degree tolerance is stricter than the tipping angle and was
fixed before any expert or learner ran.
"""

import math

import numpy as np

# ----------------------------------------------------------------------------
# object, table, and target (frozen by the kinematic probe, see docstring)
# ----------------------------------------------------------------------------
OBJECT_EXTENTS = (0.055, 0.055, 0.180)  # x, y, z in the object frame
OBJECT_HALF_HEIGHT = OBJECT_EXTENTS[2] / 2
OBJECT_MASS = 0.120  # the v1 parcel mass, holding object mass fixed across tasks
OBJECT_FRICTION = 0.5  # the v1 physics material
TABLE_TOP = 0.70  # the v1 table
START_YAW_DEG = 45.0
# Lying on a 55 x 180 face, center 1 mm above rest height, the v1 start
# convention. The 55 mm width is the aperture floor the synthesis
# established: FRoGGeR returned no seated force-closed grasp at 40 or
# 50 mm and three at 55 mm, the width the v1 parcel already proved. The
# 180 mm length admits the probed +50 mm grasp shift with the contact
# span ending 10 mm inside the shaft.
START_POS = (0.35, 0.0, TABLE_TOP + OBJECT_EXTENTS[0] / 2 + 0.001)
# The lift height covers the in-hand pivot worst case: an object that
# pivots to hang from the end-shifted grasp extends 162 mm below the
# grasp point, and at 0.18 m of lift its hanging end still clears the
# table by 46 mm at the start of the reorientation.
LIFT_DZ = 0.18
# Probe-chosen on the robot's right of the transport axis, 0.207 m clear
# of the idle left hand (the left-side candidates put the placement
# inside the idle hand's zone); transport 0.180 m at margin 0.108.
TARGET_CENTER = (0.527, 0.035)
TARGET_RADIUS = 0.030
# The commanded place pose seats the base at its rest height: the object
# arrives pitched a few degrees in the grasp (in-hand pivot grows with the
# speedup factor), and seating presses the leading base edge onto the
# table, which rights the object while it is still held; releasing from a
# positive drop instead plants the tilted base off-center (measured, 36 to
# 38 mm offsets at r 1.25 against the 30 mm target radius).
PLACE_DROP = 0.000
PLACE_Z = TABLE_TOP + OBJECT_HALF_HEIGHT + PLACE_DROP
RETREAT_DISTANCE = 0.10  # hand withdrawal along -d after release, the v1 value
# Frozen grasp-region offset along the shaft toward the future top end (a
# centered grasp saturates the waist roll while lowering, margin 0.001);
# the bank realizes a +0.052 m contact centroid, the synthesized +0.072 m
# grasp slid 20 mm toward the center of mass so every fingertip lands on
# the shaft.
GRASP_SHIFT = 0.050

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
# extended to 1.0 s so tipping resolves inside the episode. The scaled
# nominal durations were set by the Gate B expert-only calibration of
# 2026-09-01: at half these durations the expert's placement bias (the
# in-hand pitch accumulated under the gravity moment of the end-shifted
# grasp) exceeds the 30 mm target radius from r = 1, so the nominal
# anchors where the expert holds at least 0.9 success over a usable
# demonstrated range, the v1 procedure for the rate grid.
PHASES = [
    ("PARK", 0.5, False),
    ("APPROACH", 2.5, False),
    ("PREGRASP_DWELL", 0.6, False),
    ("CLOSE", 1.5, False),
    ("GRASP_DWELL", 0.6, False),
    ("LIFT", 2.4, True),
    ("REORIENT", 3.2, True),
    ("TRANSFER", 3.2, True),
    ("LOWER", 2.0, True),
    ("PLACE_DWELL", 0.8, True),
    ("RELEASE", 1.2, True),
    ("RETREAT", 2.0, True),
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


def quat_from_mat(m):
    """wxyz quaternion of a rotation matrix, the Isaac Lab convention."""
    m = np.asarray(m, dtype=np.float64)
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


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


def make_tf(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def inv_tf(T):
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def hand_pose_from_object(T_WO, X_OH):
    """T_WH^d = T_WO^d X_OH, the v1 convention."""
    return np.asarray(T_WO) @ np.asarray(X_OH)


def grasp_in_object_frame(T_WO, T_WH):
    """X_OH = T_WO^-1 T_WH, the hand pose expressed in the object frame."""
    return inv_tf(np.asarray(T_WO)) @ np.asarray(T_WH)


def path_frames(start_yaw_deg=START_YAW_DEG, target_center=TARGET_CENTER,
                lift_dz=LIFT_DZ, goal_yaw_deg=None):
    """Waypoint frames of the object path for one candidate parameter set.

    The defaults are the module constants; the kinematic probe passes
    candidate values over its grid. The goal yaw of the standing object
    defaults to the start yaw but is a free parameter: the success
    predicate constrains tilt and base position only, and the base is
    C4 symmetric, so the goal yaw sets the hand azimuth at placement
    without changing the task. d is the unit horizontal transport
    direction from the start toward the target.
    """
    p_start = np.asarray(START_POS, dtype=np.float64)
    yaw = math.radians(start_yaw_deg)
    goal_yaw = yaw if goal_yaw_deg is None else math.radians(goal_yaw_deg)
    p_lift = p_start + [0.0, 0.0, lift_dz]
    p_above = np.array([target_center[0], target_center[1], p_lift[2]])
    p_place = np.array([target_center[0], target_center[1], PLACE_Z])
    d = np.array([target_center[0] - p_start[0], target_center[1] - p_start[1], 0.0])
    n = np.linalg.norm(d)
    d = d / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    return {"start_yaw_deg": float(start_yaw_deg), "lift_dz": float(lift_dz),
            "goal_yaw_deg": float(math.degrees(goal_yaw)),
            "target_center": (float(target_center[0]), float(target_center[1])),
            "p_start": p_start, "R_start": rotz(yaw) @ roty(math.pi / 2),
            "R_upright": rotz(goal_yaw), "p_lift": p_lift, "p_above": p_above,
            "p_place": p_place, "d": d}


_DEFAULT_FRAMES = path_frames()


def object_pose(k, f, frames=None):
    """Desired object pose (p, R) at phase index k, in-phase fraction f.

    The path is a function of (k, f) alone; the speedup factor enters
    only through the phase clock, the v1 invariant. Moving phases blend
    with the cosine ease of the v1 expert. frames overrides the module
    constants with one candidate parameter set from path_frames.
    """
    fr = _DEFAULT_FRAMES if frames is None else frames
    s = float(smoothstep(f))
    if k <= PHASE_INDEX["GRASP_DWELL"]:
        return fr["p_start"].copy(), fr["R_start"].copy()
    if k == PHASE_INDEX["LIFT"]:
        return fr["p_start"] + s * (fr["p_lift"] - fr["p_start"]), fr["R_start"].copy()
    if k == PHASE_INDEX["REORIENT"]:
        return fr["p_lift"].copy(), slerp(fr["R_start"], fr["R_upright"], s)
    if k == PHASE_INDEX["TRANSFER"]:
        return fr["p_lift"] + s * (fr["p_above"] - fr["p_lift"]), fr["R_upright"].copy()
    if k == PHASE_INDEX["LOWER"]:
        return fr["p_above"] + s * (fr["p_place"] - fr["p_above"]), fr["R_upright"].copy()
    return fr["p_place"].copy(), fr["R_upright"].copy()


def retreat_hand_pose(X_OH, f, frames=None):
    """Hand pose during RETREAT, the place-pose hand backed out along -d
    by RETREAT_DISTANCE times the eased fraction, the v1 pattern."""
    fr = _DEFAULT_FRAMES if frames is None else frames
    T = hand_pose_from_object(make_tf(fr["R_upright"], fr["p_place"]), X_OH)
    T = T.copy()
    T[:3, 3] = T[:3, 3] - fr["d"] * (RETREAT_DISTANCE * smoothstep(f))
    return T


# ----------------------------------------------------------------------------
# knot schedule of the IK trajectory, (phase, fraction) pairs
# ----------------------------------------------------------------------------
KNOT_FRACTIONS = {
    "LIFT": [0.0, 0.25, 0.5, 0.75, 1.0],
    "REORIENT": [0.0, 1 / 9, 2 / 9, 3 / 9, 4 / 9, 5 / 9, 6 / 9, 7 / 9, 8 / 9, 1.0],
    "TRANSFER": [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
    "LOWER": [0.0, 0.25, 0.5, 0.75, 1.0],
    "PLACE_DWELL": [0.0, 1.0],
    "RELEASE": [0.0, 1.0],
    "RETREAT": [0.0, 0.25, 0.5, 0.75, 1.0],
}


def knot_list():
    """Ordered list of (phase_index, fraction) knots of the manipulation."""
    out = []
    for name, fr in KNOT_FRACTIONS.items():
        k = PHASE_INDEX[name]
        for f in fr:
            out.append((k, float(f)))
    return out
