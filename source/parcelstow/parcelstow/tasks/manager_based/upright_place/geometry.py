"""Upright placement task geometry.

A tall rigid cuboid starts lying on its side at a known pose; the task
stands it upright on a marked circular target region on the table. The
terminal predicate is quasi-static stability, final tilt from vertical
at most FINAL_TILT_TOL_DEG with the base center inside the target
region, rather than the geometric containment of ParcelStow
(docs/EXTENSION_PLAN.md).

Geometry frozen by the kinematic probe of 2026-09-01
(scripts/manipulation/probe_upright_geometry.py, report
outputs/probe/upright_probe_final.json), following the v1 freeze
protocol of docs/TASK_SPEC.md: kinematic criteria only, no learner
outcome. At the frozen candidate (start yaw +45 deg, goal yaw equal to
the start yaw, target (0.457, 0.107), lift 0.12 m, grasp shift
+0.05 m) all 38 IK knots solve within 3.7 mm and 1.3 deg, the minimum
joint-limit margin over the manipulation is 0.118 (worst at
mid-retreat, wrist yaw), the grasp margin is 0.236, the hand stays at
least 52 mm above the table after lift, and the retreat clears the
placed object by 16 mm. The probe found the goal yaw must equal the
start yaw (goal-yaw offsets drive the wrist yaw to its limit during
reorientation) and that the grasp point must sit 50 mm toward the
future top end of the shaft (a centered grasp saturates the waist roll
while lowering, minimum margin 0.001). The grasp transform itself is a
probe hypothesis derived from the frozen v1 grasp; the FRoGGeR bank,
synthesized over the GRASP_SHIFT region, replaces it before any expert
or learner runs. Phase durations await the Gate B expert-only
calibration. The module is self-contained (numpy only) so the pure
tests load it by file path, the convention of the ParcelStow geometry
module.

The tilt tolerance derivation mirrors the v1 derived tolerances: the
resting cuboid tips when the center of mass leaves the base, at
atan(half width / half height) = atan(20 / 70) = 15.9 degrees, so the
5 degree tolerance is stricter than the tipping angle and was fixed
before any expert or learner ran.
"""

import math

import numpy as np

# ----------------------------------------------------------------------------
# object, table, and target (frozen by the kinematic probe, see docstring)
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
TARGET_CENTER = (0.457, 0.107)  # probe-chosen, transport 0.151 m at margin 0.118
TARGET_RADIUS = 0.030
PLACE_DROP = 0.005  # release height of the base above the table
PLACE_Z = TABLE_TOP + OBJECT_HALF_HEIGHT + PLACE_DROP
RETREAT_DISTANCE = 0.10  # hand withdrawal along -d after release, the v1 value
# Grasp point offset along the object long axis toward the future top end,
# an input to the bank synthesis; a centered grasp saturates the waist roll
# while lowering (probe, minimum margin 0.001 vs 0.118 at +0.05).
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
