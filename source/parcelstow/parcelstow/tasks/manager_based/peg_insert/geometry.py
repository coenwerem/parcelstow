"""Keyed-peg insertion task geometry.

The upright task's 55 x 55 x 180 mm cuboid, one object across both
new tasks, starts lying on its side at the proven start; the task stands it up and inserts it into a
square pocket with 3 mm of clearance per side, the tight-clearance
containment regime of the arXiv-v2 suite (docs/EXTENSION_PLAN.md).
The 55 mm width is the RealHand L6 aperture floor the grasp synthesis
established for the upright task; the pocket sits at the probed
right-side location, clear of the idle left hand, and the pocket top
at 120 mm above the table keeps the hand in the higher workspace
the v1 receptacle already proved. The descent releases the peg 10 mm
above the pocket floor inside the guided cavity, the v1 receptacle's
release convention. Derived tolerance: a
square peg of side a in a square pocket with clearance c admits a yaw
of about 2 c / a = 6.2 degrees, so the 5 degree final tilt tolerance
is stricter and was fixed before any expert or learner ran.

The module is self-contained (numpy only) so the pure tests load it
by file path, the house convention.
"""

import math

import numpy as np

# ----------------------------------------------------------------------------
# object, table, and pocket
# ----------------------------------------------------------------------------
OBJECT_EXTENTS = (0.055, 0.055, 0.180)  # the upright task object, shared
OBJECT_HALF_HEIGHT = OBJECT_EXTENTS[2] / 2
OBJECT_MASS = 0.120  # the v1 parcel mass, fixed across the task suite
OBJECT_FRICTION = 0.5
TABLE_TOP = 0.70
START_YAW_DEG = 45.0
START_POS = (0.35, 0.0, TABLE_TOP + OBJECT_EXTENTS[0] / 2 + 0.001)
LIFT_DZ = 0.22  # pivot-to-hang clearance over the raised pocket block
POCKET_CENTER = (0.527, 0.035)  # the probed right-side location of the suite
CLEARANCE = 0.003
POCKET_W = OBJECT_EXTENTS[0] + 2 * CLEARANCE  # 61 mm across the cavity
POCKET_DEPTH = 0.060
WALL_T = 0.030
FLOOR_T = 0.010
# The pocket top sits 120 mm above the table so the hand works at the
# heights the v1 receptacle proved; at 70 mm the waist roll saturates at
# the bottom of the descent (trajectory margin 0.000, the recurring
# low-reach bind of this arm).
BLOCK_TOP = TABLE_TOP + 0.120
POCKET_FLOOR_Z = BLOCK_TOP - POCKET_DEPTH
# The descent ends with the base 10 mm above the pocket floor and the
# release drops the peg inside the guided cavity, the v1 receptacle's
# release convention; commanding a full seat instead lowers the hand
# onto the mouth hardware (measured, fingertip-fixture jams).
RELEASE_DROP = 0.010
SEAT_Z = POCKET_FLOOR_Z + RELEASE_DROP + OBJECT_HALF_HEIGHT
INSERTED_MIN_DEPTH = 0.040  # base at least 40 mm below the pocket top
RETREAT_DISTANCE = 0.10
# Frozen grasp-region offset along the shaft toward the future top end,
# the upright task's grasp on the shared object (synthesized centroid
# +0.072 m slid 20 mm toward the center of mass); at the release height
# the fingertips clear the lead-in hardware by about 50 mm.
GRASP_SHIFT = 0.050

FINAL_TILT_TOL_DEG = 5.0
SETTLE_LIN = 0.02
SETTLE_ANG = 0.2

# ----------------------------------------------------------------------------
# phase schedule, (name, nominal seconds, rate scaled)
# ----------------------------------------------------------------------------
# Acquisition keeps the v1 timings; the scaled nominals carry over the
# upright Gate B calibration (INSERT takes the LOWER slot), pending this
# task's own calibration.
PHASES = [
    ("PARK", 0.5, False),
    ("APPROACH", 2.5, False),
    ("PREGRASP_DWELL", 0.6, False),
    ("CLOSE", 1.5, False),
    ("GRASP_DWELL", 0.6, False),
    ("LIFT", 2.4, True),
    ("REORIENT", 3.2, True),
    ("TRANSFER", 3.2, True),
    ("INSERT", 2.0, True),
    ("INSERT_DWELL", 0.8, True),
    ("RELEASE", 1.2, True),
    ("RETREAT", 2.0, True),
    ("SETTLE", 1.0, False),
]
PHASE_INDEX = {name: i for i, (name, _, _) in enumerate(PHASES)}

STAGES = ["acquired", "lifted_clear", "reoriented_upright",
          "aligned", "inserted", "released", "settled"]
FAILURE_REASONS = ["acquisition_failure", "dropped_during_transport",
                   "alignment_failure", "insertion_jam",
                   "timeout", "other"]


# ----------------------------------------------------------------------------
# SO(3) helpers (numpy, self-contained)
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


R_START = rotz(math.radians(START_YAW_DEG)) @ roty(math.pi / 2)
R_UPRIGHT = rotz(math.radians(START_YAW_DEG))
R_POCKET = rotz(math.radians(START_YAW_DEG))  # cavity yaw-aligned with the goal


def tilt_deg(R):
    return math.degrees(math.acos(min(1.0, max(-1.0, float(R[2, 2])))))


def base_depth(p):
    """Depth of the peg base below the pocket top, m (negative above)."""
    return float(BLOCK_TOP - (np.asarray(p, dtype=np.float64)[2] - OBJECT_HALF_HEIGHT))


def inside_pocket(p, R):
    """All four base corners inside the pocket cross-section, which
    encodes the yaw and tilt tolerances the clearance derives."""
    p = np.asarray(p, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    hx, hy = OBJECT_EXTENTS[0] / 2, OBJECT_EXTENTS[1] / 2
    half = POCKET_W / 2
    Ry = R_POCKET.T
    for sx in (-1, 1):
        for sy in (-1, 1):
            corner = p + R @ [sx * hx, sy * hy, -OBJECT_HALF_HEIGHT]
            rel = Ry @ (corner - np.array([*POCKET_CENTER, 0.0]))
            if abs(rel[0]) > half or abs(rel[1]) > half:
                return False
    return True


# Entry lead-in: at 3 mm of clearance an open-loop insertion catches the
# rim (measured, sustained 45 N wedging with a 1 mm, 1 deg arrival), so
# the mouth carries the chamfer every engineered fixture has, four
# slanted slabs widening the entry by LEAD_H tan(LEAD_ANGLE) = 14 mm per
# side and funneling the peg into the tight containment.
LEAD_ANGLE_DEG = 35.0
LEAD_H = 0.020
LEAD_T = 0.008


def pocket_slabs():
    """Kinematic slab definitions of the pocket block, the v1 receptacle
    pattern: a floor, four walls, and four lead-in slabs, yaw-aligned
    with the goal."""
    q = [float(v) for v in quat_from_mat(R_POCKET)]
    cx, cy = POCKET_CENTER
    outer = POCKET_W + 2 * WALL_T
    wall_off = POCKET_W / 2 + WALL_T / 2

    def center(dx, dy, z):
        d = R_POCKET @ np.array([dx, dy, 0.0])
        return [cx + float(d[0]), cy + float(d[1]), z]

    wall_zc = POCKET_FLOOR_Z + POCKET_DEPTH / 2
    slabs = {
        "floor": {"size": [outer, outer, FLOOR_T],
                  "center": [cx, cy, POCKET_FLOOR_Z - FLOOR_T / 2], "quat_wxyz": q},
        "wall_a": {"size": [outer, WALL_T, POCKET_DEPTH],
                   "center": center(0.0, wall_off, wall_zc), "quat_wxyz": q},
        "wall_b": {"size": [outer, WALL_T, POCKET_DEPTH],
                   "center": center(0.0, -wall_off, wall_zc), "quat_wxyz": q},
        "wall_c": {"size": [WALL_T, POCKET_W, POCKET_DEPTH],
                   "center": center(wall_off, 0.0, wall_zc), "quat_wxyz": q},
        "wall_d": {"size": [WALL_T, POCKET_W, POCKET_DEPTH],
                   "center": center(-wall_off, 0.0, wall_zc), "quat_wxyz": q},
    }
    # Lead-in slabs: the inner-lower edge of each slanted face sits on the
    # pocket rim; the face rises outward at LEAD_ANGLE from the vertical.
    A = math.radians(LEAD_ANGLE_DEG)
    L = LEAD_H / math.cos(A)  # slab length along the slanted face
    rim = POCKET_W / 2
    # surface direction up-outward and inward-facing normal, in the (u, z)
    # plane of the pocket frame where u points outward across the rim
    d_u, d_z = math.sin(A), math.cos(A)
    n_u, n_z = -math.cos(A), math.sin(A)
    # face midpoint plus half the thickness behind the face (away from the peg)
    u_c = rim + (L / 2) * d_u - (LEAD_T / 2) * n_u
    z_c = BLOCK_TOP + 0.001 + (L / 2) * d_z - (LEAD_T / 2) * n_z
    # The leads cover only the straight mouth edges and sit 1 mm above
    # the wall tops: overlapping kinematic slab pairs (mitred corners,
    # face-on-face rim contact) disturb the GPU contact pipeline enough
    # to break the arm's contact behavior at the start (measured by scene
    # bisection); the corner regions stay backed by the flat wall tops.
    lead_len = POCKET_W
    for name, R_side in (("lead_a", rotz(0.0)), ("lead_b", rotz(math.pi)),
                         ("lead_c", rotz(math.pi / 2)), ("lead_d", rotz(-math.pi / 2))):
        # R_side maps the +y wall's lead into each side; tilt about the
        # slab's local x by -A so the top leans outward
        R_slab = R_POCKET @ R_side @ rotx_local(-A)
        off = R_POCKET @ R_side @ np.array([0.0, u_c, 0.0])
        slabs[name] = {"size": [lead_len, LEAD_T, L],
                       "center": [cx + float(off[0]), cy + float(off[1]), z_c],
                       "quat_wxyz": [float(v) for v in quat_from_mat(R_slab)]}
    return slabs


def rotx_local(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


# ----------------------------------------------------------------------------
# object path
# ----------------------------------------------------------------------------
def path_frames(start_yaw_deg=START_YAW_DEG, pocket_center=POCKET_CENTER,
                lift_dz=LIFT_DZ):
    p_start = np.asarray(START_POS, dtype=np.float64)
    yaw = math.radians(start_yaw_deg)
    p_lift = p_start + [0.0, 0.0, lift_dz]
    p_above = np.array([pocket_center[0], pocket_center[1], p_lift[2]])
    p_seat = np.array([pocket_center[0], pocket_center[1], SEAT_Z])
    d = np.array([pocket_center[0] - p_start[0], pocket_center[1] - p_start[1], 0.0])
    n = np.linalg.norm(d)
    d = d / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    return {"p_start": p_start, "R_start": rotz(yaw) @ roty(math.pi / 2),
            "R_upright": rotz(yaw), "p_lift": p_lift, "p_above": p_above,
            "p_seat": p_seat, "d": d}


_DEFAULT_FRAMES = path_frames()


def object_pose(k, f, frames=None):
    """Desired object pose (p, R) at phase index k, in-phase fraction f;
    a function of (k, f) alone, the suite invariant."""
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
    if k == PHASE_INDEX["INSERT"]:
        return fr["p_above"] + s * (fr["p_seat"] - fr["p_above"]), fr["R_upright"].copy()
    return fr["p_seat"].copy(), fr["R_upright"].copy()


def hand_pose_from_object(T_WO, X_OH):
    return np.asarray(T_WO) @ np.asarray(X_OH)


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


def retreat_hand_pose(X_OH, f, frames=None):
    fr = _DEFAULT_FRAMES if frames is None else frames
    T = hand_pose_from_object(make_tf(fr["R_upright"], fr["p_seat"]), X_OH)
    T = T.copy()
    T[:3, 3] = T[:3, 3] - fr["d"] * (RETREAT_DISTANCE * smoothstep(f))
    return T


KNOT_FRACTIONS = {
    "LIFT": [0.0, 0.25, 0.5, 0.75, 1.0],
    "REORIENT": [0.0, 1 / 9, 2 / 9, 3 / 9, 4 / 9, 5 / 9, 6 / 9, 7 / 9, 8 / 9, 1.0],
    "TRANSFER": [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
    "INSERT": [0.0, 0.25, 0.5, 0.75, 1.0],
    "INSERT_DWELL": [0.0, 1.0],
    "RELEASE": [0.0, 1.0],
    "RETREAT": [0.0, 0.25, 0.5, 0.75, 1.0],
}


def knot_list():
    out = []
    for name, fr in KNOT_FRACTIONS.items():
        k = PHASE_INDEX[name]
        for f in fr:
            out.append((k, float(f)))
    return out
