"""Pure-math pieces of the ParcelStow task, the phase schedule with the
task-rate law, SO(3) helpers, the receptacle geometry derived from the
frozen task specification, and the object task-space path T_WO(k, f) of the
manipulation.

Every constant here follows docs/TASK_SPEC.md. The
geometry file assets/parcel_stow_geometry.json (written by
scripts/manipulation/probe_stow_geometry.py --finalize) fixes the pending
values, and load_geometry() reads it. The module imports only numpy, so
the pure tests load it by file path without the simulator, and the task
package and the scripts import it after the app launches.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", ".."))
GEOMETRY_PATH = os.path.join(REPO, "assets", "parcel_stow_geometry.json")
TRAJECTORY_PATH = os.path.join(REPO, "assets", "parcel_stow_trajectory.json")
BANK_PATH = os.path.join(REPO, "assets", "gdf_bank_parcel.json")

# ----------------------------------------------------------------------------
# frozen task constants (TASK_SPEC.md sections 2 to 4)
# ----------------------------------------------------------------------------
PARCEL_EXTENTS = (0.080, 0.055, 0.040)
PARCEL_HALF = tuple(0.5 * e for e in PARCEL_EXTENTS)
PARCEL_MASS = 0.120
PARCEL_FRICTION = 0.5
TABLE_TOP = 0.70
PARCEL_START = (0.35, 0.0, TABLE_TOP + PARCEL_HALF[2] + 0.001)
PELVIS_POS = (0.0, 0.0, 0.75)
LIFT_DZ = 0.08
LIFT_DX_DEFAULT = 0.0
SHELF_HEIGHT_DEFAULT = 0.0
WALL_THICKNESS = 0.020
C_TIGHT = 0.010
C_LOOSE = 0.055
DEPTH_SLACK = 0.070
INSERT_BACK_MARGIN = 0.030
PREINSERT_STANDOFF = 0.030
RETREAT_DISTANCE = 0.10
CONTROL_DT = 0.02

# ----------------------------------------------------------------------------
# phase schedule (TASK_SPEC.md section 7)
# ----------------------------------------------------------------------------
PHASES = [
    ("PARK", 0.5, False),
    ("APPROACH", 2.5, False),
    ("PREGRASP_DWELL", 0.6, False),
    ("CLOSE", 1.5, False),
    ("GRASP_DWELL", 0.6, False),
    ("LIFT", 1.2, True),
    ("REORIENT", 1.6, True),
    ("TRANSFER", 1.6, True),
    ("PREINSERT_DWELL", 0.4, True),
    ("INSERT", 1.0, True),
    ("INSERT_DWELL", 0.4, True),
    ("RELEASE", 0.6, True),
    ("RETREAT", 1.0, True),
    ("SETTLE", 0.6, False),
]
PHASE_NAMES = [p[0] for p in PHASES]
PHASE_INDEX = {n: i for i, n in enumerate(PHASE_NAMES)}
N_PHASES = len(PHASES)
NOMINAL_DURATIONS = np.array([p[1] for p in PHASES], dtype=np.float64)
RATE_SCALED = np.array([p[2] for p in PHASES], dtype=bool)
T_ACQ = float(NOMINAL_DURATIONS[:PHASE_INDEX["LIFT"]].sum())
T_MANIP = float(NOMINAL_DURATIONS[RATE_SCALED].sum())
T_SETTLE = float(NOMINAL_DURATIONS[PHASE_INDEX["SETTLE"]])


def phase_durations(rate: float) -> np.ndarray:
    """Per-phase durations in seconds at task rate r."""
    d = NOMINAL_DURATIONS.copy()
    d[RATE_SCALED] = d[RATE_SCALED] / float(rate)
    return d


def cycle_time(rate: float) -> float:
    return float(phase_durations(rate).sum())


def phase_of_time(t: float, rate: float):
    """(k, f) at elapsed time t for rate r, f clipped to [0, 1], k clipped to
    the last phase once the cycle is over."""
    d = phase_durations(rate)
    cum = np.cumsum(d)
    k = int(np.searchsorted(cum, t, side="right"))
    if k >= N_PHASES:
        return N_PHASES - 1, 1.0
    start = cum[k - 1] if k > 0 else 0.0
    f = (t - start) / d[k]
    return k, float(min(max(f, 0.0), 1.0))


def smoothstep(f):
    """Cosine ease in [0, 1] -> [0, 1], the blend the cube expert used."""
    f = np.clip(f, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(math.pi * f))


# ----------------------------------------------------------------------------
# SO(3) helpers (numpy, wxyz quaternions as in Isaac Lab)
# ----------------------------------------------------------------------------
def rotx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def roty(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


AXIS_ROT = {"x": rotx, "y": roty, "z": rotz}


def mat_from_quat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def quat_from_mat(m):
    m = np.asarray(m, dtype=np.float64)
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def so3_log(R):
    """Rotation vector of R (axis times angle)."""
    c = (np.trace(R) - 1.0) * 0.5
    c = min(1.0, max(-1.0, c))
    a = math.acos(c)
    if a < 1e-9:
        return np.zeros(3)
    if abs(math.pi - a) < 1e-6:
        A = (R + np.eye(3)) * 0.5
        axis = np.sqrt(np.clip(np.diag(A), 0.0, 1.0))
        if A[0, 1] < 0:
            axis[1] = -axis[1]
        if A[0, 2] < 0:
            axis[2] = -axis[2]
        return axis / (np.linalg.norm(axis) + 1e-12) * a
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (a / (2.0 * math.sin(a)))


def so3_exp(w):
    a = np.linalg.norm(w)
    if a < 1e-12:
        return np.eye(3)
    k = w / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


def so3_angle(R_a, R_b) -> float:
    """Geodesic angle between two rotations, radians."""
    return float(np.linalg.norm(so3_log(np.asarray(R_a).T @ np.asarray(R_b))))


def slerp(R_a, R_b, s: float):
    """Geodesic interpolation from R_a (s = 0) to R_b (s = 1)."""
    return np.asarray(R_a) @ so3_exp(s * so3_log(np.asarray(R_a).T @ np.asarray(R_b)))


def make_tf(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def inv_tf(T):
    R = T[:3, :3]
    p = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p
    return out


# ----------------------------------------------------------------------------
# receptacle geometry
# ----------------------------------------------------------------------------
UNIT = {"+x": np.array([1.0, 0, 0]), "-x": np.array([-1.0, 0, 0]),
        "+y": np.array([0, 1.0, 0]), "-y": np.array([0, -1.0, 0])}


class StowGeometry:
    """Frozen task geometry, either a candidate under evaluation by the probe
    or the finalized asset. Everything derives from the start yaw of the
    parcel, the reorientation rotation, the insertion axis, and the entrance
    plane point.

    The task frame is the world frame rotated by start_yaw about the
    vertical axis through the parcel start. In the task frame the parcel
    starts axis aligned, the reorientation is AXIS_ROT[rot_axis](rot_deg),
    and the insertion axis is UNIT[insert_axis]. World quantities follow by
    the rotation R_yaw about p_start.

    Attributes (world frame unless noted),
      R_start (3x3) parcel start orientation, R_stow (3x3) stow orientation,
      d (3,) insertion axis, entrance (3,) center of the interior opening,
      i_d, i_loose, i_tight task-frame axis indices,
      L_d, L_tight, L_loose parcel extents along d, tight, loose axes,
      slabs, dict name -> (center_world, size_local, quat_wxyz).
    """

    def __init__(self, rot_axis: str, rot_deg: float, insert_axis: str,
                 entrance_xy: tuple, family: str = "", shelf_height: float = SHELF_HEIGHT_DEFAULT,
                 lift_dx: float = LIFT_DX_DEFAULT, start_yaw_deg: float = 0.0, lift_dz: float = LIFT_DZ,
                 reorient_travel: float = 0.0):
        self.family = family
        self.shelf_height = float(shelf_height)
        self.lift_dx = float(lift_dx)
        self.lift_dz = float(lift_dz)
        # fraction of the lift-to-preinsert translation performed during
        # REORIENT (the parcel turns while it starts to travel), the rest
        # during TRANSFER
        self.reorient_travel = float(reorient_travel)
        self.start_yaw_deg = float(start_yaw_deg)
        self.rot_axis = rot_axis
        self.rot_deg = float(rot_deg)
        self.insert_axis = insert_axis
        self.R_yaw = rotz(math.radians(self.start_yaw_deg))
        self.R_start = self.R_yaw.copy()
        self.R_local = AXIS_ROT[rot_axis](math.radians(rot_deg))
        self.R_stow = self.R_yaw @ self.R_local
        self.d_local = UNIT[insert_axis].copy()
        self.d = self.R_yaw @ self.d_local
        self.p_start = np.array(PARCEL_START)
        ext_local = np.abs(self.R_local @ np.array(PARCEL_EXTENTS))
        n_f = np.abs(self.R_local @ np.array([0.0, 1.0, 0.0]))
        cross = [i for i in range(3) if abs(self.d_local[i]) < 0.5]
        i_loose = max(cross, key=lambda i: n_f[i])
        i_tight = [i for i in cross if i != i_loose][0]
        self.i_d = int(np.argmax(np.abs(self.d_local)))
        self.i_loose, self.i_tight = i_loose, i_tight
        self.L_d = float(ext_local[self.i_d])
        self.L_loose = float(ext_local[i_loose])
        self.L_tight = float(ext_local[i_tight])
        self.W_loose = self.L_loose + 2 * C_LOOSE
        self.W_tight = self.L_tight + 2 * C_TIGHT
        self.D_in = self.L_d + DEPTH_SLACK
        floor_top = TABLE_TOP + self.shelf_height + WALL_THICKNESS
        self.floor_top = floor_top
        if i_tight == 2:
            zc = floor_top + 0.5 * self.W_tight
        elif i_loose == 2:
            zc = floor_top + 0.5 * self.W_loose
        else:
            raise ValueError("insertion axis must be horizontal")
        self.entrance = np.array([entrance_xy[0], entrance_xy[1], zc])
        self.entrance_local = self.to_local(self.entrance)
        half_h = 0.5 * ext_local[2]
        self.z_insert = floor_top + (C_TIGHT if i_tight == 2 else C_LOOSE) + half_h
        self.p_lift = self.p_start + self.R_yaw @ np.array([self.lift_dx, 0.0, self.lift_dz])
        depth_in = self.D_in - 0.5 * self.L_d - INSERT_BACK_MARGIN
        base = self.entrance.copy()
        base[2] = self.z_insert
        self.p_insert = base + self.d * depth_in
        self.p_preinsert = base - self.d * (0.5 * self.L_d + PREINSERT_STANDOFF)
        self.insert_depth_target = depth_in
        self.inserted_min_depth = 0.5 * self.L_d + 0.020
        self.transport_distance = float(np.linalg.norm(self.p_insert - self.p_start))
        self.slabs = self._slabs()

    # ------------------------------------------------------------------
    def to_local(self, p):
        return self.p_start + self.R_yaw.T @ (np.asarray(p) - self.p_start)

    def to_world(self, p):
        return self.p_start + self.R_yaw @ (np.asarray(p) - self.p_start)

    def _slabs(self):
        """Boxes (center_world, size_local, quat_wxyz) of floor, side_a,
        side_b, back, top. The floor slab reaches down to the table (a
        pedestal when the shelf is elevated)."""
        i_d, i_l, i_t = self.i_d, self.i_loose, self.i_tight
        s = np.sign(self.d_local[i_d])
        size = np.zeros(3)
        size[i_d] = self.D_in
        size[i_l] = self.W_loose
        size[i_t] = self.W_tight
        center = self.entrance_local + self.d_local * (0.5 * self.D_in)
        t = WALL_THICKNESS
        boxes = []
        for i in (i_l, i_t):
            for sign in (-1.0, 1.0):
                c = center.copy()
                c[i] += sign * (0.5 * size[i] + 0.5 * t)
                sz = size.copy()
                sz[i] = t
                other = i_t if i == i_l else i_l
                if i != 2:
                    sz[other] = size[other] + 2 * t
                boxes.append((i, sign, c, sz))
        c = center.copy()
        c[i_d] += s * (0.5 * size[i_d] + 0.5 * t)
        sz = size.copy()
        sz[i_d] = t
        sz[i_l] = size[i_l] + 2 * t
        sz[i_t] = size[i_t] + 2 * t
        local = {"back": (c, sz)}
        horiz = [b for b in boxes if b[0] != 2]
        vert = [b for b in boxes if b[0] == 2]
        local["side_a"] = (horiz[0][2], horiz[0][3])
        local["side_b"] = (horiz[1][2], horiz[1][3])
        lo = min(vert, key=lambda b: b[2][2])
        hi = max(vert, key=lambda b: b[2][2])
        c_lo, sz_lo = lo[2].copy(), lo[3].copy()
        floor_top = c_lo[2] + 0.5 * sz_lo[2]
        sz_lo[2] = floor_top - TABLE_TOP
        c_lo[2] = TABLE_TOP + 0.5 * sz_lo[2]
        local["floor"] = (c_lo, sz_lo)
        local["top"] = (hi[2], hi[3])
        assert local["floor"][0][2] < local["top"][0][2]
        quat = quat_from_mat(self.R_yaw)
        return {k: (self.to_world(c), sz, quat) for k, (c, sz) in local.items()}

    def interior_box(self):
        """Center (world) and half size (local axes) of the interior."""
        size = np.zeros(3)
        size[self.i_d] = self.D_in
        size[self.i_loose] = self.W_loose
        size[self.i_tight] = self.W_tight
        center = self.entrance + self.d * (0.5 * self.D_in)
        return center, 0.5 * size

    def depth_of(self, p):
        """Signed depth of a point past the entrance plane along d."""
        return float(np.dot(np.asarray(p) - self.entrance, self.d))

    def inside_interior(self, p, margin=0.0):
        c, h = self.interior_box()
        rel = self.R_yaw.T @ (np.asarray(p) - c)
        return bool(np.all(np.abs(rel) <= h + margin))

    def slab_clearance(self, p, r):
        """Smallest signed distance of a sphere (p, r) to the outside of any
        slab, negative when penetrating, and the slab name."""
        best = (float("inf"), None)
        for name, (c, sz, _) in self.slabs.items():
            rel = self.R_yaw.T @ (np.asarray(p) - np.asarray(c))
            h = 0.5 * np.asarray(sz)
            d = np.abs(rel) - h
            outside = np.linalg.norm(np.maximum(d, 0.0))
            inside = min(0.0, float(d.max()))
            dist = outside + inside - r
            if dist < best[0]:
                best = (dist, name)
        return best

    def to_dict(self):
        return {
            "family": self.family, "rot_axis": self.rot_axis, "rot_deg": self.rot_deg,
            "start_yaw_deg": self.start_yaw_deg,
            "shelf_height": self.shelf_height, "lift_dx": self.lift_dx, "lift_dz": self.lift_dz,
            "reorient_travel": self.reorient_travel, "floor_top": self.floor_top,
            "insert_axis": self.insert_axis, "entrance": self.entrance.tolist(),
            "R_start": self.R_start.tolist(), "start_quat_wxyz": quat_from_mat(self.R_start).tolist(),
            "R_stow": self.R_stow.tolist(), "d": self.d.tolist(),
            "i_d": self.i_d, "i_loose": self.i_loose, "i_tight": self.i_tight,
            "L_d": self.L_d, "L_loose": self.L_loose, "L_tight": self.L_tight,
            "W_loose": self.W_loose, "W_tight": self.W_tight, "D_in": self.D_in,
            "z_insert": self.z_insert, "p_start": self.p_start.tolist(),
            "p_lift": self.p_lift.tolist(), "p_preinsert": self.p_preinsert.tolist(),
            "p_insert": self.p_insert.tolist(), "insert_depth_target": self.insert_depth_target,
            "inserted_min_depth": self.inserted_min_depth,
            "transport_distance": self.transport_distance,
            "slabs": {k: {"center": v[0].tolist(), "size": list(v[1]), "quat_wxyz": v[2].tolist()}
                      for k, v in self.slabs.items()},
            "constants": {"parcel_extents": PARCEL_EXTENTS, "parcel_mass": PARCEL_MASS,
                          "parcel_friction": PARCEL_FRICTION, "table_top": TABLE_TOP,
                          "lift_dz": LIFT_DZ, "wall_thickness": WALL_THICKNESS,
                          "c_tight": C_TIGHT, "c_loose": C_LOOSE, "depth_slack": DEPTH_SLACK,
                          "insert_back_margin": INSERT_BACK_MARGIN,
                          "preinsert_standoff": PREINSERT_STANDOFF,
                          "retreat_distance": RETREAT_DISTANCE},
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["rot_axis"], d["rot_deg"], d["insert_axis"], tuple(d["entrance"][:2]), d.get("family", ""),
                   shelf_height=d.get("shelf_height", SHELF_HEIGHT_DEFAULT), lift_dx=d.get("lift_dx", LIFT_DX_DEFAULT),
                   start_yaw_deg=d.get("start_yaw_deg", 0.0), lift_dz=d.get("lift_dz", LIFT_DZ),
                   reorient_travel=d.get("reorient_travel", 0.0))


def load_geometry(path: str = GEOMETRY_PATH) -> StowGeometry:
    with open(path) as fh:
        return StowGeometry.from_dict(json.load(fh))


def load_geometry_dict(path: str = GEOMETRY_PATH) -> dict:
    with open(path) as fh:
        return json.load(fh)


# ----------------------------------------------------------------------------
# object task-space path
# ----------------------------------------------------------------------------
def object_pose(geom: StowGeometry, k: int, f: float, p_start=None):
    """Desired parcel pose T_WO^d at phase k and in-phase fraction f during
    the held segment (LIFT through RELEASE). p_start overrides the nominal
    start center (start jitter). Returns a 4x4 matrix, or None during
    RETREAT and SETTLE where the object is not on a desired path."""
    p0 = np.array(PARCEL_START if p_start is None else p_start)
    p_lift = p0 + geom.R_yaw @ np.array([geom.lift_dx, 0.0, geom.lift_dz])
    p_mid = p_lift + geom.reorient_travel * (geom.p_preinsert - p_lift)
    name = PHASE_NAMES[k]
    s = smoothstep(f)
    R0 = geom.R_start
    if name in ("PARK", "APPROACH", "PREGRASP_DWELL", "CLOSE", "GRASP_DWELL"):
        return make_tf(R0, p0)
    if name == "LIFT":
        return make_tf(R0, p0 + s * (p_lift - p0))
    if name == "REORIENT":
        return make_tf(slerp(R0, geom.R_stow, s), p_lift + s * (p_mid - p_lift))
    if name == "TRANSFER":
        return make_tf(geom.R_stow, p_mid + s * (geom.p_preinsert - p_mid))
    if name == "PREINSERT_DWELL":
        return make_tf(geom.R_stow, geom.p_preinsert)
    if name == "INSERT":
        return make_tf(geom.R_stow, geom.p_preinsert + s * (geom.p_insert - geom.p_preinsert))
    if name in ("INSERT_DWELL", "RELEASE"):
        return make_tf(geom.R_stow, geom.p_insert)
    return None


def hand_pose_from_object(T_WO, X_OH):
    """T_WH^d = T_WO^d X_OH."""
    return np.asarray(T_WO) @ np.asarray(X_OH)


def retreat_hand_pose(geom: StowGeometry, X_OH, f: float):
    """Hand pose during RETREAT, the insert hand pose backed out along -d by
    RETREAT_DISTANCE times the eased fraction."""
    T = hand_pose_from_object(make_tf(geom.R_stow, geom.p_insert), X_OH)
    T = T.copy()
    T[:3, 3] = T[:3, 3] - geom.d * (RETREAT_DISTANCE * smoothstep(f))
    return T


# ----------------------------------------------------------------------------
# knot schedule of the IK trajectory, (phase, fraction) pairs
# ----------------------------------------------------------------------------
KNOT_FRACTIONS = {
    "LIFT": [0.0, 0.25, 0.5, 0.75, 1.0],
    "REORIENT": [0.0, 1 / 9, 2 / 9, 3 / 9, 4 / 9, 5 / 9, 6 / 9, 7 / 9, 8 / 9, 1.0],
    "TRANSFER": [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
    "PREINSERT_DWELL": [0.0, 1.0],
    "INSERT": [0.0, 0.25, 0.5, 0.75, 1.0],
    "INSERT_DWELL": [0.0, 1.0],
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


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for k successes of n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)
