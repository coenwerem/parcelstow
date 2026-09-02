"""Build the pregrasp and grasp bank of the upright placement cuboid
from its synthesis record, the build_parcel_bank.py procedure.

The synthesis record stores X_WB, the hand-root pose in the mesh frame
of the lying cuboid (long axis along mesh x, the resting pose of the
synthesis scene), and the eleven hand joint values. The task object
frame carries the long axis on z with the grasped end on +z, so
X_OH = Ry(-90) X_WB and the recorded contact x coordinates map to the
grasp shift along the shaft. The builder places the cuboid at the
frozen start pose, solves the arm chain by DLS IK for the grasp pose
T_WH = T_WO X_OH and the standoff pregrasp, and fills the planar
start-offset grid with pregrasp, grasp, and lift-end knots per entry,
for the pose-conditioned acquisition of the expert. Certificate values
are copied verbatim as provenance and enter no decision here.

Run,
  python scripts/manipulation/build_upright_bank.py \
      --record <frogger outputs>/cuboid_180x55x55.json
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--record", type=str, required=True)
parser.add_argument("--out", type=str, default="assets/upright_place_bank.json")
parser.add_argument("--standoff", type=float, default=0.10)
parser.add_argument("--grid_mm", type=float, nargs="*", default=[-15, -10, -5, 0, 5, 10, 15])
parser.add_argument("--slide_z", type=float, default=0.0,
                    help="slide the grasp along the object long axis, m; the constant "
                         "cross-section keeps the contact geometry, negative moves toward "
                         "the center of mass")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parcelstow.tasks.manager_based.upright_place import geometry as ug  # noqa: E402
from stow_ik import CHAIN_NAMES, HAND_NAME_MAP, ChainIK  # noqa: E402

PELVIS_POS = (0.0, 0.0, 0.75)


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def named(q):
    return {n: float(v) for n, v in zip(CHAIN_NAMES, q)}


def main():
    with open(args.record) as fh:
        rec = json.load(fh)
    # The synthesized contact points live in the rank-0 rich sibling record.
    rec_dir = os.path.dirname(args.record)
    riches = sorted(f for f in os.listdir(rec_dir) if "__g00" in f and f.endswith(".json"))
    rich0 = {}
    if riches:
        with open(os.path.join(rec_dir, riches[0])) as fh:
            rich0 = json.load(fh)
    # mesh frame (lying, long axis x) -> object frame (long axis z, +z top)
    X_OH = ug.make_tf(ug.roty(-math.pi / 2)) @ np.array(rec["X_WB"], dtype=np.float64)
    if args.slide_z:
        X_OH = ug.make_tf(p=[0.0, 0.0, args.slide_z]) @ X_OH
    pts_mesh = np.array(rec.get("contact_points_mm") or rich0.get("contact_points_mm") or [],
                        dtype=np.float64)
    grasp_shift = float(pts_mesh[:, 0].mean() * 1e-3) + args.slide_z if len(pts_mesh) else None
    hand_grasp = {HAND_NAME_MAP[n]: float(v) for n, v in zip(rec["dof_order"], rec["q"])}
    solver = ChainIK(pelvis_pos=PELVIS_POS, iters=300)
    lower = solver.robot.data.joint_pos_limits[0, :, 0]
    hand_ids, _ = solver.robot.find_joints(list(hand_grasp.keys()), preserve_order=True)
    hand_open = {n: float(min(lower[i].item(), 0.0)) for n, i in zip(hand_grasp.keys(), hand_ids)}
    q_seed = np.array([rec["q_arm"][n] for n in CHAIN_NAMES])
    q_default = solver.robot.data.default_joint_pos[0, solver.chain_ids].cpu().numpy()
    p0 = np.asarray(ug.START_POS)

    def solve_pair(T_WO, seed, multi=False):
        T_grasp = ug.hand_pose_from_object(T_WO, X_OH)
        solver.set_hand(hand_grasp)
        r_g = solver.solve_multi(T_grasp, [seed, q_default], n_random=12) if multi else solver.solve(T_grasp, seed)
        hand_z_g = float(min(v[2] for v in solver.body_positions().values()))
        ray = T_grasp[:3, 3] - T_WO[:3, 3]
        ray = ray / np.linalg.norm(ray)
        T_pre = T_grasp.copy()
        T_pre[:3, 3] = T_grasp[:3, 3] + args.standoff * ray
        solver.set_hand(hand_open)
        r_p = solver.solve(T_pre, r_g["q"])
        hand_z_p = float(min(v[2] for v in solver.body_positions().values()))
        return r_g, r_p, hand_z_g, hand_z_p

    t0 = time.time()
    T_WO0 = ug.make_tf(ug.R_START, p0)
    r_g, r_p, hz_g, hz_p = solve_pair(T_WO0, q_seed, multi=True)
    ok = r_g["ok"] and r_p["ok"] and hz_p > ug.TABLE_TOP + 0.02
    print(f"[grasp] ok {r_g['ok']} ({r_g['pos_err']*1e3:.1f} mm, {r_g['ori_err_deg']:.1f} deg) "
          f"pregrasp ok {r_p['ok']} ({r_p['pos_err']*1e3:.1f} mm) hand z grasp {hz_g:.3f} pre {hz_p:.3f} "
          f"margin {solver.limit_margin(r_g['q'])[0]:.3f} admitted {ok}", flush=True)
    if not ok:
        raise SystemExit("grasp candidate not admitted")
    candidates = [{
        "yaw_deg": 0.0, "q_chain": named(r_p["q"]), "q_chain_grasp": named(r_g["q"]),
        "pre_pos_error_m": r_p["pos_err"], "grasp_pos_error_m": r_g["pos_err"],
        "pre_ori_error_deg": r_p["ori_err_deg"], "grasp_ori_error_deg": r_g["ori_err_deg"],
        "limit_margin_grasp": solver.limit_margin(r_g["q"])[0],
        "limit_margin_pre": solver.limit_margin(r_p["q"])[0],
        "min_hand_body_z_grasp": hz_g,
    }]
    # planar start-offset grid with the lift-end knot
    grid = []
    q_g0 = r_g["q"].copy()
    for dx_mm in args.grid_mm:
        for dy_mm in args.grid_mm:
            p = p0 + np.array([dx_mm * 1e-3, dy_mm * 1e-3, 0.0])
            T_WO = ug.make_tf(ug.R_START, p)
            r_gi, r_pi, hz_gi, hz_pi = solve_pair(T_WO, q_g0)
            solver.set_hand(hand_grasp)
            T_lift = ug.hand_pose_from_object(ug.make_tf(ug.R_START, p + [0.0, 0.0, ug.LIFT_DZ]), X_OH)
            r_l = solver.solve(T_lift, r_gi["q"])
            okg = r_gi["ok"] and r_pi["ok"] and r_l["ok"]
            grid.append({"dx": dx_mm * 1e-3, "dy": dy_mm * 1e-3, "ok": bool(okg),
                         "q_chain": named(r_pi["q"]), "q_chain_grasp": named(r_gi["q"]),
                         "q_chain_lift": named(r_l["q"]),
                         "pre_pos_error_m": r_pi["pos_err"], "grasp_pos_error_m": r_gi["pos_err"],
                         "lift_pos_error_m": r_l["pos_err"], "grasp_ori_error_deg": r_gi["ori_err_deg"],
                         "lift_ori_error_deg": r_l["ori_err_deg"]})
            print(f"[grid {dx_mm:+.0f} {dy_mm:+.0f}] ok {okg} grasp {r_gi['pos_err']*1e3:.1f} mm "
                  f"pre {r_pi['pos_err']*1e3:.1f} mm lift {r_l['pos_err']*1e3:.1f} mm", flush=True)
    out = {
        "object": rec.get("object"),
        "source_record": os.path.relpath(args.record),
        "source_record_sha256": sha256(args.record),
        "synthesis": {
            "tool": "frogger scripts/g1_l6_runner.py (local checkout)",
            "arguments": {"table_height_m": rec.get("table_height_m"), "object_xy": rec.get("object_xy"),
                          "riser_mm": rec.get("riser_mm"), "clearance_mm": rec.get("clearance_mm"),
                          "objective": rec.get("objective"), "grid": "thorough"},
            "scores_from_record": {"eps_nominal": rec.get("eps_nominal"),
                                   "eps_dropout_objective": rec.get("eps_dropout_objective"),
                                   "l_star": rec.get("l_star"), "n_registered": rec.get("n_registered")},
            "contact_points_mm_mesh": rec.get("contact_points_mm") or rich0.get("contact_points_mm"),
            "contact_finger_ids": rich0.get("contact_finger_ids"),
            "git_commit": rich0.get("provenance", {}).get("git_commit"),
            "note": "Certificate values describe the synthesized contact set in the synthesis tool. "
                    "They enter no task predicate.",
        },
        "X_OH": X_OH.tolist(),
        "mesh_to_object_frame": "object = Ry(90) mesh (long axis z, grasped end +z)",
        "grasp_slide_m": args.slide_z,
        "grasp_shift_measured_m": grasp_shift,
        "grasp_shift_frozen_m": ug.GRASP_SHIFT,
        "object_pose_world": {"pos": p0.tolist(), "quat_wxyz": [float(v) for v in ug.quat_from_mat(ug.R_START)]},
        "pelvis_pose_world": {"pos": list(PELVIS_POS), "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
        "chain_joint_names": CHAIN_NAMES,
        "hand_joint_names": list(hand_grasp.keys()),
        "hand_grasp": hand_grasp,
        "hand_pregrasp": hand_open,
        "candidates": candidates,
        "grid": {"dx_dy_mm": args.grid_mm, "entries": grid},
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[bank] {len(candidates)} candidate, {sum(g['ok'] for g in grid)}/{len(grid)} grid entries, "
          f"grasp shift {grasp_shift}, written {args.out} in {time.time()-t0:.0f}s", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
