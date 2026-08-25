"""Scan the parcel start pose for the arm comfort of the grasp and its
standoff pregrasp (M3 support). For each candidate start (x, y) on the
table and each yaw symmetry element of the grasp record, the probe solves
the grasp and pregrasp IK from several seeds and reports the position and
orientation error, the joint-limit margin, the binding joint, and the
lowest hand body height. The choice of the start pose uses these kinematic
numbers only.

Run,
  python scripts/manipulation/probe_start_pose.py --out outputs/paper/start_pose_probe.json
"""

import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--record", type=str,
                    default="assets/provenance/frogger_parcel/scene_lab_x0.35_riser15/parcel_80x55x40.json")
parser.add_argument("--xs", type=float, nargs="*", default=[0.30, 0.35, 0.40])
parser.add_argument("--ys", type=float, nargs="*", default=[0.05, 0.0, -0.05, -0.10, -0.15])
parser.add_argument("--yaws", type=float, nargs="*", default=[0.0, 180.0])
parser.add_argument("--standoff", type=float, default=0.10)
parser.add_argument("--out", type=str, default="outputs/paper/start_pose_probe.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stow_common as sc  # noqa: E402
from stow_ik import ChainIK, CHAIN_NAMES, HAND_NAME_MAP  # noqa: E402


def main():
    rec = json.load(open(args.record))
    X_rec = np.array(rec["X_WB"], dtype=np.float64)
    hand_grasp = {HAND_NAME_MAP[n]: float(v) for n, v in zip(rec["dof_order"], rec["q"])}
    solver = ChainIK(pelvis_pos=sc.PELVIS_POS, iters=200)
    lower = solver.robot.data.joint_pos_limits[0, :, 0]
    hand_ids, _ = solver.robot.find_joints(list(hand_grasp.keys()), preserve_order=True)
    hand_open = {n: float(min(lower[i].item(), 0.0)) for n, i in zip(hand_grasp.keys(), hand_ids)}
    q_seed = np.array([rec["q_arm"][n] for n in CHAIN_NAMES])
    q_default = solver.robot.data.default_joint_pos[0, solver.chain_ids].cpu().numpy()
    rows = []
    z0 = sc.PARCEL_START[2]
    for yaw in args.yaws:
        X_OH = sc.make_tf(sc.rotz(math.radians(yaw))) @ X_rec
        for x in args.xs:
            for y in args.ys:
                p0 = np.array([x, y, z0])
                T_WO = sc.make_tf(np.eye(3), p0)
                T_g = sc.hand_pose_from_object(T_WO, X_OH)
                solver.set_hand(hand_grasp)
                r_g = solver.solve_multi(T_g, [q_seed, q_default], n_random=10)
                m_g, per = solver.limit_margin(r_g["q"])
                bind_g = CHAIN_NAMES[int(np.argmin(per))]
                hz = float(min(v[2] for v in solver.body_positions().values()))
                ray = T_g[:3, 3] - p0
                ray /= np.linalg.norm(ray)
                T_p = T_g.copy()
                T_p[:3, 3] += args.standoff * ray
                solver.set_hand(hand_open)
                r_p = solver.solve_multi(T_p, [r_g["q"]], n_random=6)
                m_p, per_p = solver.limit_margin(r_p["q"])
                row = {"yaw": yaw, "x": x, "y": y, "grasp_ok": bool(r_g["ok"]), "grasp_pos_mm": r_g["pos_err"] * 1e3,
                       "grasp_ori_deg": r_g["ori_err_deg"], "grasp_margin": m_g, "grasp_binding": bind_g,
                       "min_hand_z": hz, "pre_ok": bool(r_p["ok"]), "pre_pos_mm": r_p["pos_err"] * 1e3,
                       "pre_margin": m_p, "pre_binding": CHAIN_NAMES[int(np.argmin(per_p))],
                       "q_grasp": {n: float(v) for n, v in zip(CHAIN_NAMES, r_g["q"])}}
                rows.append(row)
                print(f"[yaw {yaw:.0f} x {x:.2f} y {y:+.2f}] grasp ok {r_g['ok']} {r_g['pos_err']*1e3:.1f}mm "
                      f"{r_g['ori_err_deg']:.1f}deg margin {m_g:.3f} ({bind_g}) hz {hz:.3f} | pre ok {r_p['ok']} "
                      f"{r_p['pos_err']*1e3:.1f}mm margin {m_p:.3f}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"record": args.record, "rows": rows}, open(args.out, "w"), indent=1)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
