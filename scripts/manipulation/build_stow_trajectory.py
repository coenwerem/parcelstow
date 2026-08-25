"""Build the IK-verified manipulation trajectory of the ParcelStow expert
(M3), assets/parcel_stow_trajectory.json.

For every knot (phase, fraction) of the manipulation, the builder forms the
desired object pose T_WO^d(k, f) on the frozen path (geometry.object_pose),
the desired hand pose T_WH^d = T_WO^d X_OH, and solves the arm chain by
damped least squares IK seeded from the previous knot, with the hand held
at the grasp shape (open during RETREAT). Knots are denser than the probe,
LIFT 5, REORIENT 19 (5 deg steps), TRANSFER 17, PREINSERT 2, INSERT 9,
INSERT_DWELL 2, RELEASE 2, RETREAT 9. Each knot records the desired hand pose, the solved
joint target, the IK position and orientation error, and the joint-limit
margin. The object is never attached to the desired transform, the file
only shapes the commanded hand trajectory.

Run,
  python scripts/manipulation/build_stow_trajectory.py
"""

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--geometry", type=str, default="assets/parcel_stow_geometry.json")
parser.add_argument("--bank", type=str, default="assets/gdf_bank_parcel.json")
parser.add_argument("--out", type=str, default="assets/parcel_stow_trajectory.json")
parser.add_argument("--pos_tol", type=float, default=0.002)
parser.add_argument("--ori_tol_deg", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stow_common as sc  # noqa: E402
from stow_ik import ChainIK, CHAIN_NAMES  # noqa: E402

DENSE = {
    "LIFT": np.linspace(0, 1, 5), "REORIENT": np.linspace(0, 1, 19), "TRANSFER": np.linspace(0, 1, 17),
    "PREINSERT_DWELL": np.array([0.0, 1.0]), "INSERT": np.linspace(0, 1, 9), "INSERT_DWELL": np.array([0.0, 1.0]),
    "RELEASE": np.array([0.0, 1.0]),
    "RETREAT": np.linspace(0, 1, 9),
}


def main():
    gd = json.load(open(args.geometry))
    geom = sc.StowGeometry.from_dict(gd)
    X_OH = np.array(gd["X_OH"], dtype=np.float64)
    bank = json.load(open(args.bank))
    hand_grasp = bank["hand_grasp"]
    hand_open = bank["hand_pregrasp"]
    solver = ChainIK(pelvis_pos=sc.PELVIS_POS, iters=400, pos_tol=args.pos_tol, ori_tol_deg=args.ori_tol_deg)
    q = np.array([bank["candidates"][0]["q_chain_grasp"][n] for n in CHAIN_NAMES])
    knots = []
    t0 = time.time()
    worst = {"pos": 0.0, "ori": 0.0}
    min_margin = 1.0
    for name, fracs in DENSE.items():
        k = sc.PHASE_INDEX[name]
        solver.set_hand(hand_open if name == "RETREAT" else hand_grasp)
        for f in fracs:
            f = float(f)
            if name == "RETREAT":
                T_WH = sc.retreat_hand_pose(geom, X_OH, f)
                T_WO = None
            else:
                T_WO = sc.object_pose(geom, k, f)
                T_WH = sc.hand_pose_from_object(T_WO, X_OH)
            r = solver.solve(T_WH, q)
            if not r["ok"]:
                r = solver.solve_multi(T_WH, [q], n_random=6, scale=0.3)
            q = r["q"]
            margin, per = solver.limit_margin(q)
            worst["pos"] = max(worst["pos"], r["pos_err"])
            worst["ori"] = max(worst["ori"], r["ori_err_deg"])
            min_margin = min(min_margin, margin)
            knots.append({
                "phase": name, "k": k, "f": f,
                "T_WO": T_WO.tolist() if T_WO is not None else None,
                "T_WH": T_WH.tolist(),
                "q_chain": {n: float(v) for n, v in zip(CHAIN_NAMES, q)},
                "ik_ok": bool(r["ok"]), "pos_err_m": r["pos_err"], "ori_err_deg": r["ori_err_deg"],
                "limit_margin": margin, "binding_joint": CHAIN_NAMES[int(np.argmin(per))],
            })
            print(f"[{name:16s} f {f:.3f}] ok {r['ok']} pos {r['pos_err']*1e3:.2f} mm ori {r['ori_err_deg']:.2f} deg "
                  f"margin {margin:.3f} ({knots[-1]['binding_joint']})", flush=True)
    out = {
        "geometry_file": os.path.abspath(args.geometry), "bank_file": os.path.abspath(args.bank),
        "X_OH": X_OH.tolist(), "hand_grasp": hand_grasp, "hand_open": hand_open,
        "chain_joint_names": CHAIN_NAMES, "pos_tol_m": args.pos_tol, "ori_tol_deg": args.ori_tol_deg,
        "worst_pos_err_m": worst["pos"], "worst_ori_err_deg": worst["ori"], "min_limit_margin": min_margin,
        "all_ok": all(k["ik_ok"] for k in knots), "knots": knots,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[trajectory] {len(knots)} knots, all_ok {out['all_ok']}, worst pos {worst['pos']*1e3:.2f} mm, "
          f"worst ori {worst['ori']:.2f} deg, min margin {min_margin:.3f}, written {args.out} ({time.time()-t0:.0f}s)",
          flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
