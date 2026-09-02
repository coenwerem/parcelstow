"""Build the IK-verified manipulation trajectory of the upright
placement expert, assets/upright_place_trajectory.json, the
build_stow_trajectory.py procedure on the upright object path.

For every knot (phase, fraction) the builder forms the desired object
pose on the frozen path, the desired hand pose T_WH = T_WO X_OH with
X_OH from the bank, and solves the arm chain by DLS IK seeded from the
previous knot, hand at the grasp shape (open during RETREAT). Each knot
records the desired hand pose, the solved joint target, the IK errors,
and the joint-limit margin. The object is never attached to the desired
transform.

Run,
  python scripts/manipulation/build_upright_trajectory.py
"""

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bank", type=str, default="assets/upright_place_bank.json")
parser.add_argument("--out", type=str, default="assets/upright_place_trajectory.json")
parser.add_argument("--pos_tol", type=float, default=0.002)
parser.add_argument("--ori_tol_deg", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parcelstow.tasks.manager_based.upright_place import geometry as ug  # noqa: E402
from stow_ik import CHAIN_NAMES, ChainIK  # noqa: E402

DENSE = {
    "LIFT": np.linspace(0, 1, 5), "REORIENT": np.linspace(0, 1, 19), "TRANSFER": np.linspace(0, 1, 17),
    "LOWER": np.linspace(0, 1, 9), "PLACE_DWELL": np.array([0.0, 1.0]), "RELEASE": np.array([0.0, 1.0]),
    "RETREAT": np.linspace(0, 1, 9),
}


def main():
    with open(args.bank) as fh:
        bank = json.load(fh)
    X_OH = np.array(bank["X_OH"], dtype=np.float64)
    hand_grasp = bank["hand_grasp"]
    hand_open = bank["hand_pregrasp"]
    solver = ChainIK(pelvis_pos=tuple(bank["pelvis_pose_world"]["pos"]), iters=400,
                     pos_tol=args.pos_tol, ori_tol_deg=args.ori_tol_deg)
    q = np.array([bank["candidates"][0]["q_chain_grasp"][n] for n in CHAIN_NAMES])
    knots = []
    t0 = time.time()
    worst = {"pos": 0.0, "ori": 0.0}
    min_margin = 1.0
    for name, fracs in DENSE.items():
        k = ug.PHASE_INDEX[name]
        solver.set_hand(hand_open if name == "RETREAT" else hand_grasp)
        for f in fracs:
            f = float(f)
            if name == "RETREAT":
                T_WH = ug.retreat_hand_pose(X_OH, f)
                T_WO = None
            else:
                p_o, R_o = ug.object_pose(k, f)
                T_WO = ug.make_tf(R_o, p_o)
                T_WH = ug.hand_pose_from_object(T_WO, X_OH)
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
            print(f"[{name:12s} f {f:.3f}] ok {r['ok']} pos {r['pos_err']*1e3:.2f} mm ori {r['ori_err_deg']:.2f} deg "
                  f"margin {margin:.3f} ({knots[-1]['binding_joint']})", flush=True)
    out = {
        "bank_file": os.path.relpath(args.bank),
        "X_OH": X_OH.tolist(), "hand_grasp": hand_grasp, "hand_open": hand_open,
        "chain_joint_names": CHAIN_NAMES, "pos_tol_m": args.pos_tol, "ori_tol_deg": args.ori_tol_deg,
        "worst_pos_err_m": worst["pos"], "worst_ori_err_deg": worst["ori"], "min_limit_margin": min_margin,
        "all_ok": all(kn["ik_ok"] for kn in knots), "knots": knots,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[trajectory] {len(knots)} knots, all_ok {out['all_ok']}, worst pos {worst['pos']*1e3:.2f} mm, "
          f"worst ori {worst['ori']:.2f} deg, min margin {min_margin:.3f}, written {args.out} "
          f"({time.time()-t0:.0f}s)", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
