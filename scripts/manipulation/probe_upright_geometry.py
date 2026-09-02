"""Kinematic feasibility probe of the upright placement geometry.

For every candidate task configuration (start yaw of the lying cuboid,
target-region placement, lift height) the probe solves the grasp and a
standoff pregrasp by DLS IK at the start pose, then solves the full
manipulation knot list, lift, reorientation to vertical, transfer,
lower, place dwell, release, and retreat, with the hand held at the
grasp shape. Each candidate reports IK errors, the joint-limit margin
(min over knots), the binding joint, hand-table clearance, and the
hand-object clearance during retreat. The ranking uses these kinematic
criteria only, never a learner, the v1 protocol of
probe_stow_geometry.py.

Provisional grasp hypothesis: the frozen v1 acquisition hand pose
(assets/parcel_stow_geometry.json, X_OH on the 80 x 55 x 40 mm parcel)
re-expressed in the upright object's frame at the shared start pose.
Both objects start lying at (0.35, 0, 0.721) with the grasped width
horizontal, so the v1 hand pose is a reachable side grasp of the
cuboid shaft; the FRoGGeR bank replaces this hypothesis before any
expert or learner runs.

Run,
  python scripts/manipulation/probe_upright_geometry.py --out outputs/probe/upright_probe.json
"""

import argparse
import json
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--geometry", type=str, default="assets/parcel_stow_geometry.json",
                    help="frozen v1 geometry supplying the grasp hypothesis")
parser.add_argument("--bank", type=str, default=None,
                    help="probe the X_OH, hand shapes, and grasp seed of a built bank "
                         "instead of the v1-derived hypothesis")
parser.add_argument("--out", type=str, default="outputs/probe/upright_probe.json")
parser.add_argument("--start_yaws", type=float, nargs="*",
                    default=[0.0, 45.0, 90.0, 135.0, 180.0, -45.0, -90.0, -135.0])
parser.add_argument("--along", type=float, nargs="*", default=[0.14, 0.20, 0.26],
                    help="target distance from the start along the yawed +x axis, m")
parser.add_argument("--lateral", type=float, nargs="*", default=[-0.08, 0.0, 0.08],
                    help="target offset across the transport direction (z cross d), m")
parser.add_argument("--lift_dzs", type=float, nargs="*", default=[0.12])
parser.add_argument("--goal_yaw_offsets", type=float, nargs="*", default=[0.0],
                    help="goal yaw of the standing object relative to the start yaw, deg")
parser.add_argument("--grasp_shifts", type=float, nargs="*", default=[0.0],
                    help="grasp point offset along the object long axis, m; positive "
                         "toward the future top end, raising the hand at placement")
parser.add_argument("--iters", type=int, default=150)
parser.add_argument("--null_gain", type=float, default=0.3)
parser.add_argument("--knot_log", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stow_common as sc  # noqa: E402
from parcelstow.tasks.manager_based.upright_place import geometry as ug  # noqa: E402
from stow_ik import CHAIN_NAMES, ChainIK  # noqa: E402

STANDOFF = 0.10  # pregrasp standoff along the grasp ray, the v1 value
TABLE_CLEAR_MIN = 0.003  # min hand-body height above the table after lift, the v1 value
OBJECT_HALF = np.array(ug.OBJECT_EXTENTS) / 2.0


def target_from(start_yaw, along, lateral):
    d = sc.rotz(math.radians(start_yaw)) @ np.array([1.0, 0.0, 0.0])
    lat = np.cross(np.array([0.0, 0.0, 1.0]), d)
    t = np.array(ug.START_POS) + d * along + lat * lateral
    return (float(t[0]), float(t[1]))


def object_clearance(p_world, frames, radius):
    """Signed distance of a body sphere from the placed object box, m."""
    T_OW = ug.inv_tf(ug.make_tf(frames["R_upright"], frames["p_place"]))
    q = (T_OW @ np.append(p_world, 1.0))[:3]
    outside = np.maximum(np.abs(q) - OBJECT_HALF, 0.0)
    dist = float(np.linalg.norm(outside))
    if dist == 0.0:
        dist = float(np.max(np.abs(q) - OBJECT_HALF))  # negative, inside the box
    return dist - radius


def main():
    if args.bank:
        with open(args.bank) as fh:
            bank = json.load(fh)
        X_OH = np.array(bank["X_OH"], dtype=np.float64)
        hand_grasp = {n: float(v) for n, v in bank["hand_grasp"].items()}
        hand_open = {n: float(v) for n, v in bank["hand_pregrasp"].items()}
        q_seed = np.array([bank["candidates"][0]["q_chain_grasp"][n] for n in CHAIN_NAMES])
        hypothesis = f"bank X_OH from {args.bank}"
    else:
        with open(args.geometry) as fh:
            gd = json.load(fh)
        X_OH_parcel = np.array(gd["X_OH"], dtype=np.float64)
        T_WH_v1 = ug.make_tf(np.array(gd["R_start"]), np.array(gd["p_start"])) @ X_OH_parcel
        fr_nominal = ug.path_frames(start_yaw_deg=float(gd["start_yaw_deg"]))
        T_WO_up = ug.make_tf(fr_nominal["R_start"], fr_nominal["p_start"])
        X_OH = ug.grasp_in_object_frame(T_WO_up, T_WH_v1)
        hand_grasp = {n: float(v) for n, v in gd["hand_grasp"].items()}
        hand_open = {n: float(v) for n, v in gd["hand_open"].items()}
        q_seed = np.array([gd["kinematic_summary"]["grasp_q"][n] for n in CHAIN_NAMES])
        hypothesis = "v1 acquisition hand pose in the upright object frame"

    solver = ChainIK(pelvis_pos=sc.PELVIS_POS, iters=args.iters, null_gain=args.null_gain)
    q_default = solver.robot.data.default_joint_pos[0, solver.chain_ids].cpu().numpy()
    knots = ug.knot_list()
    report = {"geometry": args.geometry, "bank": args.bank, "X_OH_provisional": X_OH.tolist(),
              "grasp_hypothesis": hypothesis,
              "hand_grasp": hand_grasp, "hand_open": hand_open, "starts": {}, "candidates": []}
    t0 = time.time()

    combos = [(sy, target_from(sy, a, b), dz, sy + gy, g)
              for sy in args.start_yaws for dz in args.lift_dzs
              for gy in args.goal_yaw_offsets for g in args.grasp_shifts
              for a in args.along for b in args.lateral]

    grasp_cache = {}
    for (sy, txy, dz, gy, g) in combos:
        frames = ug.path_frames(start_yaw_deg=sy, target_center=txy, lift_dz=dz,
                                goal_yaw_deg=gy)
        X_OH_g = ug.make_tf(p=[0.0, 0.0, g]) @ X_OH
        # -------- grasp and pregrasp at this start yaw and shift (cached) --------
        if (sy, g) not in grasp_cache:
            T_grasp = ug.hand_pose_from_object(ug.make_tf(frames["R_start"], frames["p_start"]), X_OH_g)
            solver.set_hand(hand_grasp)
            r_g = solver.solve_multi(T_grasp, [q_seed, q_default], n_random=16)
            m_g, per = solver.limit_margin(r_g["q"])
            ray = T_grasp[:3, 3] - frames["p_start"]
            ray = ray / np.linalg.norm(ray)
            T_pre = T_grasp.copy()
            T_pre[:3, 3] = T_grasp[:3, 3] + STANDOFF * ray
            solver.set_hand(hand_open)
            r_p = solver.solve_multi(T_pre, [r_g["q"]], n_random=6)
            m_p, _ = solver.limit_margin(r_p["q"])
            grasp_cache[(sy, g)] = (r_g, r_p)
            report["starts"][f"{sy}_g{g}"] = {
                "grasp": {"ok": bool(r_g["ok"]), "pos_err_m": r_g["pos_err"],
                          "ori_err_deg": r_g["ori_err_deg"], "limit_margin": m_g,
                          "binding_joint": CHAIN_NAMES[int(np.argmin(per))],
                          "q_chain": {n: float(v) for n, v in zip(CHAIN_NAMES, r_g["q"])}},
                "pregrasp": {"ok": bool(r_p["ok"]), "pos_err_m": r_p["pos_err"],
                             "ori_err_deg": r_p["ori_err_deg"], "limit_margin": m_p},
            }
            print(f"[start yaw {sy:+.0f} shift {g:+.3f}] grasp ok {r_g['ok']} {r_g['pos_err']*1e3:.1f} mm "
                  f"{r_g['ori_err_deg']:.1f} deg margin {m_g:.3f} | pregrasp ok {r_p['ok']} "
                  f"{r_p['pos_err']*1e3:.1f} mm margin {m_p:.3f}", flush=True)
        r_g, r_p = grasp_cache[(sy, g)]
        if not r_g["ok"]:
            print(f"[start yaw {sy:+.0f} shift {g:+.3f}] grasp infeasible, skipping candidates", flush=True)
            continue
        solver.set_hand(hand_grasp)
        q = r_g["q"].copy()
        rows = []
        feasible = True
        worst_pos = worst_ori = 0.0
        min_margin = min(solver.limit_margin(r_g["q"])[0], solver.limit_margin(r_p["q"])[0])
        min_hand_z = 10.0
        min_obj_clear = 10.0
        for (k, f) in knots:
            name = ug.PHASES[k][0]
            if name == "RETREAT":
                T_WH = ug.retreat_hand_pose(X_OH_g, f, frames=frames)
            else:
                p_o, R_o = ug.object_pose(k, f, frames=frames)
                T_WH = ug.hand_pose_from_object(ug.make_tf(R_o, p_o), X_OH_g)
            r = solver.solve(T_WH, q)
            q = r["q"]
            margin, per_joint = solver.limit_margin(q)
            binding = CHAIN_NAMES[int(np.argmin(per_joint))]
            pos = solver.body_positions()
            hz = float(min(p[2] for n, p in pos.items() if n.startswith("rh_")))
            oc = 10.0
            if name == "RETREAT" and f >= 0.5:
                oc = float(min(object_clearance(p, frames, 0.010) for p in pos.values()))
            rows.append({"phase": name, "f": round(f, 3), "ok": bool(r["ok"]),
                         "pos_err_mm": r["pos_err"] * 1e3, "ori_err_deg": r["ori_err_deg"],
                         "limit_margin": margin, "binding_joint": binding, "min_hand_z": hz,
                         "object_clearance": None if oc == 10.0 else oc,
                         "q": [round(float(v), 4) for v in q]})
            if args.knot_log:
                print(f"    {name:12s} f {f:.2f} ok {r['ok']} pos {r['pos_err']*1e3:5.1f} "
                      f"ori {r['ori_err_deg']:4.1f} margin {margin:.3f} {binding}", flush=True)
            feasible = feasible and r["ok"]
            worst_pos = max(worst_pos, r["pos_err"])
            worst_ori = max(worst_ori, r["ori_err_deg"])
            min_margin = min(min_margin, margin)
            if name != "LIFT" or f > 0.3:
                min_hand_z = min(min_hand_z, hz)
            min_obj_clear = min(min_obj_clear, oc)
            if r["pos_err"] > 0.03:
                break  # hopeless candidate, stop the knot loop early
        cand = {"start_yaw_deg": sy, "target_center": list(txy), "lift_dz": dz,
                "goal_yaw_deg": gy, "grasp_shift": g,
                "transport_distance": float(np.linalg.norm(np.array(txy) - np.array(ug.START_POS[:2]))),
                "grasp_ok": bool(r_g["ok"]), "pregrasp_ok": bool(r_p["ok"]),
                "ik_all_ok": feasible, "worst_pos_err_mm": worst_pos * 1e3,
                "worst_ori_err_deg": worst_ori, "min_limit_margin": min_margin,
                "min_hand_z_after_lift": min_hand_z,
                "table_clear": bool(min_hand_z > ug.TABLE_TOP + TABLE_CLEAR_MIN),
                "min_retreat_object_clearance": min_obj_clear, "knots": rows}
        cand["limit_bound_knots"] = int(sum(1 for row in rows if row["limit_margin"] < 0.005))
        cand["feasible"] = bool(feasible and r_g["ok"] and r_p["ok"] and cand["table_clear"]
                                and min_margin > -1e-6
                                and cand["transport_distance"] >= 0.15)  # the v1 minimum transport
        report["candidates"].append(cand)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"[yaw {sy:+.0f} gy={gy:+.0f} g={g:+.3f} t=({txy[0]:.3f},{txy[1]:.3f}) dz={dz:.2f}] ik {feasible} "
              f"pos {worst_pos*1e3:.1f}mm ori {worst_ori:.1f}deg margin {min_margin:.3f} "
              f"handz {min_hand_z:.3f} objclear {min_obj_clear*1e3:.1f}mm "
              f"FEASIBLE {cand['feasible']} ({time.time()-t0:.0f}s)", flush=True)

    feas = [c for c in report["candidates"] if c["feasible"]]
    feas.sort(key=lambda c: (c["limit_bound_knots"], -c["min_limit_margin"], c["worst_pos_err_mm"]))
    report["ranking"] = [{k: c[k] for k in ("start_yaw_deg", "goal_yaw_deg", "grasp_shift",
                                            "target_center", "lift_dz", "min_limit_margin",
                                            "worst_pos_err_mm", "min_retreat_object_clearance",
                                            "transport_distance")}
                        for c in feas]
    print("[ranking] " + json.dumps(report["ranking"][:12], indent=1), flush=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"[written] {args.out}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
