"""Kinematic feasibility probe of the ParcelStow geometry (M3).

The probe loads the parcel synthesis record, and for every candidate task
configuration (start yaw of the parcel on the table, reorientation family,
receptacle placement, shelf height) it solves the grasp and standoff
pregrasp by DLS IK at the start pose (several seeds), verifies the X_WB
convention numerically (the recovered object pose and the distal frames of
the solved grasp near the recorded contact points), and then solves the
full manipulation knot list, lift, reorientation, transfer, pre-insertion,
insertion, release, and retreat, with the hand held at the grasp shape.
Each candidate reports IK errors, the joint-limit margin (min over knots),
hand-table clearance, hand-receptacle clearance, the palm-trails condition,
the release clearance (no phalanx under the parcel bottom face at release),
and the transport distance. The ranking uses these kinematic criteria only,
never a learner. --finalize writes assets/parcel_stow_geometry.json for the
chosen candidate.

Run,
  python scripts/manipulation/probe_stow_geometry.py --out outputs/paper/geometry_probe.json
  python scripts/manipulation/probe_stow_geometry.py --finalize C --start_yaw -90 --entrance 0.40 -0.30 --shelf_height 0.10
"""

import argparse
import json
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--record", type=str,
                    default="assets/provenance/frogger_parcel/scene_lab_x0.35_riser15/parcel_80x55x40.json")
parser.add_argument("--out", type=str, default="outputs/paper/geometry_probe.json")
parser.add_argument("--families", type=str, nargs="*", default=["C"])
parser.add_argument("--start_yaws", type=float, nargs="*", default=[0.0, 45.0, 90.0, 135.0, 180.0, -45.0, -90.0, -135.0])
parser.add_argument("--along", type=float, nargs="*", default=[0.14, 0.20, 0.26],
                    help="entrance distance from the start along the insertion axis, m")
parser.add_argument("--lateral", type=float, nargs="*", default=[-0.08, 0.0, 0.08],
                    help="entrance offset across the insertion axis (z cross d), m")
parser.add_argument("--shelf_heights", type=float, nargs="*", default=[0.0, 0.10])
parser.add_argument("--lift_dxs", type=float, nargs="*", default=[0.0])
parser.add_argument("--lift_dzs", type=float, nargs="*", default=[0.08])
parser.add_argument("--travels", type=float, nargs="*", default=[0.0])
parser.add_argument("--lift_dz", type=float, default=0.08)
parser.add_argument("--travel", type=float, default=0.0)
parser.add_argument("--grasp_yaw", type=float, default=0.0, help="C2 element applied to the record")
parser.add_argument("--iters", type=int, default=150)
parser.add_argument("--grasp_branches", type=int, default=1,
                    help="number of distinct grasp IK solutions (arm branches) to evaluate per start yaw")
parser.add_argument("--null_gain", type=float, default=0.3)
parser.add_argument("--knot_log", action="store_true")
parser.add_argument("--finalize", type=str, default=None, help="family letter to freeze")
parser.add_argument("--start_yaw", type=float, default=None)
parser.add_argument("--entrance", type=float, nargs=2, default=None, help="entrance xy (world) to freeze")
parser.add_argument("--shelf_height", type=float, default=0.0)
parser.add_argument("--lift_dx", type=float, default=0.0)
parser.add_argument("--geometry_out", type=str, default="assets/parcel_stow_geometry.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stow_common as sc  # noqa: E402
from stow_ik import ChainIK, CHAIN_NAMES, HAND_NAME_MAP  # noqa: E402

STANDOFF = 0.10
BODY_RADIUS = {"rh_hand_base_link": 0.025}
DEFAULT_RADIUS = 0.010
ARM_BODIES = ["right_wrist_yaw_link", "right_wrist_pitch_link", "right_wrist_roll_link", "right_elbow_link"]
ARM_RADIUS = 0.04

# family: (rot_axis, rot_deg, insert_axis in the task frame)
FAMILIES = {
    "A": ("x", -90.0, "-y"),
    "B": ("x", 90.0, "+y"),
    "C": ("y", -90.0, "+x"),
    "D": ("y", 90.0, "-x"),
}
TIP_NAMES = ["rh_thumb_tip", "rh_index_tip", "rh_middle_tip", "rh_ring_tip", "rh_pinky_tip"]


def entrance_from(start_yaw, insert_axis, along, lateral):
    R = sc.rotz(math.radians(start_yaw))
    d = R @ sc.UNIT[insert_axis]
    lat = np.cross(np.array([0.0, 0.0, 1.0]), d)
    e = np.array(sc.PARCEL_START) + d * along + lat * lateral
    return (float(e[0]), float(e[1]))


def main():
    rec = json.load(open(args.record))
    X_OH = sc.make_tf(sc.rotz(math.radians(args.grasp_yaw))) @ np.array(rec["X_WB"], dtype=np.float64)
    hand_grasp = {HAND_NAME_MAP[n]: float(v) for n, v in zip(rec["dof_order"], rec["q"])}
    solver = ChainIK(pelvis_pos=sc.PELVIS_POS, iters=args.iters, null_gain=args.null_gain)
    lower = solver.robot.data.joint_pos_limits[0, :, 0]
    hand_ids, _ = solver.robot.find_joints(list(hand_grasp.keys()), preserve_order=True)
    hand_open = {n: float(min(lower[i].item(), 0.0)) for n, i in zip(hand_grasp.keys(), hand_ids)}
    q_seed = np.array([rec["q_arm"][n] for n in CHAIN_NAMES])
    q_default = solver.robot.data.default_joint_pos[0, solver.chain_ids].cpu().numpy()
    riches = sorted(f for f in os.listdir(os.path.dirname(args.record)) if "__g00" in f and f.endswith(".json"))
    rich = json.load(open(os.path.join(os.path.dirname(args.record), riches[0]))) if riches else None
    p_obj = X_OH[:3, 3] / np.linalg.norm(X_OH[:3, 3])
    knots = sc.knot_list()
    report = {"record": args.record, "grasp_yaw": args.grasp_yaw, "X_OH": X_OH.tolist(), "hand_grasp": hand_grasp,
              "hand_open": hand_open, "starts": {}, "candidates": []}
    t0 = time.time()

    if args.finalize:
        combos = [(args.start_yaw, args.finalize, args.entrance, args.shelf_height, args.lift_dx, b, args.lift_dz, args.travel)
                  for b in range(args.grasp_branches)]
    else:
        combos = []
        for sy in args.start_yaws:
            for br in range(args.grasp_branches):
                for fam in args.families:
                    for h in args.shelf_heights:
                        for dx in args.lift_dxs:
                            for dz in args.lift_dzs:
                                for tv in args.travels:
                                    for a in args.along:
                                        for b in args.lateral:
                                            combos.append((sy, fam, entrance_from(sy, FAMILIES[fam][2], a, b), h, dx, br, dz, tv))

    grasp_cache = {}
    for (sy, fam, exy, h, dx, br, dz, tv) in combos:
        rot_axis, rot_deg, ins_axis = FAMILIES[fam]
        geom = sc.StowGeometry(rot_axis, rot_deg, ins_axis, exy, family=fam, shelf_height=h, lift_dx=dx,
                               start_yaw_deg=sy, lift_dz=dz, reorient_travel=tv)
        # -------- grasp and pregrasp at this start yaw (cached) --------
        if sy not in grasp_cache:
            T_WO0 = sc.make_tf(geom.R_start, geom.p_start)
            solver.set_hand(hand_grasp)
            T_grasp = sc.hand_pose_from_object(T_WO0, X_OH)
            sols = solver.solve_multi(T_grasp, [q_seed, q_default], n_random=16, return_all=True)
            # distinct arm branches among the converged solutions
            branches = []
            for r_i in sols:
                if not r_i["ok"]:
                    continue
                if all(np.linalg.norm(r_i["q"] - b["q"]) > 0.3 for b in branches):
                    branches.append(r_i)
                if len(branches) >= args.grasp_branches:
                    break
            if not branches:
                branches = [sols[0]]
            r_g = branches[0]
            solver.settle(r_g["q"])
            T_back = solver.hand_pose() @ sc.inv_tf(X_OH)
            conv = {"recovered_object_pos_err_m": float(np.linalg.norm(T_back[:3, 3] - geom.p_start)),
                    "recovered_object_ang_err_deg": math.degrees(sc.so3_angle(geom.R_start, T_back[:3, :3]))}
            tips = solver.body_positions(TIP_NAMES)
            T_OW = sc.inv_tf(T_WO0)
            tip_check = {}
            if rich is not None:
                Ry = sc.rotz(math.radians(args.grasp_yaw))
                for fid, c in zip(rich["contact_finger_ids"], rich["contact_points_mm"]):
                    n = TIP_NAMES[fid]
                    tip_obj = (T_OW @ np.append(tips[n], 1.0))[:3] * 1e3
                    c_yaw = Ry @ np.array(c)
                    tip_check[n] = {"tip_obj_mm": tip_obj.round(1).tolist(), "contact_obj_mm": c_yaw.round(1).tolist(),
                                    "dist_mm": float(np.linalg.norm(tip_obj - c_yaw))}
            hz_g = float(min(v[2] for v in solver.body_positions().values()))
            m_g, per = solver.limit_margin(r_g["q"])
            ray = T_grasp[:3, 3] - geom.p_start
            ray = ray / np.linalg.norm(ray)
            T_pre = T_grasp.copy()
            T_pre[:3, 3] = T_grasp[:3, 3] + STANDOFF * ray
            solver.set_hand(hand_open)
            r_p = solver.solve_multi(T_pre, [r_g["q"]], n_random=6)
            m_p, _ = solver.limit_margin(r_p["q"])
            pre_list = []
            for b_i in branches:
                solver.set_hand(hand_open)
                pre_list.append(solver.solve_multi(T_pre, [b_i["q"]], n_random=6))
            grasp_cache[sy] = (branches, pre_list)
            report["starts"][str(sy)] = {
                "n_branches": len(branches),
                "branches": [{"q_chain": {n: float(v) for n, v in zip(CHAIN_NAMES, b_i["q"])},
                              "limit_margin": solver.limit_margin(b_i["q"])[0], "pos_err_m": b_i["pos_err"]}
                             for b_i in branches],
                "grasp": {"ok": bool(r_g["ok"]), "pos_err_m": r_g["pos_err"], "ori_err_deg": r_g["ori_err_deg"],
                          "limit_margin": m_g, "binding_joint": CHAIN_NAMES[int(np.argmin(per))],
                          "q_chain": {n: float(v) for n, v in zip(CHAIN_NAMES, r_g["q"])}, "min_hand_body_z": hz_g},
                "pregrasp": {"ok": bool(r_p["ok"]), "pos_err_m": r_p["pos_err"], "ori_err_deg": r_p["ori_err_deg"],
                             "limit_margin": m_p, "q_chain": {n: float(v) for n, v in zip(CHAIN_NAMES, r_p["q"])}},
                "convention_check": conv, "tip_vs_recorded_contact": tip_check,
            }
            print(f"[start yaw {sy:+.0f}] grasp ok {r_g['ok']} {r_g['pos_err']*1e3:.1f} mm {r_g['ori_err_deg']:.1f} deg "
                  f"margin {m_g:.3f} hz {hz_g:.3f} | pregrasp ok {r_p['ok']} {r_p['pos_err']*1e3:.1f} mm margin {m_p:.3f} "
                  f"| convention pos {conv['recovered_object_pos_err_m']*1e3:.2f} mm ang {conv['recovered_object_ang_err_deg']:.2f} deg",
                  flush=True)
        branches, pre_list = grasp_cache[sy]
        if br >= len(branches):
            continue
        r_g, r_p = branches[br], pre_list[br]
        if not r_g["ok"]:
            print(f"[start yaw {sy:+.0f} {fam}] grasp infeasible, skipping", flush=True)
            continue
        solver.set_hand(hand_grasp)
        palm_dot = float(np.dot(geom.R_stow @ p_obj, geom.d))
        q = r_g["q"].copy()
        rows = []
        feasible = True
        worst_pos = worst_ori = 0.0
        min_margin = min(r_g["ok"] and solver.limit_margin(r_g["q"])[0], solver.limit_margin(r_p["q"])[0])
        min_hand_z = 10.0
        min_wall = 10.0
        release_clear = 10.0
        for (k, f) in knots:
            name = sc.PHASE_NAMES[k]
            if name == "RETREAT":
                T_WH = sc.retreat_hand_pose(geom, X_OH, f)
            else:
                T_WH = sc.hand_pose_from_object(sc.object_pose(geom, k, f), X_OH)
            r = solver.solve(T_WH, q)
            q = r["q"]
            margin, per_joint = solver.limit_margin(q)
            binding = CHAIN_NAMES[int(np.argmin(per_joint))]
            pos = solver.body_positions(ARM_BODIES)
            hz = float(min(p[2] for n, p in pos.items() if n.startswith("rh_")))
            walls = {n: geom.slab_clearance(p, BODY_RADIUS.get(n, DEFAULT_RADIUS if n.startswith("rh_") else ARM_RADIUS))
                     for n, p in pos.items()}
            worst_body = min(walls, key=lambda n: walls[n][0])
            wall = float(walls[worst_body][0])
            worst_slab = walls[worst_body][1]
            if name == "RELEASE":
                bottom = geom.z_insert - 0.5 * abs((geom.R_stow @ np.array(sc.PARCEL_EXTENTS))[2])
                distal_z = [pos[n][2] for n in pos if n.endswith("_distal") or n.endswith("_tip")]
                release_clear = float(min(distal_z) - bottom)
            rows.append({"phase": name, "f": round(f, 3), "ok": bool(r["ok"]), "pos_err_mm": r["pos_err"] * 1e3,
                         "ori_err_deg": r["ori_err_deg"], "limit_margin": margin, "min_hand_z": hz,
                         "wall_clearance": wall, "hand_pos": T_WH[:3, 3].round(4).tolist(),
                         "binding_joint": binding, "q": [round(float(v), 4) for v in q],
                         "worst_body": worst_body, "worst_slab": worst_slab})
            if args.knot_log:
                print(f"    {name:16s} f {f:.2f} ok {r['ok']} pos {r['pos_err']*1e3:5.1f} ori {r['ori_err_deg']:4.1f} "
                      f"margin {margin:.3f} {binding:26s} wall {wall*1e3:6.1f} {worst_body} {worst_slab}", flush=True)
            feasible = feasible and r["ok"]
            worst_pos = max(worst_pos, r["pos_err"])
            worst_ori = max(worst_ori, r["ori_err_deg"])
            min_margin = min(min_margin, margin)
            if name != "LIFT" or f > 0.3:
                min_hand_z = min(min_hand_z, hz)
            if name in ("TRANSFER", "PREINSERT_DWELL", "INSERT", "RELEASE", "RETREAT"):
                min_wall = min(min_wall, wall)
            if r["pos_err"] > 0.03 and not args.finalize:
                # hopeless candidate, stop the knot loop early
                break
        cand = {"family": fam, "start_yaw_deg": sy, "grasp_branch": br,
                "grasp_q": {n: float(v) for n, v in zip(CHAIN_NAMES, r_g["q"])},
                "pregrasp_q": {n: float(v) for n, v in zip(CHAIN_NAMES, r_p["q"])},
                "grasp_yaw": args.grasp_yaw, "rot_axis": rot_axis,
                "rot_deg": rot_deg, "insert_axis": ins_axis, "entrance_xy": list(exy), "shelf_height": h,
                "lift_dx": dx, "lift_dz": dz, "reorient_travel": tv, "geometry": geom.to_dict(), "palm_dot_d": palm_dot,
                "grasp_ok": bool(r_g["ok"]), "pregrasp_ok": bool(r_p["ok"]),
                "ik_all_ok": feasible, "worst_pos_err_mm": worst_pos * 1e3, "worst_ori_err_deg": worst_ori,
                "min_limit_margin": min_margin, "min_hand_z_after_lift": min_hand_z,
                "table_clear": bool(min_hand_z > sc.TABLE_TOP + 0.003),
                "min_wall_clearance": min_wall, "wall_clear": bool(min_wall > 0.0),
                "release_clearance": release_clear, "release_clear": bool(release_clear > -0.005),
                "palm_trails": bool(palm_dot < -0.5),
                "transport_distance": geom.transport_distance, "knots": rows}
        cand["limit_bound_knots"] = int(sum(1 for k in rows if k["limit_margin"] < 0.005))
        cand["feasible"] = bool(feasible and r_g["ok"] and r_p["ok"] and cand["table_clear"] and cand["wall_clear"]
                                and cand["release_clear"] and cand["palm_trails"] and geom.transport_distance >= 0.15
                                and min_margin > -1e-6)
        report["candidates"].append(cand)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"[yaw {sy:+.0f} b{br} {fam} e=({exy[0]:.3f},{exy[1]:.3f}) h={h:.2f} dx={dx:.2f} dz={dz:.2f} tv={tv:.2f}] ik {feasible} pos {worst_pos*1e3:.1f}mm "
              f"ori {worst_ori:.1f}deg margin {min_margin:.3f} handz {min_hand_z:.3f} wall {min_wall*1e3:.1f}mm "
              f"release {release_clear*1e3:.1f}mm palm {palm_dot:+.2f} transport {geom.transport_distance:.3f} "
              f"FEASIBLE {cand['feasible']} ({time.time()-t0:.0f}s)", flush=True)
    finish(report)


def finish(report):
    feas = [c for c in report["candidates"] if c["feasible"]]
    feas.sort(key=lambda c: (c["limit_bound_knots"], -c["min_limit_margin"], c["worst_pos_err_mm"]))
    report["ranking"] = [{"family": c["family"], "start_yaw_deg": c["start_yaw_deg"], "grasp_branch": c["grasp_branch"],
                          "entrance_xy": c["entrance_xy"],
                          "shelf_height": c["shelf_height"], "lift_dx": c["lift_dx"], "lift_dz": c["lift_dz"],
                          "reorient_travel": c["reorient_travel"],
                          "min_limit_margin": c["min_limit_margin"], "worst_pos_err_mm": c["worst_pos_err_mm"],
                          "min_wall_clearance": c["min_wall_clearance"], "transport_distance": c["transport_distance"]}
                         for c in feas]
    print("[ranking] " + json.dumps(report["ranking"][:12], indent=1), flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"[written] {args.out}", flush=True)
    if args.finalize:
        cands = sorted(report["candidates"], key=lambda c: (not c["feasible"], c["limit_bound_knots"], -c["min_limit_margin"]))
        chosen = cands[0]
        if not chosen["feasible"]:
            print("[finalize] refused, candidate infeasible", flush=True)
        else:
            gd = chosen["geometry"]
            gd["frozen_from"] = {"probe": args.out, "record": args.record, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            gd["grasp_yaw"] = args.grasp_yaw
            gd["X_OH"] = report["X_OH"]
            gd["hand_grasp"] = report["hand_grasp"]
            gd["hand_open"] = report["hand_open"]
            gd["kinematic_summary"] = {k: v for k, v in chosen.items() if k not in ("knots", "geometry")}
            gd["kinematic_summary"]["grasp_q"] = chosen["grasp_q"]
            gd["kinematic_summary"]["pregrasp_q"] = chosen["pregrasp_q"]
            with open(args.geometry_out, "w") as fh:
                json.dump(gd, fh, indent=1)
            print(f"[finalize] wrote {args.geometry_out}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
