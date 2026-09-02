"""Build the GDF pregrasp and grasp bank of the parcel from its synthesis
record, the procedure of scripts/tools/build_gdf_bank.py applied to the
parcel (M2).

The record stores X_WB, the hand-root pose in the object frame, and the
eleven hand joint values of the synthesized grasp. The builder places the
parcel at the frozen start pose, solves the arm chain by damped least
squares IK for the grasp pose T_WH = T_WO X_WB and for the standoff
pregrasp (the grasp pose translated STANDOFF along the ray from the parcel
center with the hand open), for the yaw symmetry set {0, 180} deg of the
resting cuboid, and admits a candidate when both solves converge inside the
joint limits with the hand above the table. A second section holds the
planar start-offset grid (dx, dy) of TASK_SPEC.md section 11.1 with the
pregrasp, grasp, and lift-end knots per entry, for the pose-conditioned
acquisition of the expert. The certificate values of the record are copied
verbatim as provenance and enter no decision here.

Run,
  python scripts/manipulation/build_parcel_bank.py \
      --record assets/provenance/frogger_parcel/scene_lab_x0.35_riser15/parcel_80x55x40.json
"""

import argparse
import hashlib
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--record", type=str, required=True)
parser.add_argument("--out", type=str, default="assets/gdf_bank_parcel.json")
parser.add_argument("--standoff", type=float, default=0.10)
parser.add_argument("--grid_mm", type=float, nargs="*", default=[-15, -10, -5, 0, 5, 10, 15])
parser.add_argument("--yaws", type=float, nargs="*", default=[0.0, 180.0])
parser.add_argument("--geometry", type=str, default="assets/parcel_stow_geometry.json",
                    help="frozen geometry (start orientation and the C2 element of the record)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stow_common as sc  # noqa: E402
from stow_ik import ChainIK, CHAIN_NAMES, HAND_NAME_MAP  # noqa: E402


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def named(q):
    return {n: float(v) for n, v in zip(CHAIN_NAMES, q)}


def main():
    rec = json.load(open(args.record))
    rec_dir = os.path.dirname(args.record)
    gd = json.load(open(args.geometry))
    R_start = np.array(gd["R_start"], dtype=np.float64)
    grasp_yaw = float(gd.get("grasp_yaw", 0.0))
    X_OH = sc.make_tf(sc.rotz(np.radians(grasp_yaw))) @ np.array(rec["X_WB"], dtype=np.float64)
    hand_grasp = {HAND_NAME_MAP[n]: float(v) for n, v in zip(rec["dof_order"], rec["q"])}
    solver = ChainIK(pelvis_pos=sc.PELVIS_POS, iters=300)
    lower = solver.robot.data.joint_pos_limits[0, :, 0]
    hand_ids, _ = solver.robot.find_joints(list(hand_grasp.keys()), preserve_order=True)
    hand_open = {n: float(min(lower[i].item(), 0.0)) for n, i in zip(hand_grasp.keys(), hand_ids)}
    q_seed = np.array([rec["q_arm"][n] for n in CHAIN_NAMES])
    p0 = np.array(sc.PARCEL_START)

    def solve_pair(T_WO, seed, multi=False):
        T_grasp = sc.hand_pose_from_object(T_WO, X_OH)
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
    candidates = []
    q_default = solver.robot.data.default_joint_pos[0, solver.chain_ids].cpu().numpy()
    seed0 = np.array([gd["kinematic_summary"]["grasp_q"][n] for n in CHAIN_NAMES]) if "grasp_q" in gd.get("kinematic_summary", {}) else q_seed
    for yaw in args.yaws:
        T_WO = sc.make_tf(R_start @ sc.rotz(np.radians(yaw)), p0)
        r_g, r_p, hz_g, hz_p = solve_pair(T_WO, seed0 if yaw == 0.0 else q_seed, multi=True)
        ok = r_g["ok"] and r_p["ok"] and hz_p > sc.TABLE_TOP + 0.02
        print(f"[yaw {yaw:.0f}] grasp ok {r_g['ok']} ({r_g['pos_err']*1e3:.1f} mm, {r_g['ori_err_deg']:.1f} deg) "
              f"pregrasp ok {r_p['ok']} ({r_p['pos_err']*1e3:.1f} mm, {r_p['ori_err_deg']:.1f} deg) "
              f"hand z grasp {hz_g:.3f} pre {hz_p:.3f} admitted {ok}", flush=True)
        if ok:
            candidates.append({
                "yaw_deg": float(yaw), "q_chain": named(r_p["q"]), "q_chain_grasp": named(r_g["q"]),
                "pre_pos_error_m": r_p["pos_err"], "grasp_pos_error_m": r_g["pos_err"],
                "pre_ori_error_deg": r_p["ori_err_deg"], "grasp_ori_error_deg": r_g["ori_err_deg"],
                "limit_margin_grasp": solver.limit_margin(r_g["q"])[0],
                "limit_margin_pre": solver.limit_margin(r_p["q"])[0],
                "min_hand_body_z_grasp": hz_g,
            })
    if not candidates:
        raise SystemExit("no candidate admitted")
    # start-offset grid at yaw 0 with the lift-end knot
    grid = []
    q_g0 = np.array([candidates[0]["q_chain_grasp"][n] for n in CHAIN_NAMES])
    for dx_mm in args.grid_mm:
        for dy_mm in args.grid_mm:
            p = p0 + np.array([dx_mm * 1e-3, dy_mm * 1e-3, 0.0])
            T_WO = sc.make_tf(R_start, p)
            r_g, r_p, hz_g, hz_p = solve_pair(T_WO, q_g0)
            solver.set_hand(hand_grasp)
            geom = sc.StowGeometry.from_dict(gd)
            p_lift = p + geom.R_yaw @ np.array([geom.lift_dx, 0.0, sc.LIFT_DZ])
            T_lift = sc.hand_pose_from_object(sc.make_tf(R_start, p_lift), X_OH)
            r_l = solver.solve(T_lift, r_g["q"])
            ok = r_g["ok"] and r_p["ok"] and r_l["ok"]
            grid.append({"dx": dx_mm * 1e-3, "dy": dy_mm * 1e-3, "ok": bool(ok),
                         "q_chain": named(r_p["q"]), "q_chain_grasp": named(r_g["q"]), "q_chain_lift": named(r_l["q"]),
                         "pre_pos_error_m": r_p["pos_err"], "grasp_pos_error_m": r_g["pos_err"],
                         "lift_pos_error_m": r_l["pos_err"], "grasp_ori_error_deg": r_g["ori_err_deg"],
                         "lift_ori_error_deg": r_l["ori_err_deg"]})
            print(f"[grid {dx_mm:+.0f} {dy_mm:+.0f}] ok {ok} grasp {r_g['pos_err']*1e3:.1f} mm pre {r_p['pos_err']*1e3:.1f} mm lift {r_l['pos_err']*1e3:.1f} mm", flush=True)
    log_path = os.path.join(rec_dir, "run.log")
    cmd_line = None
    if os.path.exists(log_path):
        for line in open(log_path):
            if "g1_l6_runner" in line:
                cmd_line = line.strip()
                break
    riches = sorted(f for f in os.listdir(rec_dir) if "__g" in f and f.endswith(".json"))
    rich0 = json.load(open(os.path.join(rec_dir, riches[0]))) if riches else {}
    out = {
        "object": "parcel_80x55x40",
        "source_record": os.path.relpath(args.record),
        "source_record_sha256": sha256(args.record),
        "source_rich_records": [os.path.abspath(os.path.join(rec_dir, f)) for f in riches],
        "source_mesh": os.path.abspath(os.path.join(rec_dir, "..", "parcel_80x55x40.obj")),
        "source_mesh_sha256": rich0.get("provenance", {}).get("mesh_sha256"),
        "synthesis": {
            "tool": "frogger scripts/g1_l6_runner.py (local checkout)",
            "git_commit": rich0.get("provenance", {}).get("git_commit"),
            "utc": rich0.get("provenance", {}).get("utc"),
            "command_hint": cmd_line,
            "arguments": {"table_height_m": rec.get("table_height_m"), "object_xy": rec.get("object_xy"),
                          "riser_mm": rec.get("riser_mm"), "clearance_mm": rec.get("clearance_mm"),
                          "objective": rec.get("objective"), "grid": "thorough"},
            "scores_from_record": {"eps_nominal": rec.get("eps_nominal"),
                                   "eps_dropout_objective": rec.get("eps_dropout_objective"),
                                   "l_star": rec.get("l_star"), "n_registered": rec.get("n_registered"),
                                   "rigid_pen_mm": rec.get("rigid_pen_mm"),
                                   "eps_beta_confidence": rich0.get("scores", {}).get("eps_beta"),
                                   "prior": {"mean": rich0.get("scores", {}).get("prior_mean"),
                                             "std": rich0.get("scores", {}).get("prior_std")}},
            "contact_points_mm_object": rich0.get("contact_points_mm"),
            "contact_normals_object": rich0.get("contact_normals"),
            "contact_finger_ids": rich0.get("contact_finger_ids"),
            "note": "The certificate values above describe the synthesized contact set in the synthesis "
                    "tool. They enter no task predicate. The realized contact set in the simulator is "
                    "scored separately as a diagnostic.",
        },
        "X_OH": X_OH.tolist(),
        "grasp_yaw_applied_to_record": grasp_yaw,
        "geometry_file": os.path.abspath(args.geometry),
        "start_yaw_deg": gd.get("start_yaw_deg", 0.0),
        "construction": f"IK re-solve of the record at the ParcelStow start pose, standoff {args.standoff} m along the "
                        f"ray from the parcel center, open-hand pregrasp, yaw set {args.yaws}, planar grid for the expert",
        "parcel_pose_world": {"pos": p0.tolist(), "quat_wxyz": gd["start_quat_wxyz"]},
        "pelvis_pose_world": {"pos": list(sc.PELVIS_POS), "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
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
    print(f"[bank] {len(candidates)} candidates, {sum(g['ok'] for g in grid)}/{len(grid)} grid entries, "
          f"written {args.out} in {time.time()-t0:.0f}s", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
