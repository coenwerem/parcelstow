"""Relative-motion handoff diagnostic of the ParcelStow task, the second
common-controller ablation beside scripts/manipulation/stow_handoff.py
(unchanged).

The actor (expert or an ACT training seed) acquires the parcel on its own.
At stable handoff (the monitor's acquired marker plus the parcel 40 mm
above its start height, before RELEASE) the driver records the actor's
actual world hand pose T_WH^pi(t_H), freezes the actor's realized hand
joint target, and commands the waist and arm so the commanded hand pose
follows

    T_WH,d^pi(s) = T_anchor^pi (T_WH^E(t_H))^{-1} T_WH^E(s),

the expert's nominal downstream relative hand motion (T_WH^E, the forward
kinematics of the expert's nominal command) applied to the actor's own
acquired hand pose. The anchor T_anchor^pi is the forward kinematics of the
actor's arm command at t_H with its static servo offset removed (the
command minus the measured configuration at rest at the end of
GRASP_DWELL), which is the actor's commanded hand pose free of sag
compensation, the pose the actor's servo settles to at a dwell.
stow_relative.py explains why neither the measured pose nor the raw command
serves as the anchor (the servo transient and the compensation would become
permanent path offsets, measured in debug runs of the expert). The arm
command comes from the damped least squares IK iteration of the ParcelStow
trajectory builder run online on the URDF kinematic model of the chain
(checked against the PhysX hand pose in every episode), with the expert's
nominal joint configuration as the null-space attractor and the expert's
dwell integral on top. The parcel stays a free rigid body, nothing is
snapshot, restored, welded, or reset, and no object-pose feedback exists.

Primary endpoint, the free-space manipulation segment before receptacle
contact, stable lift, reorientation, transfer to pre-insertion. At the
first control step of INSERT, or at the first receptacle contact when a
preserved absolute offset brings the parcel against a slab earlier, the
record holds retained_preinsert, the hand-object transform change since
handoff (dp_preinsert_m, dR_preinsert_deg, and the segment maxima), the
contact count, the endpoint reason, and the parcel pose relative to the
hand. Secondary endpoint, the same episodes continue through
insertion and release under the relative controller and the standard task
record (task_success, inserted, settled, failure_reason) describes them,
labeled secondary because the absolute object pose still meets the
receptacle. Episodes in which the actor never acquires the parcel run to
their end with the actor in charge and are marked relative_handoff False.

Run,
  python scripts/manipulation/stow_relative_handoff.py --actors expert act_seed1 act_seed2 act_seed3 \\
      --rates 1.0 1.5 2.0 --episodes 100 --eval_seed 13345 --out_dir outputs/paper/relative_handoff
(eval_seed 13345 with the rate list 1.0 1.5 2.0 reproduces the evaluation
draws of eval_stow_policies.py, seeds 13345, 14345, 15345.)
"""

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="ParcelStow-L6-Distill-Play-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--actors", type=str, nargs="*", default=["expert", "act_seed1", "act_seed2", "act_seed3"])
parser.add_argument("--ckpt", type=str, nargs="*", default=[
    "act_seed1=outputs/paper/act/act_stow.pt",
    "act_seed2=outputs/paper/act_multiseed/act_seed2/act_seed2.pt",
    "act_seed3=outputs/paper/act_multiseed/act_seed3/act_seed3.pt",
    "act=outputs/paper/act/act_stow.pt",
    "dp=outputs/paper/dp/dp_stow.pt",
    "dagger=outputs/paper/dagger/student_final.pt"], help="name=path pairs")
parser.add_argument("--rates", type=float, nargs="*", default=[1.0, 1.5, 2.0])
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--jitter", type=float, default=0.01)
parser.add_argument("--eval_seed", type=int, default=13345)
parser.add_argument("--handoff_dz", type=float, default=0.04)
parser.add_argument("--iters", type=int, default=3)
parser.add_argument("--out_dir", type=str, default="outputs/paper/relative_handoff")
parser.add_argument("--tag", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import parcelstow.tasks  # noqa: E402, F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stow_runtime as rt  # noqa: E402
from parcelstow.tasks.manager_based.parcel_stow import geometry as G  # noqa: E402
from parcelstow.tasks.manager_based.parcel_stow.mdp.metrics import StowMonitor  # noqa: E402
from stow_relative_controller import RelativeHandoffController  # noqa: E402


def actor_kind(name):
    for kind in ("expert", "act", "dp", "dagger"):
        if name == kind or name.startswith(kind + "_"):
            return kind
    raise ValueError(name)


def summarize_relative(recs):
    s = rt.summarize(recs)
    hand = [r for r in recs if r.get("relative_handoff")]
    reached = [r for r in hand if r.get("primary_endpoint_reached")]
    kept = [r for r in hand if r.get("retained_preinsert")]
    n_h = len(hand)
    lo, hi = G.wilson(len(kept), n_h) if n_h else (float("nan"), float("nan"))

    def dist(rows, key):
        v = np.array([r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])], dtype=float)
        if v.size == 0:
            return None
        return {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "max": float(v.max()),
                "mean": float(v.mean()), "n": int(v.size)}
    s.update({
        "handoff_episodes": n_h,
        "primary_endpoint_reached": len(reached),
        "retained_preinsert": {"frac": len(kept) / n_h if n_h else None, "k": len(kept), "n": n_h, "wilson": [lo, hi]},
        "dp_preinsert_m": dist(reached, "dp_preinsert_m"),
        "dR_preinsert_deg": dist(reached, "dR_preinsert_deg"),
        "dp_max_segment_m": dist(hand, "dp_max_segment_m"),
        "dR_max_segment_deg": dist(hand, "dR_max_segment_deg"),
        "dp_preinsert_from_acquisition_m": dist(reached, "dp_preinsert_from_acquisition_m"),
        "dR_preinsert_from_acquisition_deg": dist(reached, "dR_preinsert_from_acquisition_deg"),
        "contact_count_preinsert_endpoint": dist(reached, "contact_count_preinsert_endpoint"),
        "contact_count_handoff": dist(hand, "contact_count_handoff"),
        "epsilon_handoff": dist(hand, "epsilon_handoff"),
        "receptacle_force_before_endpoint": dist(hand, "receptacle_force_before_endpoint"),
        "receptacle_touched_before_endpoint": int(sum(1 for r in hand if r.get("receptacle_force_before_endpoint", 0) > 0)),
        "primary_endpoint_by_contact": int(sum(1 for r in hand if r.get("primary_endpoint_reason") == "receptacle_contact")),
        "primary_endpoint_by_insert_start": int(sum(1 for r in hand if r.get("primary_endpoint_reason") == "insert_start")),
        "max_kinematic_residual_pos_m": dist(hand, "max_kinematic_residual_pos_m"),
        "max_kinematic_residual_rot_deg": dist(hand, "max_kinematic_residual_rot_deg"),
        "max_pose_tracking_error_pos_m": dist(hand, "max_pose_tracking_error_pos_m"),
        "max_pose_tracking_error_rot_deg": dist(hand, "max_pose_tracking_error_rot_deg"),
        "max_kinematic_model_error_pos_m": dist(hand, "max_kinematic_model_error_pos_m"),
        "max_kinematic_model_error_rot_deg": dist(hand, "max_kinematic_model_error_rot_deg"),
        "handoff_anchor_minus_measured_m": dist(hand, "handoff_anchor_minus_measured_m"),
        "handoff_anchor_minus_measured_deg": dist(hand, "handoff_anchor_minus_measured_deg"),
        "handoff_command_lead_m": dist(hand, "handoff_command_lead_m"),
        "hand_hold_violation_rad": dist(hand, "hand_hold_violation_rad"),
        "handoff_step": dist(hand, "handoff_step"),
        "secondary_task_success_given_handoff": {"k": sum(1 for r in hand if r["task_success"]), "n": n_h},
    })
    return s


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    n = base.num_envs
    os.makedirs(args_cli.out_dir, exist_ok=True)
    geom = G.load_geometry()
    monitor = StowMonitor(base, geom)
    expert = rt.ExpertActor(base)
    switches = rt.EnvSwitches(base)
    stamp = rt.config_stamp(base)
    ckpts = dict(kv.split("=", 1) for kv in args_cli.ckpt)
    summary_path = os.path.join(args_cli.out_dir, f"summary{args_cli.tag}.jsonl")
    t0 = time.time()
    for name in args_cli.actors:
        kind = actor_kind(name)
        actor = expert if kind == "expert" else rt.load_actor(kind, ckpts[name], base, n)
        controller = RelativeHandoffController(base, expert, geom, monitor, dz=args_cli.handoff_dz, iters=args_cli.iters)
        rec_path = os.path.join(args_cli.out_dir, f"{name}{args_cli.tag}.jsonl")
        for ri, r in enumerate(args_cli.rates):
            seed = args_cli.eval_seed + 1000 * ri
            recs, _ = rt.run_episodes(env, base, actor, monitor, args_cli.episodes, {"mode": "fixed", "value": r},
                                      args_cli.jitter, seed, switches, expert=expert, corrupt=False, stamp=stamp,
                                      tag=f"relhandoff_{name}_r{r:g}", step_hook=controller.hook,
                                      after_step_hook=controller.after_step, record_hook=controller.record_hook,
                                      extra={"handoff": "relative", "handoff_dz": args_cli.handoff_dz,
                                             "acquisition_actor": name, "checkpoint": ckpts.get(name)})
            for rec in recs:
                rec["policy"] = name
            rt.write_jsonl(rec_path, recs)
            s = summarize_relative(recs)
            s.update({"policy": name, "actor_kind": kind, "rate": r, "cycle_time_s": G.cycle_time(r), "seed": seed,
                      "handoff": "relative", "jitter": args_cli.jitter, "checkpoint": ckpts.get(name),
                      "params": controller.params, "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
            rt.write_jsonl(summary_path, [s])
            rp = s["retained_preinsert"]
            print(f"[RELHANDOFF {name} r={r:g}] handoff {s['handoff_episodes']}/{len(recs)} retained "
                  f"{rp['k']}/{rp['n']} dp_med {s['dp_preinsert_m'] and round(s['dp_preinsert_m']['median'], 4)} "
                  f"dR_med {s['dR_preinsert_deg'] and round(s['dR_preinsert_deg']['median'], 2)} "
                  f"endpoint_by_contact {s['primary_endpoint_by_contact']} residual "
                  f"{s['max_kinematic_residual_pos_m'] and round(s['max_kinematic_residual_pos_m']['max'], 4)} "
                  f"secondary success {s['task_success']['k']}/{s['task_success']['n']} reasons {s['failure_reasons']} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
