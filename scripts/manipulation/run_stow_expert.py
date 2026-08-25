"""Scripted-expert driver of the ParcelStow task, expert validation (M4),
the task-rate sweep (M5), and demonstration collection (M7).

Every episode runs the full manipulation under the physical monitor and
writes one JSON line (policy expert, seed, task_rate, task_duration_s, the
stage markers, the failure reason, slip diagnostics, realized contact sets
with their certificate diagnostics, actuator utilization). Success is the
physical predicate of TASK_SPEC.md section 8, nothing else.

Modes,
  validate   fixed rate, fixed jitter, N episodes (M4 wants 20/20 at r 0.5)
  sweep      the same over a rate list (calibration of the rate grid)
  demos      uniform rate in [lo, hi] and planar jitter, collects complete
             episodes with (obs, expert action) and saves the physically
             successful ones to --demo_out (the shared demonstration set)

Run,
  python scripts/manipulation/run_stow_expert.py --mode validate --rate 0.5 --episodes 20 \
      --out outputs/paper/expert/validate_r0.5.jsonl
  python scripts/manipulation/run_stow_expert.py --mode sweep --rates 0.5 0.75 1 1.25 1.5 2 2.5 3 \
      --episodes 50 --out outputs/paper/expert/sweep.jsonl
  python scripts/manipulation/run_stow_expert.py --mode demos --rate_lo 0.75 --rate_hi 1.5 --jitter 0.01 \
      --episodes 300 --demo_out outputs/paper/demos/expert_episodes.pt
"""

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="ParcelStow-L6-Distill-Play-v0")
parser.add_argument("--mode", choices=["validate", "sweep", "demos"], default="validate")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--rate", type=float, default=0.5)
parser.add_argument("--rates", type=float, nargs="*", default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])
parser.add_argument("--rate_lo", type=float, default=0.75)
parser.add_argument("--rate_hi", type=float, default=1.5)
parser.add_argument("--jitter", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--corrupt", action="store_true", help="observation corruption on (demos default on)")
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--demo_out", type=str, default=None)
parser.add_argument("--trace_envs", type=int, default=0, help="record dense traces of the first k envs")
parser.add_argument("--trace_dir", type=str, default=None)
parser.add_argument("--tag", type=str, default=None)
parser.add_argument("--env_spacing", type=float, default=None, help="override the scene env spacing (probe)")
parser.add_argument("--no_joint_noise", action="store_true", help="zero the reset joint offset noise (probe)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import parcelstow.tasks  # noqa: E402, F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stow_runtime as rt  # noqa: E402
from parcelstow.tasks.manager_based.parcel_stow import geometry as G  # noqa: E402
from parcelstow.tasks.manager_based.parcel_stow.mdp.metrics import StowMonitor  # noqa: E402


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.env_spacing is not None:
        env_cfg.scene.env_spacing = args_cli.env_spacing
    if args_cli.no_joint_noise:
        env_cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    geom = G.load_geometry()
    trace_envs = list(range(args_cli.trace_envs))
    monitor = StowMonitor(base, geom, trace_envs=trace_envs)
    expert = rt.ExpertActor(base)
    switches = rt.EnvSwitches(base)
    stamp = rt.config_stamp(base)
    out = args_cli.out
    tag = args_cli.tag or args_cli.mode
    t0 = time.time()

    if args_cli.mode in ("validate", "sweep"):
        rates = [args_cli.rate] if args_cli.mode == "validate" else args_cli.rates
        summaries = []
        for r in rates:
            recs, _ = rt.run_episodes(env, base, expert, monitor, args_cli.episodes, {"mode": "fixed", "value": r},
                                      args_cli.jitter, args_cli.seed, switches, expert=expert,
                                      corrupt=args_cli.corrupt, stamp=stamp, tag=f"{tag}_r{r:g}",
                                      trace_dir=args_cli.trace_dir)
            s = rt.summarize(recs)
            s.update({"rate": r, "cycle_time_s": G.cycle_time(r), "jitter": args_cli.jitter, "seed": args_cli.seed})
            summaries.append(s)
            print(f"[SUMMARY r={r:g}] success {s['task_success']['k']}/{s['task_success']['n']} "
                  f"reasons {s['failure_reasons']} slip_t p90 {s['max_hand_object_translation_m']} "
                  f"vel_util {s['max_joint_velocity_utilization']} ({time.time()-t0:.0f}s)", flush=True)
            if out:
                rt.write_jsonl(out, recs)
        if out:
            rt.write_jsonl(out.replace(".jsonl", "_summary.jsonl"), summaries, mode="a")
        print("[RESULT] " + json.dumps([{k: v for k, v in s.items() if k in ("rate", "task_success", "failure_reasons",
                                                                             "acquired", "inserted", "settled")}
                                        for s in summaries]), flush=True)
    else:
        spec = {"mode": "uniform", "lo": args_cli.rate_lo, "hi": args_cli.rate_hi}
        recs, episodes = rt.run_episodes(env, base, expert, monitor, args_cli.episodes, spec, args_cli.jitter,
                                         args_cli.seed, switches, expert=expert, record_data=True,
                                         corrupt=True, stamp=stamp, tag=tag)
        s = rt.summarize(recs)
        admitted = [(o, a, rec) for (o, a, rec) in episodes if rec["task_success"]]
        print(f"[DEMOS] {len(admitted)}/{len(episodes)} episodes admitted by physical task success, "
              f"reasons of the rest {s['failure_reasons']}", flush=True)
        if args_cli.demo_out:
            os.makedirs(os.path.dirname(args_cli.demo_out) or ".", exist_ok=True)
            torch.save({"episodes": [(o, a, True) for (o, a, _) in admitted],
                        "records": [rt.light_record(r) for (_, _, r) in admitted],
                        "all_records": [rt.light_record(r) for r in recs],
                        "rate_spec": spec, "jitter": args_cli.jitter, "seed": args_cli.seed,
                        "obs_dim": int(admitted[0][0].shape[1]) if admitted else None,
                        "act_dim": int(admitted[0][1].shape[1]) if admitted else None,
                        "config": stamp}, args_cli.demo_out)
            print(f"[DEMOS] written {args_cli.demo_out}", flush=True)
        if out:
            rt.write_jsonl(out, recs)
            s.update({"rate_spec": spec, "jitter": args_cli.jitter, "seed": args_cli.seed, "admitted": len(admitted)})
            rt.write_jsonl(out.replace(".jsonl", "_summary.jsonl"), [s], mode="a")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
