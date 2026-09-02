"""Scripted-expert driver of the keyed-peg insertion task, expert
validation, the speedup-factor sweep, and demonstration collection,
the run_stow_expert.py pattern.

Every episode runs under the physical monitor
(peg_insert/mdp/monitor.py) and writes one JSON line. Success is
the physical predicate of the task geometry, nothing else.

Modes,
  validate   fixed speed, fixed jitter, N episodes
  sweep      the same over a speed list (expert-only speed calibration)
  demos      uniform rate in [lo, hi] and planar jitter, saves the
             physically successful episodes to --demo_out

Run,
  python scripts/manipulation/run_peg_expert.py --mode validate --rate 0.5 --episodes 20 \
      --out outputs/peg/expert/validate_r0.5.jsonl
  python scripts/manipulation/run_peg_expert.py --mode sweep --rates 0.5 0.75 1 1.25 1.5 2 2.5 3 \
      --episodes 64 --jitter 0.01 --out outputs/peg/expert/sweep.jsonl
"""

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="PegInsert-L6-Play-v0")
parser.add_argument("--mode", choices=["validate", "sweep", "demos"], default="validate")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--rate", type=float, default=0.5)
parser.add_argument("--rates", type=float, nargs="*", default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])
# Frozen after expert-only calibration: the >=0.9 contiguous range of the 64-episode sweep
# (63, 62, 58 of 64 at r in {0.5, 0.75, 1.0}; the envelope dips to 0.73
# at r = 1.5 and recovers to about 0.9 at r >= 2).
parser.add_argument("--rate_lo", type=float, default=0.5)
parser.add_argument("--rate_hi", type=float, default=1.0)
parser.add_argument("--jitter", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--corrupt", action="store_true")
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--demo_out", type=str, default=None)
parser.add_argument("--trace_envs", type=int, default=0)
parser.add_argument("--trace_dir", type=str, default=None)
parser.add_argument("--tag", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import parcelstow.tasks  # noqa: E402, F401
import torch  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stow_runtime as rt  # noqa: E402
from parcelstow.tasks.manager_based.peg_insert.mdp.monitor import STAGE_KEYS, PegMonitor  # noqa: E402
from peg_runtime import SCHED, PegExpertActor, config_stamp  # noqa: E402


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    trace_envs = list(range(args_cli.trace_envs))
    monitor = PegMonitor(base, trace_envs=trace_envs)
    expert = PegExpertActor(base)
    switches = rt.EnvSwitches(base, reset_term="reset_object")
    stamp = config_stamp(base, task_id=args_cli.task)
    out = args_cli.out
    tag = args_cli.tag or args_cli.mode
    t0 = time.time()

    def run(n_eps, spec, jitter, seed, record_data=False, corrupt=False, tag_i=""):
        return rt.run_episodes(env, base, expert, monitor, n_eps, spec, jitter, seed, switches,
                               expert=expert, record_data=record_data, corrupt=corrupt, stamp=stamp,
                               tag=tag_i, trace_dir=args_cli.trace_dir,
                               task_id=args_cli.task, cycle_time=SCHED.cycle_time)

    if args_cli.mode in ("validate", "sweep"):
        rates = [args_cli.rate] if args_cli.mode == "validate" else args_cli.rates
        summaries = []
        for r in rates:
            recs, _ = run(args_cli.episodes, {"mode": "fixed", "value": r}, args_cli.jitter,
                          args_cli.seed, corrupt=args_cli.corrupt, tag_i=f"{tag}_r{r:g}")
            s = rt.summarize(recs, stage_keys=STAGE_KEYS)
            s.update({"rate": r, "cycle_time_s": SCHED.cycle_time(r), "jitter": args_cli.jitter,
                      "seed": args_cli.seed})
            summaries.append(s)
            print(f"[SUMMARY r={r:g}] success {s['task_success']['k']}/{s['task_success']['n']} "
                  f"reasons {s['failure_reasons']} ({time.time()-t0:.0f}s)", flush=True)
            if out:
                rt.write_jsonl(out, recs)
        if out:
            rt.write_jsonl(out.replace(".jsonl", "_summary.jsonl"), summaries, mode="a")
        print("[RESULT] " + json.dumps([{k: v for k, v in s.items()
                                         if k in ("rate", "task_success", "failure_reasons",
                                                  "acquired", "placed", "settled")}
                                        for s in summaries]), flush=True)
    else:
        spec = {"mode": "uniform", "lo": args_cli.rate_lo, "hi": args_cli.rate_hi}
        recs, episodes = run(args_cli.episodes, spec, args_cli.jitter, args_cli.seed,
                             record_data=True, corrupt=True, tag_i=tag)
        s = rt.summarize(recs, stage_keys=STAGE_KEYS)
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
            s.update({"rate_spec": spec, "jitter": args_cli.jitter, "seed": args_cli.seed,
                      "admitted": len(admitted)})
            rt.write_jsonl(out.replace(".jsonl", "_summary.jsonl"), [s], mode="a")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
