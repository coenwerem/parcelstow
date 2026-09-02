"""Matched expert-learner evaluation of the upright placement task
across execution speeds, the eval_stow_policies.py protocol: the same
per-speed seed for every policy pairs their initial-condition draws,
episode records go to <out_dir>/<actor><tag>.jsonl and one summary row
per (policy, rate) to <out_dir>/summary<tag>.jsonl.

Run,
  python scripts/manipulation/eval_upright_policies.py --actors expert act \
      --act_ckpt outputs/upright/act/act_upright.pt \
      --rates 0.5 1.0 1.5 2.0 2.25 2.5 3.0 --episodes 100
"""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="UprightPlace-L6-Play-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--actors", type=str, nargs="*", default=["expert", "act"])
parser.add_argument("--act_ckpt", type=str, default="outputs/upright/act/act_upright.pt")
parser.add_argument("--dagger_ckpt", type=str, default=None)
parser.add_argument("--dp_ckpt", type=str, default=None)
parser.add_argument("--custom_ckpt", type=str, default=None)
parser.add_argument("--rates", type=float, nargs="*",
                    default=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5])
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--jitter", type=float, default=0.01)
parser.add_argument("--eval_seed", type=int, default=12345)
parser.add_argument("--out_dir", type=str, default="outputs/upright/eval")
parser.add_argument("--tag", type=str, default="")
parser.add_argument("--trace_envs", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import parcelstow.tasks  # noqa: E402, F401

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stow_runtime as rt  # noqa: E402
from parcelstow.tasks.manager_based.upright_place.mdp.monitor import STAGE_KEYS, UprightMonitor  # noqa: E402
from upright_runtime import SCHED, UprightExpertActor, config_stamp  # noqa: E402


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    trace_envs = list(range(args_cli.trace_envs))
    monitor = UprightMonitor(base, trace_envs=trace_envs)
    expert = UprightExpertActor(base)
    switches = rt.EnvSwitches(base, reset_term="reset_object")
    stamp = config_stamp(base, task_id=args_cli.task)
    os.makedirs(args_cli.out_dir, exist_ok=True)
    tag = args_cli.tag
    ckpts = {"dagger": args_cli.dagger_ckpt, "dp": args_cli.dp_ckpt, "act": args_cli.act_ckpt}
    t0 = time.time()

    for name in args_cli.actors:
        actor = expert if name == "expert" else rt.load_actor(name, ckpts.get(name, args_cli.custom_ckpt),
                                                              base, args_cli.num_envs)
        ep_path = os.path.join(args_cli.out_dir, f"{name.replace(':', '_').replace('.', '_')}{tag}.jsonl")
        for ri, r in enumerate(args_cli.rates):
            seed = args_cli.eval_seed + 1000 * ri
            recs, _ = rt.run_episodes(env, base, actor, monitor, args_cli.episodes,
                                      {"mode": "fixed", "value": r}, args_cli.jitter, seed, switches,
                                      expert=expert, corrupt=False, stamp=stamp, tag=f"{name}_r{r:g}",
                                      task_id=args_cli.task, cycle_time=SCHED.cycle_time)
            rt.write_jsonl(ep_path, recs)
            row = rt.summarize(recs, stage_keys=STAGE_KEYS)
            row.update({"policy": name, "rate": r, "cycle_time_s": SCHED.cycle_time(r), "seed": seed,
                        "jitter": args_cli.jitter, "episodes_requested": args_cli.episodes,
                        "checkpoint": ckpts.get(name, args_cli.custom_ckpt) if name != "expert" else None,
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
            rt.write_jsonl(os.path.join(args_cli.out_dir, f"summary{tag}.jsonl"), [row])
            print(f"[EVAL {name} r={r:g}] success {row['task_success']['k']}/{row['task_success']['n']} "
                  f"reasons {row['failure_reasons']} ({time.time()-t0:.0f}s)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
