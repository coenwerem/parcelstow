"""Final capability evaluation of the ParcelStow actors (M10, M11), the
same episodes for every actor.

For every actor (expert, dagger, dp, act) and every rate of the frozen
grid, the driver runs N episodes with the identical evaluation seed per
rate, the same start-jitter law, corruption off, under the physical
monitor. Every episode record holds the stage markers, the failure reason,
slip diagnostics, the realized contact sets at acquisition, end of
reorientation, and insertion start with their certificate diagnostics
(epsilon at the parcel friction, epsilon^(beta) at beta 0.95, Gaussian
prior std 0.15), actuator utilization, and the configuration stamp. The
records go to one JSONL per actor and a summary JSONL, figures and the
certificate analysis read them.

Run,
  python scripts/manipulation/eval_stow_policies.py --actors expert dagger dp act \
      --rates 0.5 0.75 1.0 1.25 1.5 2.0 2.5 --episodes 100 --out_dir outputs/paper/eval
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
parser.add_argument("--actors", type=str, nargs="*", default=["expert", "dagger", "dp", "act"])
parser.add_argument("--dagger_ckpt", type=str, default="outputs/paper/dagger/student_final.pt")
parser.add_argument("--dp_ckpt", type=str, default="outputs/paper/dp/dp_stow.pt")
parser.add_argument("--act_ckpt", type=str, default="outputs/paper/act/act_stow.pt")
parser.add_argument("--custom_ckpt", type=str, default=None,
                    help="checkpoint handed to a module.path:ClassName actor")
parser.add_argument("--rates", type=float, nargs="*", default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5])
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--jitter", type=float, default=0.01)
parser.add_argument("--eval_seed", type=int, default=12345)
parser.add_argument("--out_dir", type=str, default="outputs/paper/eval")
parser.add_argument("--tag", type=str, default="")
parser.add_argument("--trace_envs", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import parcelstow.tasks  # noqa: E402, F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stow_runtime as rt  # noqa: E402
from parcelstow.tasks.manager_based.parcel_stow import geometry as G  # noqa: E402
from parcelstow.tasks.manager_based.parcel_stow.mdp.metrics import StowMonitor  # noqa: E402


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    n = base.num_envs
    os.makedirs(args_cli.out_dir, exist_ok=True)
    geom = G.load_geometry()
    monitor = StowMonitor(base, geom, trace_envs=range(args_cli.trace_envs))
    expert = rt.ExpertActor(base)
    switches = rt.EnvSwitches(base)
    stamp = rt.config_stamp(base)
    ckpts = {"dagger": args_cli.dagger_ckpt, "dp": args_cli.dp_ckpt, "act": args_cli.act_ckpt}
    summary_path = os.path.join(args_cli.out_dir, f"summary{args_cli.tag}.jsonl")
    t0 = time.time()
    for name in args_cli.actors:
        actor = expert if name == "expert" else rt.load_actor(name, ckpts.get(name, args_cli.custom_ckpt), base, n)
        fname = name.replace(":", "_").replace(".", "_")
        rec_path = os.path.join(args_cli.out_dir, f"{fname}{args_cli.tag}.jsonl")
        for ri, r in enumerate(args_cli.rates):
            seed = args_cli.eval_seed + 1000 * ri
            recs, _ = rt.run_episodes(env, base, actor, monitor, args_cli.episodes, {"mode": "fixed", "value": r},
                                      args_cli.jitter, seed, switches, expert=expert, corrupt=False, stamp=stamp,
                                      tag=f"{name}_r{r:g}", extra={"checkpoint": ckpts.get(name, args_cli.custom_ckpt)},
                                      trace_dir=os.path.join(args_cli.out_dir, "traces") if args_cli.trace_envs else None)
            rt.write_jsonl(rec_path, recs)
            s = rt.summarize(recs)
            s.update({"policy": name, "rate": r, "cycle_time_s": G.cycle_time(r), "seed": seed,
                      "jitter": args_cli.jitter, "episodes_requested": args_cli.episodes,
                      "checkpoint": ckpts.get(name, args_cli.custom_ckpt), "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
            rt.write_jsonl(summary_path, [s])
            print(f"[EVAL {name} r={r:g}] success {s['task_success']['k']}/{s['task_success']['n']} "
                  f"wilson {[round(v, 3) for v in s['task_success']['wilson']]} reasons {s['failure_reasons']} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
