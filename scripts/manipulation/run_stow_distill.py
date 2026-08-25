"""Full-task DAgger driver of the ParcelStow task (M8), the scheme of
scripts/distill/run_distill.py applied to the whole manipulation.

Stage 1 loads the shared demonstration set (successful complete expert
episodes from run_stow_expert.py --mode demos, admitted by physical task
success only). Stage 2 fits the MLP student by behavior cloning (512-256-128
ELU, standardized inputs, MSE on the 16 absolute joint actions, Adam 1e-3,
40 epochs, batch 4096). Stage 3 runs DAgger rounds where the student acts
for the entire task (approach, grasp, reorient, transfer, insert, release,
retreat) with Gaussian action noise on half the environments, the expert
relabels every visited state through its own phase-clock plan and parallel
integrator, and every visited state enters the aggregate. After every fit a
diagnostic evaluation runs at the nominal rate. The final student is
outputs/paper/dagger/student_final.pt. No scripted controller takes
over at any point of the student rollouts.

Run,
  python scripts/manipulation/run_stow_distill.py --demos outputs/paper/demos/expert_episodes.pt \
      --out_dir outputs/paper/dagger --rate_lo 0.75 --rate_hi 1.5 --jitter 0.01
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
parser.add_argument("--demos", type=str, required=True)
parser.add_argument("--out_dir", type=str, default="outputs/paper/dagger")
parser.add_argument("--dagger_rounds", type=int, default=4)
parser.add_argument("--dagger_episodes", type=int, default=100)
parser.add_argument("--diag_episodes", type=int, default=50)
parser.add_argument("--diag_rate", type=float, default=1.0)
parser.add_argument("--rate_lo", type=float, default=0.75)
parser.add_argument("--rate_hi", type=float, default=1.5)
parser.add_argument("--jitter", type=float, default=0.01)
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch", type=int, default=4096)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--train_seed", type=int, default=1)
parser.add_argument("--eval_seed", type=int, default=12345)
parser.add_argument("--tag", type=str, default="")
parser.add_argument("--init_aggregate", type=str, default=None,
                    help="resume from an aggregate_dataset.pt of an earlier run (its samples seed the aggregate)")
parser.add_argument("--round_offset", type=int, default=0, help="round numbering offset when resuming")
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


def fit_student(obs, act, device, epochs, batch, lr, seed):
    torch.manual_seed(seed)
    mean = obs.mean(dim=0)
    std = obs.std(dim=0).clamp(min=1e-3)
    model = rt.Student(obs.shape[1], act.shape[1], mean, std).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = obs.shape[0]
    obs = obs.to(device)
    act = act.to(device)
    last = float("nan")
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = ((model(obs[idx]) - act[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        last = total / n
    return model.eval(), last


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    device = base.device
    os.makedirs(args_cli.out_dir, exist_ok=True)
    results_path = os.path.join(args_cli.out_dir, f"results{args_cli.tag}.jsonl")
    geom = G.load_geometry()
    monitor = StowMonitor(base, geom)
    expert = rt.ExpertActor(base)
    switches = rt.EnvSwitches(base)
    stamp = rt.config_stamp(base)
    spec = {"mode": "uniform", "lo": args_cli.rate_lo, "hi": args_cli.rate_hi}

    def log(rec):
        rec["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        print("[RESULT] " + json.dumps(rec), flush=True)
        with open(results_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    demos = torch.load(args_cli.demos)
    episodes = demos["episodes"]
    agg_obs = torch.cat([o for o, _, _ in episodes])
    agg_act = torch.cat([a for _, a, _ in episodes])
    if args_cli.init_aggregate:
        prev = torch.load(args_cli.init_aggregate)
        agg_obs, agg_act = prev["obs"], prev["act"]
        log({"stage": "resume", "path": args_cli.init_aggregate, "samples": int(agg_obs.shape[0])})
    log({"stage": "demos", "path": args_cli.demos, "episodes": len(episodes), "samples": int(agg_obs.shape[0]),
         "rate_spec": demos.get("rate_spec"), "jitter": demos.get("jitter"), "obs_dim": int(agg_obs.shape[1])})

    def evaluate(model, tag):
        actor = rt.DaggerActor(None, device, model=model)
        recs, _ = rt.run_episodes(env, base, actor, monitor, args_cli.diag_episodes,
                                  {"mode": "fixed", "value": args_cli.diag_rate}, args_cli.jitter,
                                  args_cli.eval_seed, switches, expert=expert, corrupt=False, stamp=stamp, tag=tag)
        s = rt.summarize(recs)
        log({"stage": tag, "diag_rate": args_cli.diag_rate, "jitter": args_cli.jitter,
             "task_success": s["task_success"], "acquired": s["acquired"], "inserted": s["inserted"],
             "settled": s["settled"], "failure_reasons": s["failure_reasons"]})
        rt.write_jsonl(os.path.join(args_cli.out_dir, f"diag{args_cli.tag}.jsonl"), [rt.light_record(r) for r in recs])
        return s["task_success"]["frac"]

    model, loss = fit_student(agg_obs, agg_act, device, args_cli.epochs, args_cli.batch, args_cli.lr, args_cli.train_seed)
    torch.save(model.state_dict(), os.path.join(args_cli.out_dir, f"student_round0{args_cli.tag}.pt"))
    log({"stage": "bc_fit", "samples": int(agg_obs.shape[0]), "final_mse": loss})
    evaluate(model, "eval_round0")

    for k in range(1 + args_cli.round_offset, args_cli.dagger_rounds + args_cli.round_offset + 1):
        actor = rt.DaggerActor(None, device, model=model)
        recs, eps = rt.run_episodes(env, base, actor, monitor, args_cli.dagger_episodes, spec, args_cli.jitter,
                                    args_cli.train_seed + 100 * k, switches, expert=expert, record_data=True,
                                    corrupt=True, action_noise=args_cli.action_noise, noise_half=True,
                                    stamp=stamp, tag=f"dagger_round{k}")
        s = rt.summarize(recs)
        new_obs = torch.cat([o for o, _, _ in eps])
        new_act = torch.cat([a for _, a, _ in eps])
        agg_obs = torch.cat([agg_obs, new_obs])
        agg_act = torch.cat([agg_act, new_act])
        log({"stage": f"dagger_round{k}_rollout", "student_success_noisy_half": s["task_success"],
             "failure_reasons": s["failure_reasons"], "episodes": len(recs), "aggregate_samples": int(agg_obs.shape[0])})
        model, loss = fit_student(agg_obs, agg_act, device, args_cli.epochs, args_cli.batch, args_cli.lr,
                                  args_cli.train_seed + k)
        torch.save(model.state_dict(), os.path.join(args_cli.out_dir, f"student_round{k}{args_cli.tag}.pt"))
        log({"stage": f"dagger_round{k}_fit", "samples": int(agg_obs.shape[0]), "final_mse": loss})
        evaluate(model, f"eval_round{k}")
    torch.save(model.state_dict(), os.path.join(args_cli.out_dir, f"student_final{args_cli.tag}.pt"))
    torch.save({"obs": agg_obs, "act": agg_act}, os.path.join(args_cli.out_dir, f"aggregate_dataset{args_cli.tag}.pt"))
    log({"stage": "final", "checkpoint": os.path.join(args_cli.out_dir, f"student_final{args_cli.tag}.pt"),
         "aggregate_samples": int(agg_obs.shape[0]), "train_seed": args_cli.train_seed})
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
