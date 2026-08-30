"""ACT on the ParcelStow demonstrations (M9), the released simulated-task
configuration of scripts/baselines/run_act.py (state-only DETRVAE, chunk
100, KL weight 10, hidden 512, feedforward 3200, 4 encoder and 7 decoder
layers, 8 heads, dropout 0.1, latent 32, AdamW 1e-5 weight decay 1e-4,
batch 8, one random chunk start per episode per epoch, z-score
normalization with a 1e-2 floor, L1 over the unpadded chunk plus KL, 2000
epochs, temporal ensembling exponent 0.01), trained on the same successful
full-task expert episodes as the other learners, then evaluated with a
diagnostic set at nominal speed. Checkpoint outputs/paper/act/act_stow.pt.

Run,
  python scripts/manipulation/run_stow_act.py --demos outputs/paper/demos/expert_episodes.pt \
      --out_dir outputs/paper/act --epochs 2000

Training-seed replication (act_seed1 is the run above with model_seed 0),
  python scripts/manipulation/run_stow_act.py --demos outputs/paper/demos/expert_episodes.pt \
      --out_dir outputs/paper/act_multiseed/act_seed2 --tag seed2 --model_seed 1 --epochs 2000
  python scripts/manipulation/run_stow_act.py --demos outputs/paper/demos/expert_episodes.pt \
      --out_dir outputs/paper/act_multiseed/act_seed3 --tag seed3 --model_seed 2 --epochs 2000
Only model_seed (the torch and numpy training seed) changes between the
seeds, and every record and checkpoint holds model_seed.
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
parser.add_argument("--out_dir", type=str, default="outputs/paper/act")
parser.add_argument("--epochs", type=int, default=2000)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--chunk_size", type=int, default=100)
parser.add_argument("--kl_weight", type=float, default=10.0)
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--dim_feedforward", type=int, default=3200)
parser.add_argument("--temporal_agg", type=int, default=1)
parser.add_argument("--model_seed", type=int, default=0)
parser.add_argument("--diag_episodes", type=int, default=50)
parser.add_argument("--diag_rate", type=float, default=1.0)
parser.add_argument("--jitter", type=float, default=0.01)
parser.add_argument("--eval_seed", type=int, default=12345)
parser.add_argument("--tag", type=str, default="stow")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import parcelstow.tasks  # noqa: E402, F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stow_runtime as rt  # noqa: E402
from parcelstow.tasks.manager_based.parcel_stow import geometry as G  # noqa: E402
from parcelstow.tasks.manager_based.parcel_stow.mdp.metrics import StowMonitor  # noqa: E402
from state_act import StateACT, kl_divergence  # noqa: E402


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    device = base.device
    os.makedirs(args_cli.out_dir, exist_ok=True)
    results_path = os.path.join(args_cli.out_dir, "results.jsonl")

    def log(rec):
        rec["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        rec["model_seed"] = args_cli.model_seed
        rec["tag"] = args_cli.tag
        print("[RESULT] " + json.dumps(rec), flush=True)
        with open(results_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    demos = torch.load(args_cli.demos)
    kept = [(o, a) for o, a, _ in demos["episodes"]]
    log({"stage": "demos", "path": args_cli.demos, "episodes": len(kept)})
    all_obs = torch.cat([o for o, _ in kept])
    all_act = torch.cat([a for _, a in kept])
    obs_mean, obs_std = all_obs.mean(0), all_obs.std(0).clamp(min=1e-2)
    act_mean, act_std = all_act.mean(0), all_act.std(0).clamp(min=1e-2)
    obs_mean, obs_std, act_mean, act_std = [t.to(device) for t in (obs_mean, obs_std, act_mean, act_std)]
    eps_obs = [o.to(device) for o, _ in kept]
    eps_act = [a.to(device) for _, a in kept]
    state_dim = all_obs.shape[1]
    action_dim = all_act.shape[1]
    Tq = args_cli.chunk_size
    torch.manual_seed(args_cli.model_seed)
    np.random.seed(args_cli.model_seed)
    model = StateACT(state_dim, action_dim, Tq, hidden_dim=args_cli.hidden_dim,
                     dim_feedforward=args_cli.dim_feedforward).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args_cli.lr, weight_decay=1e-4)

    def sample_batch(idx):
        qs, acts, pads = [], [], []
        for i in idx:
            L = eps_obs[i].shape[0]
            s = np.random.randint(L)
            q = (eps_obs[i][s] - obs_mean) / obs_std
            a = torch.zeros(Tq, action_dim, device=device)
            seg = eps_act[i][s:s + Tq]
            a[:seg.shape[0]] = (seg - act_mean) / act_std
            pad = torch.ones(Tq, dtype=torch.bool, device=device)
            pad[:seg.shape[0]] = False
            qs.append(q)
            acts.append(a)
            pads.append(pad)
        return torch.stack(qs), torch.stack(acts), torch.stack(pads)

    n_ep = len(kept)
    t0 = time.time()
    last = float("nan")
    for ep in range(args_cli.epochs):
        model.train()
        perm = np.random.permutation(n_ep)
        total = 0.0
        for i in range(0, n_ep, args_cli.batch):
            idx = perm[i:i + args_cli.batch]
            qpos, actions, is_pad = sample_batch(idx)
            a_hat, _, (mu, logvar) = model(qpos, actions, is_pad)
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            kl = kl_divergence(mu, logvar)[0]
            loss = l1 + args_cli.kl_weight * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        last = total / n_ep
        if ep % 100 == 0 or ep == args_cli.epochs - 1:
            print(f"[TRAIN] epoch {ep} loss {last:.5f} elapsed {time.time() - t0:.0f}s", flush=True)
    log({"stage": "train", "epochs": args_cli.epochs, "final_loss": last, "train_seconds": time.time() - t0,
         "params_m": sum(p.numel() for p in model.parameters()) / 1e6})
    ckpt = os.path.join(args_cli.out_dir, f"act_{args_cli.tag}.pt")
    torch.save({"model": model.state_dict(), "obs_mean": obs_mean.cpu(), "obs_std": obs_std.cpu(),
                "act_mean": act_mean.cpu(), "act_std": act_std.cpu(), "args": vars(args_cli)}, ckpt)
    model.eval()

    geom = G.load_geometry()
    monitor = StowMonitor(base, geom)
    expert = rt.ExpertActor(base)
    switches = rt.EnvSwitches(base)
    stamp = rt.config_stamp(base)
    actor = rt.ACTActor(ckpt, device, base.num_envs)
    recs, _ = rt.run_episodes(env, base, actor, monitor, args_cli.diag_episodes,
                              {"mode": "fixed", "value": args_cli.diag_rate}, args_cli.jitter, args_cli.eval_seed,
                              switches, expert=expert, corrupt=False, stamp=stamp, tag="act_diag")
    s = rt.summarize(recs)
    log({"stage": "diag_eval", "diag_rate": args_cli.diag_rate, "task_success": s["task_success"],
         "acquired": s["acquired"], "inserted": s["inserted"], "settled": s["settled"],
         "failure_reasons": s["failure_reasons"], "checkpoint": ckpt})
    rt.write_jsonl(os.path.join(args_cli.out_dir, "diag.jsonl"), [rt.light_record(r) for r in recs])
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
