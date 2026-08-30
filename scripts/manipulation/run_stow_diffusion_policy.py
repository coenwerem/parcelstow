"""Diffusion Policy on the ParcelStow demonstrations (M9), the released
low-dimensional configuration of scripts/baselines/run_diffusion_policy.py
(ConditionalUnet1D, diffusion step embedding 256, down dims 256-512-1024,
kernel 5, 8 groups, FiLM global conditioning on 2 observation steps, DDPM
100 train steps, squaredcos_cap_v2, epsilon prediction, 100 inference
steps, horizon 16, 8 executed action steps, AdamW 1e-4 betas (0.95, 0.999)
weight decay 1e-6, cosine schedule with 500 warmup steps, batch 256, EMA
power 0.75), trained on the same successful full-task expert episodes the
DAgger student and ACT receive, then evaluated with a diagnostic set at
nominal speed. Checkpoint outputs/paper/dp/dp_stow.pt.

Run,
  python scripts/manipulation/run_stow_diffusion_policy.py --demos outputs/paper/demos/expert_episodes.pt \
      --out_dir outputs/paper/dp --epochs 300
"""

import argparse
import copy
import json
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="ParcelStow-L6-Distill-Play-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--demos", type=str, required=True)
parser.add_argument("--out_dir", type=str, default="outputs/paper/dp")
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--batch", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--horizon", type=int, default=16)
parser.add_argument("--n_obs_steps", type=int, default=2)
parser.add_argument("--n_action_steps", type=int, default=8)
parser.add_argument("--num_inference_steps", type=int, default=100)
parser.add_argument("--model_seed", type=int, default=42)
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
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from diffusers.optimization import get_scheduler  # noqa: E402
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import parcelstow.tasks  # noqa: E402, F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stow_runtime as rt  # noqa: E402
from parcelstow.tasks.manager_based.parcel_stow import geometry as G  # noqa: E402
from parcelstow.tasks.manager_based.parcel_stow.mdp.metrics import StowMonitor  # noqa: E402
from third_party.diffusion_policy.conditional_unet1d import ConditionalUnet1D  # noqa: E402
from third_party.diffusion_policy.ema_model import EMAModel  # noqa: E402


class MinMaxNormalizer:
    def __init__(self, data, eps=1e-4):
        lo = data.min(dim=0).values
        hi = data.max(dim=0).values
        rng = (hi - lo).clamp(min=eps)
        self.scale = 2.0 / rng
        self.offset = -1.0 - lo * self.scale

    def to(self, device):
        self.scale = self.scale.to(device)
        self.offset = self.offset.to(device)
        return self

    def normalize(self, x):
        return x * self.scale + self.offset

    def state_dict(self):
        return {"scale": self.scale.cpu(), "offset": self.offset.cpu()}


def build_windows(episodes, horizon, pad_before, pad_after):
    obs_all, act_all, starts = [], [], []
    base = 0
    for obs, act in episodes:
        obs_p = torch.cat([obs[:1].repeat(pad_before, 1), obs, obs[-1:].repeat(pad_after, 1)])
        act_p = torch.cat([act[:1].repeat(pad_before, 1), act, act[-1:].repeat(pad_after, 1)])
        n = obs_p.shape[0]
        for s in range(0, n - horizon + 1):
            starts.append(base + s)
        obs_all.append(obs_p)
        act_all.append(act_p)
        base += n
    return torch.cat(obs_all), torch.cat(act_all), torch.tensor(starts, dtype=torch.long)


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
        print("[RESULT] " + json.dumps(rec), flush=True)
        with open(results_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    demos = torch.load(args_cli.demos)
    kept = [(o, a) for o, a, _ in demos["episodes"]]
    log({"stage": "demos", "path": args_cli.demos, "episodes": len(kept),
         "samples": int(sum(o.shape[0] for o, _ in kept))})

    T, To, Ta = args_cli.horizon, args_cli.n_obs_steps, args_cli.n_action_steps
    obs_dim = kept[0][0].shape[1]
    act_dim = kept[0][1].shape[1]
    obs_buf, act_buf, starts = build_windows(kept, T, To - 1, Ta - 1)
    norm_obs = MinMaxNormalizer(obs_buf).to(device)
    norm_act = MinMaxNormalizer(act_buf).to(device)
    obs_buf = obs_buf.to(device)
    act_buf = act_buf.to(device)
    starts = starts.to(device)
    torch.manual_seed(args_cli.model_seed)
    model = ConditionalUnet1D(input_dim=act_dim, global_cond_dim=obs_dim * To, diffusion_step_embed_dim=256,
                              down_dims=[256, 512, 1024], kernel_size=5, n_groups=8, cond_predict_scale=True).to(device)
    ema = EMAModel(copy.deepcopy(model), update_after_step=0, inv_gamma=1.0, power=0.75, min_value=0.0, max_value=0.9999)
    scheduler = DDPMScheduler(num_train_timesteps=100, beta_start=0.0001, beta_end=0.02, beta_schedule="squaredcos_cap_v2",
                              variance_type="fixed_small", clip_sample=True, prediction_type="epsilon")
    opt = torch.optim.AdamW(model.parameters(), lr=args_cli.lr, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
    steps_per_epoch = math.ceil(starts.numel() / args_cli.batch)
    lr_sched = get_scheduler("cosine", optimizer=opt, num_warmup_steps=500,
                             num_training_steps=steps_per_epoch * args_cli.epochs)
    win = torch.arange(T, device=device)
    t0 = time.time()
    last = float("nan")
    for ep in range(args_cli.epochs):
        perm = starts[torch.randperm(starts.numel(), device=device)]
        total = 0.0
        for i in range(0, perm.numel(), args_cli.batch):
            idx = perm[i:i + args_cli.batch].unsqueeze(1) + win.unsqueeze(0)
            nobs = norm_obs.normalize(obs_buf[idx])
            nact = norm_act.normalize(act_buf[idx])
            global_cond = nobs[:, :To].reshape(nobs.shape[0], -1)
            noise = torch.randn_like(nact)
            timesteps = torch.randint(0, 100, (nact.shape[0],), device=device).long()
            noisy = scheduler.add_noise(nact, noise, timesteps)
            pred = model(noisy, timesteps, global_cond=global_cond)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_sched.step()
            ema.step(model)
            total += float(loss) * nact.shape[0]
        last = total / starts.numel()
        if ep % 25 == 0 or ep == args_cli.epochs - 1:
            print(f"[TRAIN] epoch {ep} loss {last:.5f} elapsed {time.time() - t0:.0f}s", flush=True)
    log({"stage": "train", "epochs": args_cli.epochs, "windows": int(starts.numel()), "final_loss": last,
         "train_seconds": time.time() - t0, "params_m": sum(p.numel() for p in model.parameters()) / 1e6})
    ckpt = os.path.join(args_cli.out_dir, f"dp_{args_cli.tag}.pt")
    torch.save({"model": model.state_dict(), "ema": ema.averaged_model.state_dict(),
                "norm_obs": norm_obs.state_dict(), "norm_act": norm_act.state_dict(),
                "obs_dim": obs_dim, "act_dim": act_dim, "args": vars(args_cli)}, ckpt)

    geom = G.load_geometry()
    monitor = StowMonitor(base, geom)
    expert = rt.ExpertActor(base)
    switches = rt.EnvSwitches(base)
    stamp = rt.config_stamp(base)
    actor = rt.DPActor(ckpt, device, base.num_envs)
    recs, _ = rt.run_episodes(env, base, actor, monitor, args_cli.diag_episodes,
                              {"mode": "fixed", "value": args_cli.diag_rate}, args_cli.jitter, args_cli.eval_seed,
                              switches, expert=expert, corrupt=False, stamp=stamp, tag="dp_diag")
    s = rt.summarize(recs)
    log({"stage": "diag_eval", "diag_rate": args_cli.diag_rate, "task_success": s["task_success"],
         "acquired": s["acquired"], "inserted": s["inserted"], "settled": s["settled"],
         "failure_reasons": s["failure_reasons"], "checkpoint": ckpt})
    rt.write_jsonl(os.path.join(args_cli.out_dir, "diag.jsonl"), [rt.light_record(r) for r in recs])
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
