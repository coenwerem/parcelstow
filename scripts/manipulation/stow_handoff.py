"""Common-controller handoff diagnostic of the ParcelStow task (M12), an
ablation, not the main method.

The actor (expert, DAgger, DP, or ACT) acquires the parcel on its own. At
stable lift onset (the monitor's acquired marker plus the parcel 40 mm
above its start height) the driver freezes the actor's realized hand
target and replaces the waist and arm outputs with one common downstream
trajectory, the expert's IK path for LIFT through RETREAT at the episode's
speedup factor, blended in from the actor's arm target over 0.5 s. From RELEASE
onward the hand follows the expert's opening. The parcel stays a free rigid
body throughout, and the handoff runs in the live episode (no snapshot or
restore). The question the run answers, given the grasp this actor
acquired, how much downstream demand does that grasp tolerate under the
same arm motion.

Records hold handoff True and handoff_step. Episodes in which the actor
never acquires the parcel run to their end with the actor in charge and are
marked handoff False.

Run,
  python scripts/manipulation/stow_handoff.py --actors expert dagger dp act \
      --rates 0.5 1.0 1.5 2.0 2.5 --episodes 50 --out_dir outputs/paper/handoff
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
parser.add_argument("--rates", type=float, nargs="*", default=[0.5, 1.0, 1.5, 2.0, 2.5])
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--jitter", type=float, default=0.01)
parser.add_argument("--eval_seed", type=int, default=12345)
parser.add_argument("--handoff_dz", type=float, default=0.04)
parser.add_argument("--blend_steps", type=int, default=25)
parser.add_argument("--out_dir", type=str, default="outputs/paper/handoff")
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
from parcelstow.tasks.manager_based.parcel_stow.mdp import task_clock  # noqa: E402
from parcelstow.tasks.manager_based.parcel_stow.mdp.metrics import StowMonitor  # noqa: E402
import parcel_stow_expert as pse  # noqa: E402


class HandoffController:
    """Step hook, replaces the arm action after stable lift onset with the
    common expert trajectory and freezes the actor's hand target."""

    def __init__(self, base, expert, dz, blend_steps):
        self.base = base
        self.expert = expert
        self.dz = dz
        self.blend = blend_steps
        n = base.num_envs
        d = base.device
        self.active = torch.zeros(n, dtype=torch.bool, device=d)
        self.since = torch.zeros(n, dtype=torch.long, device=d)
        self.hand_hold = torch.zeros(n, 6, device=d)
        self.arm_offset = torch.zeros(n, 10, device=d)
        self.handoff_step = torch.full((n,), -1, dtype=torch.long, device=d)
        self.last_steps = torch.zeros(n, dtype=torch.long, device=d)
        self.arm_idx = expert.expert.arm_idx
        self.hand_idx = expert.expert.hand_idx
        self.k_release = G.PHASE_INDEX["RELEASE"]

    def hook(self, base, obs, act, monitor):
        # detect environment resets through the monitor step counter
        reset = monitor.steps < self.last_steps
        if reset.any():
            self.active[reset] = False
            self.handoff_step[reset] = -1
        self.last_steps = monitor.steps.clone()
        k, f, _, _ = task_clock.phase_state(base)
        q_default = self.expert.q_default
        q_target_actor = 0.5 * act + q_default
        parcel = base.scene["parcel"]
        z_rel = parcel.data.root_pos_w[:, 2] - base.scene.env_origins[:, 2] - base._stow_start_pos[:, 2]
        trigger = monitor.acquired & (z_rel >= self.dz) & ~self.active & (k < self.k_release)
        # the common controller is the expert's own command at the visited state,
        # phase target plus its dwell integral correction (run_episodes calls
        # expert.act every step, so the correction follows the visited states)
        q_exp = self.expert.expert.target(k, f, q_default) + self.expert.expert.corr
        if trigger.any():
            self.hand_hold[trigger] = q_target_actor[trigger][:, self.hand_idx]
            self.arm_offset[trigger] = (q_target_actor[trigger][:, self.arm_idx] - q_exp[trigger][:, self.arm_idx])
            self.since[trigger] = 0
            self.handoff_step[trigger] = monitor.steps[trigger]
            self.active[trigger] = True
        if not self.active.any():
            return act
        # expert arm targets with a decaying offset, the actor's frozen hand
        # target until RELEASE, then the expert hand
        w = (1.0 - self.since.float() / float(self.blend)).clamp(0.0, 1.0).unsqueeze(1)
        q_new = q_target_actor.clone()
        arm = q_exp[:, self.arm_idx] + w * self.arm_offset
        hand = torch.where((k >= self.k_release).unsqueeze(1), q_exp[:, self.hand_idx], self.hand_hold)
        q_new[:, self.arm_idx] = torch.where(self.active.unsqueeze(1), arm, q_new[:, self.arm_idx])
        q_new[:, self.hand_idx] = torch.where(self.active.unsqueeze(1), hand, q_new[:, self.hand_idx])
        self.since += self.active.long()
        return pse.to_action(q_new, q_default)


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
    ckpts = {"dagger": args_cli.dagger_ckpt, "dp": args_cli.dp_ckpt, "act": args_cli.act_ckpt}
    summary_path = os.path.join(args_cli.out_dir, "summary.jsonl")
    t0 = time.time()
    for name in args_cli.actors:
        actor = expert if name == "expert" else rt.load_actor(name, ckpts[name], base, n)
        controller = HandoffController(base, expert, args_cli.handoff_dz, args_cli.blend_steps)
        rec_path = os.path.join(args_cli.out_dir, f"{name}.jsonl")
        for ri, r in enumerate(args_cli.rates):
            seed = args_cli.eval_seed + 1000 * ri
            recs, _ = rt.run_episodes(env, base, actor, monitor, args_cli.episodes, {"mode": "fixed", "value": r},
                                      args_cli.jitter, seed, switches, expert=expert, corrupt=False, stamp=stamp,
                                      tag=f"handoff_{name}_r{r:g}", step_hook=controller.hook,
                                      extra={"handoff": True, "handoff_dz": args_cli.handoff_dz,
                                             "acquisition_actor": name, "checkpoint": ckpts.get(name)})
            # attach handoff step from the controller (episodes finished in order of done events)
            rt.write_jsonl(rec_path, recs)
            s = rt.summarize(recs)
            s.update({"policy": name, "rate": r, "cycle_time_s": G.cycle_time(r), "seed": seed, "handoff": True,
                      "jitter": args_cli.jitter, "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
            rt.write_jsonl(summary_path, [s])
            print(f"[HANDOFF {name} r={r:g}] success {s['task_success']['k']}/{s['task_success']['n']} "
                  f"reasons {s['failure_reasons']} ({time.time()-t0:.0f}s)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
