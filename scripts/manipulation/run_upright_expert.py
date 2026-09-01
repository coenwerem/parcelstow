"""Scripted-expert driver of the upright placement task, expert
validation, the speedup-factor sweep, and demonstration collection,
the run_stow_expert.py pattern.

Every episode runs under the physical monitor
(upright_place/mdp/monitor.py) and writes one JSON line. Success is
the physical predicate of the task geometry, nothing else.

Modes,
  validate   fixed speed, fixed jitter, N episodes
  sweep      the same over a speed list (Gate B expert calibration)
  demos      uniform rate in [lo, hi] and planar jitter, saves the
             physically successful episodes to --demo_out

Run,
  python scripts/manipulation/run_upright_expert.py --mode validate --rate 0.5 --episodes 20 \
      --out outputs/upright/expert/validate_r0.5.jsonl
  python scripts/manipulation/run_upright_expert.py --mode sweep --rates 0.5 0.75 1 1.25 1.5 2 2.5 3 \
      --episodes 64 --jitter 0.01 --out outputs/upright/expert/sweep.jsonl
"""

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="UprightPlace-L6-Play-v0")
parser.add_argument("--mode", choices=["validate", "sweep", "demos"], default="validate")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--rate", type=float, default=0.5)
parser.add_argument("--rates", type=float, nargs="*", default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])
parser.add_argument("--rate_lo", type=float, default=0.5)
parser.add_argument("--rate_hi", type=float, default=2.0)
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
import upright_place_expert as upe  # noqa: E402
from parcelstow.phase_schedule import PhaseSchedule  # noqa: E402
from parcelstow.tasks.manager_based.parcel_stow.mdp import task_clock  # noqa: E402
from parcelstow.tasks.manager_based.upright_place import geometry as U  # noqa: E402
from parcelstow.tasks.manager_based.upright_place.mdp.monitor import STAGE_KEYS, UprightMonitor  # noqa: E402

SCHED = PhaseSchedule(U.PHASES)


class UprightExpertActor:
    name = "expert"

    def __init__(self, base, bank=None, trajectory=None, candidate=0):
        self.base = base
        self.robot = base.scene["robot"]
        from parcelstow.tasks.manager_based.upright_place.upright_place_env_cfg import CHAIN_ACTUATED
        self.jids, self.jnames = self.robot.find_joints(CHAIN_ACTUATED, preserve_order=True)
        self.q_default = self.robot.data.default_joint_pos[:, self.jids]
        self.expert = upe.UprightExpert(self.jnames, bank=bank, trajectory=trajectory,
                                        device=base.device, candidate=candidate)
        self.expert.allocate(base.num_envs)
        self.start_xy = torch.tensor(U.START_POS[:2], device=base.device)

    def reset(self, ids, obs=None):
        ids = torch.as_tensor(list(ids), dtype=torch.long, device=self.base.device)
        if len(ids) == 0:
            return
        off = self.base._stow_start_pos[ids, :2] - self.start_xy
        self.expert.reset(ids, off)

    @torch.no_grad()
    def act(self, obs):
        k, f, _, _ = task_clock.phase_state(self.base)
        q_meas = self.robot.data.joint_pos[:, self.jids]
        return self.expert.act(k, f, self.q_default, q_meas)


def config_stamp(base):
    return {
        "git_sha": rt.git_sha(),
        "task": args_cli.task,
        "object_extents": list(U.OBJECT_EXTENTS), "object_mass": U.OBJECT_MASS,
        "object_friction": U.OBJECT_FRICTION, "object_start": list(U.START_POS),
        "target_center": list(U.TARGET_CENTER), "target_radius": U.TARGET_RADIUS,
        "place_z": U.PLACE_Z, "final_tilt_tol_deg": U.FINAL_TILT_TOL_DEG,
        "grasp_shift": U.GRASP_SHIFT,
        "phases": [[n, d, s] for n, d, s in U.PHASES],
        "control_dt": float(base.step_dt),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    trace_envs = list(range(args_cli.trace_envs))
    monitor = UprightMonitor(base, trace_envs=trace_envs)
    expert = UprightExpertActor(base)
    switches = rt.EnvSwitches(base, reset_term="reset_object")
    stamp = config_stamp(base)
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
