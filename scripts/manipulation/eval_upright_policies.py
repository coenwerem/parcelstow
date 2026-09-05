"""Matched expert-learner evaluation of the upright placement task
across execution speeds, the eval_stow_policies.py protocol: an indexed
initial-condition bank pairs each policy's robot and object reset state,
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from evaluation_protocol import (  # noqa: E402
    actor_record_filename,
    evaluation_output_paths,
    reject_existing_protected_outputs,
    resolve_checkpoint_bindings,
    validate_evaluation_role,
)
from task_registry import get_task_by_gym_id  # noqa: E402

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
parser.add_argument("--bank_role", choices=["unclassified", "development", "final", "diagnostic"],
                    default="unclassified")
parser.add_argument("--diagnostic_fixed_phase_rate", type=float, default=None,
                    help="replace the policy rate input during phases 0 through 4")
parser.add_argument("--out_dir", type=str, default="outputs/upright/eval")
parser.add_argument("--tag", type=str, default="")
parser.add_argument("--trace_envs", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
task_spec = get_task_by_gym_id(args_cli.task)
validate_evaluation_role(
    bank_role=args_cli.bank_role,
    rates=args_cli.rates,
    registered_rates=task_spec.default_rates,
    episodes=args_cli.episodes,
    eval_seed=args_cli.eval_seed,
    jitter=args_cli.jitter,
    corrupt=False,
    action_noise=0.0,
    fixed_phase_rate_override=args_cli.diagnostic_fixed_phase_rate,
    actors=args_cli.actors,
)
reject_existing_protected_outputs(
    args_cli.bank_role,
    evaluation_output_paths(args_cli.out_dir, args_cli.actors, args_cli.tag),
)
checkpoint_bindings = resolve_checkpoint_bindings(
    args_cli.actors,
    {
        "dagger": args_cli.dagger_ckpt,
        "dp": args_cli.dp_ckpt,
        "act": args_cli.act_ckpt,
    },
    args_cli.custom_ckpt,
)
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import parcelstow.tasks  # noqa: E402, F401
import stow_runtime as rt  # noqa: E402
from act_diagnostics import FIXED_ACQUISITION_PHASES, FixedPhaseRateActor  # noqa: E402
from parcelstow.tasks.manager_based.upright_place.mdp.monitor import STAGE_KEYS, UprightMonitor  # noqa: E402
from upright_runtime import SCHED, UprightExpertActor, config_stamp  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.seed = args_cli.eval_seed
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
    t0 = time.time()

    for name in args_cli.actors:
        binding = checkpoint_bindings[name]
        actor = expert if name == "expert" else rt.load_actor(
            name, binding.checkpoint, base, args_cli.num_envs
        )
        if args_cli.diagnostic_fixed_phase_rate is not None:
            if name == "expert":
                raise ValueError("the fixed-phase rate intervention applies only to learned policies")
            actor = FixedPhaseRateActor(actor, args_cli.diagnostic_fixed_phase_rate)
        ep_path = os.path.join(
            args_cli.out_dir,
            actor_record_filename(name, tag),
        )
        for ri, r in enumerate(args_cli.rates):
            seed = args_cli.eval_seed + 1000 * ri
            recs, _ = rt.run_episodes(env, base, actor, monitor, args_cli.episodes,
                                      {"mode": "fixed", "value": r}, args_cli.jitter, seed, switches,
                                      expert=expert, corrupt=False, stamp=stamp, tag=f"{name}_r{r:g}",
                                      task_id=args_cli.task, cycle_time=SCHED.cycle_time,
                                      indexed_initial_conditions=True,
                                      extra={"actor_spec": name, "num_envs": args_cli.num_envs,
                                             "bank_role": args_cli.bank_role,
                                             "fixed_phase_rate_override":
                                                 args_cli.diagnostic_fixed_phase_rate,
                                             "fixed_phase_indices":
                                                 list(FIXED_ACQUISITION_PHASES)
                                                 if args_cli.diagnostic_fixed_phase_rate is not None
                                                 else None,
                                             "checkpoint": binding.checkpoint,
                                             "checkpoint_sha256": binding.checkpoint_sha256})
            rt.write_jsonl(ep_path, recs)
            row = rt.summarize(recs, stage_keys=STAGE_KEYS)
            row.update({"policy": actor.name, "actor_spec": name, "rate": r,
                        "cycle_time_s": SCHED.cycle_time(r), "seed": seed,
                        "num_envs": args_cli.num_envs,
                        "bank_role": args_cli.bank_role,
                        "fixed_phase_rate_override": args_cli.diagnostic_fixed_phase_rate,
                        "fixed_phase_indices": list(FIXED_ACQUISITION_PHASES)
                        if args_cli.diagnostic_fixed_phase_rate is not None else None,
                        "jitter": args_cli.jitter, "episodes_requested": args_cli.episodes,
                        "checkpoint": binding.checkpoint,
                        "checkpoint_sha256": binding.checkpoint_sha256,
                        "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
            rt.write_jsonl(os.path.join(args_cli.out_dir, f"summary{tag}.jsonl"), [row])
            print(f"[EVAL {name} r={r:g}] success {row['task_success']['k']}/{row['task_success']['n']} "
                  f"reasons {row['failure_reasons']} ({time.time()-t0:.0f}s)", flush=True)
    env.close()


if __name__ == "__main__":
    rt.run_simulation_main(main, simulation_app)
