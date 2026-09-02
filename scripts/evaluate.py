"""Evaluate a policy on one ParcelStow benchmark task. Needs Isaac Lab.

The command wraps the validated evaluation driver
scripts/manipulation/eval_stow_policies.py, which runs every requested
actor over every requested speed on the shared frozen evaluation draws and
writes one episode-record JSONL per actor plus a summary JSONL.

Actors,
  expert                     scripted expert policy
  act | dp | dagger          released learner policies, checkpoints via
                             scripts/download_artifacts.py --paper
  module.path:ClassName      your policy, see docs/POLICY_INTERFACE.md

Run,
  python scripts/evaluate.py --task parcel --actor expert
  python scripts/evaluate.py --task upright --actor expert
  python scripts/evaluate.py --task peg --actor expert
  python scripts/evaluate.py --actor expert --rates 1.0 2.0 --episodes 100
  python scripts/evaluate.py --actor examples.custom_policy:HoldPosturePolicy \
      --rates 1.0 --episodes 5 --num_envs 8
"""

import argparse
import os
import sys

from task_registry import ALIASES, REPO, get_task, task_output_dir


def build_command(args, passthrough, *, task_explicit):
    spec = get_task(args.task)
    rates = spec.default_rates if args.rates is None else args.rates
    out_dir = task_output_dir(spec, args.out_dir, legacy_default=not task_explicit)
    command = [sys.executable, str(spec.driver_path), "--task", spec.gym_id,
               "--actors", *args.actor, "--rates", *[f"{rate:g}" for rate in rates],
               "--episodes", str(args.episodes), "--num_envs", str(args.num_envs),
               "--out_dir", out_dir, "--eval_seed", str(args.eval_seed)]
    if args.custom_ckpt:
        command += ["--custom_ckpt", args.custom_ckpt]
    return command + passthrough


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    task_explicit = "--task" in sys.argv or any(arg.startswith("--task=") for arg in sys.argv)
    ap.add_argument("--task", choices=ALIASES, default="parcel",
                    help="benchmark task alias (default: parcel)")
    ap.add_argument("--actor", nargs="+", required=True,
                    help="expert, act, dp, dagger, or module.path:ClassName")
    ap.add_argument("--rates", type=float, nargs="+", default=None,
                    help="speedup factors (default: released grid for the selected task)")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--num_envs", type=int, default=32)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--custom_ckpt", default=None, help="checkpoint handed to a custom actor")
    ap.add_argument("--eval_seed", type=int, default=12345,
                    help="12345 reproduces the frozen paper draws")
    args, passthrough = ap.parse_known_args()

    cmd = build_command(args, passthrough, task_explicit=task_explicit)
    os.chdir(str(REPO))
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
