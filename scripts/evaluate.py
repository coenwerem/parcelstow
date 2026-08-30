"""Evaluate a policy on the ParcelStow execution-speed grid. Needs Isaac Lab.

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
  python scripts/evaluate.py --actor expert --rates 1.0 2.0 --episodes 100
  python scripts/evaluate.py --actor examples.custom_policy:HoldPosturePolicy \
      --rates 1.0 --episodes 5 --num_envs 8
"""

import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DRIVER = os.path.join(REPO, "scripts", "manipulation", "eval_stow_policies.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actor", nargs="+", required=True,
                    help="expert, act, dp, dagger, or module.path:ClassName")
    ap.add_argument("--rates", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--num_envs", type=int, default=32)
    ap.add_argument("--out_dir", default="outputs/eval")
    ap.add_argument("--custom_ckpt", default=None, help="checkpoint handed to a custom actor")
    ap.add_argument("--eval_seed", type=int, default=12345,
                    help="12345 reproduces the frozen paper draws")
    args, passthrough = ap.parse_known_args()

    cmd = [sys.executable, DRIVER,
           "--actors", *args.actor,
           "--rates", *[f"{r:g}" for r in args.rates],
           "--episodes", str(args.episodes),
           "--num_envs", str(args.num_envs),
           "--out_dir", args.out_dir,
           "--eval_seed", str(args.eval_seed)]
    if args.custom_ckpt:
        cmd += ["--custom_ckpt", args.custom_ckpt]
    cmd += passthrough
    os.chdir(REPO)
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
