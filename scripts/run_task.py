"""Short ParcelStow run, the Isaac entry point with low-friction defaults.

Runs the scripted expert (or any actor of scripts/evaluate.py) for a few
episodes at one rate and prints the per-condition summary. Use it as the
first simulator check after installation.

Run,
  python scripts/run_task.py
  python scripts/run_task.py --actor act --rate 2.0 --episodes 10
"""

import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DRIVER = os.path.join(REPO, "scripts", "manipulation", "eval_stow_policies.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actor", default="expert")
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--num_envs", type=int, default=8)
    ap.add_argument("--out_dir", default="outputs/quickstart")
    args, passthrough = ap.parse_known_args()

    cmd = [sys.executable, DRIVER,
           "--actors", args.actor,
           "--rates", f"{args.rate:g}",
           "--episodes", str(args.episodes),
           "--num_envs", str(args.num_envs),
           "--out_dir", args.out_dir]
    cmd += passthrough
    os.chdir(REPO)
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
