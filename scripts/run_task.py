"""Short ParcelStow run, the Isaac entry point with low-friction defaults.

Runs the scripted expert (or any actor of scripts/evaluate.py) for a few
episodes at one rate and prints the per-condition summary. Use it as the
first simulator check after installation.

Run,
  python scripts/run_task.py
  python scripts/run_task.py --task parcel
  python scripts/run_task.py --task upright
  python scripts/run_task.py --task peg
  python scripts/run_task.py --actor act --rate 2.0 --episodes 10
"""

import argparse
import os
import sys

from task_registry import ALIASES, REPO, get_task


def build_command(args, passthrough, *, task_explicit):
    spec = get_task(args.task)
    if args.out_dir:
        out_dir = args.out_dir
    elif task_explicit:
        out_dir = f"outputs/quickstart/{spec.alias}"
    else:
        out_dir = "outputs/quickstart"
    return [sys.executable, str(spec.driver_path), "--task", spec.gym_id,
            "--actors", args.actor, "--rates", f"{args.rate:g}",
            "--episodes", str(args.episodes), "--num_envs", str(args.num_envs),
            "--out_dir", out_dir, *passthrough]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    task_explicit = "--task" in sys.argv or any(arg.startswith("--task=") for arg in sys.argv)
    ap.add_argument("--task", choices=ALIASES, default="parcel",
                    help="benchmark task alias (default: parcel)")
    ap.add_argument("--actor", default="expert")
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--num_envs", type=int, default=8)
    ap.add_argument("--out_dir", default=None)
    args, passthrough = ap.parse_known_args()

    cmd = build_command(args, passthrough, task_explicit=task_explicit)
    os.chdir(str(REPO))
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
