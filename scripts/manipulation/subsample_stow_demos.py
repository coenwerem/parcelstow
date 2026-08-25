"""Rate-stratified demonstration subsets for the ACT demo-scaling sweep
(WRL workshop addition of 2026-08-21). The full set holds 297 successful
expert episodes with task_rate drawn from U[0.5, 2.0]. A subset of size n
sorts the episodes by task_rate and takes the ranks round(j (N-1) / (n-1))
for j in 0..n-1, so every subset spans the training rate range evenly and
smaller subsets nearly nest inside larger ones. The subset file preserves
the schema of the source file (episodes, records, rate_spec, jitter, seed,
obs_dim, act_dim, config) and adds a subsample provenance block, so
run_stow_act.py trains on a subset unchanged.

Run,
  python scripts/manipulation/subsample_stow_demos.py \
      --demos outputs/paper/demos/expert_episodes.pt \
      --sizes 50 100 --out_dir outputs/paper/act_demoscale
"""

import argparse
import json
import os

import numpy as np
import torch


def stratified_ranks(n_total, n_subset):
    js = np.arange(n_subset, dtype=np.float64)
    ranks = np.round(js * (n_total - 1) / (n_subset - 1)).astype(int)
    assert len(np.unique(ranks)) == n_subset
    return ranks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=str, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    data = torch.load(args.demos, map_location="cpu", weights_only=False)
    episodes = data["episodes"]
    records = data["records"]
    assert len(episodes) == len(records)
    rates = np.array([r["task_rate"] for r in records])
    order = np.argsort(rates, kind="stable")

    os.makedirs(args.out_dir, exist_ok=True)
    for n in args.sizes:
        ranks = stratified_ranks(len(episodes), n)
        idx = order[ranks]
        sub_rates = rates[idx]
        subset = {k: v for k, v in data.items() if k not in ("episodes", "records", "all_records")}
        subset["episodes"] = [episodes[i] for i in idx]
        subset["records"] = [records[i] for i in idx]
        subset["subsample"] = {
            "source": args.demos,
            "method": "rate_stratified_ranks",
            "n": int(n),
            "source_episodes": len(episodes),
            "indices": [int(i) for i in idx],
            "rate_min": float(sub_rates.min()),
            "rate_median": float(np.median(sub_rates)),
            "rate_max": float(sub_rates.max()),
        }
        out_path = os.path.join(args.out_dir, f"demos_n{n}.pt")
        torch.save(subset, out_path)
        print(json.dumps({k: v for k, v in subset["subsample"].items() if k != "indices"}))
        print(f"wrote {out_path} with {n} episodes")


if __name__ == "__main__":
    main()
