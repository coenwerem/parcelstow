"""Plot task success across execution speeds from evaluation records.

Runs without Isaac. Reads the per-condition summary records (JSONL, one row
per policy and execution speed with success counts and Wilson intervals) and
draws success fraction against the speedup factor r for every policy. With
--gap the script also computes a paired bootstrap confidence interval for
the success difference between two policies at one speed from the per-episode
records, which share their evaluation draws across policies.

Run,
  python scripts/plot_envelope.py
  python scripts/plot_envelope.py --gap expert act --gap_rate 2.0
"""

from __future__ import annotations

import argparse
import gzip
import json
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

LABELS = {"expert": "Expert", "act": "ACT-A", "dp": "Diffusion Policy", "dagger": "DAgger"}
COLORS = {"expert": "#2e7d32", "act": "#0d47a1", "dp": "#bf360c", "dagger": "#9e9e9e"}
ORDER = ["expert", "act", "dp", "dagger"]


def read_jsonl(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_records(name, override=None):
    """Locate a released record file, preferring a decompressed local copy."""
    if override:
        return override
    candidates = [
        os.path.join(REPO, "outputs", "paper", "eval", name + ".jsonl"),
        os.path.join(REPO, "data", "records", name + "_episodes.jsonl.gz"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"no released records for {name}, looked at {candidates}")


def paired_gap_ci(recs_a, recs_b, rate, n_boot=20000, seed=0):
    """Percentile bootstrap interval for the success gap of actor A over
    actor B at one speed, paired over the shared evaluation draws."""
    def successes(recs):
        rows = sorted(
            (r for r in recs if abs(r["task_rate"] - rate) < 1e-9),
            key=lambda r: r["episode"],
        )
        return np.array([bool(r["task_success"]) for r in rows], dtype=float)

    sa, sb = successes(recs_a), successes(recs_b)
    if len(sa) != len(sb) or len(sa) == 0:
        raise ValueError(f"unpaired records at rate {rate}, {len(sa)} vs {len(sb)}")
    diff = sa - sb
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), len(diff)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default=None, help="summary JSONL, defaults to the released records")
    ap.add_argument("--actors", nargs="*", default=ORDER)
    ap.add_argument("--out", default=os.path.join(REPO, "media", "operating_envelope"))
    ap.add_argument("--gap", nargs=2, metavar=("ACTOR_A", "ACTOR_B"), default=None)
    ap.add_argument("--gap_rate", type=float, default=2.0)
    ap.add_argument("--records_a", default=None, help="episode records of the first --gap actor")
    ap.add_argument("--records_b", default=None, help="episode records of the second --gap actor")
    ap.add_argument("--no_figure", action="store_true")
    args = ap.parse_args()

    summary_path = args.summary or os.path.join(REPO, "data", "records", "eval_summary.jsonl")
    rows = read_jsonl(summary_path)
    print(f"[envelope] {summary_path}, {len(rows)} conditions")
    print(f"{'actor':<10}{'rate':>6}{'success':>10}{'wilson 95%':>20}")
    by_actor = {}
    for r in sorted(rows, key=lambda r: (ORDER.index(r["policy"]) if r["policy"] in ORDER else 99, r["rate"])):
        ts = r["task_success"]
        by_actor.setdefault(r["policy"], []).append((r["rate"], ts["frac"], ts["wilson"][0], ts["wilson"][1]))
        print(f"{r['policy']:<10}{r['rate']:>6g}{ts['k']:>6}/{ts['n']:<3}"
              f"{'[' + format(ts['wilson'][0], '.3f') + ', ' + format(ts['wilson'][1], '.3f') + ']':>20}")

    if not args.no_figure:
        from pathlib import Path

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.font_manager as fm
        import matplotlib.pyplot as plt

        for p in ("/usr/share/fonts/truetype/cmu/cmunsx.ttf", "/usr/share/fonts/truetype/cmu/cmunss.ttf"):
            if Path(p).exists():
                fm.fontManager.addfont(p)
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans", "Arial"],
            "font.size": 13, "axes.labelsize": 18, "axes.titlesize": 20,
            "axes.titleweight": "bold", "axes.linewidth": 1,
            "legend.fontsize": 12, "xtick.labelsize": 14, "ytick.labelsize": 14,
            "figure.dpi": 600, "grid.alpha": 0.12, "grid.linewidth": 0.3,
            "text.usetex": False, "axes.unicode_minus": False,
        })
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for actor in args.actors:
            if actor not in by_actor:
                continue
            pts = by_actor[actor]
            x = [p[0] for p in pts]
            y = [p[1] for p in pts]
            lo = [max(0.0, p[1] - p[2]) for p in pts]
            hi = [max(0.0, p[3] - p[1]) for p in pts]
            ax.errorbar(x, y, yerr=[lo, hi], marker="o", ms=4, lw=1.6, capsize=2,
                        color=COLORS.get(actor, "#444444"), label=LABELS.get(actor, actor))
        ax.set_xlabel("speedup factor r")
        ax.set_ylabel("success fraction")
        ax.set_ylim(-0.03, 1.05)
        ax.grid(True)
        ax.legend(frameon=False)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        fig.savefig(args.out + ".png", dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.02)
        fig.savefig(args.out + ".pdf", bbox_inches="tight", facecolor="white", pad_inches=0.02)
        print(f"[envelope] wrote {args.out}.png and .pdf")

    if args.gap:
        a, b = args.gap
        recs_a = read_jsonl(find_records(a, args.records_a))
        recs_b = read_jsonl(find_records(b, args.records_b))
        mean, lo, hi, n = paired_gap_ci(recs_a, recs_b, args.gap_rate)
        print(f"[gap] {LABELS.get(a, a)} minus {LABELS.get(b, b)} at r={args.gap_rate:g}, "
              f"n={n} paired draws, mean {mean:.2f}, bootstrap 95% [{lo:.2f}, {hi:.2f}]")


if __name__ == "__main__":
    main()
