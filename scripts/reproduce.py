"""Reproduce the reported quantitative results from the released episode
records, no Isaac.

Every target reads the frozen episode records shipped in data/records/ (or
the artifacts fetched by scripts/download_artifacts.py) and writes derived
analyses to outputs/reproduce/. Principal evaluation plots regenerate to
media/ as well. Nothing here reruns the simulator, and exact camera-ready
paper figures are not part of the release contract,
docs/REPRODUCING_THE_PAPER.md maps every reported quantity to its record
and command.

Targets,
  envelope     task-success counts and plot across execution speeds, with
               Wilson intervals, and the 20000-resample paired bootstrap
               interval of the expert over ACT-A success gap at r=2
  stages       per-stage completion against execution speed for every policy
  certificate  realized-contact force-closure analysis of the acquisition
               diagnostic
  certificate-oos  held-out force-closure ranking and calibration
  expert-ceiling  arm joint-velocity utilization and target-tracking error
               measured as the expert's success decreases at higher speeds
  all          every target above

Run,
  python scripts/reproduce.py envelope
  python scripts/reproduce.py all
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RECORDS = os.path.join(REPO, "data", "records")
EVAL_DIR = os.path.join(REPO, "outputs", "paper", "eval")
ACTORS = ["expert", "act", "dagger", "dp"]


REPLICATION = {
    "act_seed1_rerun.jsonl.gz": "act_multiseed/eval/act_seed1_rerun.jsonl",
    "act_seed2.jsonl.gz": "act_multiseed/eval/act_seed2.jsonl",
    "act_seed3.jsonl.gz": "act_multiseed/eval/act_seed3.jsonl",
    "act_n50.jsonl.gz": "act_demoscale/eval/act_n50.jsonl",
    "act_n100.jsonl.gz": "act_demoscale/eval/act_n100.jsonl",
}


def _gunzip(src, dst):
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        print(f"[records] {src} -> {dst}")


def ensure_records():
    """Materialize the shipped records where the analyzers expect them."""
    os.makedirs(EVAL_DIR, exist_ok=True)
    for a in ACTORS:
        _gunzip(os.path.join(RECORDS, a + "_episodes.jsonl.gz"),
                os.path.join(EVAL_DIR, a + ".jsonl"))
    for src, rel in REPLICATION.items():
        _gunzip(os.path.join(RECORDS, "replication", src),
                os.path.join(REPO, "outputs", "paper", rel))
    dst = os.path.join(EVAL_DIR, "summary.jsonl")
    if not os.path.exists(dst):
        shutil.copyfile(os.path.join(RECORDS, "eval_summary.jsonl"), dst)


def run(cmd):
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def target_envelope():
    run([sys.executable, "scripts/plot_envelope.py", "--gap", "expert", "act", "--gap_rate", "2.0"])


def target_stages():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(RECORDS, "eval_summary.jsonl")) as f:
        rows = [json.loads(line) for line in f]
    stages = ["acquired", "lifted_clear", "reoriented", "inserted", "settled", "task_success"]
    labels = {"expert": "Expert", "act": "ACT-A", "dp": "Diffusion Policy", "dagger": "DAgger"}
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4), sharey=True)
    for ax, actor in zip(axes, ["expert", "act", "dp", "dagger"]):
        sub = sorted((r for r in rows if r["policy"] == actor), key=lambda r: r["rate"])
        for st in stages:
            ax.plot([r["rate"] for r in sub], [r[st]["frac"] for r in sub],
                    marker="o", ms=3, lw=1.2, label=st)
        ax.set_title(labels[actor])
        ax.set_xlabel("speedup factor r")
        ax.grid(True, alpha=0.15)
    axes[0].set_ylabel("stage completion")
    axes[0].legend(fontsize=8, frameon=False)
    out = os.path.join(REPO, "media", "stages_vs_rate")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out + ".pdf", bbox_inches="tight", facecolor="white")
    print(f"[stages] wrote {out}.png and .pdf")


def target_certificate():
    ensure_records()
    out = os.path.join(REPO, "outputs", "reproduce", "certificate_analysis.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    run([sys.executable, "scripts/manipulation/analyze_stow_certificate.py",
         "--eval_dir", EVAL_DIR, "--out", out])


def target_certificate_oos():
    ensure_records()
    out_dir = os.path.join(REPO, "outputs", "reproduce")
    os.makedirs(out_dir, exist_ok=True)
    run([sys.executable, "scripts/manipulation/analyze_certificate_oos.py",
         "--eval_dir", EVAL_DIR,
         "--multiseed_dir", os.path.join(REPO, "outputs", "paper", "act_multiseed", "eval"),
         "--out_dir", out_dir])


def target_expert_ceiling():
    ensure_records()
    out_dir = os.path.join(REPO, "outputs", "reproduce")
    os.makedirs(out_dir, exist_ok=True)
    run([sys.executable, "scripts/manipulation/analyze_expert_ceiling.py",
         "--eval_path", os.path.join(EVAL_DIR, "expert.jsonl"), "--out_dir", out_dir])


TARGETS = {
    "envelope": target_envelope,
    "stages": target_stages,
    "certificate": target_certificate,
    "certificate-oos": target_certificate_oos,
    "expert-ceiling": target_expert_ceiling,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", choices=[*TARGETS, "all"])
    args = ap.parse_args()
    names = list(TARGETS) if args.target == "all" else [args.target]
    for n in names:
        TARGETS[n]()


if __name__ == "__main__":
    main()
