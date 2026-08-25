"""ACT training-seed replication summary (part A of the 2026-08-18 evening
protocol).

Reads the per-episode evaluation records of the expert and of every ACT
training seed at the shared evaluation draws (seeds 13345, 14345, 15345 for
r 1.0, 1.5, 2.0, jitter 1 cm, corruption off), checks that the parcel
start draws are identical across actors, and writes per seed and rate the
task success, stage markers, failure-reason counts, in-hand slip, and arm
velocity utilization, then across seeds the mean, min, max, and every
individual value. Nothing is averaged away, the per-seed rows stay in the
tables.

Inputs (defaults),
  outputs/paper/eval/expert.jsonl              expert, all grid rates
  outputs/paper/eval/act.jsonl                 act_seed1 (model_seed 0), all grid rates
  outputs/paper/act_multiseed/eval/act_seed2.jsonl   act_seed2 (model_seed 1)
  outputs/paper/act_multiseed/eval/act_seed3.jsonl   act_seed3 (model_seed 2)
  outputs/paper/act_multiseed/act_seed{2,3}/results.jsonl   training logs and r 1 diagnostics

Outputs,
  experiments/paper/results/act_multiseed_summary.json
  experiments/paper/results/act_multiseed_summary.csv

Run,
  python3 scripts/manipulation/summarize_act_multiseed.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "manipulation"))
import stow_common as G  # noqa: E402

RATES = [1.0, 1.5, 2.0]
STAGES = ["task_success", "acquired", "lifted_clear", "reoriented", "preinsert_reached", "inserted", "released", "settled"]
DISTS = ["max_hand_object_translation_m", "max_hand_object_rotation_deg", "max_arm_velocity_utilization",
         "epsilon_lift", "peak_hand_linear_velocity"]


def load(path):
    p = os.path.join(REPO, path)
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p)]


def dist(rows, key):
    v = np.array([r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])], dtype=float)
    if v.size == 0:
        return None
    return {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "mean": float(v.mean()),
            "max": float(v.max()), "n": int(v.size)}


def cell(rows):
    n = len(rows)
    out = {"n": n}
    for st in STAGES:
        k = sum(1 for r in rows if r.get(st))
        lo, hi = G.wilson(k, n)
        out[st] = {"k": k, "frac": k / n if n else None, "wilson": [lo, hi]}
    reasons = {}
    for r in rows:
        reasons[r["failure_reason"]] = reasons.get(r["failure_reason"], 0) + 1
    out["failure_reasons"] = dict(sorted(reasons.items(), key=lambda kv: -kv[1]))
    for d in DISTS:
        out[d] = dist(rows, d)
    acq = [r for r in rows if r.get("acquired")]
    out["max_hand_object_translation_m_acquired"] = dist(acq, "max_hand_object_translation_m")
    out["max_hand_object_rotation_deg_acquired"] = dist(acq, "max_hand_object_rotation_deg")
    return out


def draws(rows):
    return [(r["env"], tuple(round(v, 6) for v in r["parcel_initial_pose"]["pos"])) for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expert", default="outputs/paper/eval/expert.jsonl")
    ap.add_argument("--seeds", nargs="*", default=[
        "act_seed1=0=outputs/paper/eval/act.jsonl=outputs/paper/act/results.jsonl",
        "act_seed2=1=outputs/paper/act_multiseed/eval/act_seed2.jsonl=outputs/paper/act_multiseed/act_seed2/results.jsonl",
        "act_seed3=2=outputs/paper/act_multiseed/eval/act_seed3.jsonl=outputs/paper/act_multiseed/act_seed3/results.jsonl"],
        help="name=model_seed=eval_records=training_results")
    ap.add_argument("--rates", type=float, nargs="*", default=RATES)
    ap.add_argument("--out_json", default="experiments/paper/results/act_multiseed_summary.json")
    ap.add_argument("--out_csv", default="experiments/paper/results/act_multiseed_summary.csv")
    args = ap.parse_args()

    expert_rows = load(args.expert)
    seeds = []
    for spec in args.seeds:
        name, ms, ev, tr = spec.split("=")
        rows = load(ev)
        train = load(tr)
        if not rows:
            print(f"[skip] {name}, no evaluation records at {ev}")
            continue
        seeds.append({"name": name, "model_seed": int(ms), "eval_records": ev, "training_results": tr,
                      "rows": rows, "train": train})
    out = {"rates": args.rates, "protocol": {
        "episodes_per_cell": 100, "jitter_m": 0.01, "corruption": False,
        "eval_seeds": {f"{r:g}": 12345 + 1000 * (1 + i) for i, r in enumerate([1.0, 1.5, 2.0])},
        "note": "eval_stow_policies.py seeds 12345 + 1000 x rate index of the frozen grid, r 1.0, 1.5, 2.0 are indices 1, 2, 3"},
        "expert": {}, "seeds": [], "across_seeds": {}, "draw_check": {}}
    # per rate cells
    for r in args.rates:
        er = [x for x in expert_rows if float(x["task_rate"]) == r]
        out["expert"][f"{r:g}"] = cell(er)
    for sd in seeds:
        entry = {"name": sd["name"], "model_seed": sd["model_seed"], "eval_records": sd["eval_records"],
                 "training": {}, "cells": {}}
        for t in sd["train"]:
            if t.get("stage") == "train":
                entry["training"]["final_loss"] = t.get("final_loss")
                entry["training"]["train_seconds"] = t.get("train_seconds")
                entry["training"]["epochs"] = t.get("epochs")
                entry["training"]["params_m"] = t.get("params_m")
                entry["training"]["model_seed_logged"] = t.get("model_seed", 0)
            if t.get("stage") == "diag_eval":
                entry["training"]["diag_r1"] = {"task_success": t["task_success"], "acquired": t["acquired"],
                                                "inserted": t["inserted"], "settled": t["settled"],
                                                "failure_reasons": t["failure_reasons"], "checkpoint": t["checkpoint"]}
        for r in args.rates:
            rr = [x for x in sd["rows"] if float(x["task_rate"]) == r]
            entry["cells"][f"{r:g}"] = cell(rr)
            entry["cells"][f"{r:g}"]["eval_seed"] = rr[0]["seed"] if rr else None
            entry["cells"][f"{r:g}"]["checkpoint"] = rr[0].get("checkpoint") if rr else None
            er = [x for x in expert_rows if float(x["task_rate"]) == r]
            same = draws(rr) == draws(er) if rr and er else None
            out["draw_check"][f"{sd['name']}@{r:g}"] = {"identical_to_expert": same, "n": len(rr)}
        out["seeds"].append(entry)
    # across seeds
    for r in args.rates:
        vals = {}
        for st in ("task_success", "acquired", "inserted", "settled"):
            v = [sd_e["cells"][f"{r:g}"][st]["frac"] for sd_e in out["seeds"] if sd_e["cells"][f"{r:g}"]["n"]]
            vals[st] = {"per_seed": v, "mean": float(np.mean(v)) if v else None,
                        "min": float(np.min(v)) if v else None, "max": float(np.max(v)) if v else None,
                        "expert": out["expert"][f"{r:g}"][st]["frac"]}
        out["across_seeds"][f"{r:g}"] = vals
    # replication question, degradation from r 1 to r 2 relative to the expert
    rep = {}
    for sd_e in out["seeds"]:
        c = sd_e["cells"]
        if all(c[f"{r:g}"]["n"] for r in (1.0, 2.0)):
            s1, s2 = c["1"]["task_success"]["frac"], c["2"]["task_success"]["frac"]
            e1, e2 = out["expert"]["1"]["task_success"]["frac"], out["expert"]["2"]["task_success"]["frac"]
            rep[sd_e["name"]] = {"success_r1": s1, "success_r1.5": c["1.5"]["task_success"]["frac"], "success_r2": s2,
                                 "drop_r1_to_r2": s1 - s2, "expert_drop_r1_to_r2": e1 - e2,
                                 "gap_to_expert_r1": e1 - s1, "gap_to_expert_r2": e2 - s2,
                                 "gap_grows_with_rate": (e2 - s2) > (e1 - s1)}
    out["replication"] = rep
    os.makedirs(os.path.dirname(os.path.join(REPO, args.out_json)), exist_ok=True)
    with open(os.path.join(REPO, args.out_json), "w") as fh:
        json.dump(out, fh, indent=1)
    # csv, one row per actor and rate
    cols = ["actor", "model_seed", "rate", "n", "eval_seed"] + STAGES + ["failure_reasons",
            "slip_t_median_m", "slip_t_p90_m", "slip_r_median_deg", "slip_r_p90_deg", "arm_util_median", "arm_util_max",
            "epsilon_lift_median"]
    with open(os.path.join(REPO, args.out_csv), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)

        def row(actor, ms, r, c):
            def d(k, f):
                return c[k][f] if c.get(k) else ""
            w.writerow([actor, ms, r, c["n"], c.get("eval_seed", "")] + [c[st]["frac"] for st in STAGES]
                       + [json.dumps(c["failure_reasons"]), d("max_hand_object_translation_m", "median"),
                          d("max_hand_object_translation_m", "p90"), d("max_hand_object_rotation_deg", "median"),
                          d("max_hand_object_rotation_deg", "p90"), d("max_arm_velocity_utilization", "median"),
                          d("max_arm_velocity_utilization", "max"), d("epsilon_lift", "median")])
        for r in args.rates:
            row("expert", "", r, out["expert"][f"{r:g}"])
        for sd_e in out["seeds"]:
            for r in args.rates:
                row(sd_e["name"], sd_e["model_seed"], r, sd_e["cells"][f"{r:g}"])
    print(json.dumps({"across_seeds": out["across_seeds"], "replication": out["replication"],
                      "draw_check": out["draw_check"]}, indent=1))
    print(f"[written] {args.out_json} {args.out_csv}")


if __name__ == "__main__":
    main()
