"""Relative-motion handoff summary tables (part B of the 2026-08-18 evening
protocol). Recomputes every per-cell summary from the per-episode records
of stow_relative_handoff.py (so the tables and the records agree even if
the driver's summary rows were appended by several runs), checks the start
draws across actors, and writes

  experiments/paper/results/relative_handoff_summary.jsonl
  experiments/paper/results/relative_handoff_summary.csv

Run,
  python3 scripts/manipulation/summarize_relative_handoff.py
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


def load(path):
    p = os.path.join(REPO, path)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def dist(rows, key):
    v = np.array([r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])], dtype=float)
    if v.size == 0:
        return None
    return {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "max": float(v.max()),
            "mean": float(np.mean(v)), "n": int(v.size)}


def frac(rows, key):
    n = len(rows)
    k = sum(1 for r in rows if r.get(key))
    lo, hi = G.wilson(k, n)
    return {"k": k, "n": n, "frac": k / n if n else None, "wilson": [lo, hi]}


def summarize(recs, name, rate):
    hand = [r for r in recs if r.get("relative_handoff")]
    reached = [r for r in hand if r.get("primary_endpoint_reached")]
    reasons = {}
    for r in recs:
        reasons[r["failure_reason"]] = reasons.get(r["failure_reason"], 0) + 1
    ep_reason = {}
    for r in hand:
        ep_reason[r.get("primary_endpoint_reason")] = ep_reason.get(r.get("primary_endpoint_reason"), 0) + 1
    out = {
        "policy": name, "rate": rate, "episodes": len(recs), "seed": recs[0]["seed"] if recs else None,
        "checkpoint": recs[0].get("checkpoint") if recs else None,
        "params": recs[0].get("relative_handoff_params") if recs else None,
        "acquired": frac(recs, "acquired"), "handoff_episodes": len(hand),
        "retained_preinsert": frac(hand, "retained_preinsert"),
        "primary_endpoint_reason": ep_reason,
        "dp_preinsert_m": dist(reached, "dp_preinsert_m"), "dR_preinsert_deg": dist(reached, "dR_preinsert_deg"),
        "dp_max_segment_m": dist(hand, "dp_max_segment_m"), "dR_max_segment_deg": dist(hand, "dR_max_segment_deg"),
        "dp_preinsert_from_acquisition_m": dist(reached, "dp_preinsert_from_acquisition_m"),
        "dR_preinsert_from_acquisition_deg": dist(reached, "dR_preinsert_from_acquisition_deg"),
        "contact_count_handoff": dist(hand, "contact_count_handoff"),
        "contact_count_preinsert_endpoint": dist(reached, "contact_count_preinsert_endpoint"),
        "epsilon_handoff": dist(hand, "epsilon_handoff"),
        "epsilon_handoff_positive_frac": (float(np.mean([r["epsilon_handoff"] > 0 for r in hand])) if hand else None),
        "epsilon_lift": dist(hand, "epsilon_lift"),
        "receptacle_force_before_endpoint_max": dist(hand, "receptacle_force_before_endpoint"),
        "handoff_anchor_minus_measured_m": dist(hand, "handoff_anchor_minus_measured_m"),
        "handoff_anchor_minus_measured_deg": dist(hand, "handoff_anchor_minus_measured_deg"),
        "handoff_command_lead_m": dist(hand, "handoff_command_lead_m"),
        "handoff_phase": {ph: sum(1 for r in hand if r.get("handoff_phase") == ph) for ph in set(r.get("handoff_phase") for r in hand)},
        "max_kinematic_residual_pos_m": dist(hand, "max_kinematic_residual_pos_m"),
        "max_kinematic_residual_rot_deg": dist(hand, "max_kinematic_residual_rot_deg"),
        "max_kinematic_model_error_pos_m": dist(hand, "max_kinematic_model_error_pos_m"),
        "max_pose_tracking_error_pos_m": dist(hand, "max_pose_tracking_error_pos_m"),
        "hand_hold_violation_rad": dist(hand, "hand_hold_violation_rad"),
        # secondary endpoint, the same episodes through insertion and release
        "secondary": {"task_success_all": frac(recs, "task_success"), "task_success_given_handoff": frac(hand, "task_success"),
                      "inserted_given_handoff": frac(hand, "inserted"), "settled_given_handoff": frac(hand, "settled"),
                      "failure_reasons": reasons},
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_dir", default="outputs/paper/relative_handoff")
    ap.add_argument("--actors", nargs="*", default=["expert", "act_seed1", "act_seed2", "act_seed3"])
    ap.add_argument("--out_jsonl", default="experiments/paper/results/relative_handoff_summary.jsonl")
    ap.add_argument("--out_csv", default="experiments/paper/results/relative_handoff_summary.csv")
    args = ap.parse_args()
    rows_out = []
    draws = {}
    for name in args.actors:
        recs = load(os.path.join(args.in_dir, f"{name}.jsonl"))
        if not recs:
            print(f"[skip] {name}")
            continue
        for rate in sorted(set(r["task_rate"] for r in recs)):
            rr = [r for r in recs if r["task_rate"] == rate]
            rows_out.append(summarize(rr, name, rate))
            draws[(name, rate)] = [(r["env"], tuple(round(v, 6) for v in r["parcel_initial_pose"]["pos"])) for r in rr]
    # draw identity across actors per rate
    check = {}
    for (name, rate), d in draws.items():
        ref = draws.get(("expert", rate))
        check[f"{name}@{rate:g}"] = (d == ref) if ref is not None else None
    for row in rows_out:
        row["draws_identical_to_expert"] = check.get(f"{row['policy']}@{row['rate']:g}")
    os.makedirs(os.path.dirname(os.path.join(REPO, args.out_jsonl)), exist_ok=True)
    with open(os.path.join(REPO, args.out_jsonl), "w") as fh:
        for row in rows_out:
            fh.write(json.dumps(row) + "\n")
    cols = ["policy", "rate", "episodes", "acquired", "handoff_episodes", "retained_k", "retained_frac", "retained_wilson_lo",
            "retained_wilson_hi", "endpoint_insert_start", "endpoint_receptacle_contact", "endpoint_not_reached",
            "dp_med_mm", "dp_p90_mm", "dR_med_deg", "dR_p90_deg", "dp_maxseg_med_mm", "dp_maxseg_p90_mm",
            "contacts_handoff_med", "contacts_endpoint_med", "eps_handoff_med", "eps_handoff_pos_frac",
            "anchor_minus_measured_med_mm", "kin_residual_max_mm", "model_err_max_mm", "secondary_success_given_handoff",
            "secondary_success_all", "failure_reasons", "draws_identical_to_expert"]
    with open(os.path.join(REPO, args.out_csv), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for s in rows_out:
            def d(k, f, scale=1.0):
                return round(s[k][f] * scale, 4) if s.get(k) else ""
            er = s["primary_endpoint_reason"]
            w.writerow([s["policy"], s["rate"], s["episodes"], s["acquired"]["k"], s["handoff_episodes"],
                        s["retained_preinsert"]["k"], s["retained_preinsert"]["frac"], s["retained_preinsert"]["wilson"][0],
                        s["retained_preinsert"]["wilson"][1], er.get("insert_start", 0), er.get("receptacle_contact", 0),
                        er.get("not_reached", 0), d("dp_preinsert_m", "median", 1e3), d("dp_preinsert_m", "p90", 1e3),
                        d("dR_preinsert_deg", "median"), d("dR_preinsert_deg", "p90"), d("dp_max_segment_m", "median", 1e3),
                        d("dp_max_segment_m", "p90", 1e3), d("contact_count_handoff", "median"),
                        d("contact_count_preinsert_endpoint", "median"), d("epsilon_handoff", "median"),
                        s["epsilon_handoff_positive_frac"], d("handoff_anchor_minus_measured_m", "median", 1e3),
                        d("max_kinematic_residual_pos_m", "max", 1e3), d("max_kinematic_model_error_pos_m", "max", 1e3),
                        s["secondary"]["task_success_given_handoff"]["frac"], s["secondary"]["task_success_all"]["frac"],
                        json.dumps(s["secondary"]["failure_reasons"]), s["draws_identical_to_expert"]])
    for s in rows_out:
        rp = s["retained_preinsert"]
        print(f"{s['policy']:10s} r {s['rate']:<4g} acquired {s['acquired']['k']:3d}/{s['episodes']} handoff {s['handoff_episodes']:3d} "
              f"retained {rp['k']:3d}/{rp['n']:3d} ({rp['frac'] if rp['frac'] is None else round(rp['frac'], 3)}) endpoint {s['primary_endpoint_reason']} "
              f"dp {s['dp_preinsert_m'] and round(s['dp_preinsert_m']['median'] * 1e3, 1)}/{s['dp_preinsert_m'] and round(s['dp_preinsert_m']['p90'] * 1e3, 1)} mm "
              f"dR {s['dR_preinsert_deg'] and round(s['dR_preinsert_deg']['median'], 1)}/{s['dR_preinsert_deg'] and round(s['dR_preinsert_deg']['p90'], 1)} deg "
              f"secondary {s['secondary']['task_success_given_handoff']['k']}/{s['secondary']['task_success_given_handoff']['n']} "
              f"draws_ok {s['draws_identical_to_expert']}")
    print(f"[written] {args.out_jsonl} {args.out_csv}")


if __name__ == "__main__":
    main()
