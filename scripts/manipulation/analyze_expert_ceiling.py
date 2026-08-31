"""Analyze expert arm motion, grasp outcomes, and target tracking across speeds.

This WRL workshop analysis (added 2026-08-21) reads the expert rows of the
final evaluation. At each execution speed, it reports measurements relevant
to three candidate explanations for the expert's success decrease at
`r >= 2.5`: actuator saturation, grasp failure, and target-tracking error at
the insertion interface.

Reported per rate (median and p90 over 100 episodes),
- max_arm_velocity_utilization, the arm-joint velocity fraction of the
  limit (saturation check). max_joint_velocity_utilization is reported
  once with a note, the near-1 values come from the finger squeeze during
  the rate-fixed CLOSE phase and do not vary with rate.
- max_hand_object_translation_m and rotation (grasp-mechanics check).
- min_orientation_error_deg before insertion against the frozen 10 deg
  slot tolerance and insertion_depth against the frozen 50 mm inserted
  predicate (tracking-accuracy check).
- max_receptacle_force and the failure-reason counts.

Outputs,
  experiments/paper/results/expert_ceiling_analysis.json
  experiments/paper/results/expert_ceiling_analysis.txt

Run,
  python3 scripts/manipulation/analyze_expert_ceiling.py
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

KEYS = ["max_arm_velocity_utilization", "max_joint_velocity_utilization",
        "max_hand_object_translation_m", "max_hand_object_rotation_deg",
        "min_orientation_error_deg", "insertion_depth", "max_receptacle_force",
        "peak_hand_linear_velocity"]
ORIENT_TOL_DEG = 10.0
INSERT_PRED_M = 0.050


def cell(sub):
    res = {"n": len(sub), "success": float(np.mean([r["task_success"] for r in sub]))}
    for k in KEYS:
        vals = np.array([r[k] for r in sub if r.get(k) is not None], dtype=float)
        res[k] = {"p50": float(np.median(vals)), "p90": float(np.percentile(vals, 90))} if len(vals) else None
    fails = {}
    for r in sub:
        if not r["task_success"]:
            fails[r["failure_reason"]] = fails.get(r["failure_reason"], 0) + 1
    res["failure_reasons"] = dict(sorted(fails.items(), key=lambda x: -x[1]))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval_path", type=str, default="outputs/paper/eval/expert.jsonl")
    ap.add_argument("--out_dir", type=str, default="experiments/paper/results")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.eval_path)]
    out = {"note_joint_velocity": "max_joint_velocity_utilization sits near 1 at every rate including r 0.5 "
                                  "because the finger squeeze of the rate-fixed CLOSE phase touches the hand "
                                  "velocity limit, the value is rate-invariant and does not explain the ceiling",
           "orientation_tolerance_deg": ORIENT_TOL_DEG,
           "inserted_predicate_m": INSERT_PRED_M,
           "per_rate": {}}
    for rate in sorted(set(r["task_rate"] for r in rows)):
        out["per_rate"][f"{rate:g}"] = cell([r for r in rows if r["task_rate"] == rate])

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "expert_ceiling_analysis.json")
    with open(json_path, "w") as fh:
        json.dump(out, fh, indent=1)

    lines = []
    lines.append("=" * 118)
    lines.append("Expert measurements across execution speeds: median (p90), 100 episodes per speed")
    lines.append("Frozen orientation and insertion thresholds: 10 deg and 50 mm")
    lines.append("=" * 118)
    lines.append(f"{'rate':>5s} {'succ':>5s} {'arm_util':>15s} {'slip_mm':>15s} {'slip_deg':>15s} "
                 f"{'orient_err_deg':>15s} {'depth_mm':>15s} {'recept_N':>15s}  failure reasons")
    for rate, res in out["per_rate"].items():
        def m(k, scale=1.0):
            d = res[k]
            return f"{d['p50']*scale:6.2f} ({d['p90']*scale:6.2f})" if d else "na"
        fr = ", ".join(f"{k}:{v}" for k, v in res["failure_reasons"].items()) or "none"
        lines.append(f"{rate:>5s} {res['success']:5.2f} {m('max_arm_velocity_utilization'):>15s} "
                     f"{m('max_hand_object_translation_m',1000):>15s} {m('max_hand_object_rotation_deg'):>15s} "
                     f"{m('min_orientation_error_deg'):>15s} {m('insertion_depth',1000):>15s} "
                     f"{m('max_receptacle_force'):>15s}  {fr}")
    lines.append("")
    lines.append(out["note_joint_velocity"])
    txt_path = os.path.join(args.out_dir, "expert_ceiling_analysis.txt")
    with open(txt_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[written] {json_path}")
    print(f"[written] {txt_path}")


if __name__ == "__main__":
    main()
