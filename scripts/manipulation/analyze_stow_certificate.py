"""Force-closure diagnostics of the ParcelStow evaluation (M11).

Reads the per-episode records of eval_stow_policies.py and asks whether the
realized analytical margins predict task success. Per
policy and per rate (and pooled), the script reports Spearman rank
correlations of epsilon and epsilon^(beta) at acquisition, at the end of
reorientation, and at insertion start against task success, insertion, the
maximum in-hand translation and rotation, and the success probability
stratified by margin quantile, plus AUROC of the margin for task success as
a secondary diagnostic. Episodes without a recorded contact set (never
acquired) enter with margin -1 and are also reported separately.

Run,
  python scripts/manipulation/analyze_stow_certificate.py --eval_dir outputs/paper/eval \
      --out outputs/paper/eval/certificate_analysis.json
"""

import argparse
import json
import os

import numpy as np

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None

MARGINS = ["epsilon_lift", "epsilon_beta_lift", "epsilon_reorient", "epsilon_beta_reorient",
           "epsilon_preinsert", "epsilon_beta_preinsert"]
TARGETS = ["task_success", "inserted", "max_hand_object_translation_m", "max_hand_object_rotation_deg"]


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return None, None, int(ok.sum())
    if spearmanr is not None:
        r, p = spearmanr(x[ok], y[ok])
        return float(r), float(p), int(ok.sum())
    rx = np.argsort(np.argsort(x[ok]))
    ry = np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1]), None, int(ok.sum())


def auroc(score, label):
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=bool)
    ok = np.isfinite(score)
    score, label = score[ok], label[ok]
    if label.sum() == 0 or (~label).sum() == 0:
        return None
    pos = score[label]
    neg = score[~label]
    # Mann-Whitney U
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def quantile_strata(margin, success, n_bins=4):
    margin = np.asarray(margin, dtype=float)
    success = np.asarray(success, dtype=float)
    ok = np.isfinite(margin)
    m, s = margin[ok], success[ok]
    if len(m) < n_bins * 3:
        return []
    qs = np.quantile(m, np.linspace(0, 1, n_bins + 1))
    out = []
    for i in range(n_bins):
        lo, hi = qs[i], qs[i + 1]
        sel = (m >= lo) & (m <= hi) if i == n_bins - 1 else (m >= lo) & (m < hi)
        if sel.sum() == 0:
            continue
        out.append({"bin": i, "margin_lo": float(lo), "margin_hi": float(hi), "n": int(sel.sum()),
                    "success_frac": float(s[sel].mean())})
    return out


def analyze(rows, label):
    res = {"label": label, "n": len(rows)}
    if not rows:
        return res
    acquired = np.array([r["acquired"] for r in rows], dtype=bool)
    res["n_acquired"] = int(acquired.sum())
    res["margin_positive_frac"] = {}
    for m in MARGINS:
        vals = np.array([r.get(m) if r.get(m) is not None else np.nan for r in rows], dtype=float)
        vals_acq = vals[acquired]
        res["margin_positive_frac"][m] = float(np.mean(vals_acq > 0)) if len(vals_acq) else None
        res[m] = {}
        for t in TARGETS:
            y = np.array([float(r[t]) if r.get(t) is not None else np.nan for r in rows], dtype=float)
            r_all, p_all, n_all = spearman(vals, y)
            r_acq, p_acq, n_acq = spearman(vals_acq, y[acquired])
            res[m][t] = {"spearman_all": r_all, "p_all": p_all, "n_all": n_all,
                         "spearman_acquired": r_acq, "p_acquired": p_acq, "n_acquired": n_acq}
        succ = np.array([r["task_success"] for r in rows], dtype=bool)
        res[m]["auroc_task_success_acquired"] = auroc(vals_acq, succ[acquired])
        res[m]["success_by_margin_quantile_acquired"] = quantile_strata(vals_acq, succ[acquired])
        res[m]["success_if_margin_positive"] = float(succ[acquired][vals_acq > 0].mean()) if (vals_acq > 0).any() else None
        res[m]["success_if_margin_nonpositive"] = float(succ[acquired][vals_acq <= 0].mean()) if (vals_acq <= 0).any() else None
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval_dir", type=str, default="outputs/paper/eval")
    ap.add_argument("--actors", type=str, nargs="*", default=["expert", "dagger", "dp", "act"])
    ap.add_argument("--out", type=str, default="outputs/paper/eval/certificate_analysis.json")
    args = ap.parse_args()
    out = {"per_policy": {}, "per_policy_rate": {}, "pooled": None}
    all_rows = []
    for name in args.actors:
        path = os.path.join(args.eval_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            continue
        rows = [json.loads(l) for l in open(path)]
        all_rows += rows
        out["per_policy"][name] = analyze(rows, name)
        for rate in sorted(set(r["task_rate"] for r in rows)):
            sub = [r for r in rows if r["task_rate"] == rate]
            out["per_policy_rate"][f"{name}@{rate:g}"] = analyze(sub, f"{name}@{rate:g}")
    out["pooled"] = analyze(all_rows, "pooled")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    # short console table
    print(f"{'set':22s} {'n':>5s} {'acq':>5s} {'eb_lift>0':>9s} {'rho(eb_lift,success)':>21s} {'rho(eb_lift,slip_t)':>20s} {'AUROC':>6s} {'P(s|eb>0)':>9s} {'P(s|eb<=0)':>10s}")
    for key, res in list(out["per_policy"].items()) + [("pooled", out["pooled"])]:
        if not res or "epsilon_beta_lift" not in res:
            continue
        m = res["epsilon_beta_lift"]
        def f(v, w=6):
            return f"{v:{w}.3f}" if isinstance(v, float) else f"{str(v):>{w}s}"
        print(f"{key:22s} {res['n']:5d} {res.get('n_acquired', 0):5d} {f(res['margin_positive_frac']['epsilon_beta_lift'],9)} "
              f"{f(m['task_success']['spearman_acquired'],21)} {f(m['max_hand_object_translation_m']['spearman_acquired'],20)} "
              f"{f(m['auroc_task_success_acquired'],6)} {f(m['success_if_margin_positive'],9)} {f(m['success_if_margin_nonpositive'],10)}")
    print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
