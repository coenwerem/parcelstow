"""Out-of-sample validation of the realized-contact certificate as a
success predictor (WRL workshop addition of 2026-08-21).

Every margin is read from the evaluation records as stored (mdp/metrics.py,
score_contact_set at the parcel friction 0.5), nothing is recomputed. The
population is the acquired episodes, whose contact set defines the margin,
and never-acquired episodes are counted separately. The stored nominal
margin takes the frogger sentinel -1 on acquired grasps without force
closure, so the a priori rule FC (predict success iff epsilon_lift > 0)
needs no fitted parameter and is out-of-sample in every cell by
construction.

Protocols,
  1. FC rule per (actor, rate) cell and pooled, P(success | FC) and
     P(success | not FC) with Wilson 95 percent intervals, balanced
     accuracy.
  2. Rate holdout, fit on the main actors at r in {0.5, 1.0, 1.5}
     (inside the training range), test at r in {2.0, 2.25, 2.5, 3.0}.
     Fitted objects, a Youden-J threshold on the stored margin and a
     one-dimensional logistic model on the z-scored margin. Test-set
     report, AUROC, balanced accuracy at the learned threshold and at
     the a priori threshold 0, Brier score, expected calibration error
     over quantile bins, reliability table.
  3. Actor holdout, leave-one-actor-out over {expert, dagger, dp, act}
     with all rates on both sides, plus fit on all four main actors and
     test on the unseen ACT training seeds 2 and 3.
  4. Joint holdout, fit on the expert at r in {0.5, 1.0, 1.5}, test on
     the learners at r in {2.0, 2.25, 2.5, 3.0}.

Outputs,
  experiments/paper/results/certificate_oos_analysis.json
  experiments/paper/results/certificate_oos_analysis.txt

Run,
  python3 scripts/manipulation/analyze_certificate_oos.py
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

MAIN_ACTORS = ["expert", "dagger", "dp", "act"]
SEED_ACTORS = ["act_seed2", "act_seed3"]
FIT_RATES = [0.5, 1.0, 1.5]
TEST_RATES = [2.0, 2.25, 2.5, 3.0]
PRIMARY = "epsilon_lift"
SECONDARY = ["epsilon_reorient", "epsilon_preinsert", "epsilon_beta_lift"]


def wilson(k, n, z=1.959964):
    if n == 0:
        return None, None, None
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(p), float(c - h), float(c + h)


def auroc(score, label):
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=bool)
    ok = np.isfinite(score)
    score, label = score[ok], label[ok]
    if label.sum() == 0 or (~label).sum() == 0:
        return None
    pos, neg = score[label], score[~label]
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def balanced_accuracy(margin, success, tau):
    margin = np.asarray(margin, dtype=float)
    success = np.asarray(success, dtype=bool)
    pred = margin > tau
    if success.sum() == 0 or (~success).sum() == 0:
        return None
    tpr = float(pred[success].mean())
    tnr = float((~pred[~success]).mean())
    return 0.5 * (tpr + tnr)


def youden_threshold(margin, success):
    margin = np.asarray(margin, dtype=float)
    success = np.asarray(success, dtype=bool)
    vals = np.unique(margin)
    if len(vals) < 2:
        return 0.0
    mids = 0.5 * (vals[:-1] + vals[1:])
    best_tau, best_j = 0.0, -np.inf
    for tau in mids:
        pred = margin > tau
        tpr = pred[success].mean() if success.any() else 0.0
        fpr = pred[~success].mean() if (~success).any() else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j, best_tau = j, float(tau)
    return best_tau


def fit_logistic_1d(z, y, iters=100):
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.stack([np.ones_like(z), z], axis=1)
    w = np.zeros(2)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        g = X.T @ (y - p)
        s = np.clip(p * (1 - p), 1e-9, None)
        H = X.T @ (X * s[:, None]) + 1e-8 * np.eye(2)
        step = np.linalg.solve(H, g)
        w = w + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def predict_logistic(w, z):
    return 1.0 / (1.0 + np.exp(-(w[0] + w[1] * np.asarray(z, dtype=float))))


def calibration(p, y, n_bins=5):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    qs = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    rows, ece = [], 0.0
    for i in range(n_bins):
        sel = (p > qs[i]) & (p <= qs[i + 1])
        if sel.sum() == 0:
            continue
        conf, freq = float(p[sel].mean()), float(y[sel].mean())
        rows.append({"bin": i, "n": int(sel.sum()), "mean_pred": conf, "frac_success": freq})
        ece += sel.mean() * abs(conf - freq)
    return rows, float(ece)


def load_rows(eval_dir, multiseed_dir):
    rows = []
    for name in MAIN_ACTORS:
        path = os.path.join(eval_dir, f"{name}.jsonl")
        for line in open(path):
            r = json.loads(line)
            r["actor"] = name
            rows.append(r)
    for name in SEED_ACTORS:
        path = os.path.join(multiseed_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            continue
        for line in open(path):
            r = json.loads(line)
            r["actor"] = name
            rows.append(r)
    return rows


def margin_pop(rows, key):
    out = []
    for r in rows:
        if not r.get("acquired"):
            continue
        v = r.get(key)
        if v is None or not np.isfinite(v):
            continue
        out.append((float(v), bool(r["task_success"])))
    m = np.array([v for v, _ in out], dtype=float)
    s = np.array([b for _, b in out], dtype=bool)
    return m, s


def fc_rule_cell(rows, key):
    m, s = margin_pop(rows, key)
    res = {"n_acquired": int(len(m)), "n_total": len(rows),
           "n_fc": int((m > 0).sum()), "n_nfc": int((m <= 0).sum())}
    if len(m) == 0:
        return res
    p, lo, hi = wilson(int(s[m > 0].sum()), int((m > 0).sum()))
    res["p_success_fc"] = {"p": p, "lo": lo, "hi": hi}
    p, lo, hi = wilson(int(s[m <= 0].sum()), int((m <= 0).sum()))
    res["p_success_nfc"] = {"p": p, "lo": lo, "hi": hi}
    res["balanced_accuracy_tau0"] = balanced_accuracy(m, s, 0.0)
    res["auroc"] = auroc(m, s)
    return res


def transfer(fit_rows, test_rows, key, label):
    m_fit, s_fit = margin_pop(fit_rows, key)
    m_test, s_test = margin_pop(test_rows, key)
    res = {"label": label, "margin": key,
           "n_fit": int(len(m_fit)), "n_test": int(len(m_test)),
           "fit_success_frac": float(s_fit.mean()) if len(m_fit) else None,
           "test_success_frac": float(s_test.mean()) if len(m_test) else None}
    if len(m_fit) < 20 or len(m_test) < 20:
        return res
    tau = youden_threshold(m_fit, s_fit)
    mu, sd = float(m_fit.mean()), float(m_fit.std() + 1e-12)
    w = fit_logistic_1d((m_fit - mu) / sd, s_fit.astype(float))
    p_test = predict_logistic(w, (m_test - mu) / sd)
    rel, ece = calibration(p_test, s_test.astype(float))
    res.update({
        "tau_youden_fit": tau,
        "test_auroc": auroc(m_test, s_test),
        "test_balanced_accuracy_tau_youden": balanced_accuracy(m_test, s_test, tau),
        "test_balanced_accuracy_tau0": balanced_accuracy(m_test, s_test, 0.0),
        "test_brier": float(np.mean((p_test - s_test.astype(float)) ** 2)),
        "test_brier_base_rate": float(np.mean((s_fit.mean() - s_test.astype(float)) ** 2)),
        "test_ece": ece,
        "test_reliability": rel,
        "logistic_w": [float(w[0]), float(w[1])],
        "fc_rule_test": fc_rule_cell(test_rows, key),
    })
    return res


def fmt(v, w=6):
    if isinstance(v, float):
        return f"{v:{w}.3f}"
    return f"{str(v):>{w}s}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval_dir", type=str, default="outputs/paper/eval")
    ap.add_argument("--multiseed_dir", type=str, default="outputs/paper/act_multiseed/eval")
    ap.add_argument("--out_dir", type=str, default="experiments/paper/results")
    args = ap.parse_args()

    rows = load_rows(args.eval_dir, args.multiseed_dir)
    main_rows = [r for r in rows if r["actor"] in MAIN_ACTORS]
    out = {"config": {"fit_rates": FIT_RATES, "test_rates": TEST_RATES,
                      "primary": PRIMARY, "secondary": SECONDARY,
                      "actors": MAIN_ACTORS + SEED_ACTORS},
           "fc_rule_cells": {}, "transfers": []}

    for actor in MAIN_ACTORS + SEED_ACTORS:
        arows = [r for r in rows if r["actor"] == actor]
        if not arows:
            continue
        for rate in sorted(set(r["task_rate"] for r in arows)):
            cell = [r for r in arows if r["task_rate"] == rate]
            out["fc_rule_cells"][f"{actor}@{rate:g}"] = fc_rule_cell(cell, PRIMARY)
        out["fc_rule_cells"][f"{actor}@pooled"] = fc_rule_cell(arows, PRIMARY)

    keys = [PRIMARY] + SECONDARY
    for key in keys:
        fit = [r for r in main_rows if r["task_rate"] in FIT_RATES]
        test = [r for r in main_rows if r["task_rate"] in TEST_RATES]
        out["transfers"].append(transfer(fit, test, key, "rate_holdout"))
    for actor in MAIN_ACTORS:
        fit = [r for r in main_rows if r["actor"] != actor]
        test = [r for r in main_rows if r["actor"] == actor]
        out["transfers"].append(transfer(fit, test, PRIMARY, f"actor_holdout_{actor}"))
    for actor in SEED_ACTORS:
        test = [r for r in rows if r["actor"] == actor]
        if test:
            out["transfers"].append(transfer(main_rows, test, PRIMARY, f"unseen_seed_{actor}"))
    fit = [r for r in main_rows if r["actor"] == "expert" and r["task_rate"] in FIT_RATES]
    test = [r for r in main_rows if r["actor"] != "expert" and r["task_rate"] in TEST_RATES]
    out["transfers"].append(transfer(fit, test, PRIMARY, "joint_expert_id_to_learners_ood"))

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "certificate_oos_analysis.json")
    with open(json_path, "w") as fh:
        json.dump(out, fh, indent=1)

    lines = []
    lines.append("=" * 110)
    lines.append("FC rule (epsilon_lift > 0, zero fitted parameters), acquired episodes, Wilson 95 percent intervals")
    lines.append("=" * 110)
    lines.append(f"{'cell':22s} {'n_acq':>5s} {'n_fc':>5s} {'n_nfc':>5s} {'P(s|FC)':>18s} {'P(s|nFC)':>18s} {'balacc':>7s} {'AUROC':>6s}")
    for cell, res in out["fc_rule_cells"].items():
        if "p_success_fc" not in res:
            lines.append(f"{cell:22s} {res.get('n_acquired', 0):5d}  (no acquired episodes with a stored margin)")
            continue
        pf, pn = res["p_success_fc"], res["p_success_nfc"]
        def ci(d):
            if d["p"] is None:
                return f"{'na':>18s}"
            return f"{d['p']:5.2f} [{d['lo']:4.2f},{d['hi']:4.2f}]"
        lines.append(f"{cell:22s} {res['n_acquired']:5d} {res['n_fc']:5d} {res['n_nfc']:5d} "
                     f"{ci(pf)} {ci(pn)} {fmt(res['balanced_accuracy_tau0'],7)} {fmt(res['auroc'],6)}")
    lines.append("")
    lines.append("=" * 110)
    lines.append("Transfer protocols, fitted on the fit set only, every number below is a held-out test-set quantity")
    lines.append("=" * 110)
    lines.append(f"{'protocol':36s} {'margin':22s} {'n_fit':>5s} {'n_test':>6s} {'AUROC':>6s} {'ba@tau*':>7s} "
                 f"{'ba@0':>6s} {'tau*':>8s} {'Brier':>6s} {'Brier0':>6s} {'ECE':>6s}")
    for t in out["transfers"]:
        if "test_auroc" not in t:
            lines.append(f"{t['label']:36s} {t['margin']:22s} {t['n_fit']:5d} {t['n_test']:6d}  (insufficient sample)")
            continue
        lines.append(f"{t['label']:36s} {t['margin']:22s} {t['n_fit']:5d} {t['n_test']:6d} {fmt(t['test_auroc'],6)} "
                     f"{fmt(t['test_balanced_accuracy_tau_youden'],7)} {fmt(t['test_balanced_accuracy_tau0'],6)} "
                     f"{fmt(t['tau_youden_fit'],8)} {fmt(t['test_brier'],6)} {fmt(t['test_brier_base_rate'],6)} {fmt(t['test_ece'],6)}")
    lines.append("")
    lines.append("Reliability of the rate-holdout logistic model on the held-out rates (primary margin)")
    prim = next(t for t in out["transfers"] if t["label"] == "rate_holdout" and t["margin"] == PRIMARY)
    if "test_reliability" in prim:
        for b in prim["test_reliability"]:
            lines.append(f"  bin {b['bin']}  n {b['n']:4d}  mean predicted {b['mean_pred']:5.3f}  observed {b['frac_success']:5.3f}")
    txt_path = os.path.join(args.out_dir, "certificate_oos_analysis.txt")
    with open(txt_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[written] {json_path}")
    print(f"[written] {txt_path}")


if __name__ == "__main__":
    main()
