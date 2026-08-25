"""Rate-conditioned analysis of the nominal realized-contact Ferrari-Canny
margin at stable lift (part C of the 2026-08-18 evening protocol).

The margin epsilon_lift is read from the evaluation records as stored
(mdp/metrics.py, score_contact_set at the parcel friction 0.5), nothing is
recomputed, thresholded, or tuned. Restricted to the expert and the ACT
training seeds at r in {1.0, 1.5, 2.0}, acquired episodes only.

Per (actor, rate[, training seed]) cell,
- Spearman(epsilon_lift, task_success), Spearman(epsilon_lift,
  max_hand_object_translation_m), Spearman(epsilon_lift,
  max_hand_object_rotation_deg), success by epsilon quartile.

Rate-controlled models on acquired episodes (per actor and per ACT seed,
plus a pooled ACT model with training-seed fixed effects),
    logit P(task_success = 1) = b0 + b_r r + b_eps z(epsilon_lift)
    d_p = a0 + a_r r + a_eps z(epsilon_lift) + error
    d_R = c0 + c_r r + c_eps z(epsilon_lift) + error
with z the standardization over the model's own sample, maximum-likelihood
logistic fits by Newton iterations, ordinary least squares for the
continuous diagnostics, and heteroskedasticity-robust (HC1 sandwich)
standard errors with normal 95 percent intervals. The same tables repeat
for epsilon^(beta) (beta 0.95, prior std 0.15, as stored) as the secondary
domain-alignment result.

Outputs,
  experiments/paper/results/rate_conditioned_margin_analysis.json
  experiments/paper/results/rate_conditioned_margin_analysis.txt

Run,
  python3 scripts/manipulation/analyze_rate_conditioned_margin.py
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RATES = [1.0, 1.5, 2.0]
SLIP_T = "max_hand_object_translation_m"
SLIP_R = "max_hand_object_rotation_deg"

BETA_STATEMENT = (
    "ParcelStow holds nominal friction at mu=0.5 and varies task-induced dynamics, not friction. "
    "The selected FIRMGrasp prior (beta 0.95, sigma_mu 0.15) evaluates an adverse low-friction tail that is rarely "
    "realized in this benchmark. Consequently, epsilon^(beta) is expected to be conservative/saturated here and is not "
    "the primary predictor tested by this experiment. Under this nominal-friction task epsilon^(beta) does not resolve "
    "outcomes, it is conservative under the selected adverse-friction prior, and the benchmark does not excite the "
    "uncertainty dimension targeted by the risk-adjusted margin."
)


def load(path):
    p = os.path.join(REPO, path)
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p)]


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return {"rho": None, "p": None, "n": int(ok.sum())}
    if spearmanr is not None:
        r, p = spearmanr(x[ok], y[ok])
        return {"rho": float(r), "p": float(p), "n": int(ok.sum())}
    rx = np.argsort(np.argsort(x[ok]))
    ry = np.argsort(np.argsort(y[ok]))
    return {"rho": float(np.corrcoef(rx, ry)[0, 1]), "p": None, "n": int(ok.sum())}


def quartiles(margin, success):
    m = np.asarray(margin, dtype=float)
    s = np.asarray(success, dtype=float)
    ok = np.isfinite(m)
    m, s = m[ok], s[ok]
    if len(m) < 12:
        return []
    qs = np.quantile(m, [0, 0.25, 0.5, 0.75, 1.0])
    out = []
    for i in range(4):
        lo, hi = qs[i], qs[i + 1]
        sel = (m >= lo) & (m <= hi) if i == 3 else (m >= lo) & (m < hi)
        if sel.sum() == 0:
            continue
        out.append({"q": i + 1, "lo": float(lo), "hi": float(hi), "n": int(sel.sum()), "success": float(s[sel].mean())})
    return out


# ----------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------
def _ci(beta, se):
    return [float(beta - 1.96 * se), float(beta + 1.96 * se)]


def logistic_fit(X, y, names, iters=100):
    """Newton maximum likelihood with HC1 sandwich standard errors."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    if y.min() == y.max():
        return {"error": "constant outcome", "n": int(n)}
    beta = np.zeros(p)
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = mu * (1 - mu)
        grad = X.T @ (y - mu)
        H = (X * W[:, None]).T @ X + 1e-9 * np.eye(p)
        step = np.linalg.solve(H, grad)
        beta = beta + step
        if np.abs(step).max() < 1e-9:
            break
    eta = np.clip(X @ beta, -30, 30)
    mu = 1.0 / (1.0 + np.exp(-eta))
    W = mu * (1 - mu)
    H = (X * W[:, None]).T @ X + 1e-9 * np.eye(p)
    Vm = np.linalg.inv(H)
    S = (X * ((y - mu) ** 2)[:, None]).T @ X
    V = Vm @ S @ Vm * (n / max(n - p, 1))
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    ll = float(np.sum(y * np.log(mu + 1e-12) + (1 - y) * np.log(1 - mu + 1e-12)))
    p0 = y.mean()
    ll0 = float(n * (p0 * math.log(p0 + 1e-12) + (1 - p0) * math.log(1 - p0 + 1e-12)))
    separated = bool(np.abs(beta).max() > 15)
    coefs = {nm: {"coef": float(b), "se_hc1": float(s), "ci95": _ci(b, s), "z": float(b / s) if s > 0 else None,
                  "odds_ratio": float(math.exp(b)) if abs(b) < 50 else None} for nm, b, s in zip(names, beta, se)}
    return {"n": int(n), "coefficients": coefs, "log_likelihood": ll, "null_log_likelihood": ll0,
            "mcfadden_r2": float(1 - ll / ll0) if ll0 != 0 else None, "quasi_separation": separated,
            "success_rate": float(p0)}


def ols_fit(X, y, names):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    X, y = X[ok], y[ok]
    n, p = X.shape
    if n <= p + 2:
        return {"error": "too few rows", "n": int(n)}
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    res = y - X @ beta
    S = (X * (res ** 2)[:, None]).T @ X
    V = XtX_inv @ S @ XtX_inv * (n / max(n - p, 1))
    se = np.sqrt(np.diag(V))
    r2 = float(1 - np.sum(res ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12))
    coefs = {nm: {"coef": float(b), "se_hc1": float(s), "ci95": _ci(b, s), "t": float(b / s) if s > 0 else None}
             for nm, b, s in zip(names, beta, se)}
    return {"n": int(n), "coefficients": coefs, "r2": r2, "residual_sd": float(res.std(ddof=p))}


def design(rows, margin_key, seed_levels=None):
    """Design matrix [1, r, z(margin)] (+ seed dummies), standardization over
    the given rows."""
    eps = np.array([r[margin_key] for r in rows], dtype=float)
    z = (eps - eps.mean()) / (eps.std() if eps.std() > 0 else 1.0)
    rate = np.array([r["task_rate"] for r in rows], dtype=float)
    cols = [np.ones(len(rows)), rate, z]
    names = ["intercept", "rate", "z_" + margin_key]
    if seed_levels:
        for lvl in seed_levels[1:]:
            cols.append(np.array([1.0 if r["_seed"] == lvl else 0.0 for r in rows]))
            names.append(f"seed_{lvl}")
    return np.stack(cols, axis=1), names, {"mean": float(eps.mean()), "std": float(eps.std())}


def fit_models(rows, margin_key, seed_levels=None):
    X, names, std = design(rows, margin_key, seed_levels)
    y_s = np.array([1.0 if r["task_success"] else 0.0 for r in rows])
    out = {"n": len(rows), "z_standardization": std,
           "logit_task_success": logistic_fit(X, y_s, names),
           "ols_dp": ols_fit(X, [r.get(SLIP_T, np.nan) for r in rows], names),
           "ols_dR": ols_fit(X, [r.get(SLIP_R, np.nan) for r in rows], names)}
    # the same models per rate with the margin alone (no rate term), for transparency
    per_rate = {}
    for rate in sorted(set(r["task_rate"] for r in rows)):
        sub = [r for r in rows if r["task_rate"] == rate]
        eps = np.array([r[margin_key] for r in sub], dtype=float)
        z = (eps - eps.mean()) / (eps.std() if eps.std() > 0 else 1.0)
        Xr = np.stack([np.ones(len(sub)), z], axis=1)
        per_rate[f"{rate:g}"] = {"n": len(sub),
                                 "logit_task_success": logistic_fit(Xr, [1.0 if r["task_success"] else 0.0 for r in sub],
                                                                    ["intercept", "z_" + margin_key]),
                                 "ols_dp": ols_fit(Xr, [r.get(SLIP_T, np.nan) for r in sub], ["intercept", "z_" + margin_key])}
    out["per_rate_margin_only"] = per_rate
    return out


def scored(rows, margin_key):
    """Analysis sample of a margin. For the nominal margin, the episodes whose
    realized contact set is in force closure (the frogger convention clamps a
    non-contained origin, or fewer than two contacts, to -1.0, so -1 codes
    no force closure, the lowest rank, and is reported as its own group).
    For epsilon^(beta) every value stays, the values at or under zero are the
    saturation of the risk-adjusted margin under the adverse prior and are
    the point of the secondary analysis."""
    if margin_key == "epsilon_beta_lift":
        return list(rows)
    return [r for r in rows if r[margin_key] > -1]


def cell_stats(rows, margin_key):
    eps = [r[margin_key] for r in rows]
    sc = scored(rows, margin_key)
    eps_s = [r[margin_key] for r in sc]
    sent = [r for r in rows if r[margin_key] <= -1] if margin_key != "epsilon_beta_lift" else []
    return {"n": len(rows),
            "success_rate": float(np.mean([r["task_success"] for r in rows])) if rows else None,
            "margin_median": float(np.median(eps)) if rows else None,
            "margin_positive_frac": float(np.mean(np.array(eps) > 0)) if rows else None,
            "no_force_closure": {"n": len(sent), "success_rate": float(np.mean([r["task_success"] for r in sent])) if sent else None},
            "spearman_success_as_recorded": spearman(eps, [r["task_success"] for r in rows]),
            "spearman_dp_as_recorded": spearman(eps, [r.get(SLIP_T) for r in rows]),
            "spearman_dR_as_recorded": spearman(eps, [r.get(SLIP_R) for r in rows]),
            "spearman_success": spearman(eps_s, [r["task_success"] for r in sc]),
            "spearman_dp": spearman(eps_s, [r.get(SLIP_T) for r in sc]),
            "spearman_dR": spearman(eps_s, [r.get(SLIP_R) for r in sc]),
            "success_by_quartile": quartiles(eps_s, [r["task_success"] for r in sc]),
            "n_scored": len(sc)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--actors", nargs="*", default=[
        "expert==outputs/paper/eval/expert.jsonl",
        "act_seed1=0=outputs/paper/eval/act.jsonl",
        "act_seed2=1=outputs/paper/act_multiseed/eval/act_seed2.jsonl",
        "act_seed3=2=outputs/paper/act_multiseed/eval/act_seed3.jsonl"], help="name=model_seed=records")
    ap.add_argument("--rates", type=float, nargs="*", default=RATES)
    ap.add_argument("--out_json", default="experiments/paper/results/rate_conditioned_margin_analysis.json")
    ap.add_argument("--out_txt", default="experiments/paper/results/rate_conditioned_margin_analysis.txt")
    args = ap.parse_args()

    actors = []
    for spec in args.actors:
        name, ms, path = spec.split("=")
        rows = load(path)
        if not rows:
            print(f"[skip] {name}, no records at {path}")
            continue
        rows = [r for r in rows if float(r["task_rate"]) in set(args.rates) and r.get("acquired")]
        for r in rows:
            r["_seed"] = name
        actors.append({"name": name, "model_seed": ms, "kind": "expert" if name == "expert" else "act", "rows": rows,
                       "records": path})
    out = {"protocol": {"rates": args.rates, "sample": "acquired episodes only, per-episode evaluation records as stored",
                        "margins": {"primary": "epsilon_lift (nominal Ferrari-Canny margin of the realized contact set at "
                                    "acquisition, mu 0.5)", "secondary": "epsilon_beta_lift (beta 0.95, prior std 0.15)"},
                        "models": "logit P(success) = b0 + b_r r + b_eps z(eps), OLS for d_p and d_R, HC1 robust SE",
                        "no_force_closure_code": "epsilon -1 is the frogger convention for a contact set without force closure "
                        "(or fewer than two contacts), the lowest rank in the Spearman statistics, its own group in the models"},
           "actors": {}, "pooled_act": {}, "epsilon_beta_statement": BETA_STATEMENT}
    lines = []
    for margin_key, label in (("epsilon_lift", "PRIMARY, nominal epsilon at lift"),
                              ("epsilon_beta_lift", "SECONDARY, epsilon^(beta) at lift, beta 0.95, sigma_mu 0.15")):
        lines.append("=" * 100)
        lines.append(label)
        lines.append("=" * 100)
        if margin_key == "epsilon_lift":
            lines.append("Spearman and quartiles over the force-closure grasps, nfc = no-force-closure count (their success), value -1 by the frogger convention")
        else:
            lines.append("Spearman and quartiles over all acquired episodes as recorded (values at or under zero are the saturated tail)")
        lines.append(f"{'cell':22s} {'n':>4s} {'succ':>5s} {'eps>0':>6s} {'nfc':>9s} {'rho(succ)':>10s} {'rho(d_p)':>9s} {'rho(d_R)':>9s}  success by quartile Q1..Q4")
        for a in actors:
            rows = [r for r in a["rows"] if r.get(margin_key) is not None and np.isfinite(r[margin_key])]
            entry = out["actors"].setdefault(a["name"], {"model_seed": a["model_seed"], "records": a["records"],
                                                          "n_acquired_r1_1.5_2": len(a["rows"])})
            cells = {}

            def f(v, w=9):
                return f"{v:{w}.2f}" if isinstance(v, float) else f"{'na':>{w}s}"
            for rate in args.rates:
                sub = [r for r in rows if r["task_rate"] == rate]
                cells[f"{rate:g}"] = cell_stats(sub, margin_key)
                c = cells[f"{rate:g}"]
                q = " ".join(f"{s['success']:.2f}(n{s['n']})" for s in c["success_by_quartile"])
                sent = f"{c['no_force_closure']['n']}({c['no_force_closure']['success_rate']:.2f})" if c["no_force_closure"]["n"] else "0"
                lines.append(f"{a['name'] + '@' + f'{rate:g}':22s} {c['n']:4d} {f(c['success_rate'], 5)} {f(c['margin_positive_frac'], 6)} {sent:>9s} "
                             f"{f(c['spearman_success']['rho'], 10)} {f(c['spearman_dp']['rho'], 9)} {f(c['spearman_dR']['rho'], 9)}  {q}")
            cells["pooled_rates"] = cell_stats(rows, margin_key)
            c = cells["pooled_rates"]
            sent = f"{c['no_force_closure']['n']}({c['no_force_closure']['success_rate']:.2f})" if c["no_force_closure"]["n"] else "0"
            lines.append(f"{a['name'] + '@pooled':22s} {c['n']:4d} {c['success_rate']:5.2f} {c['margin_positive_frac']:6.2f} {sent:>9s} "
                         f"{f(c['spearman_success']['rho'], 10)} {f(c['spearman_dp']['rho'], 9)} {f(c['spearman_dR']['rho'], 9)}"
                         f"  (all acquired as recorded, rho succ {f(c['spearman_success_as_recorded']['rho'], 5)}, d_p {f(c['spearman_dp_as_recorded']['rho'], 5)})")
            rows_s = scored(rows, margin_key)
            models = fit_models(rows_s, margin_key) if len(rows_s) >= 20 else {"error": "too few rows", "n": len(rows_s)}
            models["sample"] = ("acquired episodes in force closure at acquisition (nominal margin -1, no force closure, "
                                "reported as its own group)" if margin_key == "epsilon_lift" else "acquired episodes, all values as recorded")
            models["no_force_closure_group"] = {"n": len(rows) - len(rows_s),
                                                "success_rate": float(np.mean([r["task_success"] for r in rows if r[margin_key] <= -1]))
                                                if len(rows) > len(rows_s) else None}
            entry[margin_key] = {"cells": cells, "models": models}
        # pooled ACT with seed fixed effects
        act_rows = scored([r for a in actors if a["kind"] == "act" for r in a["rows"]
                           if r.get(margin_key) is not None and np.isfinite(r[margin_key])], margin_key)
        seed_levels = [a["name"] for a in actors if a["kind"] == "act"]
        if len(act_rows) >= 20 and len(seed_levels) >= 1:
            out["pooled_act"][margin_key] = {"seed_levels": seed_levels, "reference_seed": seed_levels[0],
                                             "models": fit_models(act_rows, margin_key, seed_levels),
                                             "cells": cell_stats(act_rows, margin_key)}
        # model table
        lines.append("")
        lines.append("Rate-controlled models, coef [95 percent CI, HC1], sample per margin as stated above")
        lines.append(f"{'model sample':22s} {'n':>4s} {'b_eps (logit)':>22s} {'b_r (logit)':>22s} {'a_eps d_p mm':>22s} {'c_eps d_R deg':>22s}")
        def coef_str(m, key, scale=1.0):
            if not m or "error" in m or key not in m.get("coefficients", {}):
                return f"{'na':>22s}"
            c = m["coefficients"][key]
            return f"{c['coef'] * scale:7.3f} [{c['ci95'][0] * scale:7.3f},{c['ci95'][1] * scale:7.3f}]"
        for a in actors:
            m = out["actors"][a["name"]][margin_key]["models"]
            if "error" in m:
                lines.append(f"{a['name']:22s} {m['n']:4d} {m['error']}")
                continue
            lines.append(f"{a['name']:22s} {m['n']:4d} {coef_str(m['logit_task_success'], 'z_' + margin_key)} "
                         f"{coef_str(m['logit_task_success'], 'rate')} {coef_str(m['ols_dp'], 'z_' + margin_key, 1e3)} "
                         f"{coef_str(m['ols_dR'], 'z_' + margin_key)}")
        if margin_key in out["pooled_act"]:
            m = out["pooled_act"][margin_key]["models"]
            lines.append(f"{'act pooled + seed FE':22s} {m['n']:4d} {coef_str(m['logit_task_success'], 'z_' + margin_key)} "
                         f"{coef_str(m['logit_task_success'], 'rate')} {coef_str(m['ols_dp'], 'z_' + margin_key, 1e3)} "
                         f"{coef_str(m['ols_dR'], 'z_' + margin_key)}")
        lines.append("")
    lines.append("=" * 100)
    lines.append("epsilon^(beta) domain alignment")
    lines.append("=" * 100)
    lines.append(BETA_STATEMENT)
    lines.append("")
    os.makedirs(os.path.dirname(os.path.join(REPO, args.out_json)), exist_ok=True)
    with open(os.path.join(REPO, args.out_json), "w") as fh:
        json.dump(out, fh, indent=1)
    with open(os.path.join(REPO, args.out_txt), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[written] {args.out_json} {args.out_txt}")


if __name__ == "__main__":
    main()
