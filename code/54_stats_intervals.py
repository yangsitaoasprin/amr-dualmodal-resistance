#!/usr/bin/env python3
"""54_stats_intervals.py -- intervals and equivalence tests cited in the manuscript.

Produces results/stats_intervals.json with:
  * the structural gain (dual minus sequence-only) on TEM-1 at 8M, with a
    position-clustered bootstrap interval. Positions, not variants, are the
    resampling unit, because cross-validation splits by position and all
    substitutions at one position are dependent;
  * the same quantity under a naive variant-level bootstrap, for comparison;
  * TOST equivalence tests for GCNv2 versus the GBM on both beta-lactamases,
    with a margin of +/-0.05 rho (the smallest difference we consider
    practically meaningful).

Run from this directory:  python 54_stats_intervals.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
N_BOOT = 2000
MARGIN = 0.05
SEED = 0


def delta_rho(idx, y, dual, seq):
    return (stats.spearmanr(y[idx], dual[idx]).statistic
            - stats.spearmanr(y[idx], seq[idx]).statistic)


def structural_gain():
    z = np.load(os.path.join(RES, "oof_predictions_tem1_8m.npz"), allow_pickle=True)
    pos, y, dual, seq = z["pos"], z["oof_true"], z["oof_dual"], z["oof_seq"]
    rng = np.random.default_rng(SEED)
    point = delta_rho(np.arange(len(y)), y, dual, seq)

    naive = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        naive.append(delta_rho(idx, y, dual, seq))

    upos = np.unique(pos)
    groups = {p: np.flatnonzero(pos == p) for p in upos}
    cluster = []
    for _ in range(N_BOOT):
        pick = rng.choice(upos, len(upos), replace=True)
        idx = np.concatenate([groups[p] for p in pick])
        cluster.append(delta_rho(idx, y, dual, seq))

    lo_v, hi_v = np.percentile(naive, [2.5, 97.5])
    lo_c, hi_c = np.percentile(cluster, [2.5, 97.5])
    return {
        "quantity": "dual-minus-sequence Spearman rho, TEM-1, ESM-2 8M",
        "point_estimate": round(float(point), 4),
        "variants": int(len(y)),
        "positions": int(len(upos)),
        "bootstrap_resamples": N_BOOT,
        "variant_bootstrap_95CI": [round(float(lo_v), 4), round(float(hi_v), 4)],
        "variant_bootstrap_width": round(float(hi_v - lo_v), 4),
        "position_cluster_bootstrap_95CI": [round(float(lo_c), 4), round(float(hi_c), 4)],
        "position_cluster_bootstrap_width": round(float(hi_c - lo_c), 4),
        "width_inflation": round(float((hi_c - lo_c) / (hi_v - lo_v)), 2),
    }


def tost(name, path):
    t = pd.read_csv(path)
    diff = (t["GCNv2_rho"] - t["GBM_rho"]).to_numpy()
    n = len(diff)
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(n))
    p_lower = float(1 - stats.t.cdf((mean + MARGIN) / se, n - 1))
    p_upper = float(stats.t.cdf((mean - MARGIN) / se, n - 1))
    # sensitivity: equivalence is only meaningful relative to the margin chosen
    sensitivity = {}
    for mar in (0.02, 0.03, 0.05, 0.10):
        pl = float(1 - stats.t.cdf((mean + mar) / se, n - 1))
        pu = float(stats.t.cdf((mean - mar) / se, n - 1))
        sensitivity["%.2f" % mar] = {"tost_p": round(max(pl, pu), 4),
                                     "equivalent_at_5pct": bool(max(pl, pu) < 0.05)}
    return {
        "margin_sensitivity": sensitivity,
        "comparison": "GCNv2 vs GBM, %s, 5 paired folds" % name,
        "mean_difference": round(mean, 4),
        "se": round(se, 4),
        "margin": MARGIN,
        "tost_p_lower": round(p_lower, 4),
        "tost_p_upper": round(p_upper, 4),
        "tost_p": round(max(p_lower, p_upper), 4),
        "equivalent_at_5pct": bool(max(p_lower, p_upper) < 0.05),
        "paired_t_p": round(float(stats.ttest_rel(t["GCNv2_rho"], t["GBM_rho"]).pvalue), 4),
    }


out = {
    "structural_gain": structural_gain(),
    "tost": [
        tost("TEM-1", os.path.join(RES, "gcnv2_5fold_cv.csv")),
        tost("VIM-2", os.path.join(RES, "vim2_gcnv2_5fold_cv.csv")),
    ],
}
path = os.path.join(RES, "stats_intervals.json")
with open(path, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)

g = out["structural_gain"]
print("structural gain %.4f | variant CI %s | cluster CI %s (%.2fx wider)"
      % (g["point_estimate"], g["variant_bootstrap_95CI"],
         g["position_cluster_bootstrap_95CI"], g["width_inflation"]))
for t in out["tost"]:
    print("%-28s mean %+.4f | TOST p=%.4f (%s) | paired t p=%.4f"
          % (t["comparison"], t["mean_difference"], t["tost_p"],
             "equivalent" if t["equivalent_at_5pct"] else "not shown", t["paired_t_p"]))
print("wrote", os.path.relpath(path, HERE))
