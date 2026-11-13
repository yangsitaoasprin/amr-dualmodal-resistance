#!/usr/bin/env python3
"""
20_random_anchor_null.py — Random-anchor null model + collinearity diagnostics
+ paired statistics + permutation importance.

Addresses the reviewer critique that the counterfactual ablation (script 16)
cannot distinguish "anchors irrelevant" from "tree-model feature redundancy":
here we REPLACE the true anchor-distance features with distances to RANDOM
surface residues / random 16-residue windows (10 seeds) and rerun the
dual-modal CV. Distances are CB-CB with a CA fallback for glycine, i.e. the
same convention as the anchor distances in the modelling tables. If random
anchors perform like true anchors, functional-site
geometry carries no information beyond burial; if they perform worse, the
true anchors do contribute.

Also reports:
  (a) Spearman correlation matrix of structural features per protein
      (collinearity diagnostic: contact_num vs d_s70/d_omega);
  (b) paired t-tests across the 5 CV folds for dual vs seq and
      dual vs no-anchors (script 16 configurations);
  (c) permutation importance (test-set, 5 repeats per fold) of the dual-modal
      GBM for TEM-1 at 8M and 650M, replacing impurity-based importance.

Output: results/random_anchor_null.csv, results/feature_collinearity.csv,
        results/paired_tests.csv, results/permutation_importance.csv
"""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
PHY = ["blosum62", "d_hydro", "d_vol", "d_chg"]

# assay -> (table, pdb, true anchor residue, true loop range)
CFG = {
    "TEM1_Stiffler2015": ("tem1_stiffler_modeling_table.csv",
                          "AF-P62593-F1-model.pdb", 70, (163, 178)),
    "VIM2_Chen2020": ("ampc_modeling_table.csv",
                      "AF-A4GRB6-F1-model.pdb", 64, (190, 215)),
    "APH3II_Melnikov2014": ("aph3ii_modeling_table.csv",
                            "AF-P00552.pdb", 190, (151, 166)),
    "DHFR_Nguyen2023": ("dhfr_modeling_table.csv",
                        "AF-P0ABQ4.pdb", 27, (9, 24)),
}
N_SEEDS = 10


def cb_coords(pdb_path):
    """Cbeta coordinates, falling back to CA for glycine.

    Same convention as the anchor distances of the modelling tables
    (01_build_features.py: CB, or CA when the residue is a glycine). CA is
    stored unconditionally and CB wins when present; the earlier version let
    the first-seen atom win, which returns CA everywhere because AF models
    list CA before CB for every residue.
    """
    ca, cb = {}, {}
    for line in open(pdb_path):
        if not line.startswith("ATOM"):
            continue
        rn = int(line[22:26])
        atom = line[12:16].strip()
        if atom == "CA":
            ca[rn] = np.array([float(line[30:38]), float(line[38:46]),
                               float(line[46:54])])
        elif atom == "CB":
            cb[rn] = np.array([float(line[30:38]), float(line[38:46]),
                               float(line[46:54])])
    return {r: cb.get(r, ca[r]) for r in ca}


def dist_to(coords, targets):
    T = np.array([coords[t] for t in targets if t in coords])
    return {r: float(np.min(np.linalg.norm(T - c, axis=1)))
            for r, c in coords.items()}


def cv_folds(m, F, k=5):
    rs = []
    for f in range(k):
        te = m[m.pos % k == f]
        tr = m[m.pos % k != f]
        g = GradientBoostingRegressor(random_state=0).fit(tr[F], tr.DMS_score)
        p = g.predict(te[F])
        rs.append(spearmanr(p, te.DMS_score).statistic)
    return rs


def load(assay, backbone, zs):
    table, pdb, _, _ = CFG[assay]
    m = pd.read_csv(os.path.join(DATA, table)).rename(
        columns={"d_act": "d_s70", "d_loop": "d_omega"})
    if backbone == "650m":
        z = zs[zs.assay == assay][["mutant", "esm2_650m"]]
        m = m.merge(z, on="mutant", how="inner")
    return m


def random_anchor_task(args):
    assay, backbone, seed = args
    _, pdb, _, (l0, l1) = CFG[assay]
    coords = cb_coords(os.path.join(DATA, pdb))
    zs = pd.concat([pd.read_csv(os.path.join(RES, "esm2_650m_zeroshot.csv")),
                    pd.read_csv(os.path.join(RES, "esm2_650m_newproteins.csv"))],
                   ignore_index=True)
    m = load(assay, backbone, zs)
    rng = np.random.default_rng(seed)
    loop_len = l1 - l0 + 1
    L = max(coords)
    # random surface anchor: contact_num below the protein median
    med = m.drop_duplicates("pos").contact_num.median()
    surface = m.drop_duplicates("pos")
    surface = surface[surface.contact_num < med].pos.to_numpy()
    r_anchor = int(rng.choice(surface))
    start = int(rng.integers(1, L - loop_len + 1))
    d1 = dist_to(coords, [r_anchor])
    d2 = dist_to(coords, list(range(start, start + loop_len)))
    m = m.copy()
    m["d_s70"] = m.pos.map(d1)
    m["d_omega"] = m.pos.map(d2)
    fc = "esm2" if backbone == "8m" else "esm2_650m"
    F = [fc] + PHY + ["pLDDT", "contact_num", "d_s70", "d_omega"]
    rs = cv_folds(m, F)
    return assay, backbone, seed, float(np.mean(rs)), float(np.std(rs))


def main():
    zs = pd.concat([pd.read_csv(os.path.join(RES, "esm2_650m_zeroshot.csv")),
                    pd.read_csv(os.path.join(RES, "esm2_650m_newproteins.csv"))],
                   ignore_index=True)

    # (a) collinearity
    rows = []
    for assay in CFG:
        m = load(assay, "8m", zs).drop_duplicates("pos")
        for x, y in [("contact_num", "d_s70"), ("contact_num", "d_omega"),
                     ("d_s70", "d_omega"), ("pLDDT", "contact_num")]:
            rows.append(dict(assay=assay, feat_x=x, feat_y=y,
                             spearman=spearmanr(m[x], m[y]).statistic))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "feature_collinearity.csv"),
                              index=False)
    print("(a) collinearity done")

    # (b) paired tests + true-anchor reference gains
    rows = []
    ref = {}
    for assay in CFG:
        for backbone in ["8m", "650m"]:
            m = load(assay, backbone, zs)
            fc = "esm2" if backbone == "8m" else "esm2_650m"
            seq = [fc] + PHY
            dual = seq + ["pLDDT", "contact_num", "d_s70", "d_omega"]
            abl = seq + ["pLDDT", "contact_num"]
            r_seq, r_dual, r_abl = (cv_folds(m, f) for f in (seq, dual, abl))
            ref[(assay, backbone)] = np.mean(r_dual)
            rows.append(dict(assay=assay, backbone=backbone,
                             contrast="dual_vs_seq",
                             mean_diff=np.mean(np.array(r_dual) - r_seq),
                             p=ttest_rel(r_dual, r_seq).pvalue))
            rows.append(dict(assay=assay, backbone=backbone,
                             contrast="dual_vs_noanchors",
                             mean_diff=np.mean(np.array(r_dual) - r_abl),
                             p=ttest_rel(r_dual, r_abl).pvalue))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "paired_tests.csv"),
                              index=False)
    print("(b) paired tests done")

    # (c) random-anchor null
    tasks = [(a, b, s) for a in CFG for b in ["8m", "650m"]
             for s in range(N_SEEDS)]
    with Pool(2) as pool:
        out = pool.map(random_anchor_task, tasks)
    rows = []
    for assay, backbone, seed, rho, std in out:
        rows.append(dict(assay=assay, backbone=backbone, seed=seed,
                         rho=rho, rho_std=std,
                         true_anchor_rho=ref[(assay, backbone)]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "random_anchor_null.csv"), index=False)
    print("(c) random-anchor null done")
    summ = df.groupby(["assay", "backbone"]).agg(
        rand_mean=("rho", "mean"), rand_max=("rho", "max"),
        true=("true_anchor_rho", "first")).reset_index()
    print(summ.to_string(index=False))

    # (d) permutation importance, TEM-1 both backbones
    rows = []
    for backbone in ["8m", "650m"]:
        m = load("TEM1_Stiffler2015", backbone, zs)
        fc = "esm2" if backbone == "8m" else "esm2_650m"
        F = [fc] + PHY + ["pLDDT", "contact_num", "d_s70", "d_omega"]
        for f in range(5):
            te = m[m.pos % 5 == f]
            tr = m[m.pos % 5 != f]
            g = GradientBoostingRegressor(random_state=0).fit(tr[F],
                                                              tr.DMS_score)
            pi = permutation_importance(g, te[F], te.DMS_score,
                                        n_repeats=5, random_state=0,
                                        scoring="neg_mean_squared_error")
            for feat, imp in zip(F, pi.importances_mean):
                rows.append(dict(backbone=backbone, fold=f, feature=feat,
                                 importance=imp))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "permutation_importance.csv"),
                              index=False)
    print("(d) permutation importance done")


if __name__ == "__main__":
    main()
