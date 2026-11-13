#!/usr/bin/env python3
"""
16_anchor_counterfactual.py — Counterfactual anchor ablation.

For each protein with encoded functional anchors (TEM-1, VIM-2, APH(3')-IIa,
DHFR), rerun the dual-modal modulo 5-fold CV with the anchor-distance features
(d_s70, d_omega) set to a constant — i.e. exactly the feature treatment that
AAC(3)-I receives. This tests the functional-anchor hypothesis directly:
if the structural gain is driven by anchor distances, removing them should
collapse the dual-modal gain to the AAC(3)-I regime.

Runs both the 8M and 650M backbones (GradientBoostingRegressor,
random_state=0, identical protocol to scripts 10/14/15).

Note on the `auc` column: it is the mean of the five per-fold AUCs, i.e. the
convention used for the 650M and 3B blocks of Tables 1/3. The AUCs printed in
the 8M block of those tables are pooled out-of-fold estimates instead
(script 51_collect_numbers.py); the two estimators differ by at most
0.004 (TEM-1 8M: 0.858 here versus 0.855 pooled). The `rho` and `rho_std`
columns are unaffected.

Output: results/anchor_counterfactual.csv
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
PHY = ["blosum62", "d_hydro", "d_vol", "d_chg"]
TABLES = {
    "TEM1_Stiffler2015": "tem1_stiffler_modeling_table.csv",
    "VIM2_Chen2020": "ampc_modeling_table.csv",
    "APH3II_Melnikov2014": "aph3ii_modeling_table.csv",
    "DHFR_Nguyen2023": "dhfr_modeling_table.csv",
}


def cv(m, F, k=5):
    rs, aucs = [], []
    for f in range(k):
        te = m[m.pos % k == f]
        tr = m[m.pos % k != f]
        g = GradientBoostingRegressor(random_state=0).fit(tr[F], tr.DMS_score)
        p = g.predict(te[F])
        rs.append(spearmanr(p, te.DMS_score).statistic)
        aucs.append(roc_auc_score(te.DMS_score_bin, p))
    return np.mean(rs), np.std(rs), np.mean(aucs)


def main():
    zs = pd.concat([
        pd.read_csv(os.path.join(RES, "esm2_650m_zeroshot.csv")),
        pd.read_csv(os.path.join(RES, "esm2_650m_newproteins.csv")),
    ], ignore_index=True)
    rows = []
    for assay, fn in TABLES.items():
        m = pd.read_csv(os.path.join(DATA, fn)).rename(
            columns={"d_act": "d_s70", "d_loop": "d_omega"})
        z = zs[zs.assay == assay][["mutant", "esm2_650m"]]
        assert not z.duplicated("mutant").any(), assay
        m650 = m.merge(z, on="mutant", how="inner")
        for tag, mm in [("8m", m), ("650m", m650)]:
            feat_col = "esm2" if tag == "8m" else "esm2_650m"
            seq = [feat_col] + PHY
            dual = seq + ["pLDDT", "contact_num", "d_s70", "d_omega"]
            abl = seq + ["pLDDT", "contact_num"]  # anchors removed
            r_seq = cv(mm, seq)
            r_dual = cv(mm, dual)
            r_abl = cv(mm, abl)
            rows.append(dict(assay=assay, backbone=tag, model="seq",
                             rho=r_seq[0], rho_std=r_seq[1], auc=r_seq[2]))
            rows.append(dict(assay=assay, backbone=tag, model="dual_full",
                             rho=r_dual[0], rho_std=r_dual[1], auc=r_dual[2]))
            rows.append(dict(assay=assay, backbone=tag, model="dual_noanchors",
                             rho=r_abl[0], rho_std=r_abl[1], auc=r_abl[2]))
            print(assay, tag,
                  f"seq {r_seq[0]:.3f} | dual {r_dual[0]:.3f} "
                  f"(+{r_dual[0]-r_seq[0]:.3f}) | no-anchors {r_abl[0]:.3f} "
                  f"(+{r_abl[0]-r_seq[0]:.3f})")
    out = pd.DataFrame(rows)
    path = os.path.join(RES, "anchor_counterfactual.csv")
    out.to_csv(path, index=False)
    print("saved", path)


if __name__ == "__main__":
    main()
