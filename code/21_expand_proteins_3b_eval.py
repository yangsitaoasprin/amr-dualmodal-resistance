#!/usr/bin/env python3
"""
21_expand_proteins_3b_eval.py — 3B evaluation of the expansion proteins.

Run AFTER 09c_esm2_3b_new_proteins.py has produced
results/esm2_3b_newproteins.csv (GPU, local). CPU-only thereafter.

Reproduces, for APH(3')-IIa and DHFR with the ESM-2 3B backbone:
  zero-shot Spearman/AUC; modulo 5-fold CV for sequence-only (M1),
  structure-only (M2) and dual-modal (M3) GBMs — identical protocol to
  scripts 14/15 (GradientBoostingRegressor, random_state=0).

Output: results/expanded_proteins_3b_evaluation.csv
Then update paper Table 3 / Supplementary Table S8a with the 3B rows.
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
STR = ["pLDDT", "contact_num", "d_s70", "d_omega"]
ASSAYS = {
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
    z = pd.read_csv(os.path.join(RES, "esm2_3b_newproteins.csv"))
    rows = []
    for assay, fn in ASSAYS.items():
        m = pd.read_csv(os.path.join(DATA, fn))
        zz = z[z.assay == assay][["mutant", "esm2_3b"]]
        assert not zz.duplicated("mutant").any()
        m = m.merge(zz, on="mutant", how="inner")
        zs_rho = spearmanr(m.esm2_3b, m.DMS_score).statistic
        zs_auc = roc_auc_score(m.DMS_score_bin, m.esm2_3b)
        rows.append(dict(assay=assay, backbone="3b", model="zeroshot",
                         rho=zs_rho, rho_std=np.nan, auc=zs_auc))
        for model, F in [("seq", ["esm2_3b"] + PHY),
                         ("struct", STR),
                         ("dual", ["esm2_3b"] + PHY + STR)]:
            r, s, a = cv(m, F)
            rows.append(dict(assay=assay, backbone="3b", model=model,
                             rho=r, rho_std=s, auc=a))
            print(f"{assay} 3b {model}: rho {r:.3f} +/- {s:.3f} (AUC {a:.3f})")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "expanded_proteins_3b_evaluation.csv"),
               index=False)
    print("wrote results/expanded_proteins_3b_evaluation.csv")


if __name__ == "__main__":
    main()
