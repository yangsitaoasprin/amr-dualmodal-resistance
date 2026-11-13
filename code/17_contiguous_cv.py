#!/usr/bin/env python3
"""
17_contiguous_cv.py — Contiguous-position 5-fold CV control.

The modulo by-position scheme assigns neighbouring positions to different
folds; neighbouring positions share correlated structural features and pLM
scores, which could leak neighbourhood information and inflate accuracy.
As a stringency control, split positions into five contiguous blocks along
the sequence (ProteinGym "contiguous" scheme) and re-evaluate the main
configurations (650M backbone: zero-shot / seq / struct / dual) for TEM-1
and VIM-2.

Output: results/contiguous_cv_650m.csv
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
}


def cv_contiguous(m, F, k=5):
    pos_sorted = np.array(sorted(m.pos.unique()))
    blocks = np.array_split(pos_sorted, k)  # contiguous position blocks
    rs, aucs = [], []
    for b in blocks:
        te = m[m.pos.isin(b)]
        tr = m[~m.pos.isin(b)]
        g = GradientBoostingRegressor(random_state=0).fit(tr[F], tr.DMS_score)
        p = g.predict(te[F])
        rs.append(spearmanr(p, te.DMS_score).statistic)
        aucs.append(roc_auc_score(te.DMS_score_bin, p))
    return np.mean(rs), np.std(rs), np.mean(aucs)


def main():
    zs = pd.read_csv(os.path.join(RES, "esm2_650m_zeroshot.csv"))
    rows = []
    for assay, fn in TABLES.items():
        m = pd.read_csv(os.path.join(DATA, fn)).rename(
            columns={"d_act": "d_s70", "d_loop": "d_omega"})
        z = zs[zs.assay == assay][["mutant", "esm2_650m"]]
        assert not z.duplicated("mutant").any(), assay
        m = m.merge(z, on="mutant", how="inner")
        rows.append(dict(assay=assay, model="zeroshot",
                         rho=spearmanr(m.esm2_650m, m.DMS_score).statistic,
                         rho_std=np.nan,
                         auc=roc_auc_score(m.DMS_score_bin, m.esm2_650m)))
        for mode, F in [("seq", ["esm2_650m"] + PHY),
                        ("struct", ["pLDDT", "contact_num", "d_s70", "d_omega"]),
                        ("dual", ["esm2_650m"] + PHY +
                                 ["pLDDT", "contact_num", "d_s70", "d_omega"])]:
            r, s, a = cv_contiguous(m, F)
            rows.append(dict(assay=assay, model=mode, rho=r, rho_std=s, auc=a))
            print(assay, mode, f"{r:.3f} +/- {s:.3f} (AUC {a:.3f})")
    out = pd.DataFrame(rows)
    path = os.path.join(RES, "contiguous_cv_650m.csv")
    out.to_csv(path, index=False)
    print("saved", path)


if __name__ == "__main__":
    main()
