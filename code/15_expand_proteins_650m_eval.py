#!/usr/bin/env python3
"""
15_expand_proteins_650m_eval.py — Evaluation of the two expansion proteins
(APH(3')-IIa, DHFR) under the 650M backbone, and the full 5-protein
cross-protein transfer matrix with 650M zero-shot scores.

Mirrors the protocol of 14_expand_proteins_eval.py exactly (modulo 5-fold
by-position CV, GradientBoostingRegressor(random_state=0)), replacing the 8M
'esm2' feature with 'esm2_650m'.

Inputs:  data/{tem1_stiffler,ampc,aac3i,aph3ii,dhfr}_modeling_table.csv
         results/esm2_650m_zeroshot.csv      (original 3 assays + Deng/Firnberg)
         results/esm2_650m_newproteins.csv   (APH3II + DHFR, from 09b)
Output:  results/expanded_proteins_650m_evaluation.csv
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
    "AAC3I_Dandage2018": "aac3i_modeling_table.csv",
    "APH3II_Melnikov2014": "aph3ii_modeling_table.csv",
    "DHFR_Nguyen2023": "dhfr_modeling_table.csv",
}
ANCHORLESS = {"AAC3I_Dandage2018"}
FEAT = "esm2_650m"


def feats(assay, mode):
    if mode == "seq":
        return [FEAT] + PHY
    struct = (["pLDDT", "contact_num"] if assay in ANCHORLESS else
              ["pLDDT", "contact_num", "d_s70", "d_omega"])
    if mode == "struct":
        return struct
    return [FEAT] + PHY + struct


def load_tables():
    zs = pd.concat([
        pd.read_csv(os.path.join(RES, "esm2_650m_zeroshot.csv")),
        pd.read_csv(os.path.join(RES, "esm2_650m_newproteins.csv")),
    ], ignore_index=True)
    assert not zs.duplicated(["assay", "mutant"]).any(), "duplicate (assay, mutant) in zero-shot scores"
    data = {}
    for a, fn in TABLES.items():
        df = pd.read_csv(os.path.join(DATA, fn)).rename(
            columns={"d_act": "d_s70", "d_loop": "d_omega"})
        z = zs[zs.assay == a][["mutant", FEAT]]
        assert not z.duplicated("mutant").any(), a
        df = df.merge(z, on="mutant", how="inner")
        assert df[FEAT].notna().all(), a
        data[a] = df
        print(a, df.shape)
    return data


def main():
    data = load_tables()
    rows = []

    def add(assay, protocol, model, rho, std=np.nan, auc=np.nan, note=""):
        rows.append(dict(assay=assay, protocol=protocol, model=model,
                         rho=rho, rho_std=std, auc=auc, note=note))

    for assay in ["APH3II_Melnikov2014", "DHFR_Nguyen2023"]:
        m = data[assay]
        add(assay, "full_assay", "esm2_650m_zeroshot",
            spearmanr(m[FEAT], m.DMS_score).statistic,
            auc=roc_auc_score(m.DMS_score_bin, m[FEAT]))
        for mode, name in [("seq", "gbm_650m_seq"), ("struct", "gbm_650m_struct"),
                           ("dual", "gbm_650m_dual")]:
            F = feats(assay, mode)
            rs, aucs = [], []
            for k in range(5):
                te = m[m.pos % 5 == k]
                tr = m[m.pos % 5 != k]
                g = GradientBoostingRegressor(random_state=0).fit(
                    tr[F], tr.DMS_score)
                p = g.predict(te[F])
                rs.append(spearmanr(p, te.DMS_score).statistic)
                aucs.append(roc_auc_score(te.DMS_score_bin, p))
            add(assay, "cv5_modulo", name, np.mean(rs), np.std(rs),
                np.mean(aucs), "folds=" + ",".join(f"{x:.3f}" for x in rs))

    for tr_a in TABLES:
        F = feats(tr_a, "dual")
        g = GradientBoostingRegressor(random_state=0).fit(
            data[tr_a][F], data[tr_a].DMS_score)
        for te_a in TABLES:
            if te_a == tr_a:
                continue
            df = data[te_a].copy()
            Ft = feats(te_a, "dual")
            if tr_a not in ANCHORLESS and te_a in ANCHORLESS:
                df["d_s70"] = 0.0
                df["d_omega"] = 0.0
                Ft = F
            if tr_a in ANCHORLESS and te_a not in ANCHORLESS:
                Ft = F
            add(f"{tr_a}->{te_a}", "cross_protein", "gbm_650m_dual",
                spearmanr(g.predict(df[Ft]), df.DMS_score).statistic)

    out = pd.DataFrame(rows)
    path = os.path.join(RES, "expanded_proteins_650m_evaluation.csv")
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print("saved", path)


if __name__ == "__main__":
    main()
