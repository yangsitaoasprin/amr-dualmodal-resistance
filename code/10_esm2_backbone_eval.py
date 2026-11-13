#!/usr/bin/env python3
"""
10_esm2_backbone_eval.py — Evaluate ESM-2 large-backbone configurations
(650M and 3B) against the dual-modal GBM, across all assays and protocols
reported in the manuscript (modulo 5-fold by-position CV, ablations,
cross-experiment external validation, cross-protein transfer).

Inputs:  results/esm2_650m_zeroshot.csv   (code/09_esm2_650m_zeroshot.py)
         results/esm2_3b_zeroshot.csv     (code/09_esm2_650m_zeroshot.py, 3B model)
         data/*_modeling_table.csv
Outputs: results/esm2_650m_evaluation.csv
         results/esm2_3b_evaluation.csv
         (one row per assay/protocol/metric)
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")

TABLES = {
    "TEM1_Stiffler2015": "tem1_stiffler_modeling_table.csv",
    "TEM1_Deng2012": "BLAT_ECOLX_Deng_2012_modeling_table.csv",
    "TEM1_Firnberg2014": "BLAT_ECOLX_Firnberg_2014_modeling_table.csv",
    "VIM2_Chen2020": "ampc_modeling_table.csv",
    "AAC3I_Dandage2018": "aac3i_modeling_table.csv",
}
PHY = ["blosum62", "d_hydro", "d_vol", "d_chg"]


def load(tag):
    zs = pd.read_csv(os.path.join(RES, f"esm2_{tag}_zeroshot.csv"))
    zcol = f"esm2_{tag}"
    merged = {}
    for assay, fn in TABLES.items():
        df = pd.read_csv(os.path.join(DATA, fn))
        df = df.rename(columns={"d_act": "d_s70", "d_loop": "d_omega"})
        zsa = zs[zs.assay == assay][["mutant", zcol]]
        assert not zsa.duplicated("mutant").any()
        merged[assay] = df.merge(zsa, on="mutant")
    return merged, zcol


def make_feats(zcol):
    def feats(assay, mode):
        if mode == "seq":
            return [zcol] + PHY
        struct = ["pLDDT", "contact_num"] if assay == "AAC3I_Dandage2018" else [
            "pLDDT", "contact_num", "d_s70", "d_omega"]
        return [zcol] + PHY + struct
    return feats


def rho_auc(pred, df):
    return (spearmanr(pred, df.DMS_score).statistic,
            roc_auc_score(df.DMS_score_bin, pred))


def evaluate(tag):
    merged, zcol = load(tag)
    feats = make_feats(zcol)
    rows = []

    def add(assay, protocol, model, rho, std, auc, note=""):
        rows.append(dict(assay=assay, protocol=protocol, model=model,
                         rho=rho, rho_std=std, auc=auc, note=note))

    for assay, m in merged.items():
        r, a = rho_auc(m[zcol], m)
        add(assay, "full_assay", f"esm2_{tag}_zeroshot", r, np.nan, a)
        for mode, name in [("seq", f"gbm_{tag}_seq"), ("dual", f"gbm_{tag}_dual")]:
            F = feats(assay, mode)
            rhos, aucs = [], []
            for k in range(5):
                te = m[m.pos % 5 == k]
                tr = m[m.pos % 5 != k]
                g = GradientBoostingRegressor(random_state=0).fit(tr[F], tr.DMS_score)
                r, a = rho_auc(g.predict(te[F]), te)
                rhos.append(r); aucs.append(a)
            add(assay, "cv5_modulo", name, np.mean(rhos), np.std(rhos),
                np.mean(aucs), "folds=" + ",".join(f"{x:.3f}" for x in rhos))
        F = feats(assay, "dual")[1 + len(PHY):]
        rhos, preds, bins = [], [], []
        for k in range(5):
            te = m[m.pos % 5 == k]; tr = m[m.pos % 5 != k]
            g = GradientBoostingRegressor(random_state=0).fit(tr[F], tr.DMS_score)
            p = g.predict(te[F])
            rhos.append(spearmanr(p, te.DMS_score).statistic)
            preds.append(p); bins.append(te.DMS_score_bin.values)
        # AUC here is a *pooled* out-of-fold AUC (one ranking over all five test
        # folds), not the mean of per-fold AUCs used for the seq/dual rows above.
        # This is the convention behind Table 1's M2 AUC (0.744, i.e.
        # canonical_numbers.json T1_*["M2_auc"] = 0.7443592984876293); the
        # mean-of-fold equivalent (0.748) is delta_rho_table.csv's auc_struct.
        pooled_auc = roc_auc_score(np.concatenate(bins), np.concatenate(preds))
        add(assay, "cv5_modulo", f"gbm_{tag}_struct_only", np.mean(rhos),
            np.std(rhos), pooled_auc)

    tr = merged["TEM1_Stiffler2015"]
    F = feats("TEM1_Stiffler2015", "dual")
    g = GradientBoostingRegressor(random_state=0).fit(tr[F], tr.DMS_score)
    for assay in ["TEM1_Deng2012", "TEM1_Firnberg2014"]:
        te = merged[assay]
        r, a = rho_auc(g.predict(te[F]), te)
        add(assay, "external_from_Stiffler", f"gbm_{tag}_dual", r, np.nan, a)

    if tag == "650m":
        for tr_a in ["TEM1_Stiffler2015", "VIM2_Chen2020", "AAC3I_Dandage2018"]:
            g = GradientBoostingRegressor(random_state=0).fit(
                merged[tr_a][feats(tr_a, "dual")], merged[tr_a].DMS_score)
            for te_a in ["TEM1_Stiffler2015", "VIM2_Chen2020", "AAC3I_Dandage2018"]:
                if te_a == tr_a:
                    continue
                te = merged[te_a].copy()
                Fte = feats(te_a, "dual")
                if tr_a != "AAC3I_Dandage2018" and te_a == "AAC3I_Dandage2018":
                    te["d_s70"] = 0.0; te["d_omega"] = 0.0
                    Fte = feats(tr_a, "dual")
                if tr_a == "AAC3I_Dandage2018" and te_a != "AAC3I_Dandage2018":
                    Fte = feats(tr_a, "dual")
                r, a = rho_auc(g.predict(te[Fte]), te)
                add(f"{tr_a}->{te_a}", "cross_protein", f"gbm_{tag}_dual",
                    r, np.nan, a)

    out = pd.DataFrame(rows)
    path = os.path.join(RES, f"esm2_{tag}_evaluation.csv")
    out.to_csv(path, index=False)
    print(f"==== {tag} ====")
    print(out.to_string(index=False))
    print("saved", path)


if __name__ == "__main__":
    evaluate("650m")
    evaluate("3b")
