#!/usr/bin/env python3
"""
02_train_gbm.py — Train/evaluate the dual-modal GBM (and single-modality
ablations) under modulo 5-fold by-position cross-validation, and the
cross-protein transfer matrix (train A -> test B).

Model: sklearn GradientBoostingRegressor with default hyperparameters
(n_estimators=100, learning_rate=0.1, max_depth=3), the configuration used
in the manuscript.

Reproduces Tables 1, 3, 4.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

FEATS_SEQ = ["esm2", "blosum62", "d_hydro", "d_vol", "d_chg"]
# structural columns differ by protein (TEM-1: catalytic Ser70 + omega-loop;
# VIM-2: residue-64 + loop distances named d_act/d_loop)
FEATS_STR_TEM1 = ["pLDDT", "contact_num", "d_s70", "d_omega"]
FEATS_STR_VIM2 = ["pLDDT", "contact_num", "d_act", "d_loop"]


def make_model():
    return GradientBoostingRegressor(random_state=0)


def modulo_folds(pos, k=5):
    """Stringent by-position split: residues with pos % k == f are held out,
    so no position (nor its other substitutions) leaks across folds."""
    return pos % k


def cv_eval(df, feats, k=5):
    """Per-fold Spearman rho (mean +/- std) plus POOLED AUC computed from the
    concatenated out-of-fold predictions (as reported in the manuscript)."""
    fold = modulo_folds(df["pos"].values, k)
    rhos, preds_all, y_all, yb_all = [], [], [], []
    for f in range(k):
        tr, te = fold != f, fold == f
        m = make_model()
        m.fit(df.loc[tr, feats], df.loc[tr, "DMS_score"])
        pred = m.predict(df.loc[te, feats])
        rhos.append(spearmanr(pred, df.loc[te, "DMS_score"]).statistic)
        preds_all.append(pred)
        y_all.append(df.loc[te, "DMS_score"].values)
        yb_all.append(df.loc[te, "DMS_score_bin"].values)
    yb = np.concatenate(yb_all)
    aucs = ([roc_auc_score(yb, np.concatenate(preds_all))]
            if np.isfinite(yb).all() else [])
    return np.array(rhos), np.array(aucs)


def main():
    tem1 = pd.read_csv("../data/tem1_stiffler_modeling_table.csv")
    vim2 = pd.read_csv("../data/vim2_modeling_table.csv")

    aac = pd.read_csv("../data/aac3i_modeling_table.csv")
    for name, df, sstr in [("TEM-1", tem1, FEATS_STR_TEM1),
                           ("VIM-2", vim2, FEATS_STR_VIM2),
                           ("AAC(3)-I", aac, ["pLDDT", "contact_num"])]:
        for label, feats in [("seq-only", FEATS_SEQ),
                             ("struct-only", sstr),
                             ("dual-modal", FEATS_SEQ + sstr)]:
            r, a = cv_eval(df, feats)
            print(f"{name:6s} {label:11s} rho = {r.mean():.3f} +/- {r.std():.3f}"
                  + (f"  AUC = {a.mean():.3f}" if len(a) else ""))

    # ---- cross-protein transfer (Table 4) ----
    # beta-lactamase <-> beta-lactamase: full 9-feature space with the
    # VIM-2 anchor columns renamed onto the TEM-1 names (d_act->d_s70,
    # d_loop->d_omega). Transfers involving AAC(3)-I use the 7-feature
    # anchor-free space (sequence features + pLDDT + contact_num),
    # since AAC(3)-I has no catalytic anchor.
    vim2_ren = vim2.rename(columns={"d_act": "d_s70", "d_loop": "d_omega"})
    full = FEATS_SEQ + ["pLDDT", "contact_num", "d_s70", "d_omega"]
    common = FEATS_SEQ + ["pLDDT", "contact_num"]

    def transfer(dtr, dte, ftr, fte):
        m = make_model()
        m.fit(dtr[ftr], dtr["DMS_score"])
        return spearmanr(m.predict(dte[fte]), dte["DMS_score"]).statistic

    print("TEM-1 -> VIM-2", round(transfer(tem1, vim2_ren, full, full), 3))
    print("VIM-2 -> TEM-1", round(transfer(vim2_ren, tem1, full, full), 3))
    print("AAC(3)-I -> TEM-1", round(transfer(aac, tem1, common, common), 3))
    print("AAC(3)-I -> VIM-2", round(transfer(aac, vim2, common, common), 3))
    print("TEM-1 -> AAC(3)-I", round(transfer(tem1, aac, common, common), 3))
    print("VIM-2 -> AAC(3)-I", round(transfer(vim2, aac, common, common), 3))


if __name__ == "__main__":
    main()
