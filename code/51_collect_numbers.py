#!/usr/bin/env python3
"""51_collect_numbers.py — collect every manuscript number from the
canonical (corrected-anchor) result bundle, and refresh the static
multiprotein_results.csv TEM-1 row.

Writes: results/canonical_numbers.json + prints a Markdown summary.
Run AFTER 50_rerun_pipeline.py has finished.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data")
R = os.path.join(HERE, "..", "results")
F = lambda fn: pd.read_csv(os.path.join(R, fn))

SEQ = ["esm2", "blosum62", "d_hydro", "d_vol", "d_chg"]
STR = ["pLDDT", "contact_num", "d_s70", "d_omega"]

def cv(df, feats, k=5):
    fold = df["pos"].values % k
    rhos, preds, yb = [], [], []
    for f in range(k):
        tr, te = fold != f, fold == f
        m = GradientBoostingRegressor(random_state=0)
        m.fit(df.loc[tr, feats], df.loc[tr, "DMS_score"])
        p = m.predict(df.loc[te, feats])
        rhos.append(spearmanr(p, df.loc[te, "DMS_score"]).statistic)
        preds.append(p); yb.append(df.loc[te, "DMS_score_bin"].values)
    return np.array(rhos), roc_auc_score(np.concatenate(yb), np.concatenate(preds))

def main():
    out = {}
    stif = pd.read_csv(os.path.join(D, "tem1_stiffler_modeling_table.csv"))
    deng = pd.read_csv(os.path.join(D, "BLAT_ECOLX_Deng_2012_modeling_table.csv"))
    firn = pd.read_csv(os.path.join(D, "BLAT_ECOLX_Firnberg_2014_modeling_table.csv"))
    ext = {"Deng2012": deng, "Firnberg2014": firn}

    # ---- Table 1: 8M block (canonical)
    r0 = spearmanr(stif.esm2, stif.DMS_score).statistic
    a0 = roc_auc_score(stif.DMS_score_bin, stif.esm2)
    r1, a1 = cv(stif, SEQ)
    r2, a2 = cv(stif, STR)
    r3, a3 = cv(stif, SEQ + STR)
    out["T1_8m"] = dict(M0=float(r0), M0_auc=float(a0),
                        M1=float(r1.mean()), M1_sd=float(r1.std()), M1_auc=float(a1),
                        M2=float(r2.mean()), M2_sd=float(r2.std()), M2_auc=float(a2),
                        M3=float(r3.mean()), M3_sd=float(r3.std()), M3_auc=float(a3))
    print("Table 1 / 8M:", {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in out["T1_8m"].items()})

    # ---- Table 1: 650M / 3B blocks from regenerated evaluation CSVs
    for tag, fn in [("650m", "esm2_650m_evaluation.csv"),
                    ("3b", "esm2_3b_evaluation.csv")]:
        e = F(fn)
        t = e[e.assay == "TEM1_Stiffler2015"]
        d = {}
        for _, row in t[t.protocol == "cv5_modulo"].iterrows():
            d[row.model] = [float(row.rho), float(row.rho_std), float(row.auc)]
        for _, row in t[t.protocol == "full_assay"].iterrows():
            d[row.model] = [float(row.rho), None, float(row.auc)]
        out["T1_" + tag] = d
        print("Table 1 /", tag, {k: [round(x, 4) if isinstance(x, float) else x for x in v]
                                  for k, v in d.items()})

    # ---- Table 2: external validation (trained on Stiffler, no retraining)
    t2 = {}
    for name, te in ext.items():
        for label, feats in [("seq", SEQ), ("dual", SEQ + STR)]:
            m = GradientBoostingRegressor(random_state=0)
            m.fit(stif[feats], stif.DMS_score)
            rho = spearmanr(m.predict(te[feats]), te.DMS_score).statistic
            t2[f"{name}_{label}"] = float(rho)
        t2[f"{name}_zeroshot8m"] = float(spearmanr(te.esm2, te.DMS_score).statistic)
    # 650M version
    z = F("esm2_650m_zeroshot.csv")
    z = z[z.assay == "TEM1_Stiffler2015"][["mutant", "esm2_650m"]]
    s650 = stif.merge(z, on="mutant"); s650["esm2"] = s650["esm2_650m"]
    for name, te0 in ext.items():
        te = te0.merge(z, on="mutant"); te["esm2"] = te["esm2_650m"]
        m = GradientBoostingRegressor(random_state=0)
        m.fit(s650[SEQ + STR], s650.DMS_score)
        t2[f"{name}_dual650m"] = float(spearmanr(m.predict(te[SEQ + STR]),
                                                 te.DMS_score).statistic)
        t2[f"{name}_zeroshot650m"] = float(spearmanr(te.esm2, te.DMS_score).statistic)
    out["T2"] = t2
    print("Table 2:", {k: round(v, 4) for k, v in t2.items()})

    # ---- Table 5/6: GCNv2 vs GBM on canonical features
    for tag, fn in [("TEM1", "gcnv2_5fold_cv.csv"), ("VIM2", "vim2_gcnv2_5fold_cv.csv")]:
        t = F(fn)
        g, b = t.GCNv2_rho.values, t.GBM_rho.values
        out["T56_" + tag] = dict(GCNv2=float(g.mean()), GCNv2_sd=float(g.std()),
                                 GBM=float(b.mean()), GBM_sd=float(b.std()),
                                 p=float(ttest_rel(g, b).pvalue),
                                 diff=float((g - b).mean()))
        print("Table 5/6", tag, {k: round(v, 4) for k, v in out["T56_" + tag].items()})

    # ---- static multiprotein_results.csv: refresh TEM-1 row
    mp_path = os.path.join(R, "multiprotein_results.csv")
    mp = pd.read_csv(mp_path)
    mp.loc[0, "seq"] = out["T1_8m"]["M1"]
    mp.loc[0, "dual"] = out["T1_8m"]["M3"]
    mp.loc[0, "auc_dual"] = out["T1_8m"]["M3_auc"]
    mp.to_csv(mp_path, index=False)
    print("refreshed multiprotein_results.csv TEM-1 row")

    with open(os.path.join(R, "canonical_numbers.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved -> results/canonical_numbers.json")


if __name__ == "__main__":
    main()
