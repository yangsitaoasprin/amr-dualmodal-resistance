#!/usr/bin/env python3
"""28_transfer_controls.py — P2 methodological controls (third review round).

A. PCA dimensionality: do 8 leading PCs really exhaust the ESM-2 embedding
   signal?  Compare 8 / 32 / 64 / 128 PCs and the full 320-D embedding.
B. Cross-protein transfer: is the ~0.77 transfer driven by the two anchor
   distance features (which are not physically equivalent across Ambler
   classes)?  Controls: (i) anchor distances set to zero for ALL proteins;
   (ii) per-protein z-scored features.  Full 5x5 matrices at 8M and 650M.
C. Robustness: five fold offsets, five GBM seeds, bootstrap 95% CI of the
   pooled out-of-fold Spearman rho, and a paired bootstrap test of the
   dual-modal vs sequence-only improvement (plus Wilcoxon on fold-level rho).

Outputs: results/p2_pca_dims.csv, p2_transfer_controls.csv,
         p2_robustness.json, p2_robustness_report.md
"""
import os, json
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data")
R = os.path.join(HERE, "..", "results")
F = lambda fn: pd.read_csv(os.path.join(R, fn))
SEQ = ["esm2", "blosum62", "d_hydro", "d_vol", "d_chg"]
STR = ["pLDDT", "contact_num", "d_s70", "d_omega"]
NOANCH = SEQ + ["pLDDT", "contact_num"]
FULL = SEQ + STR


def cv(d, feats, k=5, offset=0, seed=0, fold_map=None):
    fold = (d["pos"].map(fold_map).values if fold_map is not None
            else (d["pos"].values + offset) % k)
    rhos, preds, yb = [], [], []
    for f in range(k):
        tr, te = fold != f, fold == f
        m = GradientBoostingRegressor(random_state=seed)
        m.fit(d.loc[tr, feats], d.loc[tr, "DMS_score"])
        p = m.predict(d.loc[te, feats])
        rhos.append(spearmanr(p, d.loc[te, "DMS_score"]).statistic)
        preds.append(p); yb.append(d.loc[te, "DMS_score_bin"].values)
    return np.array(rhos), roc_auc_score(np.concatenate(yb), np.concatenate(preds))


# ------------------------------------------------------------------ A. PCA dims
base = pd.read_csv(os.path.join(D, "tem1_stiffler_modeling_table.csv"))
emb = np.load(os.path.join(D, "tem1_esm2_emb.npy"))
rows = []
r0, a0 = cv(base, FULL)
rows.append(dict(n_pc=0, rho=r0.mean(), rho_sd=r0.std(), auc=a0,
                 delta_rho=0.0, note="9-feature dual-modal baseline"))
for n in [8, 32, 64, 128, 320]:
    if n == 320:
        feats_arr = emb
    else:
        feats_arr = PCA(n_components=n, random_state=0).fit_transform(emb)
    d = base.copy()
    cols = []
    for j in range(n):
        c = f"e{j}"; cols.append(c)
        d[c] = d["pos"].map(lambda p, j=j: float(feats_arr[p - 1, j]))
    r, a = cv(d, FULL + cols)
    rows.append(dict(n_pc=n, rho=r.mean(), rho_sd=r.std(), auc=a,
                     delta_rho=r.mean() - r0.mean(), note="dual + %d embedding dims" % n))
    print("PCA %3d dims: rho %.4f +/- %.4f  AUC %.4f  dRho %+.4f"
          % (n, r.mean(), r.std(), a, r.mean() - r0.mean()))
pd.DataFrame(rows).to_csv(os.path.join(R, "p2_pca_dims.csv"), index=False)

# ------------------------------------------------- B. cross-protein transfer
TABLES = {
    "TEM-1": "tem1_stiffler_modeling_table.csv",
    "VIM-2": "vim2_modeling_table.csv",
    "AAC(3)-I": "aac3i_modeling_table.csv",
    "APH(3')-IIa": "aph3ii_modeling_table.csv",
    "DHFR": "dhfr_modeling_table.csv",
}
ASSAY_OF = {
    "TEM-1": "TEM1_Stiffler2015",
    "VIM-2": "VIM2_Chen2020",
    "AAC(3)-I": "AAC3I_Dandage2018",
    "APH(3')-IIa": "APH3II_Melnikov2014",
    "DHFR": "DHFR_Nguyen2023",
}
ZS = pd.concat([
    F("esm2_650m_zeroshot.csv"), F("esm2_650m_newproteins.csv")
], ignore_index=True)


def load(tag):
    data = {}
    for name, fn in TABLES.items():
        t = pd.read_csv(os.path.join(D, fn)).rename(
            columns={"d_act": "d_s70", "d_loop": "d_omega"})
        if name == "AAC(3)-I":
            t["d_s70"] = 0.0; t["d_omega"] = 0.0
        if tag == "650m":
            z = ZS[ZS.assay == ASSAY_OF[name]][["mutant", "esm2_650m"]]
            t = t.merge(z, on="mutant", how="inner")
            t["esm2"] = t["esm2_650m"]
        data[name] = t
    return data


ANCHORLESS = {"AAC(3)-I"}


def transfer_matrix(data, mode):
    """Reproduce scripts 14/15 exactly for as_published: the feature space is
    set by the TRAINING protein (7 features when AAC(3)-I is the trainer), the
    anchorless TEST protein is scored in the 9-feature space with zeros, and
    an AAC-trained model is applied to the 7 anchor-free features."""
    out = {}
    for a in data:
        F = NOANCH if a in ANCHORLESS else FULL
        work = {n: t.copy() for n, t in data.items()}
        if mode == "zero_anchor":
            for n in work:
                work[n]["d_s70"] = 0.0; work[n]["d_omega"] = 0.0
        elif mode == "zscore":
            for n in work:
                for c in FULL:
                    s = work[n][c].std(ddof=0)
                    work[n][c] = (work[n][c] - work[n][c].mean()) / (s if s > 0 else 1.0)
        m = GradientBoostingRegressor(random_state=0).fit(work[a][F], work[a].DMS_score)
        for b, db in work.items():
            if a == b: continue
            Ft = NOANCH if b in ANCHORLESS else FULL
            if a not in ANCHORLESS and b in ANCHORLESS: Ft = F
            if a in ANCHORLESS and b not in ANCHORLESS: Ft = F
            p = m.predict(db[Ft])
            out[f"{a}->{b}"] = float(spearmanr(p, db.DMS_score).statistic)
    return out


trows = []
for tag in ["8m", "650m"]:
    data = load(tag)
    for mode in ["as_published", "zero_anchor", "zscore"]:
        M = transfer_matrix(data, mode)
        for k, v in M.items():
            trows.append(dict(backbone=tag, mode=mode, pair=k, rho=v))
T = pd.DataFrame(trows)
T.to_csv(os.path.join(R, "p2_transfer_controls.csv"), index=False)
piv = T.pivot_table(index="pair", columns=["backbone", "mode"], values="rho")
key = ["TEM-1->VIM-2", "VIM-2->TEM-1", "APH(3')-IIa->TEM-1",
       "TEM-1->AAC(3)-I", "DHFR->TEM-1"]
print("\nCross-protein transfer controls (rho):")
print(piv.loc[[k for k in key if k in piv.index]].round(3).to_string())

# ------------------------------------------------------------- C. robustness
out = {}
off, sd = [], []
uniq_pos = np.sort(base["pos"].unique())
for s in range(5):
    # five genuinely distinct by-position partitions (no position leakage)
    perm = np.random.default_rng(1000 + s).permutation(uniq_pos)
    fmap = {int(p): i % 5 for i, p in enumerate(perm)}
    r, _ = cv(base, FULL, fold_map=fmap); off.append(r.mean())
for s in range(5):
    r, _ = cv(base, FULL, seed=s); sd.append(r.mean())
out["random_by_position_splits"] = dict(values=off, mean=float(np.mean(off)),
                                        sd=float(np.std(off)), n_splits=5)
out["modulo5_reference"] = float(r0.mean())
out["gbm_seeds"] = dict(values=sd, mean=float(np.mean(sd)), sd=float(np.std(sd)))
print("\nRobustness: random by-position splits %.4f +/- %.4f | GBM seeds %.4f +/- %.4f"
      % (np.mean(off), np.std(off), np.mean(sd), np.std(sd)))

# pooled oof bootstrap CI + paired bootstrap of dual vs seq
z = np.load(os.path.join(R, "oof_predictions_tem1_8m.npz"), allow_pickle=True)
truth, dual, seq = z["oof_true"], z["oof_dual"], z["oof_seq"]
rng = np.random.default_rng(20261113)
n = len(truth)
boots, deltas = [], []
for _ in range(2000):
    idx = rng.integers(0, n, n)
    boots.append(spearmanr(dual[idx], truth[idx]).statistic)
    deltas.append(spearmanr(dual[idx], truth[idx]).statistic
                  - spearmanr(seq[idx], truth[idx]).statistic)
lo, hi = np.percentile(boots, [2.5, 97.5])
dlo, dhi = np.percentile(deltas, [2.5, 97.5])
out["oof_bootstrap"] = dict(rho=float(spearmanr(dual, truth).statistic),
                            ci95=[float(lo), float(hi)],
                            delta_rho=float(spearmanr(dual, truth).statistic
                                            - spearmanr(seq, truth).statistic),
                            delta_ci95=[float(dlo), float(dhi)], n_boot=2000)
print("pooled oof rho %.4f (bootstrap 95%% CI %.4f-%.4f); "
      "dual-vs-seq delta %.4f (CI %.4f-%.4f)"
      % (spearmanr(dual, truth).statistic, lo, hi,
         out["oof_bootstrap"]["delta_rho"], dlo, dhi))

# fold-level Wilcoxon on Tables 5-6 (5 paired folds; low power, descriptive)
for name, fn in [("TEM-1", "gcnv2_5fold_cv.csv"), ("VIM-2", "vim2_gcnv2_5fold_cv.csv")]:
    d = F(fn)
    try:
        w = wilcoxon(d.GCNv2_rho, d.GBM_rho).pvalue
    except Exception:
        w = float("nan")
    out.setdefault("wilcoxon_gcnv2", {})[name] = float(w)
    print("Wilcoxon %-6s GCNv2 vs GBM (5 folds): P = %.3f" % (name, w))

json.dump(out, open(os.path.join(R, "p2_robustness.json"), "w"), indent=2, default=str)
with open(os.path.join(R, "p2_robustness_report.md"), "w", encoding="utf-8") as f:
    f.write("# P2 robustness and control results\n\n")
    f.write("Fold offsets: %s\n\nGBM seeds: %s\n\n" % (off, sd))
    f.write("OOF bootstrap: %s\n\n" % out["oof_bootstrap"])
    f.write("Wilcoxon: %s\n\n" % out.get("wilcoxon_gcnv2"))
    f.write("\nPCA dims:\n" + pd.DataFrame(rows).to_string(index=False))
    f.write("\n\nTransfer controls:\n" + piv.round(3).to_string())
print("\nsaved -> p2_pca_dims.csv, p2_transfer_controls.csv, p2_robustness.json/report.md")
