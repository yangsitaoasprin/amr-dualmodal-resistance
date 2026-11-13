#!/usr/bin/env python3
"""27_audit_controls.py — third-round expert-review controls.

(1) Recompute the AUC of the PCA(8) ablation (the manuscript currently quotes
    "AUC 0.855 -> 0.856" with no recorded source -> must be verified).
(2) Position-shuffle null for the structural modality: the four structural
    features are CONSTANT across all 19 substitutions at a position. Is the
    "+0.16 rho dual-modal gain" really structural biology, or just a
    position-level constant that lets the tree learn a position prior?
    We shuffle the position->structure-vector mapping (preserving the
    multivariate distribution) and retrain M1/M2/M3.
(3) Position-ID-only control: replace structural features by a single
    position index (or permutation-scrambled rank) to test the same.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data")
R = os.path.join(HERE, "..", "results")
SEQ = ["esm2", "blosum62", "d_hydro", "d_vol", "d_chg"]
STR = ["pLDDT", "contact_num", "d_s70", "d_omega"]

df = pd.read_csv(os.path.join(D, "tem1_stiffler_modeling_table.csv"))
emb = np.load(os.path.join(D, "tem1_esm2_emb.npy"))


def cv_auc(d, feats, k=5):
    fold = d["pos"].values % k
    rhos, preds, yb = [], [], []
    for f in range(k):
        tr, te = fold != f, fold == f
        m = GradientBoostingRegressor(random_state=0)
        m.fit(d.loc[tr, feats], d.loc[tr, "DMS_score"])
        p = m.predict(d.loc[te, feats])
        rhos.append(spearmanr(p, d.loc[te, "DMS_score"]).statistic)
        preds.append(p); yb.append(d.loc[te, "DMS_score_bin"].values)
    return (float(np.mean(rhos)), float(np.std(rhos)),
            float(roc_auc_score(np.concatenate(yb), np.concatenate(preds))))

# ---------------------------------------------------------------- (1) PCA AUC
pcs = PCA(n_components=8).fit_transform(emb)
pc_map = {p: pcs[p - 1] for p in range(1, emb.shape[0] + 1)}
d = df.copy()
for j in range(8):
    d[f"pc{j}"] = d["pos"].map(lambda p, j=j: pc_map[p][j])
base = cv_auc(d, SEQ + STR)
pca = cv_auc(d, SEQ + STR + [f"pc{j}" for j in range(8)])
print("(1) PCA(8) ablation  base rho/sd/AUC = %.4f %.4f %.4f" % base)
print("                     PCA  rho/sd/AUC = %.4f %.4f %.4f" % pca)
print("                     delta rho = %+.4f  delta AUC = %+.4f"
      % (pca[0] - base[0], pca[2] - base[2]))

# ------------------------------------------- (2) position-shuffle null (20x)
pos = sorted(df["pos"].unique())
vec = df.drop_duplicates("pos").set_index("pos")[STR].loc[pos].values
rng = np.random.default_rng(20261113)
rows = []
for rep in range(20):
    perm = rng.permutation(len(pos))
    shuffled = {p: vec[perm[i]] for i, p in enumerate(pos)}
    dd = df.copy()
    for j, c in enumerate(STR):
        dd[c] = dd["pos"].map({p: float(shuffled[p][j]) for p in pos})
    r1, _, _ = cv_auc(dd, SEQ)
    r2, _, _ = cv_auc(dd, STR)
    r3, _, _ = cv_auc(dd, SEQ + STR)
    rows.append(dict(rep=rep, M1=r1, M2=r2, M3=r3, gain=r3 - r1))
S = pd.DataFrame(rows)
print("\n(2) position-shuffled structure (20 permutations):")
print("    M3 mean %.4f +/- %.4f (range %.4f-%.4f)"
      % (S.M3.mean(), S.M3.std(), S.M3.min(), S.M3.max()))
print("    gain over M1 mean %+.4f +/- %.4f" % (S.gain.mean(), S.gain.std()))
print("    genuine (unshuffled) M3 = %.4f, gain = %+.4f"
      % (base[0], base[0] - cv_auc(df, SEQ)[0]))

# ------------------------------------------- (3) position-ID-only controls
for label, col in [("position index", None), ("scrambled position id", "scr")]:
    dd = df.copy()
    if col is None:
        dd["posid"] = dd["pos"].astype(float)
    else:
        mp = {p: float(v) for p, v in zip(pos, rng.permutation(len(pos)))}
        dd["posid"] = dd["pos"].map(mp)
    r1, _, _ = cv_auc(dd, SEQ)
    r3, _, _ = cv_auc(dd, SEQ + ["posid"])
    r2, _, _ = cv_auc(dd, ["posid"])
    print("(3) %-24s M1 %.4f | posid-only %.4f | M1+posid %.4f | gain %+.4f"
          % (label, r1, r2, r3, r3 - r1))

pd.DataFrame(rows).to_csv(os.path.join(R, "audit_position_shuffle.csv"), index=False)
print("\nsaved -> results/audit_position_shuffle.csv")
