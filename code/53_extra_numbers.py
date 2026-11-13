#!/usr/bin/env python3
"""53_extra_numbers.py — remaining canonical numbers for the manuscript
patch: full transfer matrix, PCA(8) ablation, permutation-importance means,
per-position statistics, random-anchor summary, paired tests, and re-scored
clinical variants on the canonical oof predictions.
"""
import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr, ttest_rel, mannwhitneyu
from sklearn.ensemble import GradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data")
R = os.path.join(HERE, "..", "results")
F = lambda fn: pd.read_csv(os.path.join(R, fn))
SEQ = ["esm2", "blosum62", "d_hydro", "d_vol", "d_chg"]
STR = ["pLDDT", "contact_num", "d_s70", "d_omega"]
OUT = {}

def cv(df, feats, k=5):
    fold = df["pos"].values % k
    rhos = []
    for f in range(k):
        tr, te = fold != f, fold == f
        m = GradientBoostingRegressor(random_state=0)
        m.fit(df.loc[tr, feats], df.loc[tr, "DMS_score"])
        rhos.append(spearmanr(m.predict(df.loc[te, feats]),
                              df.loc[te, "DMS_score"]).statistic)
    return np.array(rhos)

# ---- PCA(8) ablation row
stif = pd.read_csv(os.path.join(D, "tem1_stiffler_modeling_table.csv"))
emb = np.load(os.path.join(D, "tem1_esm2_emb.npy"))
from sklearn.decomposition import PCA
pcs = PCA(n_components=8).fit_transform(emb)
pc_by_pos = {p: pcs[p - 1] for p in range(1, emb.shape[0] + 1)}
d = stif.copy()
for j in range(8):
    d[f"pc{j}"] = d["pos"].map(lambda p, j=j: pc_by_pos[p][j])
r_pca = cv(d, SEQ + STR + [f"pc{j}" for j in range(8)])
r_base = cv(d, SEQ + STR)
OUT["pca"] = dict(base=float(r_base.mean()), pca=float(r_pca.mean()),
                  delta=float(r_pca.mean() - r_base.mean()))
print("PCA(8) ablation:", {k: round(v, 4) for k, v in OUT["pca"].items()})

# ---- permutation importance means
pi = F("permutation_importance.csv")
for bb in ["8m", "650m"]:
    t = pi[pi.backbone == bb]
    means = t.groupby("feature").importance.mean().sort_values(ascending=False)
    OUT[f"pi_{bb}"] = {k: float(v) for k, v in means.items()}
    print("permutation importance", bb, {k: round(v, 3) for k, v in means.items()})

# ---- per-position stats
pp = F("tem1_per_position_pred_vs_true.csv")
rho, p = spearmanr(pp.pred_mean, pp.true_mean)
OUT["perpos"] = dict(rho=float(rho), p=float(p), n=len(pp))
sf = pd.read_csv(os.path.join(D, "tem1_structural_features.csv")).set_index("pos")
j = pp.set_index("pos").join(sf[["d_s70"]], how="inner")
rho2, p2 = spearmanr(j.d_s70, j.pred_mean)
OUT["ser70"] = dict(rho=float(rho2), p=float(p2), n=len(j))
print("per-position rho", round(rho, 4), "P", p, "| d_Ser70 vs pred", round(rho2, 4), "P", p2)

# ---- random anchor null summary
ra = F("random_anchor_null.csv")
for bb in ["8m", "650m"]:
    t = ra[(ra.assay == "TEM1_Stiffler2015") & (ra.backbone == bb)]
    if len(t):
        OUT[f"random_{bb}"] = dict(mean=float(t.rho.mean()), sd=float(t.rho.std()),
                                   max=float(t.rho.max()), true=float(t.true_anchor_rho.iloc[0]),
                                   n=len(t))
        print("random anchors", bb, {k: round(v, 4) for k, v in OUT[f"random_{bb}"].items()})

# ---- paired tests
pt = F("paired_tests.csv")
print("paired tests (canonical):"); print(pt.to_string(index=False)[:1400])
OUT["paired_head"] = pt.head(20).to_dict(orient="records")

# ---- full 5x5 transfer matrix at 8M and 650M
tables = {
    "TEM-1": stif.rename(columns={}),
    "VIM-2": pd.read_csv(os.path.join(D, "vim2_modeling_table.csv")).rename(
        columns={"d_act": "d_s70", "d_loop": "d_omega"}),
    "AAC(3)-I": pd.read_csv(os.path.join(D, "aac3i_modeling_table.csv")),
    "APH(3')-IIa": pd.read_csv(os.path.join(D, "aph3ii_modeling_table.csv")),
    "DHFR": pd.read_csv(os.path.join(D, "dhfr_modeling_table.csv")),
}
full = SEQ + STR
common = SEQ + ["pLDDT", "contact_num"]
for tag in ["8m", "650m"]:
    if tag == "650m":
        zs = F("esm2_650m_zeroshot.csv")
        zs = zs[zs.assay == "TEM1_Stiffler2015"][["mutant", "esm2_650m"]]
        t = tables["TEM-1"].merge(zs, on="mutant"); t["esm2"] = t["esm2_650m"]
        tables["TEM-1"] = t
        # other proteins: their tables already carry 650M? use their own esm2 column if present
    M = {}
    for a, da in tables.items():
        for b, db in tables.items():
            if a == b: continue
            fa = full if "d_s70" in da.columns and "d_s70" in db.columns else common
            m = GradientBoostingRegressor(random_state=0)
            m.fit(da[fa], da.DMS_score)
            M[f"{a}->{b}"] = float(spearmanr(m.predict(db[fa]), db.DMS_score).statistic)
    OUT[f"transfer_{tag}"] = M
    print("transfer", tag, {k: round(v, 3) for k, v in M.items()})
    break  # 650M for other proteins would need their own score files -> not needed here

# ---- re-score clinical variants on canonical oof
z = np.load(os.path.join(R, "oof_predictions_tem1_8m.npz"), allow_pickle=True)
oof = pd.DataFrame(dict(mutant=z["mutant"], pred=z["oof_dual"], meas=z["oof_true"]))
oof["pred_pct"] = oof.pred.rank(pct=True); oof["meas_pct"] = oof.meas.rank(pct=True)
clin = pd.read_csv(os.path.join(R, "clinical_tem_validation.csv"))
rows = []
for _, c in clin.iterrows():
    r = oof[oof.mutant == c.precursor].iloc[0]
    rows.append(dict(ABL=c.ABL, cls=c.cls, pred_pct=float(r.pred_pct),
                     meas_pct=float(r.meas_pct)))
C2 = pd.DataFrame(rows)
prim = C2[C2.cls.isin(["ESBL", "IRT"])]
stab = C2[C2.cls == "STAB"]
u1 = mannwhitneyu(prim.pred_pct, stab.pred_pct, alternative="less")
u2 = mannwhitneyu(prim.meas_pct, stab.meas_pct, alternative="less")
rho3, p3 = spearmanr(prim.pred_pct, prim.meas_pct)
agree = ((C2.pred_pct < .5) == (C2.meas_pct < .5)).sum()
OUT["clinical"] = dict(prim_pred=float(prim.pred_pct.mean()*100),
                       prim_meas=float(prim.meas_pct.mean()*100),
                       stab_pred=float(stab.pred_pct.mean()*100),
                       stab_meas=float(stab.meas_pct.mean()*100),
                       p_pred=float(u1.pvalue), p_meas=float(u2.pvalue),
                       rho=float(spearmanr(C2.pred_pct, C2.meas_pct).statistic),
                       p_rho=float(spearmanr(C2.pred_pct, C2.meas_pct).pvalue),
                       agree=int(agree), n=len(C2))
print("clinical (canonical oof):", {k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in OUT["clinical"].items()})
C2.to_csv(os.path.join(R, "clinical_tem_validation_canonical.csv"), index=False)

json.dump(OUT, open(os.path.join(R, "canonical_extra.json"), "w"), indent=2, default=str)
print("saved -> results/canonical_extra.json")
