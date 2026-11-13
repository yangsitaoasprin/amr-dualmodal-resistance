#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
22_verify_and_dump.py — 数据核对 + 补算 + 折外预测落盘

三件事：
  (1) 核对：把 repository/results/*.csv 里的每一个关键数字重新算一遍，
      与已归档值逐项比对，输出 PASS/FAIL 表。
  (2) 补算：现归档的逐折 ρ 只覆盖 APH/DHFR 的 8M，TEM-1 / VIM-2 / AAC(3)-I
      的 8M 缺逐折值（Fig 2(a) 的 p 值、Fig 2(c) 的配对斜率都需要）。
      用与 16_anchor_counterfactual.py 完全相同的协议重跑这三者。
  (3) 落盘：TEM-1 8M 的折外预测（Fig 1a 的散点+ROC、Fig S4 的校准/DCA）。

协议必须与 16_anchor_counterfactual.py 逐字一致：
    GradientBoostingRegressor(random_state=0)
    modulo 5-fold by-position split (pos % 5 == f held out)
    8M  : seq  = [esm2] + [blosum62, d_hydro, d_vol, d_chg]
          dual = seq + [pLDDT, contact_num, d_s70, d_omega]
          abl  = seq + [pLDDT, contact_num]
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
PROJ = os.path.abspath(os.path.join(REPO, ".."))
DATA_R = os.path.join(REPO, "data")       # repository/data
DATA_P = os.path.join(PROJ, "data")        # <proj>/data
RES = os.path.join(REPO, "results")

PHY = ["blosum62", "d_hydro", "d_vol", "d_chg"]

# 归档值（用于核对），来自 anchor_counterfactual.csv / aac_anchor_experiment.csv
ARCHIVE = {
    ("TEM1_Stiffler2015", "8m", "seq"): (0.5368262258356772, 0.030330534042055365),
    # canonical anchor (Ser68 / loop 162-178); the pre-fix anchor-70 value
    # 0.7037232728331677 lives in anchor_fix_cutoff_sensitivity.csv
    ("TEM1_Stiffler2015", "8m", "dual_full"): (0.6962633904632071, 0.03168912629420877),
    ("TEM1_Stiffler2015", "8m", "dual_noanchors"): (0.6959154198912143, 0.0273485191083014),
    ("VIM2_Chen2020", "8m", "seq"): (0.506466145431021, 0.029156483964864737),
    ("VIM2_Chen2020", "8m", "dual_full"): (0.7292596493217479, 0.05296746015724983),
    ("VIM2_Chen2020", "8m", "dual_noanchors"): (0.740775111325965, 0.04413083755438296),
    ("APH3II_Melnikov2014", "8m", "seq"): (0.37479715040442724, 0.07741332305424992),
    ("APH3II_Melnikov2014", "8m", "dual_full"): (0.6431802175965997, 0.024346509037182506),
    ("APH3II_Melnikov2014", "8m", "dual_noanchors"): (0.5758562711736004, 0.06849824162550044),
    ("DHFR_Nguyen2023", "8m", "seq"): (0.2811679013990183, 0.06447208837245272),
    ("DHFR_Nguyen2023", "8m", "dual_full"): (0.30156934387732337, 0.06361167938806658),
    ("DHFR_Nguyen2023", "8m", "dual_noanchors"): (0.3467276058786862, 0.08501535470211283),
}
AAC_ARCHIVE = {
    ("8m", "seq"): 0.2428945484989223,
    ("8m", "dual_noanchors"): 0.25139356723254147,
    ("8m", "dual_with_coa_anchor"): 0.23766943875455343,
    ("650m", "seq"): 0.4767475365711231,
    ("650m", "dual_noanchors"): 0.44646828866110466,
    ("650m", "dual_with_coa_anchor"): 0.42801512727396157,
}


def cv(m, F, k=5, collect=False):
    """逐折 Spearman rho（与 16_ 完全同协议）。collect=True 时返回折外预测。"""
    rs, aucs = [], []
    oof = np.full(len(m), np.nan)
    fold = (m.pos % k).values
    for f in range(k):
        te = m[fold == f]
        tr = m[fold != f]
        g = GradientBoostingRegressor(random_state=0).fit(tr[F], tr.DMS_score)
        p = g.predict(te[F])
        rs.append(spearmanr(p, te.DMS_score).statistic)
        aucs.append(roc_auc_score(te.DMS_score_bin, p))
        if collect:
            oof[te.index.values] = p
    return (np.array(rs), np.array(aucs), oof) if collect else (np.array(rs), np.array(aucs))


def hdr(t):
    print("\n" + "=" * 96)
    print(t)
    print("=" * 96)


def load_esm2(assay, col="esm2_650m"):
    zs = pd.concat([
        pd.read_csv(os.path.join(RES, "esm2_650m_zeroshot.csv")),
        pd.read_csv(os.path.join(RES, "esm2_650m_newproteins.csv")),
    ], ignore_index=True)
    z = zs[zs.assay == assay][["mutant", col]]
    assert not z.duplicated("mutant").any(), assay
    return z


# ---------------------------------------------------------------- 0. 文件比对
hdr("0. repository/data/ampc_modeling_table.csv  vs  <proj>/data/vim2_modeling_table.csv")
a = pd.read_csv(os.path.join(DATA_R, "ampc_modeling_table.csv"))
b = pd.read_csv(os.path.join(DATA_P, "vim2_modeling_table.csv"))
print("  ampc cols :", list(a.columns))
print("  vim2 cols :", list(b.columns))
print("  shapes    :", a.shape, b.shape)
common = sorted(set(a.columns) & set(b.columns))
if len(a) == len(b) and "mutant" in common:
    same = a[common].sort_values("mutant").reset_index(drop=True).equals(
        b[common].sort_values("mutant").reset_index(drop=True))
    print("  common-cols identical (sorted by mutant):", same)
else:
    print("  (shape/keys differ -> 需人工判断)")

# ---------------------------------------------------------------- 1. 8M 逐折补算
hdr("1. 8M modulo-5 逐折 rho 补算与归档值核对")
TABLES = {
    "TEM1_Stiffler2015": "tem1_stiffler_modeling_table.csv",
    "VIM2_Chen2020": "ampc_modeling_table.csv",
    "APH3II_Melnikov2014": "aph3ii_modeling_table.csv",
    "DHFR_Nguyen2023": "dhfr_modeling_table.csv",
}
rows, folds_rows = [], []
oof_store = {}

for assay, fn in TABLES.items():
    m = pd.read_csv(os.path.join(DATA_R, fn)).rename(
        columns={"d_act": "d_s70", "d_loop": "d_omega"})
    seq = ["esm2"] + PHY
    dual = seq + ["pLDDT", "contact_num", "d_s70", "d_omega"]
    abl = seq + ["pLDDT", "contact_num"]
    for tag, feats in [("seq", seq), ("dual_full", dual), ("dual_noanchors", abl)]:
        rs, aucs = cv(m, feats)
        rows.append(dict(assay=assay, backbone="8m", model=tag,
                         rho=float(rs.mean()), rho_std=float(rs.std()),
                         auc=float(aucs.mean())))
        for i, r in enumerate(rs):
            folds_rows.append(dict(assay=assay, backbone="8m", model=tag,
                                   fold=i, rho=float(r), auc=float(aucs[i])))
        if assay == "TEM1_Stiffler2015":
            for tag2, F2 in [("seq", seq), ("struct", ["pLDDT", "contact_num", "d_s70", "d_omega"]),
                             ("dual", dual)]:
                _, _, oof = cv(m, F2, collect=True)
                oof_store[tag2] = oof

res = pd.DataFrame(rows)
print("  %-22s %-16s %10s %10s %12s %12s" % ("assay", "model", "rho_calc", "rho_arch", "absdiff", "verdict"))
ok_all = True
for _, r in res.iterrows():
    key = (r.assay, r.backbone, r.model)
    if key not in ARCHIVE:
        continue
    ar, asd = ARCHIVE[key]
    d = abs(r.rho - ar)
    ok = d < 1e-9
    ok_all &= ok
    print("  %-22s %-16s %10.7f %10.7f %12.2e  %s"
          % (r.assay, r.model, r.rho, ar, d, "PASS" if ok else "**FAIL**"))

# 归档 CSV 里同名的行也读一遍，三方交叉
arch_csv = pd.read_csv(os.path.join(RES, "anchor_counterfactual.csv"))
for _, r in res.iterrows():
    key = (r.assay, r.backbone, r.model)
    if key not in ARCHIVE:
        continue
    sub = arch_csv[(arch_csv.assay == r.assay) & (arch_csv.backbone == r.backbone)
                   & (arch_csv.model == r.model)]
    if len(sub):
        d = abs(float(sub.rho.iloc[0]) - r.rho)
        if d >= 1e-9:
            ok_all = False
            print("  **FAIL** 三方不一致 %s %s: calc=%.10f csv=%.10f" % (key, "", r.rho, float(sub.rho.iloc[0])))
print("  核对总判:", "PASS —— 全部逐位一致" if ok_all else "FAIL")

# ---------------------------------------------------------------- 2. AAC(3)-I 8M/650M
hdr("2. AAC(3)-I 反向锚点实验复算（8M 与 650M）+ 配对 t 检验")
POCKET = [50, 51, 111, 112, 113, 117, 118, 119, 120, 121, 122, 123, 124,
          125, 127, 144, 145, 147, 160]


def cb_coords(pdb_path):
    """Cbeta coordinates, falling back to CA for glycine (mirrors 19)."""
    ca, cb = {}, {}
    for line in open(pdb_path):
        if not line.startswith("ATOM"):
            continue
        rn = int(line[22:26])
        atom = line[12:16].strip()
        if atom == "CA":
            ca[rn] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        elif atom == "CB":
            cb[rn] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return {r: cb.get(r, ca[r]) for r in ca}


coords = cb_coords(os.path.join(DATA_R, "AF-P23181-F1-model.pdb"))
pocket = np.array([coords[r] for r in POCKET])
d_coa = {r: float(np.min(np.linalg.norm(pocket - c, axis=1))) for r, c in coords.items()}

m = pd.read_csv(os.path.join(DATA_R, "aac3i_modeling_table.csv"))
m["d_coa"] = m.pos.map(d_coa)
assert m.d_coa.notna().all()
z = pd.read_csv(os.path.join(RES, "esm2_650m_zeroshot.csv"))
z = z[z.assay == "AAC3I_Dandage2018"][["mutant", "esm2_650m"]]
m650 = m.merge(z, on="mutant", how="inner")

aac_rows = []
for tag, mm in [("8m", m), ("650m", m650)]:
    fc = "esm2" if tag == "8m" else "esm2_650m"
    seq = [fc] + PHY
    dual = seq + ["pLDDT", "contact_num"]
    anch = dual + ["d_coa"]
    r_seq, _ = cv(mm, seq)
    r_dual, _ = cv(mm, dual)
    r_anch, _ = cv(mm, anch)
    p1 = ttest_rel(r_dual, r_seq).pvalue
    p2 = ttest_rel(r_anch, r_dual).pvalue
    for model, rs in [("seq", r_seq), ("dual_noanchors", r_dual),
                      ("dual_with_coa_anchor", r_anch)]:
        arc = AAC_ARCHIVE[(tag, model)]
        d = abs(np.mean(rs) - arc)
        aac_rows.append(dict(assay="AAC3I_Dandage2018", backbone=tag, model=model,
                             rho=float(np.mean(rs)), rho_std=float(np.std(rs)),
                             archived=float(arc), absdiff=float(d),
                             verdict="PASS" if d < 1e-9 else "**FAIL**"))
        for i, r in enumerate(rs):
            folds_rows.append(dict(assay="AAC3I_Dandage2018", backbone=tag,
                                   model=model, fold=i, rho=float(r), auc=np.nan))
    aac_rows.append(dict(assay="AAC3I_Dandage2018", backbone=tag, model="_p_dual_vs_seq",
                         rho=float(p1), rho_std=np.nan, archived=np.nan, absdiff=np.nan,
                         verdict="p=%.4f" % p1))
    aac_rows.append(dict(assay="AAC3I_Dandage2018", backbone=tag, model="_p_anchor_vs_noanchor",
                         rho=float(p2), rho_std=np.nan, archived=np.nan, absdiff=np.nan,
                         verdict="p=%.4f" % p2))
    print("  AAC %-5s  seq %.6f | dual %.6f (paired p=%.4f) | +d_coa %.6f (paired p=%.4f)"
          % (tag, np.mean(r_seq), np.mean(r_dual), p1, np.mean(r_anch), p2))
    print("        seq folds   :", np.round(r_seq, 4).tolist())
    print("        dual folds  :", np.round(r_dual, 4).tolist())
    print("        anchor folds:", np.round(r_anch, 4).tolist())

pd.DataFrame(aac_rows).to_csv(os.path.join(RES, "aac_anchor_experiment_verified.csv"),
                               index=False)

# ---------------------------------------------------------------- 3. 落盘
hdr("3. 落盘：逐折 rho 表 + TEM-1 8M 折外预测")
pd.DataFrame(folds_rows).to_csv(os.path.join(RES, "recomputed_folds_8m.csv"), index=False)
print("  wrote", os.path.join(RES, "recomputed_folds_8m.csv"))

m = pd.read_csv(os.path.join(DATA_R, "tem1_stiffler_modeling_table.csv"))
fold = (m.pos % 5).values
np.savez_compressed(
    os.path.join(RES, "oof_predictions_tem1_8m.npz"),
    oof_seq=oof_store["seq"], oof_struct=oof_store["struct"], oof_dual=oof_store["dual"],
    oof_true=m.DMS_score.values, oof_bin=m.DMS_score_bin.values,
    mutant=m.mutant.values, pos=m.pos.values, wt=m.wt.values, mut=m.mut.values,
    fold=fold,
)
print("  wrote oof_predictions_tem1_8m.npz")

# ---------------------------------------------------------------- 4. 核对门槛
dual_oof = oof_store["dual"]
r_all = spearmanr(dual_oof, m.DMS_score.values).statistic
per_fold = []
for f in range(5):
    sel = fold == f
    per_fold.append(spearmanr(dual_oof[sel], m.DMS_score.values[sel]).statistic)
per_fold = np.array(per_fold)
print("  折外 pooled rho            = %.10f" % r_all)
print("  折外 5 折 rho mean         = %.10f" % per_fold.mean())
print("  折外 5 折 rho std (ddof=0) = %.10f" % per_fold.std())
print("  折外 pooled AUC            = %.10f"
      % roc_auc_score(m.DMS_score_bin.values, dual_oof))
ref_folds = [r["rho"] for r in folds_rows
             if r["assay"] == "TEM1_Stiffler2015" and r["backbone"] == "8m"
             and r["model"] == "dual_full"]
ref_mean = float(np.mean(ref_folds)) if ref_folds else float("nan")
ref_std = float(np.std(ref_folds)) if ref_folds else float("nan")
print("  归档口径（offset anchors）参考     = 0.7037232728")
print("  本次规范口径（corrected anchors） = %.10f" % ref_mean)
gate = (np.isfinite(ref_mean)
        and abs(per_fold.mean() - ref_mean) < 1e-9
        and abs(per_fold.std() - ref_std) < 1e-9)
print("  门槛:", "PASS —— 折外预测与逐折重算自洽，可出图"
      if gate else "**FAIL —— 折划分与现有实现不一致，不得出图**")

# 每个位置的预测/实验均值，供 Fig 1(d) 使用
pos_df = pd.DataFrame({"pos": m.pos.values, "wt": m.wt.values,
                        "pred": dual_oof, "true": m.DMS_score.values})
agg = pos_df.groupby("pos").agg(pred_mean=("pred", "mean"), true_mean=("true", "mean"),
                                 n_sub=("pred", "size")).reset_index()
agg.to_csv(os.path.join(RES, "tem1_per_position_pred_vs_true.csv"), index=False)
print("  wrote tem1_per_position_pred_vs_true.csv  (rows=%d)" % len(agg))
