#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
23_build_delta_rho_table.py — 统一构建 Δρ 全表（5 蛋白 × 3 backbone = 15 组合）

每个组合给出：
    delta      = ρ(dual-modal) − ρ(sequence-only)   （逐折配对差的均值）
    sd_delta   = 逐折配对差的标准差（ddof=0，与归档 rho_std 同口径）
    p          = 双侧配对 t 检验（scipy.stats.ttest_rel，5 折）
    n          = 该 assay 的突变数

特征组合（与 02_/16_/19_ 逐字一致）：
    seq   = [esm2*] + [blosum62, d_hydro, d_vol, d_chg]
    dual  = seq + [pLDDT, contact_num] (+ d_s70, d_omega 有锚点者)
    *8M/650M/3B 换的是 ESM-2 打分列：esm2 / esm2_650m / esm2_3b

同时把 8M/650M/3B 的逐折 rho 落盘，并交叉核对已归档值。
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel
from sklearn.ensemble import GradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(REPO, "data")
RES = os.path.join(REPO, "results")

PHY = ["blosum62", "d_hydro", "d_vol", "d_chg"]
PROTEIN_ORDER = ["TEM1_Stiffler2015", "VIM2_Chen2020", "AAC3I_Dandage2018",
                  "APH3II_Melnikov2014", "DHFR_Nguyen2023"]
TABLES = {
    "TEM1_Stiffler2015": "tem1_stiffler_modeling_table.csv",
    "VIM2_Chen2020": "ampc_modeling_table.csv",
    "AAC3I_Dandage2018": "aac3i_modeling_table.csv",
    "APH3II_Melnikov2014": "aph3ii_modeling_table.csv",
    "DHFR_Nguyen2023": "dhfr_modeling_table.csv",
}
HAS_ANCHOR = {"TEM1_Stiffler2015": True, "VIM2_Chen2020": True,
               "AAC3I_Dandage2018": False, "APH3II_Melnikov2014": True,
               "DHFR_Nguyen2023": True}
NS = {"TEM1_Stiffler2015": 4996, "VIM2_Chen2020": 5004, "AAC3I_Dandage2018": 1793,
      "APH3II_Melnikov2014": 4960, "DHFR_Nguyen2023": 2916}
SCORE_FILES = {"8m": ("esm2", ["esm2_650m_zeroshot.csv", "esm2_650m_newproteins.csv"]),
               "650m": ("esm2_650m", ["esm2_650m_zeroshot.csv", "esm2_650m_newproteins.csv"]),
               "3b": ("esm2_3b", ["esm2_3b_zeroshot.csv", "esm2_3b_newproteins.csv"])}


def cv_folds(m, F, k=5):
    """返回逐折 (rho, auc)。协议与 02_/16_/19_ 一致。"""
    rs, aucs = [], []
    fold = (m.pos % k).values
    for f in range(k):
        te = m[fold == f]
        tr = m[fold != f]
        g = GradientBoostingRegressor(random_state=0).fit(tr[F], tr.DMS_score)
        p = g.predict(te[F])
        rs.append(spearmanr(p, te.DMS_score).statistic)
        from sklearn.metrics import roc_auc_score
        aucs.append(roc_auc_score(te.DMS_score_bin, p))
    return np.array(rs), np.array(aucs)


def main():
    # --- 读 ESM-2 打分（8M 的 esm2 分已在建模表内，无需外部合并）---
    scores = {}
    for tag, (col, files) in SCORE_FILES.items():
        if tag == "8m":
            continue
        z = pd.concat([pd.read_csv(os.path.join(RES, f)) for f in files], ignore_index=True)
        scores[tag] = z[["assay", "mutant", col]]

    rows, fold_rows = [], []
    for assay in PROTEIN_ORDER:
        base = pd.read_csv(os.path.join(DATA, TABLES[assay])).rename(
            columns={"d_act": "d_s70", "d_loop": "d_omega"})
        for tag in ("8m", "650m", "3b"):
            col, _ = SCORE_FILES[tag]
            m = base
            if tag != "8m":
                z = scores[tag]
                z = z[z.assay == assay][["mutant", col]]
                assert not z.duplicated("mutant").any(), (assay, tag)
                m = base.merge(z, on="mutant", how="inner")
            seq = [col] + PHY
            struct = ["pLDDT", "contact_num"]
            dual = list(seq) + list(struct)
            if HAS_ANCHOR[assay]:
                struct = struct + ["d_s70", "d_omega"]
                dual = dual + ["d_s70", "d_omega"]
            r_seq, a_seq = cv_folds(m, seq)
            r_struct, a_struct = cv_folds(m, struct)
            r_dual, a_dual = cv_folds(m, dual)
            d = r_dual - r_seq
            p = float(ttest_rel(r_dual, r_seq).pvalue)
            rows.append(dict(assay=assay, backbone=tag,
                             rho_seq=float(r_seq.mean()),
                             rho_struct=float(r_struct.mean()),
                             rho_dual=float(r_dual.mean()),
                             delta=float(d.mean()), sd_delta=float(d.std()),
                             p=p, n=NS[assay],
                             auc_seq=float(a_seq.mean()),
                             auc_struct=float(a_struct.mean()),
                             auc_dual=float(a_dual.mean())))
            for i in range(5):
                fold_rows.append(dict(assay=assay, backbone=tag, fold=i,
                                      rho_seq=float(r_seq[i]),
                                      rho_struct=float(r_struct[i]),
                                      rho_dual=float(r_dual[i]),
                                      delta=float(d[i])))

    df = pd.DataFrame(rows)
    df["label"] = df.assay.map({
        "TEM1_Stiffler2015": "TEM-1", "VIM2_Chen2020": "VIM-2",
        "AAC3I_Dandage2018": "AAC(3)-I", "APH3II_Melnikov2014": "APH(3′)-IIa",
        "DHFR_Nguyen2023": "DHFR"})
    df["cls"] = df.assay.map({
        "TEM1_Stiffler2015": "beta", "VIM2_Chen2020": "beta",
        "AAC3I_Dandage2018": "amino", "APH3II_Melnikov2014": "amino",
        "DHFR_Nguyen2023": "folate"})

    # ---------------- 交叉核对已归档值 ----------------
    print("=" * 104)
    print("交叉核对：本次复算 vs 已归档 CSV")
    print("=" * 104)
    ok = True
    arch = []
    for fn, bk in (("expanded_proteins_evaluation.csv", "8m"),
                   ("expanded_proteins_650m_evaluation.csv", "650m"),
                   ("expanded_proteins_3b_evaluation.csv", "3b"),
                   ("esm2_650m_evaluation.csv", "650m"),
                   ("esm2_3b_evaluation.csv", "3b")):
        p = os.path.join(RES, fn)
        if not os.path.exists(p):
            continue
        a = pd.read_csv(p)
        if "protocol" in a.columns:
            a = a[a.protocol == "cv5_modulo"]
        else:
            a = a[a.model.astype(str).str.startswith("gbm_")]
        for _, rr in a.iterrows():
            if "gbm_" not in str(rr.model):
                continue
            tag = rr.model.split("_")[1]
            if tag != bk:
                continue
            kind = rr.model.split("_", 2)[2]
            kind = {"struct": "struct", "struct_only": "struct"}.get(kind, kind)
            if kind not in ("seq", "struct", "dual"):
                continue
            hit = df[(df.assay == rr.assay) & (df.backbone == bk)]
            if not len(hit):
                continue
            colname = {"seq": "rho_seq", "struct": "rho_struct", "dual": "rho_dual"}[kind]
            mine = float(hit[colname].iloc[0])
            arch.append((fn, rr.assay, bk, kind, mine, float(rr.rho)))
    seen = set()
    for fn, assay, bk, kind, mine, stored in arch:
        key = (assay, bk, kind)
        if key in seen:
            continue
        seen.add(key)
        d = abs(mine - stored)
        good = d < 1e-9
        ok &= good
        print("  %-34s %-22s %-5s %-5s  calc=%.10f  arch=%.10f  %s"
              % (fn[:34], assay, bk, kind, mine, stored, "PASS" if good else "**FAIL**"))
    print("  核对总判:", "PASS" if ok else "**FAIL**")

    # 与 paired_tests.csv 的 p 值交叉核对（8M/650M）
    pt = pd.read_csv(os.path.join(RES, "paired_tests.csv"))
    pt = pt[pt.contrast == "dual_vs_seq"]
    print("-" * 104)
    print("与 paired_tests.csv（dual_vs_seq）交叉核对：")
    for _, rr in pt.iterrows():
        hit = df[(df.assay == rr.assay) & (df.backbone == rr.backbone)]
        if not len(hit):
            continue
        d = float(hit.delta.iloc[0])
        dp = abs(d - float(rr.mean_diff))
        d_p = abs(float(hit.p.iloc[0]) - float(rr.p))
        good = dp < 1e-8 and d_p < 1e-8
        ok &= good
        print("  %-22s %-5s  delta calc=%.8f arch=%.8f (d=%.1e) | p calc=%.3e arch=%.3e (d=%.1e)  %s"
              % (rr.assay, rr.backbone, d, rr.mean_diff, dp,
                 float(hit.p.iloc[0]), rr.p, d_p, "PASS" if good else "**FAIL**"))
    print("  总判:", "PASS" if ok else "**FAIL**")

    # ---------------- 落盘 ----------------
    order = {"beta": 0, "amino": 1, "folate": 2}
    df["_o"] = df.cls.map(order)
    df = df.sort_values(["_o", "assay", "backbone"]).drop(columns="_o").reset_index(drop=True)
    df.to_csv(os.path.join(RES, "delta_rho_table.csv"), index=False)
    pd.DataFrame(fold_rows).to_csv(os.path.join(RES, "recomputed_folds_all_backbones.csv"),
                                   index=False)
    print("\nwrote delta_rho_table.csv (%d rows) & recomputed_folds_all_backbones.csv"
          % len(df))
    print(df[["label", "backbone", "n", "rho_seq", "rho_dual", "delta", "sd_delta", "p"]]
          .to_string(index=False, float_format=lambda v: "%.6f" % v))

    # ---------------- 关键结论复算 ----------------
    print("\n" + "=" * 104)
    print("关键结论复算")
    print("=" * 104)
    big = df[df.n >= 4000]
    small = df[df.n < 3000]
    print("  n>=4000 的 %d 个组合：Δρ 全为正 = %s；全部 p<0.05 = %s"
          % (len(big), bool((big.delta > 0).all()), bool((big.p < 0.05).all())))
    print("  n<3000  的 %d 个组合：Δρ 全为正 = %s；全部 p<0.05 = %s"
          % (len(small), bool((small.delta > 0).all()), bool((small.p < 0.05).all())))
    # assay 层面的相关性（避免伪重复：同一 assay 的两个 backbone 不独立）
    agg = df.groupby("assay").agg(delta=("delta", "mean"), n=("n", "first")).reset_index()
    from scipy.stats import spearmanr as sp
    r, p = sp(agg.n.values, agg.delta.values)
    print("  assay 层面（n=5）Spearman(log n, Δρ) = %.3f, P = %.3f  ← 样本量太小，仅作趋势描述"
          % (r, p))
    # 每个 backbone 内部是否一致
    for bk in ("8m", "650m", "3b"):
        s = df[df.backbone == bk]
        print("  backbone %-5s：Δρ>0 的组合 %d/%d，正的 assay %s"
              % (bk, int((s.delta > 0).sum()), len(s),
                 sorted(s[s.delta > 0].label.tolist())))


if __name__ == "__main__":
    main()
