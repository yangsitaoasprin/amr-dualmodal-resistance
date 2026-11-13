#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""46_figs_s1_s6.py — 补充图 S1–S6（每张独立导出 + 九道门禁）

S1  9 特征分布矩阵（TEM-1，n = 4,996）
S2  4 蛋白 × 4 结构特征对的共线性矩阵
S3  ESM-2 650M vs 3B 逐突变零样本分数（5 个主 assay）
S4  折外预测的校准曲线 + 决策曲线分析（post-hoc，同一数据）
S5  GNN 诊断实验全量（3 seeds 曲线 / GAT ramp / MLP 消融 / 15 runs）
S6  3B backbone 五蛋白完整评估

全部数据来自 repository/results/ 与 repository/data/ 归档文件。
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, to_hex
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as F
from figstyle import palette

S = F.setup()
ZERO, SEQ, STR, DUAL, ABL = (F.MODAL[k] for k in ("zero", "seq", "struct", "dual", "ablate"))
INK, GREY = "#1A1A1A", "#8A8A8A"
COLS = [ZERO, SEQ, STR, DUAL, ABL]
PI_LABEL = {"esm2": "ESM-2 score", "blosum62": "BLOSUM62", "d_hydro": "Δ hydrophobicity",
            "d_vol": "Δ volume", "d_chg": "Δ charge", "pLDDT": "pLDDT",
            "contact_num": "Cβ contact no.", "d_s70": "dist. to anchor",
            "d_omega": "dist. to loop"}
SEQ_FEATS = ["esm2", "blosum62", "d_hydro", "d_vol", "d_chg"]
STR_FEATS = ["pLDDT", "contact_num", "d_s70", "d_omega"]
FEATS = SEQ_FEATS + STR_FEATS
ASSAYS = ["TEM1_Stiffler2015", "VIM2_Chen2020", "AAC3I_Dandage2018",
          "APH3II_Melnikov2014", "DHFR_Nguyen2023"]


def heat_rect(ax, M, cmap, vmin, vmax):
    """用矢量矩形画热图，避免 imshow 把 PDF 栅格化。"""
    norm = Normalize(vmin, vmax)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            c = "#FFFFFF" if np.isnan(v) else to_hex(cmap(norm(v)))
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   facecolor=c, edgecolor="#FFFFFF",
                                   linewidth=0.6, zorder=1))
    ax.set_xlim(-0.5, M.shape[1] - 0.5)
    ax.set_ylim(M.shape[0] - 0.5, -0.5)


def finish(fig, stem, colors=None, allow_raster=False):
    out = os.path.join(F.FIGS, stem)
    files, res = F.save(fig, out, dpi=600, allow_raster=allow_raster,
                        colors=colors or COLS)
    plt.close(fig)
    print("  %-44s %d/%d PASS" % (stem, res["n_pass"], res["n_total"]))
    for name, ok, r in res["rows"]:
        if not ok:
            print("     FAIL:", name, r)
    return res["n_pass"] == res["n_total"]


# ================================================================ S1
def fig_s1():
    d = F.load("tem1_stiffler_modeling_table.csv", folder=F.DATA)
    fig = F.newfig("double", 132)
    gs = fig.add_gridspec(3, 3, left=0.075, right=0.985, top=0.930, bottom=0.085,
                          wspace=0.42, hspace=0.62)
    for i, f in enumerate(FEATS):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        v = pd.to_numeric(d[f], errors="coerce").dropna().values
        col = SEQ if f in SEQ_FEATS else STR
        ax.hist(v, bins=42, color=palette.tint(col, 0.30), edgecolor=col,
                linewidth=0.45, zorder=2)
        ax.set_title(PI_LABEL[f], fontsize=S["note"], loc="left",
                     x=0.12 if i == 0 else 0.0, pad=2)
        ax.tick_params(labelsize=S["note"])
        ax.set_ylabel("variants", fontsize=S["note"])
        F.tidy(ax)
        if i == 0:
            F.panel(ax, "a")
    fig.text(0.075, 0.965, "TEM-1 Stiffler 2015, n = 4,996 variants; "
             "blue = sequence side, green = structure side",
             fontsize=S["note"], color=GREY, ha="left", va="bottom")
    return fig, "figS1_feature_distributions_300dpi"


# ================================================================ S2
def fig_s2():
    c = F.load("feature_collinearity.csv")
    pairs = ["contact_num|d_s70", "contact_num|d_omega", "d_s70|d_omega",
             "pLDDT|contact_num"]
    plab = ["burial × anchor dist.", "burial × loop dist.",
            "anchor × loop dist.", "pLDDT × burial"]
    c["pair"] = c.feat_x + "|" + c.feat_y
    M = np.full((len(ASSAYS) - 1, len(pairs)), np.nan)
    have = [a for a in ASSAYS if a in set(c.assay)]
    for i, a in enumerate(have):
        for j, p in enumerate(pairs):
            s = c[(c.assay == a) & (c.pair == p)]
            if len(s):
                M[i, j] = float(s.spearman.iloc[0])
    fig = F.newfig("double", 92)
    ax = fig.add_axes([0.155, 0.20, 0.60, 0.66])
    cmap = plt.get_cmap("RdBu_r")
    heat_rect(ax, M, cmap, -1, 1)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, "%+.2f" % M[i, j], ha="center", va="center",
                        fontsize=S["label"], color=INK
                        if abs(M[i, j]) < 0.62 else "#FFFFFF", zorder=3)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(plab, fontsize=S["note"], rotation=18, ha="right")
    ax.set_yticks(range(len(have)))
    ax.set_yticklabels([F.PROTEINS[a]["short"] for a in have], fontsize=S["note"])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color("#C8C8C8")
    ax.tick_params(labelsize=S["note"], length=0)
    ax.set_title("Spearman ρ between structural features (unique positions)",
                 fontsize=S["label"], loc="left", pad=8)
    fig.text(0.155, 0.075, "Colour: blue = negative ρ, red = positive ρ (RdBu_r, −1 to +1). "
             "Burial is anti-correlated with anchor distances (ρ = −0.29 to −0.66); "
             "n = 1,793–5,004 variants per assay.",
             fontsize=S["note"], color=GREY, ha="left")
    F.panel(ax, "a")
    return fig, "figS2_collinearity_300dpi"


# ================================================================ S3
def fig_s3():
    a650 = pd.concat([F.load("esm2_650m_zeroshot.csv"),
                      F.load("esm2_650m_newproteins.csv")], ignore_index=True)
    a3b = pd.concat([F.load("esm2_3b_zeroshot.csv"),
                     F.load("esm2_3b_newproteins.csv")], ignore_index=True)
    m = a650[["assay", "mutant", "esm2_650m"]].merge(
        a3b[["assay", "mutant", "esm2_3b"]], on=["assay", "mutant"], how="inner")
    fig = F.newfig("double", 78)
    gs = fig.add_gridspec(1, 5, left=0.055, right=0.990, top=0.830, bottom=0.175,
                          wspace=0.42)
    for k, a in enumerate(ASSAYS):
        ax = fig.add_subplot(gs[0, k])
        s = m[m.assay == a]
        ax.scatter(s.esm2_650m, s.esm2_3b, s=0.7, color=GREY, linewidths=0, zorder=2)
        lo = float(min(s.esm2_650m.min(), s.esm2_3b.min()))
        hi = float(max(s.esm2_650m.max(), s.esm2_3b.max()))
        ax.plot([lo, hi], [lo, hi], color="#B8B8B8", lw=0.6,
                ls=(0, (3, 2)), zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        r = spearmanr(s.esm2_650m, s.esm2_3b)
        ax.set_title("%s\nρ = %.3f" % (F.PROTEINS[a]["short"], r.statistic),
                     fontsize=S["note"], loc="left", x=0.12, pad=2)
        ax.set_xlabel("ESM-2 650M", fontsize=S["note"])
        if k == 0:
            ax.set_ylabel("ESM-2 3B", fontsize=S["note"])
        F.panel(ax, chr(97 + k))
        ax.tick_params(labelsize=S["note"])
        F.tidy(ax)
    n_shown = int(sum((m.assay == a).sum() for a in ASSAYS))
    fig.text(0.055, 0.925, "Per-variant masked-marginal zero-shot scores, "
             "n = %s variants across the five assays shown; dashed line = identity"
             % format(n_shown, ","), fontsize=S["note"], color=GREY, ha="left")
    return fig, "figS3_zeroshot_scatter_300dpi"


# ================================================================ S4
def fig_s4():
    d = np.load(os.path.join(F.RES, "oof_predictions_tem1_8m.npz"), allow_pickle=True)
    y = (1 - d["oof_bin"]).astype(int)          # 1 = deleterious
    score = (-d["oof_dual"]).reshape(-1, 1)     # larger = more deleterious
    lr = LogisticRegression().fit(score, y)
    p = lr.predict_proba(score)[:, 1]
    brier = float(brier_score_loss(y, p))
    n = len(y)

    fig = F.newfig("double", 88)
    gs = fig.add_gridspec(1, 2, left=0.075, right=0.975, top=0.815, bottom=0.175,
                          wspace=0.34)
    # (a) 校准曲线
    ax = fig.add_subplot(gs[0, 0])
    q = np.quantile(p, np.linspace(0, 1, 11))
    q[0] -= 1e-9
    idx = np.clip(np.digitize(p, q[1:-1]), 0, 9)
    px = np.array([p[idx == k].mean() for k in range(10)])
    py = np.array([y[idx == k].mean() for k in range(10)])
    ax.plot([0, 1], [0, 1], color="#B8B8B8", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.plot(px, py, marker="o", ms=3.6, color=DUAL, lw=1.2, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted probability (decile)", fontsize=S["label"])
    ax.set_ylabel("observed deleterious fraction", fontsize=S["label"])
    ax.set_title("calibration, Brier = %.4f" % brier, fontsize=S["note"],
                 loc="left", x=0.12, pad=3)
    F.tidy(ax)
    F.panel(ax, "a")
    # (b) 决策曲线
    ax = fig.add_subplot(gs[0, 1])
    th = np.linspace(0.05, 0.80, 76)
    prev = float(y.mean())
    nb_model, nb_all = [], []
    for t in th:
        flag = p >= t
        tp = float(np.sum(flag & (y == 1))) / n
        fp = float(np.sum(flag & (y == 0))) / n
        nb_model.append(tp - fp * t / (1 - t))
        nb_all.append(prev - (1 - prev) * t / (1 - t))
    ax.plot(th, nb_model, color=DUAL, lw=1.5, zorder=3, label="dual-modal model")
    ax.plot(th, nb_all, color=ZERO, lw=1.0, ls=(0, (4, 2)), zorder=2,
            label="treat all")
    ax.axhline(0, color=INK, lw=0.9, zorder=2, label="treat none")
    ax.set_xlim(0.05, 0.80)
    ax.set_ylim(min(-0.10, min(nb_all) - 0.02), prev + 0.06)
    ax.set_xlabel("threshold probability", fontsize=S["label"])
    ax.set_ylabel("net benefit", fontsize=S["label"])
    ax.set_title("decision curve", fontsize=S["note"], loc="left",
                 x=0.12, pad=3)
    F.tidy(ax)
    ax.legend(loc="lower left", fontsize=S["note"], frameon=False,
              handlelength=1.8, labelspacing=0.25)
    F.panel(ax, "b")
    fig.text(0.075, 0.920,
             "TEM-1 out-of-fold predictions, n = %s variants. Probabilities are a "
             "post-hoc logistic map of the regression score fitted on the same "
             "out-of-fold predictions (diagnostic only, not an independent calibration)."
             % format(n, ","), fontsize=S["note"], color=GREY, ha="left")
    return fig, "figS4_calibration_dca_300dpi"


# ================================================================ S5
def fig_s5():
    ramp = F.load("gcn_extended_ramp.csv")
    gat = F.load("gat_extended_ramp.csv")
    mlp = F.load("mlp_ablation.csv")
    ng = F.load("naive_gat_5fold.csv")
    ng = ng[pd.to_numeric(ng["seed"], errors="coerce").notna()].copy()
    ng["fold_i"] = ng.fold.astype(int)
    ng["seed_i"] = ng.seed.astype(int)

    fig = F.newfig("double", 158)
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.975, top=0.915, bottom=0.085,
                          wspace=0.32, hspace=0.58)
    # (a) 3 seeds 曲线
    ax = fig.add_subplot(gs[0, 0])
    for sd, sub in ramp.groupby("seed"):
        ax.plot(sub.epoch, sub.test_rho, color=palette.tint(DUAL, 0.55),
                lw=0.8, zorder=2)
        ax.plot(sub.epoch, sub.train_rho, color=palette.tint(SEQ, 0.55),
                lw=0.8, zorder=2)
    g = ramp.groupby("epoch").agg(tr=("train_rho", "mean"), te=("test_rho", "mean"))
    ax.plot(g.index, g.tr, color=SEQ, lw=1.6, zorder=4, label="train (mean)")
    ax.plot(g.index, g.te, color=DUAL, lw=1.6, zorder=4, label="test (mean)")
    ax.set_xlabel("training epoch", fontsize=S["label"])
    ax.set_ylabel("Spearman ρ", fontsize=S["label"])
    ax.set_title("2-layer GCN, three seeds", fontsize=S["note"], loc="left",
                 x=0.12, pad=3)
    F.tidy(ax)
    ax.legend(loc="lower right", fontsize=S["note"], frameon=False, labelspacing=0.25)
    F.panel(ax, "a")
    # (b) GAT ramp
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(gat.epoch, gat.rho, marker="o", ms=2.6, color=ZERO, lw=1.2, zorder=3)
    ax.axhline(0.59, color="#C8C8C8", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.set_xlabel("training epoch", fontsize=S["label"])
    ax.set_ylabel("Spearman ρ", fontsize=S["label"])
    ax.set_title("naive GATv2 budget ramp (fold 0)", fontsize=S["note"],
                 loc="left", x=0.12, pad=3)
    F.tidy(ax)
    F.panel(ax, "b")
    # (c) MLP ablation
    ax = fig.add_subplot(gs[1, 0])
    xs = np.arange(3)
    ax.bar(xs - 0.17, mlp.train_rho, width=0.32, color=palette.tint(SEQ, 0.35),
           edgecolor=SEQ, linewidth=0.7, zorder=3, label="train")
    ax.bar(xs + 0.17, mlp.test_rho, width=0.32, color=palette.tint(DUAL, 0.35),
           edgecolor=DUAL, linewidth=0.7, zorder=3, label="test")
    ax.set_xticks(xs)
    ax.set_xticklabels(["seed 0", "seed 1", "seed 2"], fontsize=S["note"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Spearman ρ", fontsize=S["label"])
    ax.set_title("MLP ablation, no message passing (25 ep)",
                 fontsize=S["note"], loc="left", x=0.12, pad=3)
    F.tidy(ax)
    ax.legend(loc="upper right", fontsize=S["note"], frameon=False, labelspacing=0.25)
    F.panel(ax, "c")
    # (d) naive GAT 15 runs 热图
    ax = fig.add_subplot(gs[1, 1])
    H = np.full((3, 5), np.nan)
    for _, r in ng.iterrows():
        H[int(r.seed_i), int(r.fold_i)] = float(r.rho)
    heat_rect(ax, H, plt.get_cmap("cividis"), -0.1, 0.6)
    for i in range(3):
        for j in range(5):
            ax.text(j, i, "%.2f" % H[i, j], ha="center", va="center",
                    fontsize=S["note"], color=INK if H[i, j] > 0.32 else "#FFFFFF",
                    zorder=3)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["fold %d" % k for k in range(5)], fontsize=S["note"])
    ax.set_yticks(range(3))
    ax.set_yticklabels(["seed 0", "seed 1", "seed 2"], fontsize=S["note"])
    ax.set_title("naive GATv2, 15 runs (mean %.3f, SD %.3f)"
                 % (ng.rho.mean(), ng.rho.std(ddof=0)),
                 fontsize=S["note"], loc="left", x=0.12, pad=3)
    ax.tick_params(length=0, labelsize=S["note"])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color("#C8C8C8")
    F.panel(ax, "d")
    fig.text(0.075, 0.955, "All panels: TEM-1 fold 0 unless noted; "
             "n = 4,996 variants; three fixed seeds (0/1/2).",
             fontsize=S["note"], color=GREY, ha="left")
    return fig, "figS5_gnn_full_300dpi"


# ================================================================ S6
def fig_s6():
    dr = F.load("delta_rho_table.csv")
    d3 = dr[dr.backbone == "3b"].set_index("assay").loc[ASSAYS]
    ev = F.load("esm2_3b_evaluation.csv")
    zs = {}
    for a in ASSAYS:
        s = ev[(ev.assay == a) & (ev.model.astype(str).str.contains("zeroshot"))]
        if len(s):
            zs[a] = float(s.rho.iloc[0])
    # 补充 3B 新蛋白的零样本值
    e3 = F.load("expanded_proteins_3b_evaluation.csv")
    for a in ("APH3II_Melnikov2014", "DHFR_Nguyen2023"):
        s = e3[(e3.assay == a) & (e3.model == "zeroshot")]
        if len(s):
            zs[a] = float(s.rho.iloc[0])

    fig = F.newfig("double", 96)
    ax = fig.add_axes([0.085, 0.22, 0.88, 0.66])
    x = np.arange(len(ASSAYS))
    for k, (col, marker, lab, key) in enumerate([
            (ZERO, "o", "zero-shot", "zs"),
            (SEQ, "s", "sequence only", "seq"),
            (STR, "^", "structure only", "struct"),
            (DUAL, "D", "dual-modal", "dual")]):
        vals = [zs.get(a, np.nan) if key == "zs" else float(d3.loc[a, "rho_" + key])
                for a in ASSAYS]
        ax.plot(x, vals, color=col, lw=0.9, ls=(0, (3, 2)) if key == "zs" else "-",
                marker=marker, ms=4.2, zorder=3, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([F.PROTEINS[a]["short"] for a in ASSAYS], fontsize=S["note"])
    ax.set_ylabel("Spearman ρ", fontsize=S["label"])
    ax.set_ylim(0.10, 0.85)
    ax.set_title("3B backbone (esm2_t36_3B_UR50D), five DMS assays",
                 fontsize=S["label"], loc="left", pad=6)
    F.tidy(ax)
    ax.legend(loc="lower left", ncol=4, fontsize=S["note"], frameon=False,
              handlelength=1.6, columnspacing=1.1, labelspacing=0.25)
    fig.text(0.085, 0.095,
             "n = 1,793–5,004 variants per assay; error bars omitted for clarity "
             "(five-fold SD is given in Table S7); Δρ = dual − sequence.",
             fontsize=S["note"], color=GREY, ha="left")
    F.panel(ax, "a")
    return fig, "figS6_3b_panel_300dpi"


if __name__ == "__main__":
    makers = [fig_s1, fig_s2, fig_s3, fig_s4, fig_s5, fig_s6]
    ok = []
    for fn in makers:
        try:
            fig, stem = fn()
            ok.append((stem, finish(fig, stem)))
        except Exception as e:
            import traceback
            print("  %s FAILED: %r" % (fn.__name__, e))
            traceback.print_exc()
            ok.append((fn.__name__, False))
    print("\n=== summary ===")
    for stem, good in ok:
        print("  %-46s %s" % (stem, "PASS" if good else "FAIL"))
