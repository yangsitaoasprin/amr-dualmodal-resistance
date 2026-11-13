#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
41_fig1_dualmodal.py — Figure 1
！「双模态监督在语言模型之上提供了额外的结构信号」

(a) 折外预测 vs 实验 DMS 分数散点（TEM-1，8M backbone，n = 4,996）+ ROC 插图
(b) 模型 × backbone 对照图：4 个模型 × 3 个 backbone 的 Spearman ρ（±1 SD，5 折）
(c) 置换重要性发散条形（8M backbone，9 个特征；序列侧蓝 / 结构侧绿，负值斜纹）
(d) 逐残基预测有害性景观（263 个有测序位点；灰阶=埋藏度，形状=到催化 Ser68 的距离）
(e) PyMOL 结构面板：TEM-1 AlphaFold 模型，按「预测有害性」与「Cβ 接触数」各渲一张

数据全部来自 repository/results/ 与 repository/data/ 下已核对过的 CSV，零新增计算。
(e) 的位图由 47_render_pymol_tem1.py 在 pymol4 环境预渲染。
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as F
from figstyle import palette, style

S = F.setup()
ZERO = F.MODAL["zero"]
SEQ = F.MODAL["seq"]
STR = F.MODAL["struct"]
DUAL = F.MODAL["dual"]
ABL = F.MODAL["ablate"]
INK = "#1A1A1A"
GREY = "#8A8A8A"
CMAP = palette.seq_map("cividis")
OMEGA_C = "#CC79A7"
PYMOL = os.path.join(F.RES, "pymol")

# ---------------------------------------------------------------- 数据
d = np.load(os.path.join(F.RES, "oof_predictions_tem1_8m.npz"), allow_pickle=True)
oof_true, oof_dual, oof_bin = d["oof_true"], d["oof_dual"], d["oof_bin"]
rho_oof = float(spearmanr(oof_true, oof_dual).statistic)
fpr, tpr, _ = roc_curve(oof_bin, oof_dual)
auc = roc_auc_score(oof_bin, oof_dual)

rf8 = F.load("recomputed_folds_8m.csv")
TEM = "TEM1_Stiffler2015"
ev650 = F.load("esm2_650m_evaluation.csv")
ev3b = F.load("esm2_3b_evaluation.csv")


def pick(df, assay, model, protocol="cv5_modulo"):
    s = df[(df.assay == assay) & (df.model == model)]
    if protocol and "protocol" in df.columns:
        s = s[s.protocol == protocol]
    return s.iloc[0] if len(s) else None


def mean8(model):
    return float(rf8[(rf8.assay == TEM) & (rf8.model == model) & (rf8.backbone == "8m")].rho.mean())


def sd8(model):
    sub = rf8[(rf8.assay == TEM) & (rf8.model == model) & (rf8.backbone == "8m")]
    return float(sub.rho.std(ddof=0))


STRUCT_RHO = float(pick(ev650, TEM, "gbm_650m_struct_only").rho)      # 与 backbone 无关
STRUCT_SD = float(pick(ev650, TEM, "gbm_650m_struct_only").rho_std)
ZS8 = float(F.load("multiprotein_results.csv").iloc[0].zeroshot)      # TEM-1 8M 零样本
ZS650 = float(pick(ev650, TEM, "esm2_650m_zeroshot", "full_assay").rho)
ZS3B = float(pick(ev3b, TEM, "esm2_3b_zeroshot", "full_assay").rho)

MODELS4 = ["zero-shot", "sequence only", "structure only", "dual-modal"]
MODEL_COL = {"zero-shot": ZERO, "sequence only": SEQ,
             "structure only": STR, "dual-modal": DUAL}
BB = ["8m", "650m", "3b"]
BB_RHO = {
    "zero-shot": {"8m": (ZS8, np.nan), "650m": (ZS650, np.nan), "3b": (ZS3B, np.nan)},
    "sequence only": {"8m": (mean8("seq"), sd8("seq")),
                      "650m": (float(pick(ev650, TEM, "gbm_650m_seq").rho),
                               float(pick(ev650, TEM, "gbm_650m_seq").rho_std)),
                      "3b": (float(pick(ev3b, TEM, "gbm_3b_seq").rho),
                             float(pick(ev3b, TEM, "gbm_3b_seq").rho_std))},
    "structure only": {b: (STRUCT_RHO, STRUCT_SD) for b in BB},
    "dual-modal": {"8m": (mean8("dual_full"), sd8("dual_full")),
                   "650m": (float(pick(ev650, TEM, "gbm_650m_dual").rho),
                            float(pick(ev650, TEM, "gbm_650m_dual").rho_std)),
                   "3b": (float(pick(ev3b, TEM, "gbm_3b_dual").rho),
                          float(pick(ev3b, TEM, "gbm_3b_dual").rho_std))},
}

PI = F.load("permutation_importance.csv")
pi8 = PI[PI.backbone == "8m"].groupby("feature").importance.agg(["mean", "std"]).reset_index()
pi650 = PI[PI.backbone == "650m"].groupby("feature").importance.agg(["mean", "std"]).reset_index()
PI_LABEL = {"esm2": "ESM-2 score (8M)", "esm2_650m": "ESM-2 score (650M)",
            "blosum62": "BLOSUM62", "d_hydro": "hydrophobicity", "d_vol": "volume",
            "d_chg": "charge", "contact_num": "Cβ contact no.",
            "d_s70": "dist. to Ser70", "d_omega": "dist. to Ω-loop", "pLDDT": "pLDDT"}
PI_SIDE = {"esm2": "seq", "esm2_650m": "seq", "blosum62": "seq", "d_hydro": "seq",
           "d_vol": "seq", "d_chg": "seq", "contact_num": "str", "d_s70": "str",
           "d_omega": "str", "pLDDT": "str"}

PP = F.load("tem1_per_position_pred_vs_true.csv")
SF = F.load("tem1_structural_features.csv", folder=F.DATA)
PP = PP.merge(SF, on="pos", how="left").sort_values("pos").reset_index(drop=True)
rho_pos = float(spearmanr(PP.pred_mean, PP.true_mean).statistic)
p_pos = float(spearmanr(PP.pred_mean, PP.true_mean).pvalue)
top_pred = set(PP.sort_values("pred_mean", ascending=True).head(10).pos)
top_true = set(PP.sort_values("true_mean", ascending=True).head(10).pos)
both = sorted(top_pred & top_true)

# ---------------------------------------------------------------- 版面
fig = F.newfig("double", 228)
gs = fig.add_gridspec(3, 3, left=0.068, right=0.988, top=0.955, bottom=0.044,
                      height_ratios=[1.00, 0.92, 0.98], wspace=0.34, hspace=0.60)

# ================================================================ (a) 散点 + ROC
ax = fig.add_subplot(gs[0, 0:2])
tol = oof_bin == 1
ax.scatter(oof_true[~tol], oof_dual[~tol], s=4, c=ABL, linewidths=0, zorder=3)
ax.scatter(oof_true[tol], oof_dual[tol], s=4, c=SEQ, linewidths=0, zorder=3)
lim = [min(oof_true.min(), oof_dual.min()) - 0.15, max(oof_true.max(), oof_dual.max()) + 0.15]
ax.plot(lim, lim, color="#B8B8B8", lw=0.7, ls=(0, (4, 3)), zorder=2)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("experimental DMS score", fontsize=S["label"])
ax.set_ylabel("out-of-fold prediction", fontsize=S["label"])
F.tidy(ax)
leg = ax.legend(handles=[Line2D([], [], marker="o", color="none", markerfacecolor=SEQ,
                                markersize=3.4, label="tolerated (above WT)"),
                         Line2D([], [], marker="o", color="none", markerfacecolor=ABL,
                                markersize=3.4, label="deleterious (below WT)")],
                loc="upper left", bbox_to_anchor=(-0.005, 1.005), fontsize=S["note"],
                frameon=False, handletextpad=0.25, labelspacing=0.25, ncol=2,
                title="out-of-fold Spearman ρ = %.3f  (n = 4,996 variants)" % rho_oof)
leg.get_title().set_fontsize(S["note"])
leg.get_title().set_color(GREY)
leg._legend_box.align = "left"

axr = ax.inset_axes([0.700, 0.115, 0.290, 0.320])
axr.plot([0, 1], [0, 1], color="#B8B8B8", lw=0.7, ls=(0, (4, 3)), zorder=2)
axr.plot(fpr, tpr, color=DUAL, lw=1.4, zorder=3)
axr.set_xlim(-0.02, 1.02)
axr.set_ylim(-0.02, 1.02)
axr.set_xticks([0, 0.5, 1.0])
axr.set_yticks([0, 0.5, 1.0])
axr.set_xticklabels(["0", "0.5", "1"], fontsize=S["note"])
axr.set_yticklabels(["0", "0.5", "1"], fontsize=S["note"])
axr.set_xlabel("FPR", fontsize=S["note"], labelpad=1)
axr.set_ylabel("TPR", fontsize=S["note"], labelpad=1)
axr.tick_params(labelsize=S["note"])
F.tidy(axr)
# 白底细框把插图与主图刻度带在视觉上隔开（禁半透明，用实白）
axr.set_facecolor("white")
for _sp in axr.spines.values():
    _sp.set_visible(True)
    _sp.set_linewidth(0.5)
    _sp.set_color("#CFCFCF")
axr.text(0.97, 0.06, "AUC %.3f" % auc, transform=axr.transAxes, ha="right", va="bottom",
         fontsize=S["note"], color=INK)
F.panel(ax, "a")

# ================================================================ (b) 模型 × backbone
ax = fig.add_subplot(gs[0, 2])
ypos = {m: 3 - i for i, m in enumerate(MODELS4)}
for m in MODELS4:
    xs = [BB_RHO[m][b][0] for b in BB]
    ys = [ypos[m]] * 3
    ax.plot(xs, ys, color="#D2D2D2", lw=0.8, zorder=2, solid_capstyle="round")
    for b, x_ in zip(BB, xs):
        rho_, sd_ = BB_RHO[m][b]
        col = MODEL_COL[m]
        if np.isnan(sd_):
            ax.plot([x_], [ypos[m]], marker="|", ms=8, mew=1.8, color=col, zorder=4)
        else:
            ax.plot([x_ - sd_, x_ + sd_], [ypos[m], ypos[m]], color=col, lw=1.3, zorder=3,
                    solid_capstyle="butt")
            ax.plot([x_], [ypos[m]], marker=F.SHAPE[b], ms=4.4, color=col, mec="white",
                    mew=0.6, zorder=4)
ax.set_yticks([ypos[m] for m in MODELS4])
ax.set_yticklabels(MODELS4, fontsize=S["note"])
ax.set_ylim(-0.62, 3.62)
ax.set_xlim(0.30, 0.90)
ax.set_xlabel("Spearman ρ", fontsize=S["label"])
F.tidy(ax)
# 纯 ASCII：Arial 无 ◆/△ 字形；x 轴按 ρ 排序，backbone 顺序逐行不同，不写 "left to right"
ax.text(0.11, 1.050, "marker = backbone:  o 8M   D 650M   ^ 3B\n"
                     "| = zero-shot;  structure-only is backbone-independent",
        transform=ax.transAxes, fontsize=S["note"], color=GREY, ha="left", va="bottom",
        linespacing=1.35)
ax.text(0.0, 0.030, "TEM-1 Stiffler 2015; n = 4,996\nerror bars = ±1 SD over 5 folds",
        transform=ax.transAxes, fontsize=S["note"], color=GREY, ha="left", va="bottom",
        linespacing=1.4)
F.panel(ax, "b")

# ================================================================ (c) 置换重要性
ax = fig.add_subplot(gs[1, 0])
pi8 = pi8.sort_values("mean").reset_index(drop=True)
cols = [SEQ if PI_SIDE[f] == "seq" else STR for f in pi8.feature]
for i, r in pi8.iterrows():
    if r["mean"] < 0:
        ax.barh(i, r["mean"], color=palette.tint(cols[i], 0.62), edgecolor=cols[i],
                linewidth=0.8, hatch="///", zorder=3)
    else:
        ax.barh(i, r["mean"], color=cols[i], edgecolor="none", zorder=3)
    ax.plot([r["mean"] - r["std"], r["mean"] + r["std"]], [i, i], color=INK,
            lw=0.9, zorder=5, solid_capstyle="butt")
ax.axvline(0, color=INK, lw=0.9, zorder=4)
ax.set_yticks(range(len(pi8)))
ax.set_yticklabels([PI_LABEL[f] for f in pi8.feature], fontsize=S["note"])
ax.set_xlim(-0.06, 0.66)
ax.set_xlabel("permutation importance (Δ ρ)", fontsize=S["label"])
F.tidy(ax)
ax.legend(handles=[Line2D([], [], marker="s", color="none", markerfacecolor=SEQ,
                          markersize=3.4, label="sequence side"),
                   Line2D([], [], marker="s", color="none", markerfacecolor=STR,
                          markersize=3.4, label="structure side")],
          loc="lower right", bbox_to_anchor=(1.005, -0.005), fontsize=S["note"],
          frameon=False, handletextpad=0.25, labelspacing=0.2)
ax.text(0.11, 1.075, "ESM-2 8M backbone", transform=ax.transAxes, fontsize=S["note"],
        color=GREY, ha="left", va="bottom")
t650 = pi650.set_index("feature")
ax.text(1.0, 1.045, "ESM-2 650M (ref.)\nscore %.3f, burial %.3f\npLDDT ~0"
        % (t650.loc["esm2_650m", "mean"],
           t650.loc["contact_num", "mean"]),
        transform=ax.transAxes, fontsize=S["note"], color="#6E6E6E", ha="right",
        va="bottom", linespacing=1.35)
F.panel(ax, "c")

# ================================================================ (d) 逐残基景观
ax = fig.add_subplot(gs[1, 1:3])
OM_LO, OM_HI = 163, 178
ax.axvspan(OM_LO - 0.5, OM_HI + 0.5, color=palette.tint(DUAL, 0.84), lw=0, zorder=0)
cn = PP.contact_num.values.astype(float)
gray = np.clip(238 - (cn - cn.min()) / (cn.max() - cn.min()) * (238 - 96), 0, 255)
ds70 = PP.d_s70.values.astype(float)
edges = [0, 10, 20, 1e9]
shapes = ["o", "^", "s"]
for k in range(3):
    m = (ds70 >= edges[k]) & (ds70 < edges[k + 1])
    ax.scatter(PP.pos.values[m], PP.pred_mean.values[m], s=15, marker=shapes[k],
               facecolor=[("#%02X%02X%02X" % (int(g), int(g), int(g))) for g in gray[m]],
               edgecolor="#4E4E4E", linewidths=0.22, zorder=3)
ax.plot(PP.pos, PP.true_mean, color="#8C8C8C", lw=0.8, zorder=2)
for p in both:
    ax.scatter([p], [float(PP[PP.pos == p].pred_mean.iloc[0])], s=62, marker="o",
               facecolor="none", edgecolor=DUAL, linewidths=1.2, zorder=6)
ax.set_xlim(20, 290)
ax.set_ylim(-3.45, 0.72)
ax.set_xlabel("TEM-1 residue position", fontsize=S["label"])
ax.set_ylabel("predicted deleteriousness", fontsize=S["label"])
F.tidy(ax)
ax.text(0.0, 0.985, "per-position ρ = %.3f (P = %.1e, n = %d positions)\n"
        "%d of the 10 most deleterious positions are in the experimental top 10"
        % (rho_pos, p_pos, len(PP), len(both)),
        transform=ax.transAxes, ha="left", va="top", fontsize=S["note"], color=INK,
        linespacing=1.45)
ax.text((OM_LO + OM_HI) / 2, 0.02, "Ω-loop 162–177",
        transform=ax.get_xaxis_transform(), ha="center", va="bottom",
        fontsize=S["note"], color="#9A7414")
ax.legend(handles=[Line2D([], [], marker="o", color="none", markerfacecolor="#8C8C8C",
                          markeredgecolor="#4E4E4E", markersize=3.4,
                          label="< 10 Å to Ser70"),
                   Line2D([], [], marker="^", color="none", markerfacecolor="#8C8C8C",
                          markeredgecolor="#4E4E4E", markersize=3.6,
                          label="10–20 Å"),
                   Line2D([], [], marker="s", color="none", markerfacecolor="#8C8C8C",
                          markeredgecolor="#4E4E4E", markersize=3.2, label="> 20 Å"),
                   Line2D([], [], color="#8C8C8C", lw=0.8, label="experimental mean"),
                   Line2D([], [], marker="o", color="none", markerfacecolor="none",
                          markeredgecolor=DUAL, markeredgewidth=1.2,
                          label="top-10 overlap")],
          loc="upper right", bbox_to_anchor=(1.005, 1.005), ncol=3,
          fontsize=S["note"], frameon=False, handletextpad=0.3, labelspacing=0.25,
          columnspacing=1.1)
ax.text(0.11, 1.055, "point shade = Cβ contact number (burial: dark = buried)",
        transform=ax.transAxes, fontsize=S["note"], color=GREY, ha="left", va="bottom")
F.panel(ax, "d")

# ================================================================ (e) PyMOL 结构面板
gse = gs[2, 0:3].subgridspec(1, 5, width_ratios=[1.0, 0.085, 0.09, 1.0, 0.085],
                             wspace=0.05)
probe = np.load(os.path.join(PYMOL, "probe_positions.npz"))
CW, CH = int(probe["width"]), int(probe["height"])
imgs = [("crop_deleteriousness.png", "dmin", "dmax",
         "predicted deleteriousness (per residue)"),
        ("crop_burial.png", "bmin", "bmax", "Cβ contact number (burial)")]
for k, (fn, kmin, kmax, cblab) in enumerate(imgs):
    axi = fig.add_subplot(gse[0, k * 3])
    axc = fig.add_subplot(gse[0, k * 3 + 1])
    img = plt.imread(os.path.join(PYMOL, fn))
    axi.imshow(img, origin="upper", extent=[0, CW, CH, 0], zorder=1)
    axi.set_xticks([])
    axi.set_yticks([])
    for sp in axi.spines.values():
        sp.set_visible(False)
    axi.set_title("TEM-1 AlphaFold model, coloured by " + cblab, fontsize=S["note"],
                  color=INK, loc="left", pad=3)
    # 探针反解的像素坐标 -> matplotlib 矢量叠加（白描边保证在结构底色上可读）
    for (xy, col, lab, dx, dy) in [
            (probe["s70_px"], "#D55E00", "Ser68", 26, -26),
            (probe["omega_px"], OMEGA_C, "Ω-loop", 30, 24)]:
        axi.scatter([xy[0]], [xy[1]], s=155, facecolors="none", edgecolors=col,
                    linewidths=1.1, zorder=6)
        axi.text(xy[0] + dx, xy[1] + dy, lab, fontsize=S["note"], color=INK,
                 va="center", ha="left", zorder=7,
                 path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    vmin, vmax = float(probe[kmin]), float(probe[kmax])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(vmin, vmax), cmap=CMAP),
                      cax=axc, orientation="vertical")
    cb.ax.tick_params(labelsize=S["note"], width=0.6, length=2.0)
    cb.outline.set_linewidth(0.6)
    cb.set_label(cblab, fontsize=S["note"])
    if k == 0:
        F.panel(axi, "e")

fig.legend(handles=[Line2D([], [], marker="o", color="none", markerfacecolor="#D55E00",
                           markersize=4.0, label="catalytic Ser68 (sphere)"),
                    Line2D([], [], marker="s", color="none", markerfacecolor=OMEGA_C,
                           markersize=4.0, label="Ω-loop 162–177 (highlight)")],
           loc="lower center", bbox_to_anchor=(0.5, 0.006), ncol=2, fontsize=S["note"],
           frameon=False, handletextpad=0.3, columnspacing=1.2)

# ================================================================ 导出 + 门禁
OUT = os.path.join(F.FIGS, "fig1_dualmodal_300dpi")
files, res = F.save(fig, OUT, dpi=600, allow_raster=True,
                    colors=[ZERO, SEQ, STR, DUAL, ABL])
print("saved:", files)
print("门禁:", res["n_pass"], "/", res["n_total"], "PASS")
for name, ok, r in res["rows"]:
    if not ok:
        print("   FAIL:", name, r)

print("\n(a) 折外预测：n = %d, AUC = %.4f, ρ = %.4f"
      % (len(oof_true), auc, rho_oof))
print("    tolerated %d / deleterious %d" % (int(tol.sum()), int((~tol).sum())))
print("\n(b) 模型 × backbone：")
for m in MODELS4:
    print("    %-15s " % m + "  ".join("%s %.4f%s" % (b, BB_RHO[m][b][0],
          "" if np.isnan(BB_RHO[m][b][1]) else "±%.4f" % BB_RHO[m][b][1]) for b in BB))
print("\n(c) 置换重要性（8M）：")
for _, r in pi8.sort_values("mean", ascending=False).iterrows():
    print("    %-18s %+.4f ± %.4f  (%s)" % (PI_LABEL[r.feature], r["mean"], r["std"],
                                           PI_SIDE[r.feature]))
print("\n(d) 逐位点：n = %d，top-10 重合 %d 个 %s；逐位点 ρ = %.4f (P = %.3g)"
      % (len(PP), len(both), both, rho_pos, p_pos))
print("    d_s70：min %.1f / median %.1f / max %.1f"
      % (ds70.min(), np.median(ds70), ds70.max()))
