#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
44_fig4_gnn_cycle.py — Figure 4「诊断 — 修复 — 验证」循环

目标论点：GNN 部分的价值在方法论结论，不在性能。
  (a) 训练/测试曲线：更多算力不能修复结构性缺陷（TEM-1 fold 0，3 seeds）
  (b) 逐折配对：修复后 GCNv2 与 GBM 在 TEM-1 / VIM-2 上统计平局（不是胜出）
  (c) 三处针对性修复的架构示意（中性灰蓝体系，朱红编号）
  (d) 朴素 GATv2 的失败分布（3 seeds × 5 folds = 15 runs）

数据源全部为 repository/results/ 下已归档 CSV，零新增计算：
  gcn_extended_ramp.csv / gat_extended_ramp.csv /
  gcnv2_5fold_cv.csv / vim2_gcnv2_5fold_cv.csv /
  naive_gat_5fold.csv / mlp_ablation.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as F
from figstyle import palette

S = F.setup()
ZERO = F.MODAL["zero"]
SEQ = F.MODAL["seq"]
DUAL = F.MODAL["dual"]
ABL = F.MODAL["ablate"]
INK = "#1A1A1A"
GREY = "#8A8A8A"
EDGE = "#C8C8C8"
BAND = "#46697A"
FIX = "#D55E00"

# ---------------------------------------------------------------- 数据 + 核对
ramp = F.load("gcn_extended_ramp.csv")
gat = F.load("gat_extended_ramp.csv")
t1 = F.load("gcnv2_5fold_cv.csv")
v2 = F.load("vim2_gcnv2_5fold_cv.csv")
ng = F.load("naive_gat_5fold.csv")
mlp = F.load("mlp_ablation.csv")


def sd0(s):
    return float(s.std(ddof=0))


g = ramp.groupby("epoch").agg(tr=("train_rho", "mean"), tr_sd=("train_rho", sd0),
                              te=("test_rho", "mean"), te_sd=("test_rho", sd0))
EP = np.asarray(g.index, dtype=float)
PLATEAU = float(g.te.loc[60])                 # test rho at the 60-epoch plateau
GBM_FOLD0 = float(t1.GBM_rho.iloc[0])          # 0.668639，与正文 §3.6 一致
GBM_5FOLD = float(t1.GBM_rho.mean())           # 0.6963，(b)/(d) 的 5 折均值口径
NG = ng.iloc[:15].copy()                       # 3 seeds × 5 folds（末两行是汇总）
NG["fold_i"] = NG.fold.astype(int)
NG["seed_i"] = NG.seed.astype(int)
NG_MEAN = float(NG.rho.mean())
NG_SD = sd0(NG.rho)                            # population SD (ddof=0), the convention used
                                               # for every other SD in the paper. Derived from
                                               # the 15 runs, not from the CSV summary row, so
                                               # the figure cannot drift from the printed table.
assert abs(NG_MEAN - float(ng.iloc[-2].rho)) < 1e-9, "CSV mean row disagrees with 15 runs"
assert abs(NG_SD - float(ng.iloc[-1].rho)) < 1e-9, "CSV std row disagrees with ddof=0 recompute"
MLP_TR, MLP_TE = float(mlp.train_rho.mean()), float(mlp.test_rho.mean())

# ---------------------------------------------------------------- 版面
fig = F.newfig("double", 178)
gs = fig.add_gridspec(2, 3, left=0.070, right=0.985, top=0.945, bottom=0.075,
                      height_ratios=[1.0, 1.0], wspace=0.38, hspace=0.55)

# ================================================================ (a) 训练曲线
ax = fig.add_subplot(gs[0, 0:2])
ax.fill_between(EP, g.te - g.te_sd, g.te + g.te_sd,
                color=palette.tint(DUAL, 0.82), lw=0, zorder=1)
ax.plot(EP, g.tr.values, color=palette.shade(DUAL, 0.72), lw=1.2, zorder=3,
        label="2-layer GCN, train")
ax.plot(EP, g.te.values, color=DUAL, lw=1.7, ls=(0, (5, 2)), zorder=4,
        label="2-layer GCN, test")
ax.plot(gat.epoch, gat.rho, color=ZERO, lw=1.2, zorder=3,
        label="naive GATv2, test")
ax.axhline(GBM_FOLD0, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=2,
           label="dual-modal GBM (fold 0)")
ax.set_xlim(0, 63)
ax.set_ylim(-0.14, 0.98)
ax.set_xlabel("training epoch (TEM-1, fold 0)", fontsize=S["label"])
ax.set_ylabel("Spearman ρ", fontsize=S["label"])
F.tidy(ax)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.995), fontsize=S["note"],
         frameon=False, handlelength=1.9, handletextpad=0.4,
         labelspacing=0.28, ncol=2)
ax.annotate("test plateau %.3f" % PLATEAU, xy=(60, float(g.te.loc[60])),
            xytext=(44, 0.628), fontsize=S["note"], color=DUAL,
            arrowprops=dict(arrowstyle="-", color=DUAL, lw=0.7),
            ha="left", va="center")
ax.text(62.5, GBM_FOLD0 + 0.022, "GBM (fold 0) %.3f" % GBM_FOLD0, ha="right", va="bottom",
        fontsize=S["note"], color=INK)
ax.text(0.11, 1.045, "TEM-1 fold 0; three seeds; shaded band = ±1 SD (test)",
        transform=ax.transAxes, fontsize=S["note"], color=GREY,
        ha="left", va="bottom")
ax.text(0.99, 0.03, "more compute alone: fold-0 parity at best",
        transform=ax.transAxes, fontsize=S["note"], color=INK,
        ha="right", va="bottom")
F.panel(ax, "a")

# ================================================================ (b) 逐折配对
gsb = gs[0, 2].subgridspec(2, 1, hspace=0.62)
for k, (df, name, n) in enumerate([(t1, "TEM-1", 4996), (v2, "VIM-2", 5004)]):
    axb = fig.add_subplot(gsb[k])
    folds = df.fold.values.astype(float)
    gcn, gbm = df.GCNv2_rho.values, df.GBM_rho.values
    # 逐折配对连线：GCNv2 胜出用橙，落败用浅灰
    for x_, a_, b_ in zip(folds, gbm, gcn):
        axb.plot([x_, x_], [a_, b_], color=DUAL if b_ > a_ else "#C6C6C6",
                 lw=1.1, zorder=2)
    axb.plot(folds, gbm, color=SEQ, lw=0.9, zorder=2)
    axb.plot(folds, gcn, color=DUAL, lw=0.9, zorder=2)
    axb.scatter(folds, gbm, s=15, marker="o", color=SEQ, zorder=4)
    axb.scatter(folds, gcn, s=15, marker="D", color=DUAL, zorder=4)
    axb.axhline(gcn.mean(), color=DUAL, lw=0.7, ls=(0, (3, 2)), zorder=1)
    axb.axhline(gbm.mean(), color=SEQ, lw=0.7, ls=(0, (3, 2)), zorder=1)
    axb.set_xlim(-0.35, 4.35)
    axb.set_ylim(0.63, 0.84)
    axb.set_xticks(range(5))
    axb.set_title("%s  (n = %s)" % (name, format(n, ",")), fontsize=S["note"],
                  loc="left", x=0.12 if k == 0 else 0.0, pad=2, color=INK)
    if k == 1:
        axb.set_xlabel("CV fold", fontsize=S["label"])
    axb.set_ylabel("ρ", fontsize=S["label"], labelpad=1)
    F.tidy(axb)
    from scipy.stats import ttest_rel
    pv = float(ttest_rel(gcn, gbm).pvalue)
    axb.text(0.02, 0.96, "Δ = %+.3f,  P = %.3f (n.s.)" % (gcn.mean() - gbm.mean(), pv),
             transform=axb.transAxes, fontsize=S["note"], color=INK,
             ha="left", va="top")
    if k == 0:
        axb.legend(handles=[
            Line2D([], [], marker="D", color="none", markerfacecolor=DUAL,
                   markersize=3.2, label="GCNv2"),
            Line2D([], [], marker="o", color="none", markerfacecolor=SEQ,
                   markersize=3.2, label="GBM")],
            loc="lower right", bbox_to_anchor=(1.005, -0.01), ncol=2,
            fontsize=S["note"], frameon=False, handletextpad=0.2,
            columnspacing=0.8, labelspacing=0.2)
        F.panel(axb, "b")
axb_prev = axb
fig.text(0.845, 0.505, "5 paired folds per protein", fontsize=S["note"],
         color=GREY, ha="center", va="bottom")

# ================================================================ (c) 架构 / 三处修复
ax = fig.add_subplot(gs[1, 0:2])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

BOX = dict(boxstyle="round,pad=0.012", facecolor="#FFFFFF", edgecolor=EDGE,
           linewidth=0.9)
YB, YH = 0.60, 0.26


def card(x, w, lines, edge=EDGE, fc="#FFFFFF", tcol=INK, fs=None):
    p = FancyBboxPatch((x, YB), w, YH, boxstyle="round,pad=0.012",
                       facecolor=fc, edgecolor=edge, linewidth=0.9,
                       transform=ax.transAxes, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, YB + YH / 2, lines, transform=ax.transAxes,
            fontsize=fs or S["note"], color=tcol, ha="center", va="center",
            linespacing=1.45, zorder=3)
    return p


def arrow(x0, y0, x1, y1, color=BAND, lw=1.0, rad=0.0, ls="-"):
    a = FancyArrowPatch((x0, y0), (x1, y1), transform=ax.transAxes,
                        arrowstyle="-|>", mutation_scale=7, color=color,
                        linewidth=lw, linestyle=ls,
                        connectionstyle="arc3,rad=%.3f" % rad,
                        shrinkA=0, shrinkB=0, zorder=4)
    ax.add_patch(a)


card(0.015, 0.205, "node features\nESM-2 320-d + 4 structural\n+ 5-d mutation channel")
card(0.295, 0.170, "2 GCN layers\nCβ contact graph")
card(0.545, 0.150, "mean + max\nreadout")
card(0.775, 0.205, "MLP head\n→ DMS score")
arrow(0.222, 0.73, 0.293, 0.73)
arrow(0.467, 0.73, 0.543, 0.73)
arrow(0.697, 0.73, 0.773, 0.73)

# ① 突变节点 skip -> readout（绕过图传播），② 表格特征 passthrough -> head
arrow(0.115, 0.862, 0.620, 0.862, color=FIX, lw=1.3, rad=-0.16)
arrow(0.045, 0.570, 0.875, 0.570, color=FIX, lw=1.3, rad=0.13)
ax.text(0.365, 0.955, "1", transform=ax.transAxes, fontsize=S["panel"],
        fontweight="bold", color=FIX, ha="center", va="center", zorder=6)
ax.text(0.455, 0.512, "2", transform=ax.transAxes, fontsize=S["panel"],
        fontweight="bold", color=FIX, ha="center", va="center", zorder=6,
        bbox=dict(boxstyle="circle,pad=0.16", fc="#FFFFFF", ec="none"))
# ③ embedding-channel dropout 直接标在输入箭头上
ax.text(0.258, 0.775, "3", transform=ax.transAxes, fontsize=S["panel"],
        fontweight="bold", color=FIX, ha="center", va="center", zorder=6,
        bbox=dict(boxstyle="circle,pad=0.16", fc="#FFFFFF", ec=FIX, lw=0.8))

FIXTXT = [
    ("1", "mutated-node skip connection", "bypasses neighbourhood dilution"),
    ("2", "9-d tabular passthrough", "head learns a residual correction"),
    ("3", "channel dropout (p = 0.3)", "blocks position memorisation"),
]
xs = [0.015, 0.350, 0.685]
for (num, head, sub), x in zip(FIXTXT, xs):
    ax.add_patch(FancyBboxPatch((x, 0.075), 0.300, 0.335,
                                boxstyle="round,pad=0.010",
                                facecolor=palette.tint(FIX, 0.93),
                                edgecolor=FIX, linewidth=0.8,
                                transform=ax.transAxes, zorder=1))
    ax.add_patch(Circle((x + 0.032, 0.345), 0.018, transform=ax.transAxes,
                        facecolor=FIX, edgecolor="none", zorder=3))
    ax.text(x + 0.032, 0.345, num, transform=ax.transAxes,
            fontsize=S["note"], fontweight="bold", color="#FFFFFF",
            ha="center", va="center", zorder=4)
    ax.text(x + 0.062, 0.345, head, transform=ax.transAxes,
            fontsize=S["note"], color=INK, ha="left", va="center", zorder=4)
    ax.text(x + 0.016, 0.185, sub, transform=ax.transAxes,
            fontsize=S["note"], color="#6E6E6E", ha="left", va="center",
            linespacing=1.35, zorder=4)
ax.text(0.11, 1.045, "GCNv2: three targeted changes, one per diagnosed root cause",
        transform=ax.transAxes, fontsize=S["note"], color=GREY,
        ha="left", va="bottom")
F.panel(ax, "c")

# ================================================================ (d) 朴素 GAT 分布
ax = fig.add_subplot(gs[1, 2])
ax.axhspan(NG_MEAN - NG_SD, NG_MEAN + NG_SD, color=palette.tint(ZERO, 0.86),
           lw=0, zorder=0)
ax.axhline(NG_MEAN, color=ZERO, lw=0.9, ls=(0, (4, 2)), zorder=1)
ax.axhline(GBM_5FOLD, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=1)
for sd in (0, 1, 2):
    sub = NG[NG.seed_i == sd]
    ax.scatter(sub.fold_i + (sd - 1) * 0.17, sub.rho, s=17, marker="o",
               facecolor=ZERO, edgecolor="white", linewidths=0.5, zorder=3)
worst = NG.nsmallest(1, "rho").iloc[0]
ax.scatter([worst.fold_i - 0.17], [worst.rho], s=52, facecolors="none",
           edgecolors=FIX, linewidths=1.1, zorder=5)
ax.set_xlim(-0.6, 4.6)
ax.set_ylim(min(0.0, float(NG.rho.min()) - 0.04), 0.86)
ax.set_xticks(range(5))
ax.set_xlabel("CV fold", fontsize=S["label"])
ax.set_ylabel("Spearman ρ", fontsize=S["label"])
F.tidy(ax)
ax.set_title("naive GATv2, 3 seeds × 5 folds", fontsize=S["note"],
             loc="left", pad=2, color=INK)
ax.text(0.03, 0.96, "mean %.3f ± %.3f" % (NG_MEAN, NG_SD),
        transform=ax.transAxes, fontsize=S["note"], color=INK,
        ha="left", va="top")
ax.text(0.97, 0.02, "shaded = ±1 SD", transform=ax.transAxes,
        fontsize=S["note"], color=GREY, ha="right", va="bottom")
ax.annotate("ρ = %.3f" % worst.rho, xy=(worst.fold_i - 0.17, worst.rho),
            xytext=(0.9, 0.10), fontsize=S["note"], color=FIX,
            arrowprops=dict(arrowstyle="-", color=FIX, lw=0.7),
            ha="left", va="center")
ax.text(4.55, GBM_5FOLD + 0.02, "GBM %.3f (5-fold mean)" % GBM_5FOLD, ha="right", va="bottom",
        fontsize=S["note"], color=INK)
F.panel(ax, "d")

# ================================================================ 导出 + 门禁
OUT = os.path.join(F.FIGS, "fig4_gnn_cycle_300dpi")
files, res = F.save(fig, OUT, dpi=600, allow_raster=False,
                    colors=[ZERO, SEQ, DUAL, ABL])
print("saved:", files)
print("门禁:", res["n_pass"], "/", res["n_total"], "PASS")
for name, ok, r in res["rows"]:
    if not ok:
        print("   FAIL:", name, r)

print("\n(a) GCN ramp (3 seeds, fold 0):")
print(g.round(4).to_string())
print("    GBM fold 0 = %.4f ; GAT 60 ep = %.4f" % (GBM_FOLD0, float(gat.rho.iloc[-1])))
print("    MLP ablation: train %.4f / test %.4f / gap %.4f"
      % (MLP_TR, MLP_TE, MLP_TR - MLP_TE))
print("\n(b) paired 5-fold:")
for nm, d in [("TEM-1", t1), ("VIM-2", v2)]:
    from scipy.stats import ttest_rel
    tt = ttest_rel(d.GCNv2_rho, d.GBM_rho)
    print("    %-6s GCNv2 %.4f  GBM %.4f  delta %+.4f  t %+.3f  p %.4f"
          % (nm, d.GCNv2_rho.mean(), d.GBM_rho.mean(),
             d.GCNv2_rho.mean() - d.GBM_rho.mean(), tt.statistic, tt.pvalue))
print("\n(d) naive GAT: %d runs, mean %.4f, SD %.4f, min %.4f"
      % (len(NG), NG_MEAN, NG_SD, NG.rho.min()))
