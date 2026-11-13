#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
43_fig3_external_benchmark.py — Figure 3
「对住已发表预测器与外部实验，流水线在「样本量大 + 稳定性耦合」的位点胜出」

(a) 18 个已发表零样本预测器 × 5 个 assay 的 ρ 热图（按平均排名排序，格内写值）
(b) 本文最佳监督模型 vs 该 assay 最优已发表零样本（3 胜 2 负）
(c) 5×5 跨蛋白迁移矩阵（650M backbone，对角留空）
(d) 外部效度：上组 = 不重训直接跨实验/跨蛋白迁移；下组 = 更严格的 contiguous 划分

(a)(c) 共用一把色标（同一 ρ 量纲，便于互读）。
数据全部来自 repository/results/ 下已核对过的 CSV，零新增计算。
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as F
from figstyle import palette, style

S = F.setup()
ZERO = F.MODAL["zero"]
SEQ = F.MODAL["seq"]
STR = F.MODAL["struct"]
DUAL = F.MODAL["dual"]
INK = "#1A1A1A"
GREY = "#8A8A8A"
CMAP = palette.seq_map("cividis")          # CVD 友好的感知均匀顺序色标

PG_ASSAY = {"TEM-1": "TEM1_Stiffler2015", "VIM-2": "VIM2_Chen2020",
            "AAC(3)-I": "AAC3I_Dandage2018", "APH(3′)-IIa": "APH3II_Melnikov2014",
            "DHFR": "DHFR_Nguyen2023"}
ASSAY_ORDER = ["TEM-1", "VIM-2", "AAC(3)-I", "APH(3′)-IIa", "DHFR"]
PROT_SHORT = {"TEM1_Stiffler2015": "TEM-1", "VIM2_Chen2020": "VIM-2",
              "AAC3I_Dandage2018": "AAC(3)-I", "APH3II_Melnikov2014": "APH(3′)-IIa",
              "DHFR_Nguyen2023": "DHFR"}


def cell_ink(frac):
    """按格子实际底色的相对亮度选黑/白，保证数值在深浅格上都可读。"""
    rgba = CMAP(float(min(max(frac, 0.0), 1.0)))
    hx = "#%02X%02X%02X" % tuple(int(round(v * 255)) for v in rgba[:3])
    return "#FFFFFF" if palette.lum(hx) < 0.34 else INK


# ---------------------------------------------------------------- 读数据
pg = F.load("proteingym_zeroshot_benchmark.csv")
pg = pg.rename(columns={pg.columns[0]: "assay"})
pg["assay"] = pg["assay"].replace({"APH(3')-IIa": "APH(3′)-IIa"})
preds = [c for c in pg.columns if c != "assay"]
pg = pg.set_index("assay").loc[ASSAY_ORDER, preds]
ranks = pg.rank(axis=1, ascending=False, method="min")
mean_rank = ranks.mean(axis=0).sort_values()
pred_order = list(mean_rank.index)
pg = pg[pred_order]

dt = F.delta_rho_table()
ours = dt.set_index(["assay", "backbone"])
ours_best = {}
for a in PG_ASSAY.values():
    sub = ours.loc[a]
    ours_best[a] = float(max(sub.rho_seq.max(), sub.rho_struct.max(),
                              sub.rho_dual.max()))
pub_best = {a: float(pg.loc[a].max()) for a in ASSAY_ORDER}
pub_name = {a: pg.loc[a].idxmax() for a in ASSAY_ORDER}

xp = F.load("expanded_proteins_650m_evaluation.csv")
xp = xp[xp.protocol == "cross_protein"]
prot_list = F.PROTEIN_ORDER
Cm = np.full((len(prot_list), len(prot_list)), np.nan)
for _, r in xp.iterrows():
    tr, te = r.assay.split("->")
    Cm[prot_list.index(tr), prot_list.index(te)] = r.rho

vmin = float(min(pg.values.min(), np.nanmin(Cm)))
vmax = float(max(pg.values.max(), np.nanmax(Cm)))

# ---------------------------------------------------------------- 版面
fig = F.newfig("double", 200)
gs = fig.add_gridspec(2, 12, left=0.078, right=0.980, top=0.945, bottom=0.075,
                      wspace=0.95, hspace=0.55)
gs2 = gs[1, :].subgridspec(1, 3, width_ratios=[1.12, 0.86, 1.42], wspace=0.78)

# ================================================================ (a) 18 预测器热图
ax = fig.add_subplot(gs[0, 0:11])
M = pg.values
nx, ny = M.shape[1], M.shape[0]
ax.pcolormesh(np.arange(nx + 1), np.arange(ny + 1), M, cmap=CMAP, vmin=vmin,
              vmax=vmax, shading="flat", edgecolors="#FFFFFF", linewidth=0.35)
for i in range(ny):
    for j in range(nx):
        v = M[i, j]
        ax.text(j + 0.5, i + 0.5, "%.3f" % v, ha="center", va="center",
                fontsize=S["note"], color=cell_ink((v - vmin) / (vmax - vmin)))
for name, ls in (("ESM2_650M", "-"), ("ESM2_3B", (0, (3, 2)))):
    if name in pred_order:
        j = pred_order.index(name)
        ax.add_patch(Rectangle((j + 0.02, 0.02), 0.96, ny - 0.04, fill=False,
                               edgecolor=INK, linewidth=1.0, linestyle=ls))
ax.set_xticks(np.arange(nx) + 0.5)
ax.set_xticklabels(pred_order, rotation=45, ha="right", fontsize=S["tick"])
ax.set_yticks(np.arange(ny) + 0.5)
ax.set_yticklabels(ASSAY_ORDER, fontsize=S["tick"])
ax.set_xlim(0, nx)
ax.set_ylim(ny, 0)
for sp in ("top", "right", "bottom", "left"):
    ax.spines[sp].set_visible(False)
ax.tick_params(length=0)
for j, name in enumerate(pred_order):
    ax.text(j + 0.5, -0.30, "%.1f" % mean_rank[name], ha="center", va="top",
            fontsize=S["note"], color=GREY, transform=ax.get_xaxis_transform())
ax.text(0.0, -0.30, "rank", ha="left", va="top", fontsize=S["note"], color=GREY,
        transform=ax.get_xaxis_transform())
ax.text(1.0, 1.055, "predictors ordered by mean rank over the five assays",
        transform=ax.transAxes, fontsize=S["note"], color=GREY, ha="right",
        va="bottom")
F.panel(ax, "a")

fig.canvas.draw()
_pos = gs[0, 11].get_position(fig)
style.colorbar_vector(fig, [_pos.x0, _pos.y0 + 0.03, _pos.width, _pos.height * 0.92],
                       CMAP, vmin, vmax, label="Spearman ρ", vertical=True)

# ================================================================ (b) 本文 vs 已发表
ax = fig.add_subplot(gs2[0, 0])
for i, a in enumerate(ASSAY_ORDER):
    o = ours_best[PG_ASSAY[a]]
    p = pub_best[a]
    ax.plot([i, i], [p, o], color="#BFBFBF", lw=1.0, zorder=2)
    ax.scatter([i], [p], s=34, marker="s", facecolor=ZERO, edgecolor="white",
               linewidths=0.6, zorder=4)
    ax.scatter([i], [o], s=44, marker="o", facecolor=DUAL, edgecolor="white",
               linewidths=0.6, zorder=5)
    # Δ 标签放在远离另一标记的一侧，避免压住对侧方块
    if o >= p:
        ax.text(i + 0.17, (p + o) / 2, "%+.3f" % (o - p), va="center", ha="left",
                fontsize=S["note"], color=INK, zorder=6)
    else:
        ax.text(i - 0.17, (p + o) / 2, "%+.3f" % (o - p), va="center", ha="right",
                fontsize=S["note"], color=INK, zorder=6)
ax.set_xticks(np.arange(len(ASSAY_ORDER)))
ax.set_xticklabels(ASSAY_ORDER, rotation=45, ha="right", fontsize=S["tick"])
ax.set_xlim(-0.55, len(ASSAY_ORDER) - 0.10)
ax.set_ylim(0.44, 0.88)
ax.set_ylabel("Spearman ρ", fontsize=S["label"])
F.tidy(ax)
ax.legend(handles=[
    Line2D([], [], marker="s", color="none", markerfacecolor=ZERO,
           markeredgecolor="white", markersize=4.0,
           label="best published zero-shot"),
    Line2D([], [], marker="o", color="none", markerfacecolor=DUAL,
           markeredgecolor="white", markersize=4.4,
           label="our best supervised model")],
    loc="upper left", bbox_to_anchor=(-0.01, 1.005), fontsize=S["note"],
    frameon=False, handletextpad=0.3, labelspacing=0.25)
F.panel(ax, "b")

# ================================================================ (c) 5×5 迁移矩阵（650M）
ax = fig.add_subplot(gs2[0, 1])
nx2, ny2 = Cm.shape
ax.pcolormesh(np.arange(nx2 + 1), np.arange(ny2 + 1), np.ma.masked_invalid(Cm),
              cmap=CMAP, vmin=vmin, vmax=vmax, shading="flat",
              edgecolors="#FFFFFF", linewidth=0.5)
for i in range(ny2):
    for j in range(nx2):
        if np.isnan(Cm[i, j]):
            ax.text(j + 0.5, i + 0.5, "—", ha="center", va="center",
                    fontsize=S["note"], color="#BBBBBB")
        else:
            ax.text(j + 0.5, i + 0.5, "%.2f" % Cm[i, j], ha="center", va="center",
                    fontsize=S["note"],
                    color=cell_ink((Cm[i, j] - vmin) / (vmax - vmin)))
ax.set_xticks(np.arange(nx2) + 0.5)
ax.set_xticklabels([PROT_SHORT[p] for p in prot_list], rotation=45, ha="right",
                   fontsize=S["tick"])
ax.set_yticks(np.arange(ny2) + 0.5)
ax.set_yticklabels(["train " + PROT_SHORT[p] for p in prot_list],
                   fontsize=S["tick"])
ax.set_xlim(0, nx2)
ax.set_ylim(ny2, 0)
for sp in ("top", "right", "bottom", "left"):
    ax.spines[sp].set_visible(False)
ax.tick_params(length=0)
ax.set_xlabel("test protein", fontsize=S["label"], labelpad=1)
ax.text(1.0, 1.055, "shared colour scale with (a)", transform=ax.transAxes,
        fontsize=S["note"], color=GREY, ha="right", va="bottom")
F.panel(ax, "c")

# ================================================================ (d) 外部效度（单面板，2 组）
ax = fig.add_subplot(gs2[0, 2])
ex = F.load("esm2_650m_evaluation.csv")
ex = ex[ex.protocol == "external_from_Stiffler"].copy()
ex2 = F.load("expanded_proteins_650m_evaluation.csv")
ex2 = ex2[(ex2.protocol == "cross_protein")
          & (ex2.assay.str.startswith("TEM1_Stiffler2015->"))].copy()
ext = pd.concat([ex[["assay", "rho"]], ex2[["assay", "rho"]]], ignore_index=True)
ext["target"] = [a.split("->")[-1] for a in ext.assay]
short = {"TEM1_Deng2012": "TEM-1 Deng 2012", "TEM1_Firnberg2014": "TEM-1 Firnberg 2014",
         "VIM2_Chen2020": "VIM-2", "AAC3I_Dandage2018": "AAC(3)-I",
         "APH3II_Melnikov2014": "APH(3′)-IIa", "DHFR_Nguyen2023": "DHFR"}
ext["lab"] = ext.target.map(short)
ext = ext.sort_values("rho", ascending=False).reset_index(drop=True)

cv = F.load("contiguous_cv_650m.csv")
MODCOL = {"zeroshot": ZERO, "seq": SEQ, "struct": STR, "dual": DUAL}
MODMK = {"zeroshot": "o", "seq": "s", "struct": "^", "dual": "D"}

rowsd = []
yy = 0.0
for _, r in ext.iterrows():
    rowsd.append((r.lab, r.rho, DUAL, "o", yy))
    yy -= 1.0
yy -= 1.4                                   # 组间留白
n_ext = len(rowsd)
for assay, shortname in (("TEM1_Stiffler2015", "TEM-1"), ("VIM2_Chen2020", "VIM-2")):
    part = cv[cv.assay == assay]
    for mod in ("zeroshot", "seq", "struct", "dual"):
        v = float(part[part.model == mod].rho.iloc[0])
        rowsd.append(("%s  %s" % (shortname, mod), v, MODCOL[mod], MODMK[mod], yy))
        yy -= 1.0
sep_y = (rowsd[n_ext - 1][4] + rowsd[n_ext][4]) / 2.0
for (lab, v, col, mk, y_) in rowsd:
    ax.plot([0, v], [y_, y_], color="#E0E0E0", lw=0.9, zorder=2)
    ax.scatter([v], [y_], s=26, marker=mk, facecolor=col, edgecolor="white",
               linewidths=0.5, zorder=4)
    ax.text(v + 0.013, y_, "%.3f" % v, va="center", ha="left",
            fontsize=S["note"], color=INK, zorder=5)
ax.set_yticks([r[4] for r in rowsd])
ax.set_yticklabels([r[0] for r in rowsd], fontsize=S["tick"])
ax.set_ylim(rowsd[-1][4] - 0.85, rowsd[0][4] + 1.55)
ax.set_xlim(0.36, 1.00)
ax.set_xlabel("Spearman ρ", fontsize=S["label"])
F.tidy(ax)
ax.axhline(sep_y, color="#D5D5D5", lw=0.6)
ax.text(0.0, rowsd[0][4] + 0.40, "trained on TEM-1 Stiffler 2015, transferred without retraining",
        transform=ax.get_yaxis_transform(), ha="left", va="bottom",
        fontsize=S["note"], color=GREY)
ax.text(0.0, sep_y - 0.12, "contiguous-position split (stricter than modulo)",
        transform=ax.get_yaxis_transform(), ha="left", va="top",
        fontsize=S["note"], color=GREY)
ax.legend(handles=[Line2D([], [], marker=MODMK[m], color="none",
                          markerfacecolor=MODCOL[m], markeredgecolor="white",
                          markersize=3.4, label=m) for m in MODCOL],
          loc="lower left", bbox_to_anchor=(-0.01, 1.005), ncol=4,
          fontsize=S["note"], frameon=False, handletextpad=0.25, columnspacing=0.7)
F.panel(ax, "d")

# ================================================================ 导出 + 门禁
OUT = os.path.join(F.FIGS, "fig3_external_benchmark_300dpi")
files, res = F.save(fig, OUT, dpi=600, colors=[ZERO, SEQ, STR, DUAL, INK])
print("saved:", files)
print("门禁:", res["n_pass"], "/", res["n_total"], "PASS")
for name, ok, r in res["rows"]:
    if not ok:
        print("   FAIL:", name, r)
print("\n(b) 逐 assay 明细：")
for a in ASSAY_ORDER:
    print("   %-11s ours %.4f | best published %-22s %.4f | Δ %+0.4f"
          % (a, ours_best[PG_ASSAY[a]], pub_name[a], pub_best[a],
             ours_best[PG_ASSAY[a]] - pub_best[a]))
