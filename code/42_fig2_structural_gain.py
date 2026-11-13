#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
42_fig2_structural_gain.py — Figure 2
「结构增益只在「样本量足够 + 表型与稳定性紧耦合」时出现，且它并非来自功能锚点」

(a) Δρ 发散柱 —— 15 个 蛋白×backbone 组合；条长 = 结构增益，气泡面积 ∝ assay size n
(b) Δρ vs log10(n) —— 三个 backbone 上的干净分离
(c) 锚点反事实消融的哑铃图 —— 增益不来自功能锚点（APH 显著为正、DHFR 显著为负）
(d) 随机锚点零模型 —— 真锚点 vs 10 个随机种子

全部数据来自 repository/results/ 下已交叉核对过的 CSV，零新增计算。
视觉语法：颜色 = 输入模态（通道 A），形状 = backbone 规模。
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as F
from figstyle import palette

S = F.setup()
DUAL = F.MODAL["dual"]
DUAL_L = palette.tint(DUAL, 0.78)
DUAL_S = palette.shade(DUAL, 0.32)
SEED = "#7B8792"          # 随机种子（中性灰蓝，与橙/黑在 CVD 与灰度下均可辨）
INK = "#1A1A1A"
GREY = "#8A8A8A"

BB = ["8m", "650m", "3b"]
BB_TICK = {"8m": "8M", "650m": "650M", "3b": "3B"}
PROT_ORDER = ["TEM1_Stiffler2015", "VIM2_Chen2020", "APH3II_Melnikov2014",
              "AAC3I_Dandage2018", "DHFR_Nguyen2023"]
ANCH_PROT = ["TEM1_Stiffler2015", "VIM2_Chen2020", "APH3II_Melnikov2014",
             "DHFR_Nguyen2023"]

d = F.delta_rho_table()
d["_bo"] = d.backbone.map({"8m": 0, "650m": 1, "3b": 2})
d["_po"] = d.assay.map({a: i for i, a in enumerate(PROT_ORDER)})
d = d.sort_values(["_po", "_bo"]).reset_index(drop=True)

# ---------------------------------------------------------------- 版面
fig = F.newfig("double", 212)
gs = fig.add_gridspec(2, 3, left=0.112, right=0.988, top=0.945, bottom=0.072,
                      wspace=0.46, hspace=0.52, width_ratios=[1.22, 1.0, 1.0])

# ================================================================ (a) Δρ 发散柱
ax = fig.add_subplot(gs[:, 0])
rows, ylab, cls_of_row = [], [], []
y = 0.0
prev_cls = None
for _, r in d.iterrows():
    if prev_cls is not None and r.cls != prev_cls:
        y -= 1.70                       # 药类之间的额外留白
    rows.append((r, y))
    ylab.append("%s\n%s" % (r.label, BB_TICK[r.backbone]))
    cls_of_row.append(r.cls)
    y -= 1.0
    prev_cls = r.cls
ys = np.array([t[1] for t in rows])

ax.axvline(0, color=INK, lw=1.2, zorder=3)
for (r, yy), cl in zip(rows, cls_of_row):
    pos = r.delta > 0
    ax.barh(yy, r.delta, height=0.60,
            color=(DUAL if pos else DUAL_L),
            edgecolor=("none" if pos else DUAL), hatch=("" if pos else "///"),
            linewidth=0.0 if pos else 0.6, zorder=2)
    ax.scatter([r.delta], [yy], s=r.n * 0.012, facecolor="white",
               edgecolor=DUAL_S, linewidths=0.7, zorder=5)
    off = 0.032 if pos else -0.032
    ax.text(r.delta + off, yy, F.stars(r.p), va="center",
            ha="left" if pos else "right", fontsize=S["tick"], color=INK, zorder=6)

ax.set_yticks(ys)
ax.set_yticklabels(ylab, fontsize=S["tick"])
ax.set_ylim(ys.min() - 0.9, ys.max() + 1.0)
ax.set_xlim(-0.135, 0.345)
ax.set_xlabel("Δρ  (dual-modal − sequence-only)", fontsize=S["label"])
F.tidy(ax)

for i, ((r, yy), cl) in enumerate(zip(rows, cls_of_row)):
    if i + 1 < len(rows):
        nxt = rows[i + 1][0]
        mid = (yy + rows[i + 1][1]) / 2
        if nxt.cls != cl:
            ax.axhline(mid, color="#D0D0D0", lw=0.6, zorder=1)
            ax.text(0.010, mid + 0.34, F.CLASS_LABEL[cl],
                    transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                    fontsize=S["note"], color=GREY, style="italic")
        else:
            ax.axhline(yy - 0.5, color="#EFEFEF", lw=0.5, zorder=1)

for nv, txt in ((1793, "1,793"), (2916, "2,916"), (5004, "5,004")):
    ax.scatter([], [], s=nv * 0.012, facecolor="white", edgecolor=DUAL_S,
               linewidths=0.7, label=txt)
ax.legend(title="assay size n", loc="lower right", bbox_to_anchor=(1.02, -0.005),
          ncol=3, fontsize=S["note"], title_fontsize=S["note"], frameon=False,
          handletextpad=0.25, columnspacing=0.7, labelspacing=0.3,
          borderpad=0.2)
F.panel(ax, "a")

# ================================================================ (b) Δρ vs n
ax = fig.add_subplot(gs[0, 1:])
offs = {"8m": -0.016, "650m": 0.0, "3b": 0.016}
for _, r in d.iterrows():
    ax.scatter(np.log10(r.n) + offs[r.backbone], r.delta,
               s=F.SIZE_M[r.backbone], marker=F.SHAPE[r.backbone],
               facecolor=DUAL, edgecolor="white", linewidths=0.6, zorder=4)
ax.axhline(0, color=INK, lw=0.9, zorder=2)
xdiv = np.log10(4000)
ax.axvline(xdiv, color="#BFBFBF", lw=0.8, ls=(0, (3, 2)), zorder=1)
ax.text(xdiv + 0.016, 0.372, "large assays\nTEM-1, VIM-2, APH(3′)-IIa\nn = 4,960–5,004",
        ha="left", va="top", fontsize=S["note"], color=INK, zorder=6)
ax.text(xdiv - 0.016, 0.372, "small assays\nAAC(3)-I, DHFR\nn ≤ 2,916",
        ha="right", va="top", fontsize=S["note"], color=INK, zorder=6)
ax.set_xlim(np.log10(1600), np.log10(5600))
ax.set_ylim(-0.090, 0.378)
ax.set_xlabel("log10 assay size  (n variants)", fontsize=S["label"])
ax.set_ylabel("Δρ", fontsize=S["label"])
F.tidy(ax)
ax.legend(handles=[Line2D([], [], marker=F.SHAPE[b], color="none",
                          markerfacecolor=DUAL, markeredgecolor="white",
                          markersize=np.sqrt(F.SIZE_M[b]), label=F.SHAPE_LABEL[b])
                   for b in BB],
          loc="lower right", bbox_to_anchor=(1.0, 1.005), ncol=3,
          fontsize=S["note"], frameon=False, handletextpad=0.25, columnspacing=0.9)
F.panel(ax, "b")

# ================================================================ (c) 锚点消融哑铃图
ax = fig.add_subplot(gs[1, 1])
ab = F.anchor_ablation_table()
c_rows, c_lab = [], []
for a in ANCH_PROT:
    for bk in ("8m", "650m"):
        sub = ab[(ab.assay == a) & (ab.backbone == bk)]
        if len(sub):
            c_rows.append(sub.iloc[0])
            c_lab.append("%s  %s" % (F.PROTEINS[a]["short"], BB_TICK[bk]))
cys = np.arange(0, -len(c_rows), -1.0)
for r, yy in zip(c_rows, cys):
    ax.plot([r.rho_noanchors, r.rho_full], [yy, yy], color=DUAL, lw=1.2, zorder=3)
    ax.scatter([r.rho_noanchors], [yy], s=26, marker="o", facecolor="white",
               edgecolor=DUAL, linewidths=0.9, zorder=4)
    ax.scatter([r.rho_full], [yy], s=26, marker="o", facecolor=DUAL,
               edgecolor="white", linewidths=0.5, zorder=5)
    star = "" if r.p >= 0.05 else ("*" if r.p < 0.01 else "**")
    ax.text(0.885, yy, "%+.3f%s" % (r.mean_diff, star), va="center", ha="left",
            fontsize=S["note"], color=INK, zorder=6)
ax.text(0.885, cys.max() + 0.72, "Δρ", va="center", ha="left",
        fontsize=S["note"], color=GREY, style="italic")
ax.set_yticks(cys)
ax.set_yticklabels(c_lab, fontsize=S["tick"])
ax.set_ylim(cys.min() - 0.9, cys.max() + 1.1)
ax.set_xlim(0.26, 0.995)
ax.set_xlabel("Spearman ρ", fontsize=S["label"])
F.tidy(ax)
ax.legend(handles=[
    Line2D([], [], marker="o", color="none", markerfacecolor="white",
           markeredgecolor=DUAL, markeredgewidth=0.9, markersize=3.4,
           label="anchors removed"),
    Line2D([], [], marker="o", color="none", markerfacecolor=DUAL,
           markeredgecolor="white", markersize=3.4, label="true anchors")],
    loc="lower left", bbox_to_anchor=(-0.005, 1.005), ncol=2, fontsize=S["note"],
    frameon=False, handletextpad=0.3, columnspacing=0.9)
F.panel(ax, "c")

# ================================================================ (d) 随机锚点零模型
ax = fig.add_subplot(gs[1, 2])
nl = F.random_anchor_null_table()
d_rows, d_lab = [], []
for a in ANCH_PROT:
    for bk in ("8m", "650m"):
        sub = nl[(nl.assay == a) & (nl.backbone == bk)]
        if len(sub):
            d_rows.append(sub)
            d_lab.append("%s  %s" % (F.PROTEINS[a]["short"], BB_TICK[bk]))
dys = np.arange(0, -len(d_rows), -1.0)
for sub, yy in zip(d_rows, dys):
    seeds = sub.rho.values
    ax.hlines(yy, seeds.min(), seeds.max(), color=SEED, lw=0.8, zorder=2)
    for s_ in seeds:
        ax.plot([s_, s_], [yy - 0.16, yy + 0.16], color=SEED, lw=0.6, zorder=3)
    ax.plot([np.median(seeds)] * 2, [yy - 0.22, yy + 0.22], color=SEED, lw=1.4,
            zorder=4)
    t = sub.true_anchor_rho.iloc[0]
    ax.scatter([t], [yy], s=30, marker="s", facecolor=INK, edgecolor="white",
               linewidths=0.5, zorder=6)

# —— 关键结论（全部由数据现算，不写死）——
# 三分类：真锚点高于全部种子 / 落在种子区间内 / 低于全部种子
beats_all, between, below_all, marginal = [], [], [], []
for sub in d_rows:
    a, bk = sub.assay.iloc[0], sub.backbone.iloc[0]
    t = sub.true_anchor_rho.iloc[0]
    margin = t - sub.seed_max.iloc[0]
    sd = sub.seed_sd.iloc[0]
    rec = (a, bk, margin, sd)
    if t > sub.seed_max.iloc[0]:
        (marginal if margin <= sd else beats_all).append(rec)
    elif t < sub.seed_min.iloc[0]:
        below_all.append(rec)
    else:
        between.append(rec)
notes = []
def _rl(rec):
    """row label: protein + backbone, so a mixed panel is not mis-read."""
    return "%s %s" % (F.PROTEINS[rec[0]]["short"], BB_TICK[rec[1]])
if beats_all:
    notes.append("true anchor above all 10 seeds: %s"
                 % ", ".join(_rl(r) for r in beats_all))
if marginal:
    for a, bk, mg, sd in marginal:
        notes.append("%s %s: only %+.4f, below the seed SD of %.3f"
                     % (F.PROTEINS[a]["short"], BB_TICK[bk], mg, sd))
if between:
    notes.append("within the seed range: %s"
                 % ", ".join(_rl(r) for r in between))
if below_all:
    notes.append("true anchor below all 10 seeds: %s"
                 % ", ".join(_rl(r) for r in below_all))
for i, s in enumerate(notes):
    ax.text(0.268, dys.max() + 3.05 - 0.62 * i, s, ha="left", va="top",
            fontsize=S["note"], color=INK, zorder=6)

ax.set_yticks(dys)
ax.set_yticklabels(d_lab, fontsize=S["tick"])
ax.set_ylim(dys.min() - 2.4, dys.max() + 3.7)
ax.set_xlim(0.255, 0.995)
ax.set_xlabel("Spearman ρ", fontsize=S["label"])
F.tidy(ax)
ax.legend(handles=[
    Line2D([], [], marker="|", color=SEED, lw=1.0, markersize=5,
           label="random anchor (10 seeds)"),
    Line2D([], [], marker="|", color=SEED, lw=1.8, markersize=6,
           label="seed median"),
    Line2D([], [], marker="s", color="none", markerfacecolor=INK,
           markeredgecolor="white", markersize=4.0, label="true anchors")],
    loc="lower left", bbox_to_anchor=(-0.01, 0.0), ncol=3, fontsize=S["note"],
    frameon=False, handletextpad=0.35, columnspacing=1.0)
F.panel(ax, "d")

# ================================================================ 导出 + 门禁
OUT = os.path.join(F.FIGS, "fig2_structural_gain_300dpi")
files, res = F.save(fig, OUT, dpi=600, colors=[DUAL, SEED, INK])
print("saved:", files)
print("门禁:", res["n_pass"], "/", res["n_total"], "PASS")
for name, ok, r in res["rows"]:
    if not ok:
        print("   FAIL:", name, r)
