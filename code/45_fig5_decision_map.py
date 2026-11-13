#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
45_fig5_decision_map.py — Figure 5
→「实践决策图：何时用双模态、何时直接用大模型零样本、何时别碰 3B」

(a) 决策象限图：Δρ（dual − seq）按 assay 大小 n 排序分组，15 个组合（5 蛋白 × 3 backbone）
    · 实心 = P < 0.05，空心 = n.s.（不新增色相，只用填充/空心冗余编码）
    · 背景分区用 tint 预混浅色（禁 alpha）：n ≥ 4,000 = 增益区；n < 4,000 = 无增益区
(b) 成本-性能：5 个蛋白小多图，x = ESM-2 backbone（8M / 650M / 3B），y = Spearman ρ
    · 三条曲线：零样本（灰，空心）/ 序列监督（蓝，方）/ 双模态监督（橙，菱）
    · 直接给出「650M 是拐点、3B 更贵却更差」的图形证据
(c) AAC(3)-I 反向锚点实验：x = 模型（seq / dual-no-anchor / dual+CoA-anchor），
    组内两个 backbone（形状=backbone，色=模态，沿用通道 A），y = Spearman ρ
    · 标注 Δ(anchor − no-anchor) 与 Δ(dual − seq) 及其配对 t 检验 P 值

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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as F
from figstyle import palette

S = F.setup()
ZERO = F.MODAL["zero"]
SEQ = F.MODAL["seq"]
STR = F.MODAL["struct"]
DUAL = F.MODAL["dual"]
ABL = F.MODAL["ablate"]
INK = "#1A1A1A"
GREY = "#8A8A8A"

BB_LABEL = {"8m": "8M", "650m": "650M", "3b": "3B"}


# ---------------------------------------------------------------- 零样本分数汇总
def load_zeroshot():
    """从各 evaluation CSV 汇总 (assay, backbone) -> 零样本 ρ。"""
    out = {}
    for fn in ["expanded_proteins_evaluation.csv",
               "expanded_proteins_650m_evaluation.csv",
               "expanded_proteins_3b_evaluation.csv",
               "esm2_650m_evaluation.csv", "esm2_3b_evaluation.csv"]:
        df = F.load(fn)
        if "protocol" in df.columns:
            df = df[df.protocol == "full_assay"]
        else:
            df = df[df.model == "zeroshot"]
        for _, r in df.iterrows():
            m = str(r.model)
            if "zeroshot" not in m:
                continue
            bb = str(r.backbone) if "backbone" in df.columns else \
                next((b for b in ("8m", "650m", "3b") if b in m), None)
            if bb:
                out[(r.assay, bb)] = float(r.rho)
    # 8M 三个蛋白的零样本分只存在于 multiprotein_results.csv
    mp = F.load("multiprotein_results.csv")
    name_map = {"TEM-1 (Amp)": "TEM1_Stiffler2015",
                "VIM-2 PSEAI (Amp)": "VIM2_Chen2020",
                "AAC(3)-I (Gent)": "AAC3I_Dandage2018"}
    for _, r in mp.iterrows():
        out[(name_map[r.protein], "8m")] = float(r.zeroshot)
    return out


ZS = load_zeroshot()
DT = F.delta_rho_table()
SEQ_RHO = {(r.assay, r.backbone): r.rho_seq for _, r in DT.iterrows()}
DUAL_RHO = {(r.assay, r.backbone): r.rho_dual for _, r in DT.iterrows()}

# ---------------------------------------------------------------- 版面
fig = F.newfig("double", 190)
gs = fig.add_gridspec(2, 1, left=0.082, right=0.985, top=0.940, bottom=0.170,
                      height_ratios=[1.0, 0.70], hspace=0.80)
gs0 = gs[0].subgridspec(1, 2, width_ratios=[1.22, 1.0], wspace=0.44)
gs1 = gs[1].subgridspec(1, 5, wspace=0.46)

# ================================================================ (a) 决策象限图
ax = fig.add_subplot(gs0[0, 0])
order = sorted(F.PROTEIN_ORDER, key=lambda a: F.PROTEINS[a]["n"])   # 按 n 升序
xpos = {a: i for i, a in enumerate(order)}
DODGE = {"8m": -0.19, "650m": 0.0, "3b": 0.19}
DIV = 1.5                                   # n = 4,000 分界（AAC/DHFR | APH/TEM/VIM）

YMIN, YMAX = -0.095, 0.365
ax.fill_between([-0.62, DIV], YMIN, YMAX, color="#F1F1F1", lw=0, zorder=0)
ax.fill_between([DIV, 4.62], YMIN, YMAX, color=palette.tint(DUAL, 0.87), lw=0, zorder=0)
ax.axhline(0, color="#9A9A9A", lw=0.8, zorder=1)
ax.axvline(DIV, color="#B4B4B4", lw=0.8, ls=(0, (4, 3)), zorder=1)

# --- 先画点，再统一做数值标签防重叠
labels = []
for _, r in DT.iterrows():
    x = xpos[r.assay] + DODGE[r.backbone]
    sig = r.p < 0.05
    ax.errorbar([x], [r.delta], yerr=[r.sd_delta], fmt="none", ecolor="#8C8C8C",
                elinewidth=0.8, capsize=1.6, capthick=0.8, zorder=2)
    ax.scatter([x], [r.delta], s=F.SIZE_M[r.backbone], marker=F.SHAPE[r.backbone],
               facecolor=DUAL if sig else "white", edgecolor=DUAL, linewidths=1.0,
               zorder=4)
    labels.append((x, r.delta + r.sd_delta + 0.010, "%+.3f" % r.delta))
placed = []
for x, y, t in sorted(labels, key=lambda z: z[1]):
    yy = y
    for px, py in placed:
        if abs(px - x) < 0.36 and abs(py - yy) < 0.026:
            yy = py + 0.026
    placed.append((x, yy))
    ax.text(x, yy, t, ha="center", va="bottom", fontsize=S["note"],
            color="#6E6E6E", zorder=5)

ax.set_xticks(range(len(order)))
ax.set_xticklabels(["%s\nn = %s" % (F.PROTEINS[a]["short"], format(F.PROTEINS[a]["n"], ","))
                    for a in order], fontsize=S["tick"], linespacing=1.35)
ax.set_xlim(-0.62, 4.62)
ax.set_ylim(YMIN, YMAX)
ax.set_ylabel("structural gain  Δρ  (dual − seq)", fontsize=S["label"])
ax.set_xlabel("assay (ordered by assay size n)", fontsize=S["label"])
F.tidy(ax)
ax.text(DIV, YMIN + 0.008, "n = 4,000", ha="center", va="bottom", fontsize=S["note"],
        color="#8C8C8C")
ax.text(0.45, 0.185, "n < 4,000:  Δρ not significantly\n"
                     "different from 0 (6/6 n.s.)",
        ha="center", va="center", fontsize=S["note"], color="#6E6E6E", linespacing=1.5)
ax.text(3.05, -0.052, "n ≥ 4,000:  Δρ > 0 in 9/9 combinations\n"
                     "across all three backbones (all P < 0.05)",
        ha="center", va="center", fontsize=S["note"], color="#8A6A10", linespacing=1.5)
leg = ax.legend(handles=[Line2D([], [], marker=F.SHAPE[b], color="none",
                                markerfacecolor=DUAL, markeredgecolor=DUAL,
                                markersize=4.0, label=F.SHAPE_LABEL[b])
                       for b in ("8m", "650m", "3b")],
                loc="lower left", bbox_to_anchor=(0.015, 1.005), ncol=3,
                fontsize=S["note"], frameon=False, handletextpad=0.25,
                columnspacing=0.9,
                title="ESM-2 backbone      filled = P < 0.05,  open = n.s.")
leg.get_title().set_fontsize(S["note"])
leg.get_title().set_color("#6E6E6E")
leg._legend_box.align = "left"
F.panel(ax, "a")

# ================================================================ (c) AAC 反向锚点
ax = fig.add_subplot(gs0[0, 1])
aac = F.load("aac_anchor_experiment.csv")
ver = F.load("aac_anchor_experiment_verified.csv")
pval = {}
for _, r in ver.iterrows():
    if str(r.model).startswith("_p_"):
        pval[(r.backbone, str(r.model)[3:])] = float(r.rho)

MODELS = [("seq", "sequence only", SEQ, "s"),
          ("dual_noanchors", "dual, no anchor", DUAL, "D"),
          ("dual_with_coa_anchor", "dual + CoA-pocket anchor", ABL, "^")]
BBS = [("8m", "8M"), ("650m", "650M")]
for mi, (mkey, mlab, mcol, mmk) in enumerate(MODELS):
    xs = []
    for bi, (bb, blab) in enumerate(BBS):
        r = aac[(aac.backbone == bb) & (aac.model == mkey)].iloc[0]
        x = mi + (-0.15 if bi == 0 else 0.15)
        xs.append(x)
        ax.errorbar([x], [r.rho], yerr=[r.rho_std], fmt="none", ecolor="#8C8C8C",
                    elinewidth=0.8, capsize=2.0, capthick=0.8, zorder=2)
        ax.scatter([x], [r.rho], s=34, marker=F.SHAPE[bb], facecolor=mcol,
                   edgecolor="white", linewidths=0.6, zorder=4)
        if bi == 0:
            ax.text(x - 0.035, r.rho, "%.3f" % r.rho, ha="right", va="center",
                    fontsize=S["note"], color=INK, zorder=5)
        else:
            ax.text(x + 0.035, r.rho, "%.3f" % r.rho, ha="left", va="center",
                    fontsize=S["note"], color=INK, zorder=5)
    ax.plot(xs, [aac[(aac.backbone == bb) & (aac.model == mkey)].rho.iloc[0]
                 for bb, _ in BBS], color="#C9C9C9", lw=0.8, ls=(0, (2, 2)), zorder=1)

ann_anch, ann_seq = [], []
for bb, blab in BBS:
    g = aac[aac.backbone == bb].set_index("model")
    d_anch = g.loc["dual_with_coa_anchor", "rho"] - g.loc["dual_noanchors", "rho"]
    d_seq = g.loc["dual_noanchors", "rho"] - g.loc["seq", "rho"]
    ann_anch.append("%s: %+.3f (P = %.3f)"
                    % (blab, d_anch, pval[(bb, "anchor_vs_noanchor")]))
    ann_seq.append("%s: %+.3f (P = %.3f)"
                   % (blab, d_seq, pval[(bb, "dual_vs_seq")]))
ax.text(0.015, 0.025,
        "CoA anchor − no anchor:\n   " + "   ·   ".join(ann_anch)
        + "\ndual − seq:\n   " + "   ·   ".join(ann_seq),
        transform=ax.transAxes, ha="left", va="bottom", fontsize=S["note"],
        color="#6E6E6E", linespacing=1.5)
ax.set_xticks(range(len(MODELS)))
ax.set_xticklabels(["sequence\nonly", "dual\n(no anchor)", "dual + CoA\nanchor"],
                   fontsize=S["tick"], linespacing=1.35)
ax.set_xlim(-0.55, len(MODELS) - 0.45)
ax.set_ylim(0.0, 0.60)
ax.set_ylabel("Spearman ρ", fontsize=S["label"])
F.tidy(ax)
ax.legend(handles=[Line2D([], [], marker=F.SHAPE[bb], color="none",
                          markerfacecolor="#9A9A9A", markeredgecolor="#9A9A9A",
                          markersize=4.0, label="ESM-2 " + blab) for bb, blab in BBS],
          loc="lower right", bbox_to_anchor=(1.005, 1.005), ncol=2,
          fontsize=S["note"], frameon=False, handletextpad=0.25, columnspacing=0.9)
ax.text(0.0, 1.005, "AAC(3)-I, n = 1,793", transform=ax.transAxes, ha="left",
        va="bottom", fontsize=S["note"], color=GREY)
F.panel(ax, "c")

# ================================================================ (b) 成本-性能小多图
PANEL_ORDER = ["AAC3I_Dandage2018", "DHFR_Nguyen2023", "APH3II_Melnikov2014",
               "TEM1_Stiffler2015", "VIM2_Chen2020"]
BBS3 = ["8m", "650m", "3b"]
SERIES = [("zero-shot", ZERO, "o", False),
          ("sequence only", SEQ, "s", True),
          ("dual-modal", DUAL, "D", True)]
for pi, a in enumerate(PANEL_ORDER):
    axb = fig.add_subplot(gs1[0, pi])
    for sname, scol, smk, filled in SERIES:
        ys = []
        for bb in BBS3:
            if sname == "zero-shot":
                ys.append(ZS[(a, bb)])
            elif sname == "sequence only":
                ys.append(SEQ_RHO[(a, bb)])
            else:
                ys.append(DUAL_RHO[(a, bb)])
        axb.plot(range(3), ys, color="#7C7C7C" if sname == "zero-shot" else scol,
                 lw=1.1, zorder=2, solid_capstyle="round")
        axb.scatter(range(3), ys, s=26, marker=smk,
                    facecolor=scol if filled else "white",
                    edgecolor="#7C7C7C" if sname == "zero-shot" else scol,
                    linewidths=1.0, zorder=4)
    axb.set_xticks(range(3))
    axb.set_xticklabels([BB_LABEL[b] for b in BBS3], fontsize=S["tick"])
    axb.set_xlim(-0.32, 2.32)
    axb.set_ylim(0.05, 0.92)
    F.tidy(axb)
    if pi == 0:
        axb.set_ylabel("Spearman ρ", fontsize=S["label"])
    else:
        axb.tick_params(labelleft=False)
    axb.text(0.5, 1.025, F.PROTEINS[a]["short"], transform=axb.transAxes, ha="center",
             va="bottom", fontsize=S["label"], color=INK)
    if pi == 4:
        axb.set_xlabel("ESM-2 backbone", fontsize=S["label"], labelpad=1)
    if pi == 0:
        F.panel(axb, "b")

m_dual = {b: np.mean([DUAL_RHO[(a, b)] for a in PANEL_ORDER]) for b in BBS3}
m_zero = {b: np.mean([ZS[(a, b)] for a in PANEL_ORDER]) for b in BBS3}
fig.text(0.082, 0.086,
         "Mean over the five assays —  dual-modal: 8M %.3f  →  650M %.3f  →  3B %.3f"
         "   |   zero-shot: 8M %.3f  →  650M %.3f  →  3B %.3f"
         % (m_dual["8m"], m_dual["650m"], m_dual["3b"],
            m_zero["8m"], m_zero["650m"], m_zero["3b"]),
         ha="left", va="center", fontsize=S["note"], color="#6E6E6E")
fig.text(0.082, 0.048,
         "The 3B backbone costs %.1f× more parameters than 650M yet does not improve "
         "the mean ρ; 650M is the cost–performance knee." % (3e9 / 6.5e8),
         ha="left", va="center", fontsize=S["note"], color="#6E6E6E")

# 共享图例：放在小多图正上方
fig.canvas.draw()
_p = gs1[0, 0].get_position(fig)
fig.legend(handles=[Line2D([], [], marker=mk, color="none",
                           markerfacecolor=col if fl else "white",
                           markeredgecolor=col, markersize=4.0, label=nm)
                    for nm, col, mk, fl in SERIES],
           loc="lower left", bbox_to_anchor=(_p.x0, _p.y1 + 0.030), ncol=3,
           fontsize=S["note"], frameon=False, handletextpad=0.3, columnspacing=1.2)

# ================================================================ 导出 + 门禁
OUT = os.path.join(F.FIGS, "fig5_decision_map_300dpi")
files, res = F.save(fig, OUT, dpi=600, colors=[ZERO, SEQ, DUAL, ABL])
print("saved:", files)
print("门禁:", res["n_pass"], "/", res["n_total"], "PASS")
for name, ok, r in res["rows"]:
    if not ok:
        print("   FAIL:", name, r)

print("\n(a) 分组与 Δρ：")
for a in order:
    row = DT[DT.assay == a].set_index("backbone")
    print("   %-11s n=%5d  " % (F.PROTEINS[a]["short"], F.PROTEINS[a]["n"])
          + "  ".join("%s %+.3f(p=%.3f)" % (b, row.loc[b, "delta"], row.loc[b, "p"])
                      for b in ("8m", "650m", "3b")))
big = DT[DT.n >= 4000]
small = DT[DT.n < 4000]
print("   大样本 n>=4000: %d/%d Δρ>0, %d/%d P<0.05"
      % ((big.delta > 0).sum(), len(big), (big.p < 0.05).sum(), len(big)))
print("   小样本 n< 4000: %d/%d Δρ>0, %d/%d P<0.05"
      % ((small.delta > 0).sum(), len(small), (small.p < 0.05).sum(), len(small)))
print("\n(b) 均值：")
for b in BBS3:
    print("   %-5s  dual %.4f   seq %.4f   zero %.4f"
          % (BB_LABEL[b],
             np.mean([DUAL_RHO[(a, b)] for a in PANEL_ORDER]),
             np.mean([SEQ_RHO[(a, b)] for a in PANEL_ORDER]),
             np.mean([ZS[(a, b)] for a in PANEL_ORDER])))
print("\n(c) AAC 反向锚点：")
for bb in ("8m", "650m"):
    d = aac[aac.backbone == bb]
    print("   %-5s seq %.4f | no-anchor %.4f | +CoA %.4f | Δ(anch) %+.4f P=%.4f"
          % (bb, d[d.model == "seq"].rho.iloc[0], d[d.model == "dual_noanchors"].rho.iloc[0],
             d[d.model == "dual_with_coa_anchor"].rho.iloc[0],
             d[d.model == "dual_with_coa_anchor"].rho.iloc[0] - d[d.model == "dual_noanchors"].rho.iloc[0],
             pval[(bb, "anchor_vs_noanchor")]))
