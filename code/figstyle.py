# -*- coding: utf-8 -*-
"""
figstyle.py — shared style layer for the manuscript figures (thin wrapper).

The style primitives come from ``figstyle_core.py`` in this directory: theme,
figure sizing, panel labels, export, colour helpers and the delivery gate. There
are no other dependencies.

The primitives wrapped are:
  style.set_style("nature")      physical size / font ladder / inward ticks
  style.new_fig / panel_label / save_all / colorbar_vector
  palette.tint / shade / mix     pre-blended colours instead of alpha
  checks.audit                   delivery gate
  caption_style.audit_caption    caption gate

本文件只提供三样东西：
  ① 通道 A 色表（模态）+ backbone 形状表 —— 全篇唯一的两套视觉语法
  ② 蛋白元数据（样本量 n / 药类）—— 各图共用，杜绝口径打架
  ③ 几个小工具（显著性标记、CSV 读取、逐折 note 解析）

通道约定（FIGURE_PLAN_v3 §3）：
  颜色 = "输入模态"，全局一套；形状 = "backbone 规模"，全局一套。
  两者正交，绝不混用；流程图另用中性灰蓝体系，不占通道 A。
"""
import os
import sys

import numpy as np
import pandas as pd

# ------------------------------------------------------- style implementation
# Single implementation, shipped with the repository; no external dependency.
from figstyle_core import palette, style, checks, caption_style   # noqa: E402
BACKEND = "core"

# ---------------------------------------------------------------- 路径
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
PROJ = os.path.abspath(os.path.join(REPO, ".."))
DATA = os.path.join(REPO, "data")
RES = os.path.join(REPO, "results")
# 图件输出目录；设 AMR_FIGDIR 可把渲染结果写到别处（例如预览/对比），
# 不动工程根目录下的发表版图件。
FIGS = os.environ.get("AMR_FIGDIR") or PROJ

# ---------------------------------------------------------------- 通道 A：模态色
# 「零样本 / 单模态 / 双模态」是一条输入信息通道，颜色只表达它。
MODAL = {
    "zero":    "#999999",   # 零样本（ESM-2 直接打分）
    "seq":     "#0072B2",   # 仅序列侧特征
    "struct":  "#009E73",   # 仅结构侧特征
    "dual":    "#E69F00",   # 双模态（本文主角）
    "ablate":  "#D55E00",   # 消融 / 反事实
    "external": "#000000",  # 外部已发表方法 / 参考线
}
# 通道 B（受限次级）：蛋白 / 药类。**绝不在同一面板与通道 A 共存**（R-B1）
CLASS_B = {
    "beta":   "#3B6FB0",    # β-内酰胺酶类
    "amino":  "#7FA88F",    # 氨基糖苷类
    "folate": "#B08A6E",    # 抗叶酸类
}
# 流程图 / 架构示意的中性灰蓝体系（不占通道 A）
NEUTRAL = {"band": "#46697A", "edge": "#C8C8C8", "card": "#FFFFFF", "ink": "#1A1A1A"}

# ---------------------------------------------------------------- 形状：backbone 规模
SHAPE = {"8m": "o", "650m": "D", "3b": "^"}
SHAPE_LABEL = {"8m": "ESM-2 8M", "650m": "ESM-2 650M", "3b": "ESM-2 3B"}
SIZE_M = {"8m": 28, "650m": 34, "3b": 38}      # 标记大小（pt^2），越大模型越大

# ---------------------------------------------------------------- 蛋白元数据（唯一口径）
PROTEINS = {
    "TEM1_Stiffler2015": dict(short="TEM-1", n=4996, cls="beta", drug="Ampicillin",
                               label="TEM-1"),
    "VIM2_Chen2020":     dict(short="VIM-2", n=5004, cls="beta", drug="Ampicillin",
                               label="VIM-2"),
    "AAC3I_Dandage2018": dict(short="AAC(3)-I", n=1793, cls="amino", drug="Gentamicin",
                               label="AAC(3)-I"),
    "APH3II_Melnikov2014": dict(short="APH(3′)-IIa", n=4960, cls="amino",
                                drug="Kanamycin", label="APH(3′)-IIa"),
    "DHFR_Nguyen2023":   dict(short="DHFR", n=2916, cls="folate", drug="Trimethoprim",
                               label="DHFR"),
}
CLASS_LABEL = {"beta": "β-lactamase", "amino": "Aminoglycoside", "folate": "Antifolate"}
# 五蛋白在迁移矩阵里的固定顺序（TEM-1 打头，与正文叙述顺序一致）
PROTEIN_ORDER = ["TEM1_Stiffler2015", "VIM2_Chen2020", "AAC3I_Dandage2018",
                  "APH3II_Melnikov2014", "DHFR_Nguyen2023"]

# ---------------------------------------------------------------- 基础工具
SIZE = style.SIZE


def setup():
    """应用 Nature 主题并返回字号层级。"""
    style.set_style("nature")
    return style.SIZE


def newfig(width="double", height=None, **kw):
    """按物理尺寸建图（Nature：单栏 89 mm / 双栏 183 mm / 高 ≤247 mm）。"""
    return style.new_fig("nature", width, height, **kw)


def panel(ax, s):
    """面板号：8 pt 粗体小写，置于左上角（**不用 set_title**）。"""
    return style.panel_label(ax, s)


def tidy(ax):
    return style.tidy(ax)


def save(fig, stem, dpi=600, allow_raster=False, colors=None):
    """导出 PNG + PDF，并跑九道门禁。返回 (文件列表, 门禁结果)。

    colors：本次图件实际用到的**分类色**（用于色盲/灰度可辨性这一关）。
    """
    files = style.save_all(fig, stem, dpi=dpi)
    res = checks.audit(fig, stem=stem, colors=colors, journal="nature",
                       allow_raster=allow_raster)
    return files, res


def audit_colors(colors):
    return palette.audit_colors(list(colors))


def check_caption(text, journal="Bioinformatics"):
    """图注门禁：n / 误差棒含义 / 检验+P 三项必标。"""
    return caption_style.audit_caption(text, journal=journal)


# ---------------------------------------------------------------- 小工具
def stars(p):
    """p 值 -> 显著性标记（ns / * / ** / ***）。"""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n.s."
    if p < 1e-4:
        return "P < 1e-4"
    if p < 0.001:
        return "P = %.1e-4" % (p / 1e-4)
    return "P = %.3f" % p


def load(name, folder=RES):
    """读 results/ 或 data/ 下的 csv。"""
    return pd.read_csv(os.path.join(folder, name))


def parse_folds(note):
    """把 'folds=0.479,0.390,...' 解析成 float 数组。"""
    if not isinstance(note, str) or "folds=" not in note:
        return None
    tail = note.split("folds=", 1)[1]
    return np.array([float(x) for x in tail.split(",") if x.strip() != ""])


def fold_table():
    """汇总所有归档的逐折 rho：来源为 evaluation CSV 的 note 字段。"""
    files = ["expanded_proteins_evaluation.csv", "expanded_proteins_650m_evaluation.csv",
             "esm2_650m_evaluation.csv", "esm2_3b_evaluation.csv"]
    out = []
    for fn in files:
        p = os.path.join(RES, fn)
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        df = df[df.protocol == "cv5_modulo"]
        for _, r in df.iterrows():
            folds = parse_folds(r.get("note", ""))
            if folds is None:
                continue
            out.append(dict(assay=r.assay, model=r.model, rho=r.rho,
                            rho_std=r.rho_std, folds=folds))
    return out


# ---------------------------------------------------------------- 派生量
def delta_rho_table():
    """Fig 2(a) 的 Δρ 全表（权威来源：delta_rho_table.csv，由 23_build 脚本现算并
    与 paired_tests.csv / 各 evaluation CSV 交叉核对通过）。

    15 个组合 = 5 蛋白 × 3 backbone，含逐折配对差的 SD 与双侧配对 t 检验 p 值。
    """
    df = load("delta_rho_table.csv")
    df["label"] = df.assay.map(lambda a: PROTEINS[a]["short"])
    df["cls"] = df.assay.map(lambda a: PROTEINS[a]["cls"])
    return df


def anchor_ablation_table():
    """锚点反事实消融：ρ(full anchors) vs ρ(no anchors)，8 组合（4 蛋白 × 8M/650M）。
    Δ 与 p 取自 paired_tests.csv 的 dual_vs_noanchors 对比（已交叉核对）。"""
    pt = load("paired_tests.csv")
    d = pt[pt.contrast == "dual_vs_noanchors"].copy()
    ac = load("anchor_counterfactual.csv")
    d["rho_noanchors"] = [
        float(ac[(ac.assay == r.assay) & (ac.backbone == r.backbone)
                 & (ac.model == "dual_noanchors")].rho.iloc[0])
        for _, r in d.iterrows()]
    d["rho_full"] = [
        float(ac[(ac.assay == r.assay) & (ac.backbone == r.backbone)
                 & (ac.model == "dual_full")].rho.iloc[0])
        for _, r in d.iterrows()]
    d["label"] = d.assay.map(lambda a: PROTEINS[a]["short"])
    return d


def random_anchor_null_table():
    """随机锚点零模型：8 组 × 10 seed，含真锚点 ρ（仅覆盖 4 个**有锚点**的蛋白）。"""
    df = load("random_anchor_null.csv")
    df["label"] = df.assay.map(lambda a: PROTEINS[a]["short"])
    g = df.groupby(["assay", "backbone"])
    df["seed_max"] = g.rho.transform("max")
    df["seed_min"] = g.rho.transform("min")
    df["seed_sd"] = g.rho.transform(lambda s: s.std(ddof=0))
    df["true_margin"] = df.true_anchor_rho - df.seed_max
    df["true_beats_all"] = df.true_margin > 0
    return df



if __name__ == "__main__":
    setup()
    print("样式已加载；通道 A 色表：")
    for k, v in MODAL.items():
        print("   %-9s %s" % (k, v))
    print("配色审计:", audit_colors(list(MODAL.values()))["ok"])
    print("Δρ 表:")
    t = delta_rho_table()
    print(t.to_string(index=False))
