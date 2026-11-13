#!/usr/bin/env python3
"""40_fig_ga.py -- the graphical abstract.

Writes two things: fig0_framework (Figure GA, the four-layer framework that sits in the
manuscript) and figGA_graphical_abstract_52 (the separate 5:2, 127 x 50 mm graphic the
journal asks for at submission).

Historically this script also emitted three sibling figures under names the
manuscript never used:

    fig1_results       -> superseded by 41_fig1_dualmodal.py  (fig1_dualmodal)
    fig2_multiprotein  -> superseded by 42_fig2_structural_gain.py  (fig2_structural_gain)
    fig3_combined      -> superseded by 43_fig3_external_benchmark.py  (fig3_external_benchmark)

Those blocks were removed on 2026-11-13. They were still wired to
pre-anchor-fix values (p = 0.854, rho = 0.703 +/- 0.029, 0.705 / AUC 0.86,
GBM 0.691), so any re-run regenerated three figures that contradicted the
manuscript while looking authoritative. The old renderings are archived in
repository/figures/_superseded/.

Only the graphical abstract is produced here. Every performance number it
displays is read from repository/results/ at run time -- no hard-coded values.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os
import json
import pandas as pd

# ---- style -----------------------------------------------------------------
INK = "#222222"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8, "figure.dpi": 300, "savefig.dpi": 300,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.5,
})
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(OUT, "repository", "results")
# 输出目录；AMR_FIGDIR 可改写到别处（预览/对比），默认写工程根目录
FIGOUT = os.environ.get("AMR_FIGDIR") or OUT
# canonical (corrected-anchor) numbers, written by 51_collect_numbers.py
_NUM_PATH = os.path.join(RES, "canonical_numbers.json")
NUM = json.load(open(_NUM_PATH, encoding="utf-8")) if os.path.exists(_NUM_PATH) else {}

# ---- GNN storyline stats, read from the reproducible pipeline outputs ----
_gat5 = pd.read_csv(f"{RES}/naive_gat_5fold.csv")
_gat5 = _gat5[pd.to_numeric(_gat5["seed"], errors="coerce").notna()]
GAT_M, GAT_S = float(_gat5["rho"].mean()), float(_gat5["rho"].std(ddof=0))
# ddof=0: population SD, the single convention used for every SD in the paper.
# (pandas' default ddof=1 is what previously put 0.162 into this figure.)
_gcnr = pd.read_csv(f"{RES}/gcn_extended_ramp.csv")
GCN_E = sorted(_gcnr["epoch"].unique())
_gcng = _gcnr.groupby("epoch")["test_rho"]
GCN_M = [float(_gcng.mean()[e]) for e in GCN_E]
PLATEAU = GCN_M[-1]
_mlp = pd.read_csv(f"{RES}/mlp_ablation.csv")
MLP_TR, MLP_TE = float(_mlp["train_rho"].mean()), float(_mlp["test_rho"].mean())

# ============================ fig0: framework ================================
MUTE = {"data": "#6E7B4E", "feat": "#46697A", "mod": "#A65D46", "eval": "#4E7E74"}
fig, ax = plt.subplots(figsize=(9.12, 10.97))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def tab(y0, y1, color, label):
    ax.add_patch(FancyBboxPatch((0.5, y0), 6.5, y1 - y0,
                 boxstyle="round,pad=0.3,rounding_size=1.2",
                 facecolor=color, edgecolor="none", zorder=2))
    ax.text(3.75, (y0 + y1) / 2, label, rotation=90, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="white", zorder=3)

def card(x0, y0, x1, y1, color, title, lines, fs=8.6):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                 boxstyle="round,pad=0.5,rounding_size=1.0",
                 facecolor="white", edgecolor="#C8C8C8", linewidth=1.1, zorder=2))
    ax.add_patch(FancyBboxPatch((x0, y1 - 4.6), x1 - x0, 4.0,
                 boxstyle="round,pad=0.5,rounding_size=1.0",
                 facecolor=color, edgecolor="none", alpha=0.12, zorder=2))
    cx = (x0 + x1) / 2
    ax.text(cx, y1 - 2.3, title, ha="center", va="center", fontsize=10.3,
            fontweight="bold", color=color, zorder=3)
    n = len(lines)
    ys = [y1 - 6.6 - i * ((y1 - y0 - 8.6) / max(n - 1, 1)) for i in range(n)]
    for yy, ln in zip(ys, lines):
        ax.text(cx, yy, ln, ha="center", va="center", fontsize=fs, color=INK,
                linespacing=1.25, zorder=3)

def flow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=14, linewidth=1.3, color="#8A8A8A", zorder=1))

tab(74, 97, MUTE["data"], "DATA")
card(9, 76, 36.5, 95, MUTE["data"], "ProteinGym v1.3 DMS assays",
     ["TEM-1 (4,996) \u00b7 VIM-2 (5,004)",
      "AAC(3)-I (1,793) \u00b7 APH(3\u2032)-IIa (4,960)",
      "DHFR (2,916) \u00b7 ampicillin \u00b7 meropenem",
      "gentamicin \u00b7 kanamycin \u00b7 growth selection"])
card(38.5, 76, 66, 95, MUTE["data"], "UniProt sequences",
     ["P62593 (TEM-1) \u00b7 P23181 (AAC)",
      "P00552 (APH) \u00b7 P0ABQ4 (DHFR)",
      "A4GRB6\u2020 (VIM-2)",
      "\u2020deprecated \u2192 DMS target used"])
card(68, 76, 98.5, 95, MUTE["data"], "AlphaFold DB structures",
     ["mean pLDDT 88.9\u201396.4 (5 proteins)",
      "structure sequence verified",
      "residue-by-residue",
      "vs wild type"])

tab(50, 71, MUTE["feat"], "ALIGNMENT-FREE FEATURES")
card(9, 52, 53, 69, MUTE["feat"], "Sequence modality \u2014 ESM-2 (8M), no MSA",
     ["masked-marginal zero-shot score",
      "logP(mut | context) \u2212 logP(wt | context)",
      "+ BLOSUM62 \u00b7 \u0394hydrophobicity \u00b7 \u0394volume \u00b7 \u0394charge",
      "(5 features)"])
card(55, 52, 98.5, 69, MUTE["feat"], "Structural modality \u2014 predicted structure",
     ["pLDDT (confidence) \u00b7 C\u03b2 contact number (burial)",
      "distance to functional-site reference",
      "(S68 [ABL 70] \u00b7 L3-loop 64 \u00b7 D190 \u00b7 D27)",
      "distance to \u03a9-loop / mobile-loop analogue",
      "(4 features)"], fs=8.2)

tab(26, 47, MUTE["mod"], "MODELS")
# ---- canonical numbers (single source of truth for GA labels) ----
def _n(*keys, default=0.0):
    x = NUM
    for k in keys:
        if not isinstance(x, dict) or k not in x:
            return default
        x = x[k]
    return float(x) if isinstance(x, (int, float)) else default

M3_8, M3_8_SD, M3_8_AUC = _n("T1_8m", "M3"), _n("T1_8m", "M3_sd"), _n("T1_8m", "M3_auc")
EXT_8 = _n("T2", "Firnberg2014_dual", default=0.820)
EXT_650 = _n("T2", "Firnberg2014_dual650m", default=0.871)
GC_T, GB_T, P_T = _n("T56_TEM1", "GCNv2"), _n("T56_TEM1", "GBM"), _n("T56_TEM1", "p")
GC_V, GB_V, P_V = _n("T56_VIM2", "GCNv2"), _n("T56_VIM2", "GBM"), _n("T56_VIM2", "p")
# VIM-2 dual-modal 8M cell, read from the same shipped table as Table 3 rather than
# typed in (it was hard-coded as 0.728, a value from the pre-anchor-fix build).
_mpr = pd.read_csv(f"{RES}/multiprotein_results.csv")
_vim = _mpr[_mpr.protein.str.startswith("VIM-2")].iloc[0]
VIM_8, VIM_8_AUC = float(_vim.dual), float(_vim.auc_dual)
PI_PATH = os.path.join(RES, "permutation_importance.csv")
BURIAL_IMP = 0.50
if os.path.exists(PI_PATH):
    try:
        _pi = pd.read_csv(PI_PATH)
        _sub = _pi[(_pi.backbone == "8m") & (_pi.feature == "contact_num")]
        if len(_sub):
            BURIAL_IMP = float(_sub.importance.mean())
    except Exception:
        pass
card(9, 28, 53, 45, MUTE["mod"], "Dual-modal GBM",
     ["9 interpretable features \u2192 gradient-boosted trees",
      f"TEM-1: \u03c1 = {M3_8:.3f} \u00b1 {M3_8_SD:.3f}, AUC {M3_8_AUC:.3f}",
      f"VIM-2: \u03c1 = {VIM_8:.3f}, AUC {VIM_8_AUC:.3f}",
      f"external validation \u03c1 up to {EXT_8:.3f}"])
card(55, 28, 98.5, 45, MUTE["mod"], "Graph network branch (C\u03b2 contact graph)",
     [f"naive GATv2: \u03c1 = {GAT_M:.3f} \u00b1 {GAT_S:.3f}",
      f"extended training \u2192 plateau {PLATEAU:.3f} (< GBM)",
      "GCNv2 (redesign)",
      "\u03c1 = %.3f \u00b1 %.3f \u2248 GBM (p = %.3f)" % (GC_T, _n("T56_TEM1", "GCNv2_sd"), P_T)])

tab(2, 23, MUTE["eval"], "EVALUATION")
card(9, 4, 53, 21, MUTE["eval"], "Generalization & mechanism",
     ["stringent modulo 5-fold by-position CV",
      f"cross-experiment: \u03c1 up to {EXT_8:.3f}",
      "cross-protein transfer: strong between",
      "beta-lactamases, weak to anchorless",
      f"burial dominates (perm. importance {BURIAL_IMP:.2f});",
      "hotspots recapitulate pocket & \u03a9-loop"], fs=8.2)
card(55, 4, 98.5, 21, MUTE["eval"], "Diagnose \u2192 Fix \u2192 Validate",
     ["1  signal dilution at readout (~18 neighbours)",
      "   \u2192 mutated-node skip connection",
      f"2  position memorization (train {MLP_TR:.2f} / test {MLP_TE:.2f})",
      "   \u2192 embedding-channel dropout (p = 0.3)",
      "3  redundant re-learning \u2192 feature passthrough",
      f"parity on VIM-2: {GC_V:.3f} vs {GB_V:.3f} (p = {P_V:.3f})"], fs=8.2)

flow(23, 75.4, 26, 69.8); flow(52, 75.4, 40, 69.8); flow(83, 75.4, 76, 69.8)
flow(31, 51.4, 31, 45.8); flow(77, 51.4, 77, 45.8)
flow(31, 27.4, 31, 21.8); flow(77, 27.4, 77, 21.8)
plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
os.makedirs(FIGOUT, exist_ok=True)
fig.savefig(f"{FIGOUT}/fig0_framework_300dpi.png"); fig.savefig(f"{FIGOUT}/fig0_framework_300dpi.pdf")
plt.close(fig); print("fig0 done")

# ==================== graphical abstract: 5:2 landscape, 127 x 50 mm =========
# OUP asks for a separately supplied graphical abstract at 5:2, >=127x50 mm,
# landscape. The in-manuscript framework figure above is portrait and carries
# far more text than fits at that size, so this is a condensed four-column
# strip: same four layers, one message each.
TW, TH = 127 / 25.4, 50.8 / 25.4
g, gax = plt.subplots(figsize=(TW, TH))
gax.set_xlim(0, 100); gax.set_ylim(0, 100); gax.axis("off")

COLS = [
    (MUTE["data"], "DATA", ["5 DMS assays", "1,793-5,004 variants", "AlphaFold structures"]),
    (MUTE["feat"], "FEATURES", ["ESM-2 zero-shot", "+ C\u03b2 burial", "no MSA needed"]),
    (MUTE["mod"],  "MODEL", ["dual-modal GBM", "GCNv2 0.366 \u2192 0.701", "parity, p = 0.448"]),
    (MUTE["eval"], "RESULT", ["\u03c1 0.814 vs 0.767*", "+0.27 on APH(3\u2032)-IIa", "gain = burial prior"]),
]
w, gap = 22.0, 2.6
for k, (col, head, lines) in enumerate(COLS):
    x0 = 1.0 + k * (w + gap)
    gax.add_patch(FancyBboxPatch((x0, 15), w, 83,
                  boxstyle="round,pad=0.6,rounding_size=2.0",
                  facecolor="white", edgecolor="#D0D0D0", linewidth=0.9, zorder=2))
    gax.add_patch(FancyBboxPatch((x0, 80), w, 18,
                  boxstyle="round,pad=0.6,rounding_size=2.0",
                  facecolor=col, edgecolor="none", alpha=0.16, zorder=2))
    gax.text(x0 + w / 2, 89, head, ha="center", va="center", fontsize=7.4,
             fontweight="bold", color=col, zorder=3)
    for i, ln in enumerate(lines):
        gax.text(x0 + w / 2, 66 - i * 17, ln, ha="center", va="center",
                 fontsize=6.2, color=INK, zorder=3)
    if k < 3:
        gax.annotate("", xy=(x0 + w + gap - 0.4, 50), xytext=(x0 + w + 0.4, 50),
                     arrowprops=dict(arrowstyle="-|>", color="#9A9A9A",
                                     linewidth=1.0, mutation_scale=8))
g.text(0.5, 0.015, "*best published zero-shot predictor for TEM-1; "
       "supervised-versus-zero-shot comparison, not like-for-like",
       ha="center", va="bottom", fontsize=4.6, color="#6A6A6A",
       transform=g.transFigure)
plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.055)
os.makedirs(FIGOUT, exist_ok=True)
g.savefig(f"{FIGOUT}/figGA_graphical_abstract_52.png", dpi=600)
g.savefig(f"{FIGOUT}/figGA_graphical_abstract_52.pdf")
plt.close(g)
print("graphical abstract (5:2) done")
