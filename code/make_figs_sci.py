# Publication-quality figure suite (fig0-fig3), restyled per sci-paper skill
# figure guidelines: Okabe-Ito colorblind-safe palette, >=8pt fonts, vector PDF
# + 300-dpi PNG output, minimal spines, consistent model colors across figures.
# All values are the verified reproduction numbers (see verifier v13).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np, pandas as pd

# ---- style -----------------------------------------------------------------
GRAY, BLUE, ORANGE, RED, GREEN = "#999999", "#0072B2", "#E69F00", "#D55E00", "#009E73"
INK = "#222222"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8, "figure.dpi": 300, "savefig.dpi": 300,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.5,
})
def despine(ax, keep=("left", "bottom")):
    for s in ax.spines:
        ax.spines[s].set_visible(s in keep)
def panel_tag(ax, tag):
    ax.set_title(tag, loc="left", fontsize=11, fontweight="bold", pad=8)
OUT = "/mnt/agents/output/amr_project"
RES = "/mnt/agents/output/amr_project/repository/results"

# ---- GNN storyline stats, read from the reproducible pipeline outputs ----
_gat5 = pd.read_csv(f"{RES}/naive_gat_5fold.csv")
_gat5 = _gat5[pd.to_numeric(_gat5["seed"], errors="coerce").notna()]
GAT_M, GAT_S = float(_gat5["rho"].mean()), float(_gat5["rho"].std())
_gcnr = pd.read_csv(f"{RES}/gcn_extended_ramp.csv")
GCN_E = sorted(_gcnr["epoch"].unique())
_gcng = _gcnr.groupby("epoch")["test_rho"]
GCN_M = [float(_gcng.mean()[e]) for e in GCN_E]
GCN_SD = [float(_gcng.std()[e]) for e in GCN_E]
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
     ["TEM-1 (n = 4,996) \u00b7 VIM-2 (n = 5,004)",
      "AAC(3)-I (n = 1,793)",
      "ampicillin / meropenem / gentamicin",
      "selection fitness"])
card(38.5, 76, 66, 95, MUTE["data"], "UniProt sequences",
     ["P62593 (TEM-1) \u00b7 P23181 (AAC)",
      "A4GRB6\u2020 (VIM-2)",
      "\u2020deprecated \u2192 DMS target",
      "sequence used"])
card(68, 76, 98.5, 95, MUTE["data"], "AlphaFold DB structures",
     ["mean pLDDT 94.8 / 91.0 / 88.9",
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
      "distance to catalytic Ser (S70 / S64)",
      "distance to \u03a9-loop",
      "(4 features)"])

tab(26, 47, MUTE["mod"], "MODELS")
card(9, 28, 53, 45, MUTE["mod"], "Dual-modal GBM",
     ["9 interpretable features \u2192 gradient-boosted trees",
      "TEM-1: \u03c1 = 0.705 \u00b1 0.023, AUC 0.859",
      "VIM-2: \u03c1 = 0.728, AUC 0.901",
      "external validation \u03c1 up to 0.820"])
card(55, 28, 98.5, 45, MUTE["mod"], "Graph network branch (C\u03b2 contact graph)",
     [f"naive GATv2: \u03c1 = {GAT_M:.3f} \u00b1 {GAT_S:.3f}",
      "extended training \u2192 plateau 0.59\u20130.66 (< GBM)",
      "GCNv2 (redesign)",
      "\u03c1 = 0.703 \u00b1 0.029 \u2248 GBM (p = 0.854)"])

tab(2, 23, MUTE["eval"], "EVALUATION")
card(9, 4, 53, 21, MUTE["eval"], "Generalization & mechanism",
     ["stringent modulo 5-fold by-position CV",
      "cross-experiment: \u03c1 up to 0.820",
      "cross-protein transfer: within-mechanism",
      "0.63\u20130.71, cross-mechanism 0.27",
      "burial dominates (importance 0.43); hotspots",
      "recapitulate catalytic pocket & \u03a9-loop"], fs=8.2)
card(55, 4, 98.5, 21, MUTE["eval"], "Diagnose \u2192 Fix \u2192 Validate",
     ["1  signal dilution at readout (~18 neighbours)",
      "   \u2192 mutated-node skip connection",
      f"2  position memorization (train {MLP_TR:.2f} / test {MLP_TE:.2f})",
      "   \u2192 embedding-channel dropout (p = 0.3)",
      "3  redundant re-learning \u2192 feature passthrough",
      "parity replicated on VIM-2: 0.736 vs 0.728 (p = 0.616)"], fs=8.2)

flow(23, 75.4, 26, 69.8); flow(52, 75.4, 40, 69.8); flow(83, 75.4, 76, 69.8)
flow(31, 51.4, 31, 45.8); flow(77, 51.4, 77, 45.8)
flow(31, 27.4, 31, 21.8); flow(77, 27.4, 77, 21.8)
plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(f"{OUT}/fig0_framework_300dpi.png"); fig.savefig(f"{OUT}/fig0_framework_300dpi.pdf")
plt.close(fig); print("fig0 done")



# ============================ fig1: TEM-1 results ============================
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))

# (a) model comparison
ax = axes[0]
models = ["ESM-2 zero-shot", "Seq-only GBM", "Struct-only GBM", "Dual-modal GBM"]
vals = [0.399, 0.527, 0.538, 0.705]
errs = [0, 0, 0, 0.023]
cols = [GRAY, GRAY, GRAY, BLUE]
b = ax.bar(models, vals, yerr=errs, capsize=3, color=cols, edgecolor="none",
           error_kw=dict(ecolor=INK, linewidth=1))
for r, v, e in zip(b, vals, errs):
    ax.text(r.get_x() + r.get_width() / 2, v + e + 0.025, f"{v:.3f}",
            ha="center", fontsize=8.5)
ax.set_ylabel("Spearman \u03c1 (5-fold CV)")
ax.set_ylim(0, 0.82); despine(ax); panel_tag(ax, "a")
ax.set_xticklabels(models, rotation=20, ha="right")
ax.tick_params(axis="x", labelsize=8)
ax.grid(axis="x", visible=False)

# (b) feature importance
ax = axes[1]
feats = ["contact_num", "blosum62", "esm2", "d_s70", "d_omega",
         "d_vol", "d_hydro", "d_chg", "pLDDT"]
imps = [0.43, 0.15, 0.09, 0.09, 0.07, 0.04, 0.04, 0.02, 0.01]
order = np.argsort(imps)
cols = [BLUE if feats[i] == "contact_num" else GRAY for i in order]
ax.barh([feats[i] for i in order], [imps[i] for i in order],
        color=cols, edgecolor="none")
for y, i in enumerate(order):
    ax.text(imps[i] + 0.008, y, f"{imps[i]:.2f}", va="center", fontsize=8.5)
ax.set_xlabel("GBM feature importance")
ax.set_xlim(0, 0.5); despine(ax); panel_tag(ax, "b")
ax.grid(axis="y", visible=False)

# (c) per-position predicted effect, coloured by distance to catalytic Ser70
ax = axes[2]
from sklearn.ensemble import GradientBoostingRegressor
t = pd.read_csv(f"{OUT}/repository/data/tem1_stiffler_modeling_table.csv")
FEATS = ["esm2", "blosum62", "d_hydro", "d_vol", "d_chg",
         "pLDDT", "contact_num", "d_s70", "d_omega"]
gbm = GradientBoostingRegressor(random_state=0).fit(t[FEATS], t["DMS_score"])
t["pred"] = gbm.predict(t[FEATS])
g = t.groupby("pos").agg(pred=("pred", "mean"), d_s70=("d_s70", "first")).reset_index()
ax.axvspan(163, 178, color=ORANGE, alpha=0.15, lw=0)
sc = ax.scatter(g["pos"], g["pred"], c=g["d_s70"], cmap="viridis",
                s=9, linewidths=0, rasterized=True)
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label("Distance to Ser70 (\u00c5)", fontsize=8.5)
cb.ax.tick_params(labelsize=8)
ax.text(170.5, g["pred"].max() + 0.06, "\u03a9-loop", ha="center",
        fontsize=8.5, color="#8a5a00")
ax.set_xlabel("Residue position (TEM-1)")
ax.set_ylabel("Mean predicted DMS score")
ax.set_ylim(top=g["pred"].max() + 0.16)
despine(ax); panel_tag(ax, "c")

fig.tight_layout()
fig.savefig(f"{OUT}/fig1_results_300dpi.png", bbox_inches="tight")
fig.savefig(f"{OUT}/fig1_results_300dpi.pdf", bbox_inches="tight")
plt.close(fig); print("fig1 done")

# ==================== fig2: multi-protein + transfer =========================
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))

# (a) three-protein generalisation
ax = axes[0]
prot = ["TEM-1", "VIM-2", "AAC(3)-I"]
esm = [0.399, 0.431, 0.204]; sq = [0.527, 0.507, 0.243]; du = [0.705, 0.728, 0.251]
x = np.arange(3); w = 0.26
ax.bar(x - w, esm, w, label="ESM-2 zero-shot", color=GRAY, edgecolor="none")
ax.bar(x, sq, w, label="Sequence-only GBM", color=ORANGE, edgecolor="none")
ax.bar(x + w, du, w, label="Dual-modal GBM", color=BLUE, edgecolor="none")
for xi, v in zip(x + w, du):
    ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=8.5)
ax.set_xticks(x, prot); ax.set_ylabel("Spearman \u03c1 (5-fold CV)")
ax.set_ylim(0, 1.02); ax.legend(frameon=False, loc="upper left")
despine(ax); panel_tag(ax, "a")
ax.grid(axis="x", visible=False)

# (b) cross-protein transfer
ax = axes[1]
pairs = ["TEM-1 \u2192 VIM-2", "VIM-2 \u2192 TEM-1",
         "AAC(3)-I \u2192 VIM-2", "AAC(3)-I \u2192 TEM-1",
         "TEM-1 \u2192 AAC(3)-I", "VIM-2 \u2192 AAC(3)-I"]
tv = [0.713, 0.631, 0.635, 0.522, 0.268, 0.268]
tc = [RED, RED, ORANGE, ORANGE, GRAY, GRAY]
ax.barh(pairs[::-1], tv[::-1], color=tc[::-1], edgecolor="none")
for y, v in enumerate(tv[::-1]):
    ax.text(v + 0.012, y, f"{v:.3f}", va="center", fontsize=8.5)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=RED, label="Across \u03b2-lactamase classes (A\u2194B)"),
                   Patch(color=ORANGE, label="AAC \u2192 \u03b2-lactamase"),
                   Patch(color=GRAY, label="\u03b2-lactamase \u2192 AAC")],
          frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16),
          fontsize=8, ncol=3, columnspacing=0.9, handlelength=1.2)
ax.set_xlabel("Spearman \u03c1 (train \u2192 test)")
ax.set_xlim(0, 0.84); despine(ax); panel_tag(ax, "b")
ax.grid(axis="y", visible=False)

fig.tight_layout()
fig.savefig(f"{OUT}/fig2_multiprotein_300dpi.png", bbox_inches="tight")
fig.savefig(f"{OUT}/fig2_multiprotein_300dpi.pdf", bbox_inches="tight")
plt.close(fig); print("fig2 done")

# ===== fig4: per-model architecture diagrams (tensors, layers, data flow) ====
def _box(ax, x0, y0, x1, y1, line1, line2=None, fc="#FFFFFF", ec="#C8C8C8",
         fs=7.2, tc=INK, bold=False, lw=1.0, z=3, ls="-"):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                 boxstyle="round,pad=0.25,rounding_size=0.8",
                 facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, linestyle=ls))
    cy = (y0 + y1) / 2
    if line2 is None:
        ax.text((x0 + x1) / 2, cy, line1, ha="center", va="center",
                fontsize=fs, color=tc, fontweight="bold" if bold else "normal", zorder=z + 1)
    else:
        ax.text((x0 + x1) / 2, cy + (y1 - y0) * 0.22, line1, ha="center", va="center",
                fontsize=fs, color=tc, fontweight="bold" if bold else "normal", zorder=z + 1)
        ax.text((x0 + x1) / 2, cy - (y1 - y0) * 0.24, line2, ha="center", va="center",
                fontsize=fs - 1.0, color="#666666" if tc == INK else tc, zorder=z + 1)

def _flow(ax, x, y0, y1, label=None, color="#888888", lx=None, lc="#555555"):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=9,
                 color=color, lw=1.4, zorder=4))
    if label:
        ax.text((lx if lx is not None else x + 1.6), (y0 + y1) / 2, label, fontsize=6.2,
                color=lc, ha="left", va="center", style="italic", zorder=4)

def _mat(ax, x0, y0, x1, y1, n=5, fc="#EAF0F4", ec="#8FA9BA", z=3, hcell=None):
    """Matrix/tensor glyph with grid lines."""
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                 boxstyle="round,pad=0.15,rounding_size=0.4",
                 facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=z))
    for i in range(1, n):
        ax.plot([x0 + (x1 - x0) * i / n] * 2, [y0 + 0.6, y1 - 0.6], color=ec, lw=0.4, zorder=z + 1)
        ax.plot([x0 + 0.6, x1 - 0.6], [y0 + (y1 - y0) * i / n] * 2, color=ec, lw=0.4, zorder=z + 1)
    if hcell is not None:  # highlight one row (mutated node row)
        yc = y0 + (y1 - y0) * hcell
        ax.add_patch(plt.Rectangle((x0 + 0.4, yc - 0.9), x1 - x0 - 0.8, 1.8,
                     facecolor=RED, edgecolor="none", alpha=0.85, zorder=z + 2))

def ahead(ax, tag, name, color, result):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(-2, 104, tag, fontsize=11, fontweight="bold", color=INK, ha="left", va="center")
    ax.text(2, 97.5, name, fontsize=9.5, fontweight="bold", color=color, ha="left", va="center")
    ax.add_patch(FancyBboxPatch((58, 93.5), 40, 7.0, boxstyle="round,pad=0.3,rounding_size=0.9",
                 facecolor=color, edgecolor="none", zorder=3))
    ax.text(78, 97.0, result, fontsize=7.6, fontweight="bold", color="white",
            ha="center", va="center", zorder=4)

def anote(ax, txt, y, color="#B03A2E", mark="Diagnosis"):
    ax.text(4, y, mark, fontsize=6.8, color=color, ha="left", va="top", fontweight="bold")
    ax.text(4 + 3.4 * len(mark), y, txt, fontsize=6.8, color=INK, ha="left", va="top")

fig = plt.figure(figsize=(13.6, 13.2))
gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.3, 1.3])
axes = np.empty((2, 3), dtype=object)
for r in range(2):
    for c in range(3):
        axes[r, c] = fig.add_subplot(gs[r + 1, c])

# (a) training ramps
axr = fig.add_subplot(gs[0, :2])
gcn_e, gcn_r, gcn_s = GCN_E, GCN_M, GCN_SD  # repository/results/gcn_extended_ramp.csv (3 seeds)
gat_e = [3, 6, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
gat_r = [-0.031, 0.078, 0.364, 0.568, 0.591, 0.583, 0.589, 0.589,
         0.583, 0.602, 0.595, 0.592, 0.593]
axr.plot(gcn_e, gcn_r, "-o", color=ORANGE, markersize=4, linewidth=1.8, label="GCN, extended training (mean of 3 seeds)")
axr.fill_between(gcn_e, np.array(gcn_r) - np.array(gcn_s), np.array(gcn_r) + np.array(gcn_s), color=ORANGE, alpha=0.18, linewidth=0)
axr.plot(gat_e, gat_r, "-s", color=GRAY, markersize=4, linewidth=1.8, label="Naive GATv2, extended ramp (60 epochs)")
axr.axhline(0.691, color=BLUE, linestyle="--", linewidth=1.5, label="Dual-modal GBM (same fold)")
axr.annotate(f"plateau {PLATEAU:.3f} < GBM 0.691", xy=(57, PLATEAU + 0.004), xytext=(26, 0.60),
             fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="->", color="#666666", lw=0.9))
axr.set_xlabel("Training epochs"); axr.set_ylabel("Spearman \u03c1 (fold 0)")
axr.set_ylim(0, 0.78); axr.legend(frameon=False, loc="lower right")
despine(axr); panel_tag(axr, "a")

# (b) storyline summary bars
axb = fig.add_subplot(gs[0, 2])
vals = [GAT_M, PLATEAU, MLP_TE, 0.703, 0.705]
cols = [GRAY, ORANGE, GRAY, GREEN, BLUE]
labs = ["Naive\nGATv2", "Extended\nGCN", "MLP\nablation", "GCNv2", "Dual-modal\nGBM"]
axb.bar(range(5), vals, color=cols, width=0.62, zorder=3)
for i, v in enumerate(vals):
    axb.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8, color=INK)
axb.plot([3, 3, 4, 4], [0.80, 0.825, 0.825, 0.80], color=INK, lw=1.0)
axb.text(3.5, 0.835, "p = 0.854", ha="center", fontsize=7.6, color=INK)
axb.set_xticks(range(5), labs, fontsize=7.6)
axb.set_ylim(0, 0.92); axb.set_ylabel("Spearman \u03c1")
axb.set_title("Storyline summary (TEM-1)", fontsize=9.5, fontweight="bold", loc="left")
despine(axb); panel_tag(axb, "b")
axb.grid(axis="x", visible=False)

# ---- (a) Naive GATv2 --------------------------------------------------------
ax = axes[0, 0]
ahead(ax, "c", "Naive GATv2", GRAY, f"\u03c1 = {GAT_M:.3f} \u00b1 {GAT_S:.3f}")
_mat(ax, 10, 82, 34, 92, hcell=0.72)
ax.text(22, 80.2, "X \u2208 \u211d\u00b2\u2078\u2076\u02e3\u00b3\u00b2\u2079", fontsize=6.4, ha="center", color=INK)
ax.text(22, 76.8, "ESM-2(320)+struct(4)+mut(5)", fontsize=5.4, ha="center", color="#666666")
_mat(ax, 46, 82, 70, 92, fc="#F5F0E8", ec="#BFA77A")
ax.text(58, 80.2, "A \u2208 {0,1}\u00b2\u2078\u2076\u02e3\u00b2\u2078\u2076", fontsize=6.4, ha="center", color=INK)
ax.text(58, 76.8, "C\u03b2 contacts (< 10 \u00c5)", fontsize=5.4, ha="center", color="#666666")
_flow(ax, 30, 82, 70.5); _flow(ax, 50, 82, 70.5)
_box(ax, 22, 62, 78, 70, "GATv2 layer  (2 heads)", "\u03b1_ij = softmax_j LeakyReLU(a[Wh_i\u2016Wh_j]) \u00b7 329\u219264")
_flow(ax, 50, 62, 56.5, "H\u00b9 \u2208 \u211d\u00b2\u2078\u2076\u02e3\u2076\u2074")
_box(ax, 22, 48, 78, 56, "GATv2 layer  (2 heads)", "masked by A \u00b7 64\u219264")
_flow(ax, 50, 48, 42.5, "H\u00b2 \u2208 \u211d\u00b2\u2078\u2076\u02e3\u2076\u2074")
_box(ax, 22, 34, 78, 42, "global readout", "mean \u2016 max over 286 nodes \u2192 z \u2208 \u211d\u00b9\u00b2\u2078")
_flow(ax, 50, 34, 28.5)
_box(ax, 22, 20, 78, 28, "MLP head", "128 \u2192 64 \u2192 1", fc=GRAY, ec="none", tc="white", bold=True)
_flow(ax, 50, 20, 15.5, "\u0177 \u2208 \u211d (DMS score)")
anote(ax, "mutation signal diluted over ~18 neighbours in readout", 12)
anote(ax, "position identity memorised from embeddings", 8)
anote(ax, "re-learns the pre-digested ESM-2 score from scratch", 4)

# ---- (b) Extended-training GCN ----------------------------------------------
ax = axes[0, 1]
ahead(ax, "d", "Extended-training GCN", ORANGE, f"plateau \u03c1 = {PLATEAU:.3f}")
_mat(ax, 10, 82, 34, 92, hcell=0.72)
ax.text(22, 80.2, "X \u2208 \u211d\u00b2\u2078\u2076\u02e3\u00b3\u00b2\u2079", fontsize=6.4, ha="center", color=INK)
ax.text(22, 76.8, "same inputs as (a)", fontsize=5.4, ha="center", color="#666666")
_mat(ax, 46, 82, 70, 92, fc="#F5F0E8", ec="#BFA77A")
ax.text(58, 80.2, "A \u2208 {0,1}\u00b2\u2078\u2076\u02e3\u00b2\u2078\u2076", fontsize=6.4, ha="center", color=INK)
ax.text(58, 76.8, "\u00c2 = D^{-1/2}(A+I)D^{-1/2}", fontsize=5.4, ha="center", color="#666666")
_flow(ax, 30, 82, 70.5); _flow(ax, 50, 82, 70.5)
_box(ax, 22, 62, 78, 70, "GCN layer", "H\u2190\u03c3(\u00c2 H W) \u00b7 329\u219264")
_flow(ax, 50, 62, 56.5, "H\u00b9 \u2208 \u211d\u00b2\u2078\u2076\u02e3\u2076\u2074")
_box(ax, 22, 48, 78, 56, "GCN layer", "\u03c3(\u00c2 H W) \u00b7 64\u219264")
_flow(ax, 50, 48, 42.5, "H\u00b2 \u2208 \u211d\u00b2\u2078\u2076\u02e3\u2076\u2074")
_box(ax, 22, 34, 78, 42, "global readout", "mean \u2016 max \u2192 z \u2208 \u211d\u00b9\u00b2\u2078")
_flow(ax, 50, 34, 28.5)
_box(ax, 22, 20, 78, 28, "MLP head", "128 \u2192 64 \u2192 1", fc=ORANGE, ec="none", tc="white", bold=True)
_flow(ax, 50, 20, 15.5, "\u0177 \u2208 \u211d")
anote(ax, "architecture unchanged \u2014 only the budget grew:", 12, color="#B07D10", mark="Finding")
anote(ax, "60 epochs, lr decay, full 4,000-sample training set", 8, color="#B07D10", mark="Protocol")
anote(ax, f"{GCN_M[0]:.3f} (5 ep) \u2192 {PLATEAU:.3f} plateau, still < GBM 0.691", 4, color="#B07D10", mark="Finding")

# ---- (c) MLP ablation ---------------------------------------------------------
ax = axes[0, 2]
ahead(ax, "e", "MLP ablation", GRAY, f"test \u03c1 = {MLP_TE:.3f}")
_mat(ax, 10, 82, 34, 92, hcell=0.72)
ax.text(22, 80.2, "X \u2208 \u211d\u00b2\u2078\u2076\u02e3\u00b3\u00b2\u2079", fontsize=6.4, ha="center", color=INK)
ax.text(22, 76.8, "same node features", fontsize=5.4, ha="center", color="#666666")
_mat(ax, 46, 82, 70, 92, fc="#F5F0E8", ec="#BFA77A")
ax.plot([46, 70], [82, 92], color=RED, lw=1.6, zorder=6)
ax.plot([46, 70], [92, 82], color=RED, lw=1.6, zorder=6)
ax.text(58, 80.2, "A removed", fontsize=6.4, ha="center", color=RED)
ax.text(58, 76.8, "no message passing", fontsize=5.4, ha="center", color="#666666")
_flow(ax, 30, 82, 70.5)
_box(ax, 22, 62, 78, 70, "identity (no propagation)", "H\u2070 = X \u00b7 graph branch deleted", ls="--")
_flow(ax, 50, 62, 56.5, "X \u2208 \u211d\u00b2\u2078\u2076\u02e3\u00b3\u00b2\u2079")
_box(ax, 22, 48, 78, 56, "global readout", "mean \u2016 max over nodes \u2192 z")
_flow(ax, 50, 48, 42.5)
_box(ax, 22, 34, 78, 42, "MLP head", "25 epochs \u00b7 \u0177 \u2208 \u211d", fc=GRAY, ec="none", tc="white", bold=True)
anote(ax, f"train {MLP_TR:.3f} vs test {MLP_TE:.3f} \u2014 gap {MLP_TR - MLP_TE:.2f}", 12)
anote(ax, "message passing itself aids generalisation;", 8)
anote(ax, "position memorization from embeddings confirmed", 4)

# ---- (d) GCNv2 (redesign) -----------------------------------------------------
ax = axes[1, 0]
ahead(ax, "f", "GCNv2 (redesign)", GREEN, "\u03c1 = 0.703 \u00b1 0.029")
_mat(ax, 8, 84, 28, 92, hcell=0.72)
ax.text(18, 82.2, "X (329 ch.)", fontsize=6.0, ha="center", color=INK)
_mat(ax, 36, 84, 56, 92, fc="#F5F0E8", ec="#BFA77A")
ax.text(46, 82.2, "\u00c2", fontsize=6.4, ha="center", color=INK)
_flow(ax, 24, 84, 74.5); _flow(ax, 42, 84, 74.5)
_box(ax, 16, 66, 74, 74, "GCN layer + channel dropout (p = 0.3)", "\u03c3(\u00c2 H W) \u00b7 329\u219264, dropout on 320 ESM-2 channels")
_flow(ax, 45, 66, 60.5, "H\u00b9 \u2208 \u211d\u00b2\u2078\u2076\u02e3\u2076\u2074")
_box(ax, 16, 52, 74, 60, "GCN layer", "\u03c3(\u00c2 H W) \u00b7 64\u219264")
_flow(ax, 45, 52, 46.5, "H\u00b2 \u2208 \u211d\u00b2\u2078\u2076\u02e3\u2076\u2074")
_box(ax, 16, 34, 74, 46, "readout: mean \u2016 max \u2295 h_mut", "z = [mean \u2016 max \u2016 h_mut] \u2208 \u211d\u00b9\u2079\u00b2")
ax.add_patch(FancyArrowPatch((16.5, 56), (10.5, 40), arrowstyle="-|>", mutation_scale=9,
             color=RED, lw=1.6, connectionstyle="arc3,rad=0.35", zorder=5))
ax.text(7.5, 48, "h_mut\nskip", fontsize=6.0, color=RED, ha="center", va="center")
_box(ax, 80, 34, 97, 42, "tabular\nf \u2208 \u211d\u2079", fc="#EAF3FA", ec=BLUE, fs=6.4)
ax.add_patch(FancyArrowPatch((88, 34), (66, 28.5), arrowstyle="-|>", mutation_scale=9,
             color=BLUE, lw=1.6, connectionstyle="arc3,rad=-0.2", zorder=5))
ax.text(83, 30.0, "passthrough", fontsize=5.8, color=BLUE, ha="center")
_flow(ax, 45, 34, 28.5)
_box(ax, 16, 20, 74, 28, "MLP head", "[z \u2016 f] \u2208 \u211d\u00b2\u2070\u00b9 \u2192 64 \u2192 1", fc=GREEN, ec="none", tc="white", bold=True)
_flow(ax, 45, 20, 15.5, "\u0177 \u2208 \u211d")
anote(ax, "skip beats dilution \u00b7 passthrough ends re-learning", 12, color="#1E8449", mark="Fix")
anote(ax, "channel dropout blocks position memorization", 8, color="#1E8449", mark="Fix")

# ---- (e) Dual-modal GBM --------------------------------------------------------
ax = axes[1, 1]
ahead(ax, "g", "Dual-modal GBM", BLUE, "\u03c1 = 0.705 \u00b7 AUC 0.86")
_mat(ax, 12, 76, 40, 92, n=9, fc="#EAF3FA", ec=BLUE)
ax.text(26, 73.6, "x \u2208 \u211d\u2079 per mutation", fontsize=6.4, ha="center", color=INK)
feats9 = "ESM-2 \u00b7 BLOSUM62 \u00b7 \u0394hydro \u00b7 \u0394vol \u00b7 \u0394chg\npLDDT \u00b7 contact_num \u00b7 d(Ser70) \u00b7 d(\u03a9-loop)"
ax.text(26, 67.5, feats9, fontsize=5.8, ha="center", va="top", color="#666666")
_flow(ax, 45, 84, 70.5, None, lx=44)
_box(ax, 52, 62, 92, 70, "sequence modality", "5 seq. features", fs=6.8)
_box(ax, 52, 48, 92, 56, "structure modality", "4 struct. features", fs=6.8)
ax.text(26, 56.0, "no graph, no embeddings,\nno training budget", fontsize=6.2, ha="center", va="top", color=INK, style="italic")
_flow(ax, 60, 48, 44.5); _flow(ax, 84, 48, 44.5)
ax.text(88, 46.2, "x \u2208 \u211d\u2079", fontsize=6.2, color="#555555", style="italic")
_box(ax, 42, 32, 96, 44, "GradientBoostingRegressor", "100 trees \u00b7 depth 3 \u00b7 F(x) = \u03a3 \u03bd\u00b7T_m(x)", fc=BLUE, ec="none", tc="white", bold=True)
_flow(ax, 69, 32, 26.5, "\u0177 \u2208 \u211d,  P(resistant) for AUC", lx=70)
anote(ax, "interpretable, trains in seconds on CPU", 20, color="#1A5276", mark="Takeaway")
anote(ax, "statistical parity with GCNv2 (paired t-test p = 0.854)", 16, color="#1A5276", mark="Takeaway")
anote(ax, "the practical default; GCNv2 the validated graph alternative", 12, color="#1A5276", mark="Takeaway")

# ---- (h) reading guide -------------------------------------------------------
ax = axes[1, 2]
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
ax.text(-2, 104, "h", fontsize=11, fontweight="bold", color=INK, ha="left", va="center")
ax.text(2, 94, "How to read this figure", fontsize=9.5, fontweight="bold", color=INK)
guide = [
    ("Results", "a\u2013b: what happened \u2014 training dynamics and the final ranking on TEM-1.", "#666666"),
    ("Diagnosis", "c, e: why the naive end-to-end GNN failed \u2014 dilution, memorization, re-learning.", "#B03A2E"),
    ("Diagnosis", f"d: more compute alone does not fix it (plateau {PLATEAU:.3f} < GBM 0.691).", "#B07D10"),
    ("Fix", "f: three targeted changes restore the graph model to 0.703 \u00b1 0.029.", "#1E8449"),
    ("Validation", "g: parity with the interpretable GBM (p = 0.854) closes the loop.", "#1A5276"),
]
y = 84
for tag, txt, col in guide:
    ax.text(2, y, tag, fontsize=7.6, fontweight="bold", color=col, va="top")
    ax.text(2, y - 6, txt, fontsize=7.0, color=INK, va="top", wrap=True)
    y -= 19
ax.text(2, y - 2, "Colours match across all panels: grey = naive/ablation,\norange = extended training, green = GCNv2, blue = GBM.",
        fontsize=6.6, color="#666666", va="top", style="italic")

fig.tight_layout()
fig.savefig(f"{OUT}/fig3_combined_300dpi.png", bbox_inches="tight")
fig.savefig(f"{OUT}/fig3_combined_300dpi.pdf", bbox_inches="tight")
plt.close(fig); print("fig3 combined done")
