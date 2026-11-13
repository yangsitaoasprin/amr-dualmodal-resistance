#!/usr/bin/env python3
"""52_canonical_report.py — dump every number needed to patch the
manuscript from the canonical result bundle into a single Markdown report.
"""
import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
D = os.path.join(HERE, "..", "data")
F = lambda fn: pd.read_csv(os.path.join(R, fn))
out = []
P = out.append

P("# Canonical (corrected-anchor) numbers — full report\n")

# --- naive GAT / ramps / MLP
ng = F("naive_gat_5fold.csv")
ngn = ng[pd.to_numeric(ng["seed"], errors="coerce").notna()]
P("## GNN diagnostics")
P(f"- naive GATv2 15 runs: {ngn.rho.mean():.3f} +/- {ngn.rho.std():.3f}")
ramp = F("gcn_extended_ramp.csv")
g = ramp.groupby("epoch").agg(te=("test_rho", "mean"), tr=("train_rho", "mean"))
P(f"- extended GCN plateau (epoch 60): test {g.te.loc[60]:.3f}, train {g.tr.loc[60]:.3f}")
gat = F("gat_extended_ramp.csv")
gg = gat.groupby("epoch").agg(te=("rho", "mean"))
P(f"- naive GAT ramp at last epoch: {gg.te.iloc[-1]:.3f}; max {gg.te.max():.3f}")
mlp = F("mlp_ablation.csv")
P(f"- MLP ablation: train {mlp.train_rho.mean():.3f}, test {mlp.test_rho.mean():.3f}")

# --- oof / per-position / top10
z = np.load(os.path.join(R, "oof_predictions_tem1_8m.npz"), allow_pickle=True)
oof = pd.DataFrame(dict(pos=z["pos"], pred=z["oof_dual"], true=z["oof_true"]))
agg = oof.groupby("pos").agg(pred=("pred", "mean"), true=("true", "mean")).reset_index()
P("\n## Figure 1 / Section 3.5")
P(f"- pooled oof rho = {spearmanr(z['oof_dual'], z['oof_true']).statistic:.3f}")
P(f"- per-position rho = {spearmanr(agg.pred, agg.true).statistic:.3f} (n={len(agg)})")
pt = agg.sort_values("pred").head(10)["pos"].tolist()
et = agg.sort_values("true").head(10)["pos"].tolist()
P(f"- predicted most-deleterious top10 = {pt}")
P(f"- experimental top10 = {et}")
P(f"- overlap = {sorted(set(pt) & set(et))} ({len(set(pt)&set(et))}/10)")

# --- anchor counterfactual
ac = F("anchor_counterfactual.csv")
P("\n## Anchor counterfactual (canonical)")
P(ac[ac.backbone.isin(["8m", "650m"])].to_string(index=False))

# --- random anchor null
try:
    ra = F("random_anchor_null.csv")
    P("\n## Random-anchor null (canonical)")
    P(ra.head(16).to_string(index=False))
except Exception as e:
    P(f"\n(random_anchor_null.csv: {e})")

# --- permutation importance
pi = F("permutation_importance.csv")
P("\n## Test-set permutation importance (canonical)")
P(pi.head(12).to_string(index=False))

# --- collinearity
fc = F("feature_collinearity.csv")
P("\n## Feature collinearity (canonical)")
P(fc.head(12).to_string(index=False))

# --- contiguous CV
cc = F("contiguous_cv_650m.csv")
P("\n## Contiguous CV 650M (canonical)")
P(cc.to_string(index=False))

# --- transfer matrices
P("\n## Cross-protein transfer (canonical)")
for fn in ["expanded_proteins_evaluation.csv", "expanded_proteins_650m_evaluation.csv"]:
    e = F(fn)
    cols = [c for c in e.columns if "->" in c or c in ("assay", "protocol")]
    P(f"### {fn}"); P(e.head(12).to_string(index=False)[:2200])

# --- delta rho table
dr = F("delta_rho_table.csv")
P("\n## delta_rho_table (canonical, first 20 rows)")
P(dr.head(20).to_string(index=False))

txt = "\n".join(out)
open(os.path.join(R, "canonical_report.md"), "w", encoding="utf-8").write(txt)
print(txt)
