#!/usr/bin/env python3
"""
08_naive_gat_5fold.py — Naive GATv2 on the TEM-1 contact graph under the
full modulo 5-fold by-position CV with the CPU-realistic budget of
3 epochs x 1,200 samples per fold (paper "naive GAT" number, given as the
reference line inside Figure 4a and as the GATv2 row of Table 1).

Architecture and graph are identical to 05_gat_extended.py (two GATv2-style
attention layers, hidden 32, 2 heads, mean+max readout, MLP head). For each
(seed, fold) the model is re-initialised (seed * 10 + fold) and trained on
the 4,000-sample training split with a 1,200-sample subsample per epoch.
Three fixed seeds (0, 1, 2) are run for reproducibility; the paper reports
the mean +/- SD across the resulting 15 (seed, fold) runs, with the SD taken as a
population SD (ddof=0) to match every other SD reported in the paper.
Pure PyTorch, CPU-only. Writes ../results/naive_gat_5fold.csv.
"""
import importlib.util

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location("gat_ext", "05_gat_extended.py")
gat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gat)

MUTCH = gat.MUTCH
STRUCT = gat.STRUCT
DATA = "../data"


def train_fold(seed, fold_id, mask, base, dtr, dte):
    torch.manual_seed(seed * 10 + fold_id)
    np.random.seed(seed * 10 + fold_id)
    mut_tr = torch.tensor(dtr[MUTCH].values, dtype=torch.float32)
    mut_te = torch.tensor(dte[MUTCH].values, dtype=torch.float32)
    pos_tr = torch.tensor(dtr["pos"].values - 1)
    pos_te = torch.tensor(dte["pos"].values - 1)
    ytr = torch.tensor(dtr["DMS_score"].values, dtype=torch.float32)
    yte = dte["DMS_score"].values
    yte_bin = dte["DMS_score_bin"].values
    model = gat.NaiveGAT()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(3):  # CPU-realistic budget: 3 epochs x 1,200 samples
        model.train()
        perm = torch.randperm(len(dtr))[:1200]
        for s in range(0, len(perm), 64):
            b = perm[s:s + 64]
            Xb = gat.assemble(base, mut_tr[b], pos_tr[b])
            loss = F.mse_loss(model(Xb, mask), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(dte), 200):
            sl = slice(s, s + 200)
            preds.append(model(gat.assemble(base, mut_te[sl], pos_te[sl]),
                               mask).numpy())
    pred = np.concatenate(preds)
    return (spearmanr(pred, yte).statistic,
            roc_auc_score(yte_bin, pred))


def main():
    df = pd.read_csv(f"{DATA}/tem1_stiffler_modeling_table.csv")
    emb = np.load(f"{DATA}/tem1_esm2_emb.npy").astype(np.float32)
    sf = pd.read_csv(f"{DATA}/tem1_structural_features.csv").sort_values("pos")
    s4 = sf[STRUCT].values.astype(np.float32)
    src, dst, n = gat.build_edges(f"{DATA}/AF-P62593-F1-model.pdb")
    assert n == emb.shape[0]
    adj = torch.zeros(n, n)
    adj[src, dst] = 1.0
    mask = ((1.0 - adj).clamp(min=0) * (-1e9)).view(1, 1, n, n)
    base = gat.make_base(emb, s4)
    fold = df["pos"].values % 5
    rows = []
    for seed in (0, 1, 2):
        rhos = []
        for f in range(5):
            dtr = df[fold != f].reset_index(drop=True)
            dte = df[fold == f].reset_index(drop=True)
            r, a = train_fold(seed, f, mask, base, dtr, dte)
            rhos.append(r)
            rows.append({"seed": seed, "fold": f, "rho": r, "auc": a})
            print(f"seed {seed} fold {f}: naive GAT rho {r:.3f} AUC {a:.3f}", flush=True)
        rhos = np.array(rhos)
        print(f"seed {seed}: 5-fold mean {rhos.mean():.3f} +/- {rhos.std():.3f}",
              flush=True)
    out = pd.DataFrame(rows)
    m, s = out["rho"].mean(), out["rho"].std(ddof=0)
    ma, sa = out["auc"].mean(), out["auc"].std(ddof=0)
    # ddof=0: population SD, the convention used for every other SD in the paper
    # (44_fig4_gnn_cycle.py reads the "std" row, so its annotation follows this).
    print(f"naive GAT all runs (n=15): {m:.3f} +/- {s:.3f}; AUC {ma:.3f} +/- {sa:.3f}")
    out.loc[len(out)] = {"seed": "all", "fold": "mean", "rho": m, "auc": ma}
    out.loc[len(out)] = {"seed": "all", "fold": "std", "rho": s, "auc": sa}
    out.to_csv("../results/naive_gat_5fold.csv", index=False)
    print("wrote ../results/naive_gat_5fold.csv")


if __name__ == "__main__":
    main()
