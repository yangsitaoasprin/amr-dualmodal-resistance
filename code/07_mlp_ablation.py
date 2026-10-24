#!/usr/bin/env python3
"""
07_mlp_ablation.py — MLP ablation of the extended 2-layer GCN on TEM-1
fold 0: identical architecture and features, but graph propagation removed
(adjacency replaced by the identity matrix), 25 epochs, full 4,000-sample
training split, batch 64, Adam lr 2e-3 with 0.6 decay every 10 epochs.

Diagnosis experiment (paper Section "Diagnosis", experiment i): reports
train vs test Spearman rho to expose the overfitting gap of the graph-free
model. Pure PyTorch, CPU-only, three fixed seeds (0, 1, 2) for
reproducibility. Writes ../results/mlp_ablation.csv.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

import importlib.util
spec = importlib.util.spec_from_file_location("gcn_ramp", "06_extended_gcn_ramp.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def main():
    rows = [run_seed(s) for s in (0, 1, 2)]
    pd.DataFrame(rows).to_csv("../results/mlp_ablation.csv", index=False)
    print("wrote ../results/mlp_ablation.csv")


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    df = pd.read_csv(f"{g.DATA}/tem1_stiffler_modeling_table.csv")
    emb = np.load(f"{g.DATA}/tem1_esm2_emb.npy").astype(np.float32)
    sf = pd.read_csv(f"{g.DATA}/tem1_structural_features.csv").sort_values("pos")
    s4 = sf[g.STRUCT].values.astype(np.float32)
    assert len(s4) == emb.shape[0]
    # ablation: identity adjacency == no message passing
    A_norm = torch.eye(emb.shape[0])
    fold = df["pos"].values % 5
    dtr = df[fold != 0].reset_index(drop=True)
    dte = df[fold == 0].reset_index(drop=True)
    base = g.make_base(emb, s4)
    mut_tr = torch.tensor(dtr[g.MUTCH].values, dtype=torch.float32)
    mut_te = torch.tensor(dte[g.MUTCH].values, dtype=torch.float32)
    pos_tr = torch.tensor(dtr["pos"].values - 1)
    pos_te = torch.tensor(dte["pos"].values - 1)
    ytr = torch.tensor(dtr["DMS_score"].values, dtype=torch.float32)
    ytr_np, yte_np = dtr["DMS_score"].values, dte["DMS_score"].values
    model = g.GCN()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lr = 2e-3
    for ep in range(1, 26):
        if ep in (11, 21):
            lr *= 0.6
            for gparam in opt.param_groups:
                gparam["lr"] = lr
        model.train()
        perm = torch.randperm(len(dtr))
        for s in range(0, len(perm), 64):
            b = perm[s:s + 64]
            loss = F.mse_loss(model(A_norm, g.assemble(base, mut_tr[b], pos_tr[b])),
                              ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    r_tr = g.evaluate(model, A_norm, base, mut_tr, pos_tr, ytr_np)
    r_te = g.evaluate(model, A_norm, base, mut_te, pos_te, yte_np)
    print(f"seed {seed}: MLP ablation (25 epochs, fold 0): train rho "
          f"{r_tr:.3f} | test rho {r_te:.3f}")
    return {"seed": seed, "model": "mlp_ablation", "epochs": 25, "fold": 0,
            "train_rho": r_tr, "test_rho": r_te}


if __name__ == "__main__":
    main()
