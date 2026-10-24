#!/usr/bin/env python3
"""
06_extended_gcn_ramp.py — Extended-training 2-layer GCN on the TEM-1 Cbeta
contact graph, fold 0, up to 60 epochs (Figure 3a "extended GCN" curve).

Architecture: a *plain* 2-layer GCN (no skip connection / passthrough /
dropout -- those are the GCNv2 fixes of 03_gcnv2.py), hidden width 64,
mean+max global readout, MLP regression head. Same normalized adjacency
A_norm = D^-1/2 (A+I) D^-1/2 and node features as 03_gcnv2.py.

Protocol: full 4,000-sample training split, batch 64, Adam lr 2e-3 with
0.6 decay every 10 epochs, checkpoints at 5/15/25/35/45/60 epochs with
train and test Spearman rho. Pure PyTorch, CPU-only, three fixed seeds
(0, 1, 2) for reproducibility; results are reported as mean +/- std
across seeds. Writes ../results/gcn_extended_ramp.csv.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

MUTCH = ["esm2", "d_hydro", "d_vol", "d_chg", "blosum62"]
STRUCT = ["pLDDT", "contact_num", "d_s70", "d_omega"]
DATA = "../data"


def build_graph(pdb_path, cutoff=10.0):
    """Normalized adjacency A_norm = D^-1/2 (A+I) D^-1/2 of the Cbeta <10 A graph."""
    from Bio.PDB import PDBParser
    st = PDBParser(QUIET=True).get_structure("af", pdb_path)
    chain = next(st[0].get_chains())
    coords = []
    for res in chain:
        if res.id[0] == " ":
            coords.append(res["CB"].get_coord() if "CB" in res else res["CA"].get_coord())
    X = np.array(coords)
    dm = np.linalg.norm(X[:, None] - X[None, :], axis=-1)
    A = (dm < cutoff).astype(np.float32)
    np.fill_diagonal(A, 1.0)
    d = A.sum(1)
    return torch.tensor(A / np.sqrt(d[:, None] * d[None, :]))


class GCN(nn.Module):
    """Plain 2-layer GCN (pre-fix architecture): mean+max readout, MLP head."""
    def __init__(self, n_in=329, h=64):
        super().__init__()
        self.W1 = nn.Linear(n_in, h)
        self.W2 = nn.Linear(h, h)
        self.head = nn.Sequential(
            nn.Linear(2 * h, 64), nn.ELU(), nn.Linear(64, 1))

    def forward(self, A_norm, X):
        H = F.elu(A_norm @ X @ self.W1.weight.T + self.W1.bias)
        H = F.elu(A_norm @ H @ self.W2.weight.T + self.W2.bias)
        z = torch.cat([H.mean(1), H.max(1).values], 1)
        return self.head(z).squeeze(-1)


def make_base(emb, struct4):
    """Shared per-residue features (n_res, 324); mutation channel added per batch."""
    return torch.tensor(np.concatenate([emb, struct4], 1).astype(np.float32))


def assemble(base, mut_vals, mut_pos):
    """mut_vals: (B,5) mutation channels; mut_pos: (B,) 0-based positions."""
    B = mut_vals.shape[0]
    X = base.unsqueeze(0).expand(B, -1, -1).clone()          # (B,n_res,324)
    X = torch.cat([X, torch.zeros(B, X.size(1), 5)], 2)      # (B,n_res,329)
    X[torch.arange(B), mut_pos, 324:] = mut_vals
    return X


def evaluate(model, A_norm, base, mut, pos, y, batch=200):
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(mut), batch):
            sl = slice(s, s + batch)
            preds.append(model(A_norm, assemble(base, mut[sl], pos[sl])).numpy())
    return spearmanr(np.concatenate(preds), y).statistic


def main():
    all_rows = []
    for seed in (0, 1, 2):
        all_rows += run_seed(seed)
    out = pd.DataFrame(all_rows, columns=["seed", "epoch", "train_rho", "test_rho"])
    out.to_csv("../results/gcn_extended_ramp.csv", index=False)
    print("wrote ../results/gcn_extended_ramp.csv")


def run_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    df = pd.read_csv(f"{DATA}/tem1_stiffler_modeling_table.csv")
    emb = np.load(f"{DATA}/tem1_esm2_emb.npy").astype(np.float32)
    sf = pd.read_csv(f"{DATA}/tem1_structural_features.csv").sort_values("pos")
    s4 = sf[STRUCT].values.astype(np.float32)
    assert len(s4) == emb.shape[0], "struct/residue count mismatch"
    A_norm = build_graph(f"{DATA}/AF-P62593-F1-model.pdb")
    fold = df["pos"].values % 5
    dtr = df[fold != 0].reset_index(drop=True)
    dte = df[fold == 0].reset_index(drop=True)
    base = make_base(emb, s4)
    mut_tr = torch.tensor(dtr[MUTCH].values, dtype=torch.float32)
    mut_te = torch.tensor(dte[MUTCH].values, dtype=torch.float32)
    pos_tr = torch.tensor(dtr["pos"].values - 1)
    pos_te = torch.tensor(dte["pos"].values - 1)
    ytr = torch.tensor(dtr["DMS_score"].values, dtype=torch.float32)
    ytr_np, yte_np = dtr["DMS_score"].values, dte["DMS_score"].values
    model = GCN()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lr = 2e-3
    rows = []
    for ep in range(1, 61):
        if ep in (11, 21, 31, 41, 51):
            lr *= 0.6
            for g in opt.param_groups:
                g["lr"] = lr
        model.train()
        perm = torch.randperm(len(dtr))
        for s in range(0, len(perm), 64):
            b = perm[s:s + 64]
            loss = F.mse_loss(model(A_norm, assemble(base, mut_tr[b], pos_tr[b])),
                              ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep in (5, 15, 25, 35, 45, 60):
            r_tr = evaluate(model, A_norm, base, mut_tr, pos_tr, ytr_np)
            r_te = evaluate(model, A_norm, base, mut_te, pos_te, yte_np)
            rows.append((seed, ep, r_tr, r_te))
            print(f"seed {seed} epoch {ep:2d}: train rho {r_tr:.3f} | test rho {r_te:.3f}",
                  flush=True)
    return rows


if __name__ == "__main__":
    main()
