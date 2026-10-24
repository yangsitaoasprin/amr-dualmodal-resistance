#!/usr/bin/env python3
"""
gat_extended.py — Continue the naive GAT (GATv2-style, pure PyTorch) budget
ramp on TEM-1 fold 0 out to 60 epochs, to complete Figure 3a.

Architecture follows the paper Methods: two GATv2-style attention layers
(hidden 32, 2 heads), mean+max global readout, MLP regression head.
Node features: [320 ESM-2 | 4 structural | 5 mutation channel at mutated node].
Same modulo by-position split, fold 0 = test. Checkpoints at
3,6,10,15,20,25,30,35,45,60 epochs; Adam lr 2e-3 with 0.6 decay every 10 epochs.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

torch.manual_seed(0)
np.random.seed(0)

BASE = "/mnt/agents/output/amr_project/data"
MUTCH = ["esm2", "d_hydro", "d_vol", "d_chg", "blosum62"]
STRUCT = ["pLDDT", "contact_num", "d_s70", "d_omega"]

# ---- graph edge_index (same <10 A Cbeta graph as 03_gcnv2.py) --------------
def build_edges(pdb_path, cutoff=10.0):
    from Bio.PDB import PDBParser
    st = PDBParser(QUIET=True).get_structure("af", pdb_path)
    chain = next(st[0].get_chains())
    coords = []
    for res in chain:
        if res.id[0] == " ":
            coords.append(res["CB"].get_coord() if "CB" in res else res["CA"].get_coord())
    X = np.array(coords)
    dm = np.linalg.norm(X[:, None] - X[None, :], axis=-1)
    n = len(X)
    A = (dm < cutoff)
    A[np.arange(n), np.arange(n)] = True          # self loops
    src, dst = np.where(A)
    return torch.tensor(src), torch.tensor(dst), n

class GATLayer(nn.Module):
    """Dense masked GAT attention (matmul message passing, CPU-friendly).
    e_ij = LeakyReLU(a_s.Wh_i + a_d.Wh_j), softmax over graph neighbours."""
    def __init__(self, n_in, n_out, heads=2):
        super().__init__()
        self.heads, self.n_out = heads, n_out
        self.W = nn.Linear(n_in, heads * n_out, bias=False)
        self.a_s = nn.Parameter(torch.empty(heads, n_out))
        self.a_d = nn.Parameter(torch.empty(heads, n_out))
        nn.init.xavier_uniform_(self.a_s)
        nn.init.xavier_uniform_(self.a_d)
        self.bias = nn.Parameter(torch.zeros(heads * n_out))

    def forward(self, H, mask):
        # H: (B,N,F); mask: (1,1,N,N) additive 0/-1e9
        B, N, _ = H.shape
        Wh = self.W(H).view(B, N, self.heads, self.n_out)
        s = (Wh * self.a_s).sum(-1).permute(0, 2, 1)        # (B,heads,N)
        d = (Wh * self.a_d).sum(-1).permute(0, 2, 1)
        e = F.leaky_relu(s.unsqueeze(-1) + d.unsqueeze(-2) + mask, 0.2)
        att = torch.softmax(e, dim=-1)                      # (B,heads,N,N)
        out = torch.matmul(att, Wh.permute(0, 2, 1, 3))     # (B,heads,N,nout)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.heads * self.n_out)
        return F.elu(out + self.bias)

class NaiveGAT(nn.Module):
    def __init__(self, n_in=329, h=32, heads=2):
        super().__init__()
        self.g1 = GATLayer(n_in, h, heads)
        self.g2 = GATLayer(h * heads, h, heads)
        self.head = nn.Sequential(nn.Linear(2 * h * heads, 64), nn.ELU(),
                                  nn.Linear(64, 1))

    def forward(self, X, mask):
        H = self.g1(X, mask)
        H = self.g2(H, mask)
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

def main():
    df = pd.read_csv(f"{BASE}/tem1_stiffler_modeling_table.csv")
    emb = np.load(f"{BASE}/tem1_esm2_emb.npy").astype(np.float32)
    sf = pd.read_csv(f"{BASE}/tem1_structural_features.csv").sort_values("pos")
    s4 = sf[STRUCT].values.astype(np.float32)
    src, dst, n = build_edges(f"{BASE}/AF-P62593-F1-model.pdb")
    assert n == emb.shape[0]
    adj = torch.zeros(n, n)
    adj[src, dst] = 1.0
    mask = (1.0 - adj).clamp(min=0) * (-1e9)
    mask = mask.view(1, 1, n, n)
    fold = df["pos"].values % 5
    dtr = df[fold != 0].reset_index(drop=True)
    dte = df[fold == 0].reset_index(drop=True)
    base = make_base(emb, s4)
    mut_tr = torch.tensor(dtr[MUTCH].values, dtype=torch.float32)
    mut_te = torch.tensor(dte[MUTCH].values, dtype=torch.float32)
    pos_tr = torch.tensor(dtr["pos"].values - 1)
    pos_te = torch.tensor(dte["pos"].values - 1)
    ytr = torch.tensor(dtr["DMS_score"].values, dtype=torch.float32)
    yte = dte["DMS_score"].values
    model = NaiveGAT()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = {3, 6, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60}
    lr = 2e-3
    print("epoch,rho", flush=True)
    for ep in range(1, 61):
        if ep in (11, 21, 31, 41, 51):
            lr *= 0.6
            for g in opt.param_groups:
                g["lr"] = lr
        model.train()
        perm = torch.randperm(len(dtr))[:1200]  # naive-GAT CPU budget: 1,200 samples/epoch
        for s in range(0, len(perm), 64):
            b = perm[s:s + 64]
            Xb = assemble(base, mut_tr[b], pos_tr[b])
            loss = F.mse_loss(model(Xb, mask), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep in sched:
            model.eval()
            preds = []
            with torch.no_grad():
                for s in range(0, len(dte), 200):
                    sl = slice(s, s + 200)
                    preds.append(model(assemble(base, mut_te[sl], pos_te[sl]),
                                       mask).numpy())
            pred = np.concatenate(preds)
            print(f"{ep},{spearmanr(pred, yte).statistic:.4f}", flush=True)

if __name__ == "__main__":
    main()
