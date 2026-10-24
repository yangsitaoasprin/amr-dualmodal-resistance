#!/usr/bin/env python3
"""
03_gcnv2.py — Pure-PyTorch implementation of the GCNv2 graph model on the
AlphaFold Cbeta contact graph, with the three diagnosis-driven fixes:
  (1) mutated-node skip connection (against signal dilution)
  (2) 9-dim tabular-feature passthrough to the head (against redundant re-learning)
  (3) embedding-channel dropout p=0.3 + readout dropout (against position memorization)

Node features (per sample, n_res x 329):
  [0:320]   ESM-2 per-residue embedding (same for every sample)
  [320:324] 4 structural channels per residue (pLDDT, contact_num, d_s70, d_omega)
  [324:329] 5-dim mutation channel, non-zero only at the mutated node
            (esm2, d_hydro, d_vol, d_chg, blosum62)

Protocol per fold: 30 epochs = 10 @ lr 2e-3 + 10 @ 1e-3 + 10 @ 5e-4,
batch 64, full training split; identical modulo 5-fold by-position CV as the GBM.
Reproduces Tables 5-6 and Figure 3b. CPU-only, ~1 min per fold.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, ttest_rel
from sklearn.ensemble import GradientBoostingRegressor

torch.manual_seed(0)
np.random.seed(0)

TAB_TEM1 = ["esm2", "blosum62", "d_hydro", "d_vol", "d_chg",
            "pLDDT", "contact_num", "d_s70", "d_omega"]
TAB_VIM2 = ["esm2", "blosum62", "d_hydro", "d_vol", "d_chg",
            "pLDDT", "contact_num", "d_act", "d_loop"]
STRUCT_TEM1 = ["pLDDT", "contact_num", "d_s70", "d_omega"]
STRUCT_VIM2 = ["pLDDT", "contact_num", "d_act", "d_loop"]
MUTCH = ["esm2", "d_hydro", "d_vol", "d_chg", "blosum62"]


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


class GCNv2(nn.Module):
    def __init__(self, n_in=329, n_tab=9, h=64, p_emb=0.3, p_out=0.3,
                 emb_dim=320):
        super().__init__()
        self.W1 = nn.Linear(n_in, h)
        self.W2 = nn.Linear(h, h)
        # readout: mean + max + mutated-node state + 9 tabular features
        self.head = nn.Sequential(
            nn.Linear(3 * h + n_tab, 64), nn.ELU(), nn.Dropout(p_out),
            nn.Linear(64, 1))
        self.p_emb = p_emb
        self.emb_dim = emb_dim

    def forward(self, A_norm, X, mut_idx, tab):
        # fix (3): embedding-channel dropout (inverted), train-time only
        if self.training:
            mask = (torch.rand(X.size(0), 1, self.emb_dim, device=X.device)
                    > self.p_emb).float() / (1 - self.p_emb)
            X = torch.cat([X[..., :self.emb_dim] * mask, X[..., self.emb_dim:]], 2)
        H = F.elu(A_norm @ X @ self.W1.weight.T + self.W1.bias)
        H = F.elu(A_norm @ H @ self.W2.weight.T + self.W2.bias)
        mean = H.mean(1)
        mx = H.max(1).values
        mut = H[torch.arange(H.size(0)), mut_idx]     # fix (1): skip connection
        z = torch.cat([mean, mx, mut, tab], 1)        # fix (2): passthrough
        return self.head(z).squeeze(-1)


def make_batch(df, emb, struct4):
    """emb: (n_res, 320); struct4: (n_res, 4). Returns (B, n_res, 329)."""
    B, n_res = len(df), emb.shape[0]
    base = np.concatenate([emb, struct4], 1).astype(np.float32)     # (n_res, 324)
    X = np.repeat(base[None], B, 0)                                 # (B, n_res, 324)
    X = np.concatenate([X, np.zeros((B, n_res, 5), np.float32)], 2) # (B, n_res, 329)
    for i, (_, r) in enumerate(df.iterrows()):
        X[i, int(r["pos"]) - 1, 324:] = r[MUTCH].values.astype(np.float32)
    return torch.tensor(X)


def train_fold(A_norm, df_tr, df_te, emb, struct4, tab_cols):
    Xtr, Xte = make_batch(df_tr, emb, struct4), make_batch(df_te, emb, struct4)
    tab_tr = torch.tensor(df_tr[tab_cols].values, dtype=torch.float32)
    tab_te = torch.tensor(df_te[tab_cols].values, dtype=torch.float32)
    ytr = torch.tensor(df_tr["DMS_score"].values, dtype=torch.float32)
    midx_tr = torch.tensor(df_tr["pos"].values - 1)
    midx_te = torch.tensor(df_te["pos"].values - 1)
    model = GCNv2()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for lr in (2e-3, 1e-3, 5e-4):
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        for _ in range(10):
            perm = torch.randperm(len(Xtr))
            for s in range(0, len(perm), 64):
                b = perm[s:s + 64]
                loss = F.mse_loss(model(A_norm, Xtr[b], midx_tr[b], tab_tr[b]),
                                  ytr[b])
                opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(A_norm, Xte, midx_te, tab_te).numpy()
    return spearmanr(pred, df_te["DMS_score"]).statistic


def run(csv, emb_npy, pdb, tab_cols, struct_cols, label, struct_csv=None):
    df = pd.read_csv(csv)
    emb = np.load(emb_npy).astype(np.float32)
    if struct_csv is not None:
        # full-length per-residue structural features (covers residues absent
        # from the DMS assay, e.g. the TEM-1 signal peptide 1-23)
        sf = pd.read_csv(struct_csv).sort_values("pos")
        s4 = sf[struct_cols].values.astype(np.float32)
    else:
        s4 = (df.drop_duplicates("pos").sort_values("pos")[struct_cols]
              .values.astype(np.float32))
    assert len(s4) == emb.shape[0], "struct/residue count mismatch"
    A_norm = build_graph(pdb)
    fold = df["pos"].values % 5
    g_rhos, b_rhos = [], []
    for f in range(5):
        dtr = df[fold != f].reset_index(drop=True)
        dte = df[fold == f].reset_index(drop=True)
        r = train_fold(A_norm, dtr, dte, emb, s4, tab_cols)
        gbm = GradientBoostingRegressor(random_state=0)
        gbm.fit(dtr[tab_cols], dtr["DMS_score"])
        b = spearmanr(gbm.predict(dte[tab_cols]), dte["DMS_score"]).statistic
        g_rhos.append(r); b_rhos.append(b)
        print(f"  {label} fold {f}: GCNv2 {r:.3f} | GBM {b:.3f}", flush=True)
    g, b = np.array(g_rhos), np.array(b_rhos)
    t = ttest_rel(g, b)
    print(f"{label}: GCNv2 {g.mean():.3f} +/- {g.std():.3f} | "
          f"GBM {b.mean():.3f} +/- {b.std():.3f} | paired t p = {t.pvalue:.3f}")
    return g, b


if __name__ == "__main__":
    run("../data/tem1_stiffler_modeling_table.csv", "../data/tem1_esm2_emb.npy",
        "../data/AF-P62593-F1-model.pdb", TAB_TEM1, STRUCT_TEM1, "TEM-1",
        struct_csv="../data/tem1_structural_features.csv")
    run("../data/vim2_modeling_table.csv", "../data/vim2_esm2_emb.npy",
        "../data/AF-A4GRB6-F1-model.pdb", TAB_VIM2, STRUCT_VIM2, "VIM-2")
