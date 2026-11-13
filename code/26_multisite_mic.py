#!/usr/bin/env python3
"""26_multisite_mic.py — Item 3b: unseen multi-substitution genotypes and
clinical MIC endpoints.

Data: Knies, Cai & Weinreich (2017) Mol Biol Evol 34(5):1040-1054, Table 2 —
cefotaxime MIC (ug/mL) for all 2^4 combinations of four TEM-1 clinical
substitutions (A42G, E104K, M182T, G238S; Ambler numbering), including the
clinically named alleles TEM-19/20/15/52 and the quadruple mutant. MIC assays
were performed in triplicate (identical values); EUCAST cefotaxime breakpoints
for Enterobacterales are S <= 1 and R > 2 ug/mL (v14.0), so all 16 alleles are
clinically resistant and the binary breakpoint is saturated in this panel.

Features for multi-mutants (explicit, documented extensions of the 9-D
single-substitution space):
  * esm2 : sum of masked-marginal log-odds at each mutated position, evaluated
           in the full mutant context (all other substitutions present);
  * physicochemical: additive sums of the four per-substitution deltas;
  * structural: mean of per-position values over mutated positions (for the
           wild type, mean over the four candidate positions).
Both the corrected-anchor (catalytic Ser = precursor 68, Omega-loop 162-177)
and the released (offset) anchor definitions are evaluated.

Outputs: results/multisite_clinical_mic.csv
Requires: boltz_cuda env (fair-esm 2.0.0, torch, sklearn, biopython).
"""
import os
import numpy as np
import pandas as pd
import torch
import esm
from Bio.PDB import PDBParser
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.ensemble import GradientBoostingRegressor

ROOT = os.path.join(os.path.dirname(__file__), "..")
D = os.path.join(ROOT, "data")
FEATS_SEQ = ["esm2", "blosum62", "d_hydro", "d_vol", "d_chg"]
SSTR = ["pLDDT", "contact_num", "d_s70", "d_omega"]

WT_SEQ = "".join(open(os.path.join(D, "P62593_TEM1.fasta")).read().split("\n")[1:])
# Ambler (ABL) -> precursor: ABL = precursor + 2 for ABL <= 238
SUBS = [("A42G", 42, "A", "G"), ("E104K", 104, "E", "K"),
        ("M182T", 182, "M", "T"), ("G238S", 238, "G", "S")]
SUBS = [(name, abl - 2, w, m) for name, abl, w, m in SUBS]
for name, p, w, m in SUBS:
    assert WT_SEQ[p - 1] == w, (name, p, WT_SEQ[p - 1])
print("ABL->precursor mapping:", [(n, abl, p, w, m) for (n, abl, w, m), (_, p, _, _) in zip(
    [("A42G", 42, "A", "G"), ("E104K", 104, "E", "K"),
     ("M182T", 182, "M", "T"), ("G238S", 238, "G", "S")], SUBS)])

# Table 2 of Knies et al. 2017 (cefotaxime MIC, ug/mL), genotype bits in
# A42G / E104K / M182T / G238S order.
ALLELES = [
    ("0000", "TEM-1",      5.7),
    ("0001", "TEM-19",   260.0),
    ("0010", "TEM-135",    8.0),
    ("0011", "TEM-20",   260.0),
    ("0100", "TEM-17",    11.0),
    ("0101", "TEM-15",  2000.0),
    ("0110", "TEM-106",   11.0),
    ("0111", "TEM-52",  4100.0),
    ("1000", "None",       5.7),
    ("1001", "None",     720.0),
    ("1010", "None",       4.0),
    ("1011", "None",     720.0),
    ("1100", "None",      32.0),
    ("1101", "None",     290.0),
    ("1110", "None",      23.0),
    ("1111", "None",    8200.0),
]

# ------------------------------------------------------------ structural features
def structural_features(anchor, lo, hi, cutoff=10.0):
    st = PDBParser(QUIET=True).get_structure("af", os.path.join(D, "AF-P62593-F1-model.pdb"))
    plddt, cb = {}, {}
    for res in next(st[0].get_chains()):
        if res.id[0] != " ":
            continue
        i = res.id[1]
        plddt[i] = res["CA"].get_bfactor()
        cb[i] = res["CB"].get_coord() if "CB" in res else res["CA"].get_coord()
    idx = sorted(cb)
    coords = np.array([cb[i] for i in idx])
    dm = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    rows = {}
    for k, i in enumerate(idx):
        d = dict(pLDDT=plddt[i],
                 contact_num=int(((dm[k] > 0) & (dm[k] < cutoff)).sum()))
        d["d_s70"] = float(dm[k, idx.index(anchor)])
        d["d_omega"] = float(min(dm[k, idx.index(j)] for j in range(lo, hi + 1)
                                 if j in idx))
        rows[i] = d
    return rows

# ------------------------------------------------------------ ESM-2 multi-site
print("loading ESM-2 8M ...")
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
model.eval()
bc = alphabet.get_batch_converter()


def multisite_esm(muts):
    """Sum masked-marginal log-odds over mutated positions in mutant context."""
    if not muts:
        return 0.0
    ctx = list(WT_SEQ)
    for p, w, m in muts:
        ctx[p - 1] = m
    _, _, toks = bc([("x", "".join(ctx))])
    total = 0.0
    with torch.no_grad():
        for p, w, m in muts:
            t = toks.clone()
            t[0, p] = alphabet.mask_idx
            lp = torch.log_softmax(model(t)["logits"][0, p], -1)
            total += float(lp[alphabet.get_idx(m)] - lp[alphabet.get_idx(w)])
    return total


def main():
    base = pd.read_csv(os.path.join(D, "tem1_stiffler_modeling_table.csv"))
    single = {row["mutant"]: row for _, row in base.iterrows()}

    # per-substitution physicochemical deltas and additive ESM (WT context)
    per_sub = []
    for name, p, w, m in SUBS:
        r = single[f"{w}{p}{m}"]
        per_sub.append(dict(name=name, pos=p, wt=w, mut=m,
                            blosum62=r["blosum62"], d_hydro=r["d_hydro"],
                            d_vol=r["d_vol"], d_chg=r["d_chg"],
                            esm2_single=r["esm2"]))
    per_sub = pd.DataFrame(per_sub)

    # build 16 genotype feature rows
    genotypes = []
    for bits, allele, mic in ALLELES:
        muts = [(per_sub.iloc[k]["pos"], per_sub.iloc[k]["wt"], per_sub.iloc[k]["mut"])
                for k, b in enumerate(bits) if b == "1"]
        sub = per_sub.iloc[[i for i, b in enumerate(bits) if b == "1"]]
        genotypes.append(dict(bits=bits, allele=allele, MIC=mic,
                              log10MIC=np.log10(mic), n_mut=len(muts),
                              esm2=multisite_esm(muts),
                              esm2_additive=float(sub["esm2_single"].sum()) if len(sub) else 0.0,
                              blosum62=float(sub["blosum62"].sum()) if len(sub) else 0.0,
                              d_hydro=float(sub["d_hydro"].sum()) if len(sub) else 0.0,
                              d_vol=float(sub["d_vol"].sum()) if len(sub) else 0.0,
                              d_chg=float(sub["d_chg"].sum()) if len(sub) else 0.0,
                              mut_positions=[m[0] for m in muts]))
    G = pd.DataFrame(genotypes)
    print("\nESM-2 multi-site vs additive (WT-context) scores:")
    print(G[["allele", "n_mut", "esm2", "esm2_additive"]].to_string(index=False))

    results = []
    for label, anchor, lo, hi in [("corrected anchors (Ser68, loop 162-177)", 68, 162, 177),
                                  ("released anchors (offset)", 70, 163, 178)]:
        srows = structural_features(anchor, lo, hi)
        # train dual-modal GBM on single mutants with this anchor definition
        df = base.copy()
        for col in SSTR:
            df[col] = df["pos"].map({p: srows[p][col] for p in srows})
        gbm = GradientBoostingRegressor(random_state=0)
        gbm.fit(df[FEATS_SEQ + SSTR], df["DMS_score"])

        # structural aggregate for each genotype
        for col in SSTR:
            G[col] = [float(np.mean([srows[p][col] for p in ps]) if ps else
                            np.mean([srows[p][col] for p in per_sub["pos"]]))
                      for ps in G["mut_positions"]]
        pred = gbm.predict(G[FEATS_SEQ + SSTR])
        G[f"pred_{anchor}"] = pred

        rho, p = spearmanr(pred, G["log10MIC"])
        rho15, p15 = spearmanr(pred[1:], G["log10MIC"][1:])   # exclude WT
        rho_n, p_n = spearmanr(pred, G["n_mut"])
        u = mannwhitneyu(pred[G["MIC"] >= 260], pred[G["MIC"] < 260],
                         alternative="less")
        print(f"\n{label}")
        print(f"  pred vs log10(MIC): rho = {rho:.3f} (P = {p:.3f}, n = 16); "
              f"excluding WT rho = {rho15:.3f} (P = {p15:.3f}, n = 15)")
        print(f"  pred vs mutation count: rho = {rho_n:.3f} (P = {p_n:.3f})")
        print(f"  high-MIC (>=260 ug/mL) vs low-MIC: Mann-Whitney P = {u.pvalue:.3f}")
        results.append(dict(anchor=label, rho=rho, p=p, rho_exclWT=rho15,
                            p_exclWT=p15, rho_nmut=rho_n, p_nmut=p_n,
                            p_highMIC=u.pvalue))

    G.to_csv(os.path.join(ROOT, "results", "multisite_clinical_mic.csv"), index=False)
    pd.DataFrame(results).to_csv(os.path.join(ROOT, "results",
                                              "multisite_mic_tests.csv"), index=False)
    print("\nsaved -> results/multisite_clinical_mic.csv, multisite_mic_tests.csv")
    print("EUCAST cefotaxime breakpoints: S <= 1, R > 2 ug/mL; "
          f"panel MIC range {G.MIC.min()}-{G.MIC.max()} -> all 16 resistant "
          "(binary breakpoint saturated)")


if __name__ == "__main__":
    main()
