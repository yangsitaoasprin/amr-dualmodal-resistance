#!/usr/bin/env python3
"""25_dimer_burial_boltz.py — Item 3a: full-length dimer-aware burial for AAC(3)-I.

Boltz-2 (complex mode, local boltz2_conf.ckpt, single-sequence MSA, seed 0)
predicted the full-length AAC(3)-I homodimer (P23181 x2, 177 aa/chain).
Unlike the 1BO4 crystal (40 disordered residues), every DMS position is
covered. Recomputes Cbeta contact numbers from the dimer and re-runs the
structural ablations, testing whether interface burial was mislabelled as
'surface' in the monomeric AlphaFold model.

Input : data/boltz_out/.../boltz_aac3i_dimer_model_0.pdb
Output: results/boltz_dimer_aac3i.csv
"""
import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import spearmanr

ROOT = os.path.join(os.path.dirname(__file__), "..")
PDB = os.path.join(ROOT, "data", "boltz_out",
                   "boltz_results_boltz_aac3i_dimer", "predictions",
                   "boltz_aac3i_dimer", "boltz_aac3i_dimer_model_0.pdb")
FEATS_SEQ = ["esm2", "blosum62", "d_hydro", "d_vol", "d_chg"]


def chain_coords(structure, cid):
    st = PDBParser(QUIET=True).get_structure("d", PDB)[0]
    out = {}
    for res in st[cid]:
        if res.id[0] != " ":
            continue
        coord = res["CB"].get_coord() if "CB" in res else res["CA"].get_coord()
        out[res.id[1]] = coord
    return out


def cv_eval(df, feats, k=5):
    fold = df["pos"].values % k
    rhos = []
    for f in range(k):
        tr, te = fold != f, fold == f
        m = GradientBoostingRegressor(random_state=0)
        m.fit(df.loc[tr, feats], df.loc[tr, "DMS_score"])
        rhos.append(spearmanr(m.predict(df.loc[te, feats]),
                              df.loc[te, "DMS_score"]).statistic)
    return np.array(rhos)


def main():
    A = chain_coords(None, "A")
    B = chain_coords(None, "B")
    idxA, idxB = sorted(A), sorted(B)
    ca = np.array([A[i] for i in idxA])
    cb = np.array([B[i] for i in idxB])
    dAA = np.linalg.norm(ca[:, None] - ca[None, :], axis=-1)
    dAB = np.linalg.norm(ca[:, None] - cb[None, :], axis=-1)

    mono = {idxA[k]: int(((dAA[k] > 0) & (dAA[k] < 10)).sum()) for k in range(len(idxA))}
    cross = {idxA[k]: int((dAB[k] < 10).sum()) for k in range(len(idxA))}
    dimer = {i: mono[i] + cross[i] for i in mono}

    # sanity: sequence numbering is 1..177 = UniProt P23181 (Boltz starts at 1)
    wt_seq = "".join(open(os.path.join(ROOT, "data", "P23181.fasta")).read().split("\n")[1:])
    print(f"Boltz dimer: {len(idxA)}+{len(idxB)} residues; "
          f"UniProt length {len(wt_seq)}; numbering 1..{max(idxA)}")
    interface = [i for i in idxA if cross[i] >= 5]
    print(f"interface residues (cross-chain contacts >= 5): {len(interface)} "
          f"-> {interface[:25]}")
    print(f"mean cross-chain contacts per residue: {np.mean(list(cross.values())):.2f}")

    table = pd.read_csv(os.path.join(ROOT, "data", "aac3i_modeling_table.csv"))
    assert set(table["pos"]).issubset(set(mono)), "DMS positions not covered"

    rows = []
    for label, cmap in [("AlphaFold monomer (published)", None),
                        ("Boltz-2 monomer (chain A only)", mono),
                        ("Boltz-2 dimer (chain A + B)", dimer)]:
        d = table.copy()
        if cmap is not None:
            d["contact_num"] = d["pos"].map(cmap)
        r1 = cv_eval(d, FEATS_SEQ)
        r2 = cv_eval(d, ["pLDDT", "contact_num"])
        r3 = cv_eval(d, FEATS_SEQ + ["pLDDT", "contact_num"])
        rows.append(dict(config=label, seq=r1.mean(), struct=r2.mean(),
                         struct_sd=r2.std(), dual=r3.mean(), dual_sd=r3.std(),
                         gain=r3.mean() - r1.mean()))
        print(f"{label:36s} seq {r1.mean():.3f}  struct {r2.mean():.3f} "
              f"+/-{r2.std():.3f}  dual {r3.mean():.3f} +/-{r3.std():.3f}  "
              f"gain {r3.mean()-r1.mean():+.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results",
                                           "boltz_dimer_aac3i.csv"), index=False)
    print("saved -> results/boltz_dimer_aac3i.csv")


if __name__ == "__main__":
    main()
