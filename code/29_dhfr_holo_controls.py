#!/usr/bin/env python3
"""29_dhfr_holo_controls.py -- DHFR holo-anchor control for Table S16d.

Rebuilds the two DHFR rows of Table S16d from the NADPH-bound crystal structure
1RX1 instead of the AlphaFold model, using the same anchor convention as the rest
of the pipeline (code/01_build_features.py): Cbeta distance to the catalytic
residue (Asp27) and the minimum Cbeta distance to the Met20 loop (residues 9-24).

Input : data/1RX1.pdb, data/dhfr_modeling_table.csv
Output: results/dhfr_holo_controls.csv

Run from this directory:  python 29_dhfr_holo_controls.py
"""
import os

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")

CATALYTIC = 27          # Asp27
LOOP = (9, 24)          # Met20 loop, residues 9-24
FEATS_SEQ = ["esm2", "blosum62", "d_hydro", "d_vol", "d_chg"]
FEATS_STR = ["pLDDT", "contact_num", "d_s70", "d_omega"]


def cb_map(path):
    chain = next(PDBParser(QUIET=True).get_structure("x", path)[0].get_chains())
    return {r.id[1]: (r["CB"].get_coord() if "CB" in r else r["CA"].get_coord())
            for r in chain if r.id[0] == " "}


def geometry(cb):
    idx = sorted(cb)
    coords = np.array([cb[i] for i in idx])
    dm = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    d_anchor = {i: float(dm[k, idx.index(CATALYTIC)]) for k, i in enumerate(idx)}
    d_loop = {i: float(min(dm[k, idx.index(j)] for j in range(*LOOP) if j in idx))
              for k, i in enumerate(idx)}
    contacts = {i: int(((dm[k] > 0) & (dm[k] < 10)).sum()) for k, i in enumerate(idx)}
    return d_anchor, d_loop, contacts


def cv(df, feats, k=5):
    fold = df["pos"].values % k
    out = []
    for f in range(k):
        tr, te = fold != f, fold == f
        m = GradientBoostingRegressor(random_state=0).fit(df.loc[tr, feats], df.loc[tr, "DMS_score"])
        out.append(spearmanr(m.predict(df.loc[te, feats]), df.loc[te, "DMS_score"]).statistic)
    return np.array(out)


def main():
    d_s, d_l, cn = geometry(cb_map(os.path.join(ROOT, "data", "1RX1.pdb")))
    tab = pd.read_csv(os.path.join(ROOT, "data", "dhfr_modeling_table.csv"))

    rows = []
    for label, use_crystal_burial in (("holo anchors (AlphaFold burial)", False),
                                      ("holo anchors + crystal burial", True)):
        d = tab.copy()
        d["d_s70"] = d["pos"].map(d_s)
        d["d_omega"] = d["pos"].map(d_l)
        if use_crystal_burial:
            d["contact_num"] = d["pos"].map(cn)
        seq = cv(d, FEATS_SEQ)
        struct = cv(d, FEATS_STR)
        dual = cv(d, FEATS_SEQ + FEATS_STR)
        rows.append(dict(config=label, seq=seq.mean(), struct=struct.mean(),
                         struct_sd=struct.std(), dual=dual.mean(), dual_sd=dual.std(),
                         gain=dual.mean() - seq.mean()))
        print("%-32s seq %.3f | struct %.3f +/- %.3f | dual %.3f +/- %.3f | gain %+.3f"
              % (label, seq.mean(), struct.mean(), struct.std(),
                 dual.mean(), dual.std(), dual.mean() - seq.mean()))

    out = os.path.join(RES, "dhfr_holo_controls.csv")
    pd.DataFrame(rows).round(4).to_csv(out, index=False)
    print("saved ->", os.path.relpath(out, HERE))


if __name__ == "__main__":
    main()
