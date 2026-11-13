#!/usr/bin/env python3
"""
12_new_proteins_struct_features.py — Structural features + physicochemical
features for the two expansion proteins (no pLM scores; those are added by
09/13 scripts). Produces partial modelling tables that 13_merge_new_proteins.py
completes with esm2 scores.

Anchors (literature):
  APH(3')-IIa (P00552, 264 aa): catalytic base Asp190 (Nurizzo et al. 2003,
      PDB 1ND4); aminoglycoside-binding acidic loop residues 151-166.
  DHFR (P0ABQ4, 159 aa): catalytic Asp27; Met20 loop residues 9-24.
"""
import math
import os
import re
from collections import OrderedDict

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

BLOS = """A 4 -1 -2 -2 0 -1 -1 0 -2 -1 -1 -1 -1 -2 -1 1 0 -3 -2 0
R -1 5 0 -2 -3 1 0 -2 0 -3 -2 2 -1 -3 -2 -1 -1 -3 -2 -3
N -2 0 6 1 -3 0 0 0 1 -3 -3 0 -2 -3 -2 1 0 -4 -2 -3
D -2 -2 1 6 -3 0 2 -1 -1 -3 -4 -1 -3 -3 -1 0 -1 -4 -3 -3
C 0 -3 -3 -3 9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1 1 0 0 -3 5 2 -2 0 -3 -2 1 0 -3 -1 0 -1 -2 -1 -2
E -1 0 0 2 -4 2 5 -2 0 -3 -3 1 -2 -3 -1 0 -1 -3 -2 -2
G 0 -2 0 -1 -3 -2 -2 6 -2 -4 -4 -2 -3 -3 -2 0 -2 -2 -3 -3
H -2 0 1 -1 -3 0 0 -2 8 -3 -3 -1 -2 -1 -2 -1 -2 -2 2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3 4 2 -3 1 0 -3 -2 -1 -3 -1 3
L -1 -2 -3 -4 -1 -2 -3 -4 -3 2 4 -2 2 0 -3 -2 -1 -2 -1 1
K -1 2 0 -1 -3 1 1 -2 -1 -3 -2 5 -1 -3 -1 0 -1 -3 -2 -2
M -1 -1 -2 -3 -1 0 -2 -3 -2 1 2 -1 5 0 -2 -1 -1 -1 -1 1
F -2 -3 -3 -3 -2 -3 -3 -3 -1 0 0 -3 0 6 -4 -2 -2 1 3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4 7 -1 -1 -4 -3 -2
S 1 -1 1 0 -1 0 0 0 -1 -2 -2 0 -1 -2 -1 4 1 -3 -2 -2
T 0 -1 0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1 1 5 -2 -2 0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1 1 -4 -3 -2 11 2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3 2 -1 -1 -2 -1 3 -3 -2 -2 2 7 -1
V 0 -3 -3 -3 -1 -2 -2 -3 -3 3 1 -2 1 -1 -2 -2 0 -3 -1 4"""
_aa = "ARNDCQEGHILKMFPSTWYV"
B62 = {}
for _l in BLOS.strip().split("\n"):
    p = _l.split()
    B62[p[0]] = dict(zip(_aa, map(int, p[1:])))
HYD = dict(zip(_aa, [1.8, -4.5, -3.5, -3.5, 2.5, -3.5, -3.5, -0.4, -3.2,
                     4.5, 3.8, -3.9, 1.9, 2.8, -1.6, -0.8, -0.7, 4.2, -0.9,
                     -1.3]))
VOL = dict(zip(_aa, [88.6, 173.4, 114.1, 111.1, 108.5, 143.8, 138.4, 60.1,
                     153.2, 166.7, 166.7, 168.6, 162.9, 189.9, 112.7, 89.0,
                     116.1, 227.8, 193.6, 140.0]))
CHG = {**{a: 0 for a in _aa}, "R": 1, "K": 1, "D": -1, "E": -1, "H": 0.1}
AA3 = {"ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
       "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
       "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
       "TRP": "W", "TYR": "Y"}

PROTEINS = {
    "aph3ii": dict(dms="KKA2_KLEPN_Melnikov_2014.csv", pdb="AF-P00552.pdb",
                   anchor=190, loop=range(151, 167)),
    "dhfr": dict(dms="DYR_ECOLI_Nguyen_2023.csv", pdb="AF-P0ABQ4.pdb",
                 anchor=27, loop=range(9, 25)),
}


def parse_pdb(fn):
    res = OrderedDict()
    for l in open(fn):
        if l.startswith("ATOM"):
            n = int(l[22:26])
            res.setdefault(n, {"atoms": []})
            if l[12:16].strip() == "CA":
                res[n]["aa"] = AA3[l[17:20]]
            res[n]["atoms"].append((float(l[30:38]), float(l[38:46]),
                                    float(l[46:54]), l[12:16].strip(),
                                    float(l[60:66])))
    return res


def cb(atoms):
    for x, y, z, nm, _ in atoms:
        if nm == "CB":
            return (x, y, z)
    for x, y, z, nm, _ in atoms:
        if nm == "CA":
            return (x, y, z)


def main():
    for name, cfg in PROTEINS.items():
        res = parse_pdb(os.path.join(DATA, cfg["pdb"]))
        wt = "".join(res[n]["aa"] for n in sorted(res))
        xyz = {n: cb(a["atoms"]) for n, a in res.items()}
        pld = {n: float(np.mean([p for *_, p in a["atoms"]]))
               for n, a in res.items()}
        df = pd.read_csv(os.path.join(DATA, cfg["dms"]))
        # verify WT reconstruction against AF sequence
        wtm = {}
        for m in df.mutant:
            g = re.match(r"([A-Z])(\d+)([A-Z])", m)
            wtm[int(g.group(2))] = g.group(1)
        for n, a in res.items():
            if n in wtm:
                assert wtm[n] == a["aa"], (name, n)
        anch = xyz[cfg["anchor"]]
        loop_pts = [xyz[n] for n in cfg["loop"] if n in xyz]
        # contact numbers
        keys = sorted(xyz)
        cbs = np.array([xyz[k] for k in keys])
        d2 = ((cbs[:, None, :] - cbs[None, :, :]) ** 2).sum(-1)
        contacts = dict(zip(keys, ((d2 < 100.0).sum(1) - 1)))
        rows = []
        for _, r in df.iterrows():
            g = re.match(r"([A-Z])(\d+)([A-Z])", r.mutant)
            w, pos, mu = g.group(1), int(g.group(2)), g.group(3)
            p = xyz[pos]
            rows.append(dict(
                mutant=r.mutant, wt=w, pos=pos, mut=mu,
                blosum62=B62[w][mu], d_hydro=HYD[mu] - HYD[w],
                d_vol=VOL[mu] - VOL[w], d_chg=CHG[mu] - CHG[w],
                pLDDT=pld[pos], contact_num=int(contacts[pos]),
                d_anchor=math.dist(p, anch),
                d_loop=min(math.dist(p, lp) for lp in loop_pts),
                DMS_score=r.DMS_score, DMS_score_bin=int(r.DMS_score_bin)))
        out = pd.DataFrame(rows)
        path = os.path.join(DATA, name + "_partial_table.csv")
        out.to_csv(path, index=False)
        print(name, out.shape, "WT len", len(wt), "mean pLDDT %.1f" % out.pLDDT.mean())
        open(os.path.join(DATA, name + "_wt.fasta"), "w").write(
            ">" + name + "\n" + wt + "\n")


if __name__ == "__main__":
    main()
