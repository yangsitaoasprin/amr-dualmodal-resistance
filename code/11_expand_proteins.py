#!/usr/bin/env python3
"""
11_expand_proteins.py — Build modelling tables for the two expansion proteins,
APH(3')-IIa (KKA2_KLEPN_Melnikov_2014, kanamycin) and DHFR
(DYR_ECOLI_Nguyen_2023, trimethoprim target), following the identical
alignment-free pipeline used for TEM-1/VIM-2/AAC(3)-I:

  features per variant = ESM-2 8M masked-marginal score (Eq. 1) + BLOSUM62 +
  d_hydro/d_vol/d_chg + pLDDT + contact_num + d_s70 + d_omega
  (d_s70/d_omega keep the TEM-1-style column aliases used by every downstream
   script; here they mean distance to this protein's catalytic anchor and to
   its functional loop, not the literal TEM-1 Ser70 / omega-loop.)

Functional anchors (literature-based):
  APH(3')-IIa: catalytic base Asp190 (Nurizzo et al. 2003, PDB 1ND4);
               aminoglycoside-binding acidic loop, residues 151-166.
  DHFR:        catalytic Asp27; Met20 loop, residues 9-24.

Inputs:  ProteinGym DMS CSVs + AlphaFold PDBs in data/
Outputs: data/aph3ii_modeling_table.csv, data/dhfr_modeling_table.csv
"""
import math
import os
import re
from collections import OrderedDict

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

# BLOSUM62 (order: A R N D C Q E G H I L K M F P S T W Y V)
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
HYD = dict(zip(_aa, [1.8,-4.5,-3.5,-3.5,2.5,-3.5,-3.5,-0.4,-3.2,4.5,3.8,-3.9,1.9,2.8,-1.6,-0.8,-0.7,4.2,-0.9,-1.3]))
VOL = dict(zip(_aa, [88.6,173.4,114.1,111.1,108.5,143.8,138.4,60.1,153.2,166.7,166.7,168.6,162.9,189.9,112.7,89.0,116.1,227.8,193.6,140.0]))
CHG = {**{a: 0 for a in _aa}, "R": 1, "K": 1, "D": -1, "E": -1, "H": 0.1}

AA3 = {"ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H",
       "ILE":"I","LYS":"K","LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q",
       "ARG":"R","SER":"S","THR":"T","VAL":"V","TRP":"W","TYR":"Y"}

PROTEINS = {
    "APH3II_Melnikov2014": dict(
        dms="KKA2_KLEPN_Melnikov_2014.csv", pdb="AF-P00552.pdb",
        anchor=190, loop=range(151, 167)),
    "DHFR_Nguyen2023": dict(
        dms="DYR_ECOLI_Nguyen_2023.csv", pdb="AF-P0ABQ4.pdb",
        anchor=27, loop=range(9, 25)),
}


def parse_pdb(fn):
    res = OrderedDict()
    for l in open(fn):
        if l.startswith("ATOM"):
            n = int(l[22:26])
            if l[12:16].strip() == "CA":
                res.setdefault(n, {})["aa"] = AA3[l[17:20]]
            res.setdefault(n, {}).setdefault("xyz", []).append(
                (float(l[30:38]), float(l[38:46]), float(l[46:54]),
                 l[12:16].strip(), float(l[60:66])))
    return res


def cb_xyz(atoms):
    for x, y, z, name, _ in atoms:
        if name == "CB":
            return (x, y, z)
    for x, y, z, name, _ in atoms:
        if name == "CA":
            return (x, y, z)


def structural(res):
    out = {}
    for n, a in res.items():
        cb = cb_xyz(a["xyz"])
        pld = np.mean([p for *_, p in a["xyz"]])
        cnt = 0
        for m, b in res.items():
            if m == n:
                continue
            x2, y2, z2 = cb_xyz(b["xyz"])
            if math.dist(cb, (x2, y2, z2)) < 10.0:
                cnt += 1
        out[n] = (pld, cnt, cb)
    return out


def esm2_scores(wt, positions):
    import torch
    import esm
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    bc = alphabet.get_batch_converter()
    scores = {}
    with torch.no_grad():
        for i in positions:
            masked = wt[: i - 1] + alphabet.mask_tok_str + wt[i:]
            _, _, toks = bc([("p", masked)])
            logits = model(toks)["logits"]
            lp = torch.log_softmax(logits[0, i], dim=-1)  # BOS offset
            scores[i] = {a: float(lp[alphabet.get_idx(a)]) for a in _aa}
    return scores


def build(name, cfg):
    res = parse_pdb(os.path.join(DATA, cfg["pdb"]))
    sf = structural(res)
    wt = "".join(res[n]["aa"] for n in sorted(res))
    df = pd.read_csv(os.path.join(DATA, cfg["dms"]))
    # verify every mutant row against the AF sequence
    wt_from_mut = {}
    for _, r in df.iterrows():
        m = re.match(r"([A-Z])(\d+)([A-Z])", r.mutant)
        wt_from_mut[int(m.group(2))] = m.group(1)
    for n, a in res.items():
        if n in wt_from_mut:
            assert wt_from_mut[n] == a["aa"], (name, n)

    positions = sorted({int(re.match(r"[A-Z](\d+)[A-Z]", m).group(1))
                        for m in df.mutant})
    es = esm2_scores(wt, positions)

    anch = sf[cfg["anchor"]][2]
    loop_pts = [sf[n][2] for n in cfg["loop"]]
    rows = []
    for _, r in df.iterrows():
        m = re.match(r"([A-Z])(\d+)([A-Z])", r.mutant)
        w, pos, mu = m.group(1), int(m.group(2)), m.group(3)
        cb = sf[pos][2]
        rows.append(dict(
            mutant=r.mutant, wt=w, pos=pos, mut=mu,
            esm2=es[pos][mu] - es[pos][w],
            blosum62=B62[w][mu],
            d_hydro=HYD[mu] - HYD[w],
            d_vol=VOL[mu] - VOL[w],
            d_chg=CHG[mu] - CHG[w],
            pLDDT=sf[pos][0], contact_num=sf[pos][1],
            d_s70=math.dist(cb, anch),
            d_omega=min(math.dist(cb, lp) for lp in loop_pts),
            DMS_score=r.DMS_score, DMS_score_bin=int(r.DMS_score_bin)))
    out = pd.DataFrame(rows)
    path = os.path.join(DATA, name + ".csv")
    out.to_csv(path, index=False)
    print(name, out.shape, "mean pLDDT %.1f" % out.pLDDT.mean(), "->", path)


if __name__ == "__main__":
    for nm, cfg in PROTEINS.items():
        out_name = ("aph3ii" if "APH" in nm else "dhfr") + "_modeling_table"
        build(out_name, cfg)
