#!/usr/bin/env python3
"""
19_aac_anchor_experiment.py — Reverse anchor ablation for AAC(3)-I.

Reviewers asked: the APH(3')-IIa vs AAC(3)-I contrast is asymmetric because
anchors were encoded only for APH. Here we encode a structural anchor for
AAC(3)-I and test whether the structural gain appears.

Anchor definition (fully reproducible, no hand-picking):
  - PDB 1BO4 (Wolf et al., 1998, Cell): Serratia marcescens AAC(3)-Ia
    co-crystallised with coenzyme A.
  - CoA-contact residues = any protein heavy atom within 5 A of a CoA atom
    -> {50, 51, 111-113, 117-125, 127, 144, 145, 147, 160}.
  - 1BO4 chain A aligns to P. aeruginosa AAC(3)-I (P23181) at 99.3% identity
    with a 1:1 position mapping, so the same residue numbers apply.
  - d_coa(pos) = minimum Cb distance from residue pos to any pocket residue
    in AF-P23181-F1 (Cb = CA for glycine).

Models (identical protocol to scripts 14/15/16, GBM random_state=0,
modulo 5-fold by-position CV, 8M and 650M backbones):
  seq       : esm2 + 4 physicochemical
  dual      : seq + pLDDT + contact_num            (current AAC treatment)
  dual_anch : seq + pLDDT + contact_num + d_coa    (reverse ablation)

Output: results/aac_anchor_experiment.csv
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
POCKET = [50, 51, 111, 112, 113, 117, 118, 119, 120, 121, 122, 123, 124,
          125, 127, 144, 145, 147, 160]
PHY = ["blosum62", "d_hydro", "d_vol", "d_chg"]


def cb_coords(pdb_path):
    """Cbeta coordinates, falling back to CA for glycine.

    CA is stored unconditionally and CB wins only when present, so the
    returned value honours the documented "Cb distance" definition. (The
    previous version let CA overwrite CB, silently producing CA-CA
    distances: in AF-P23181 every residue lists CA before CB, so the CB
    branch was never reached.)
    """
    ca, cb = {}, {}
    for line in open(pdb_path):
        if not line.startswith("ATOM"):
            continue
        rn = int(line[22:26])
        atom = line[12:16].strip()
        if atom == "CA":
            ca[rn] = np.array([float(line[30:38]), float(line[38:46]),
                               float(line[46:54])])
        elif atom == "CB":
            cb[rn] = np.array([float(line[30:38]), float(line[38:46]),
                               float(line[46:54])])
    return {r: cb.get(r, ca[r]) for r in ca}


def cv(m, F, k=5):
    rs, aucs = [], []
    for f in range(k):
        te = m[m.pos % k == f]
        tr = m[m.pos % k != f]
        g = GradientBoostingRegressor(random_state=0).fit(tr[F], tr.DMS_score)
        p = g.predict(te[F])
        rs.append(spearmanr(p, te.DMS_score).statistic)
        aucs.append(roc_auc_score(te.DMS_score_bin, p))
    return rs, aucs


def main():
    coords = cb_coords(os.path.join(DATA, "AF-P23181-F1-model.pdb"))
    pocket = np.array([coords[r] for r in POCKET])
    d_coa = {r: float(np.min(np.linalg.norm(pocket - c, axis=1)))
             for r, c in coords.items()}

    m = pd.read_csv(os.path.join(DATA, "aac3i_modeling_table.csv"))
    m["d_coa"] = m.pos.map(d_coa)
    assert m.d_coa.notna().all()
    print("d_coa: min %.1f  median %.1f  max %.1f A" %
          (m.d_coa.min(), m.d_coa.median(), m.d_coa.max()))

    z = pd.read_csv(os.path.join(RES, "esm2_650m_zeroshot.csv"))
    z = z[z.assay == "AAC3I_Dandage2018"][["mutant", "esm2_650m"]]
    assert not z.duplicated("mutant").any()
    m650 = m.merge(z, on="mutant", how="inner")

    rows = []
    for tag, mm in [("8m", m), ("650m", m650)]:
        fc = "esm2" if tag == "8m" else "esm2_650m"
        seq = [fc] + PHY
        dual = seq + ["pLDDT", "contact_num"]
        anch = dual + ["d_coa"]
        r_seq, a_seq = cv(mm, seq)
        r_dual, a_dual = cv(mm, dual)
        r_anch, a_anch = cv(mm, anch)
        p1 = ttest_rel(r_dual, r_seq).pvalue
        p2 = ttest_rel(r_anch, r_dual).pvalue
        for model, rs, au in [("seq", r_seq, a_seq),
                              ("dual_noanchors", r_dual, a_dual),
                              ("dual_with_coa_anchor", r_anch, a_anch)]:
            rows.append(dict(assay="AAC3I_Dandage2018", backbone=tag,
                             model=model, rho=np.mean(rs),
                             rho_std=np.std(rs), auc=np.mean(au)))
        print(f"AAC(3)-I {tag}: seq {np.mean(r_seq):.3f} | "
              f"dual {np.mean(r_dual):.3f} (p={p1:.3f}) | "
              f"+d_coa {np.mean(r_anch):.3f} "
              f"(delta vs no-anchor {np.mean(r_anch)-np.mean(r_dual):+.3f}, "
              f"paired p={p2:.3f})")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "aac_anchor_experiment.csv"), index=False)
    print("wrote results/aac_anchor_experiment.csv")


if __name__ == "__main__":
    main()
