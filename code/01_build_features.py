#!/usr/bin/env python3
"""
01_build_features.py — Build the 9-dimensional alignment-free feature table
for a resistance protein from (i) a ProteinGym DMS substitution assay and
(ii) an AlphaFold DB predicted structure.

Output columns:
  mutant, mutated_sequence, DMS_score, DMS_score_bin, wt, pos, mut,
  esm2, blosum62, d_hydro, d_vol, d_chg, pLDDT, contact_num, d_s70, d_omega
(VIM-2 uses d_act/d_loop for the residue-64 and loop-region anchors;
AAC(3)-I lacks catalytic anchors -> structural features are pLDDT, contact_num only.)

Dependencies: fair-esm (esm2_t6_8M_UR50D), biopython, pandas, numpy, requests
"""
import numpy as np
import pandas as pd
import requests
from Bio.PDB import PDBParser
from Bio.Align import substitution_matrices

BLOSUM62 = substitution_matrices.load("BLOSUM62")

HYDRO = dict(A=1.8, C=2.5, D=-3.5, E=-3.5, F=2.8, G=-0.4, H=-3.2, I=4.5,
             K=-3.9, L=3.8, M=1.9, N=-3.5, P=-1.6, Q=-3.5, R=-4.5, S=-0.8,
             T=-0.7, V=4.2, W=-0.9, Y=-1.3)
VOL = dict(A=88.6, C=108.5, D=111.1, E=138.4, F=189.9, G=60.1, H=153.2,
           I=166.7, K=168.6, L=166.7, M=162.9, N=114.1, P=112.7, Q=143.8,
           R=173.4, S=89.0, T=116.1, V=140.0, W=227.8, Y=193.6)
CHG = dict(D=-1, E=-1, K=1, R=1, H=0.1)


# ---------------------------------------------------------------- structure
def download_af_structure(uniprot_id, out_pdb):
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    open(out_pdb, "wb").write(r.content)


def parse_structure(pdb_path):
    """Per-residue pLDDT (B-factor of CA) and Cbeta coordinates."""
    st = PDBParser(QUIET=True).get_structure("af", pdb_path)
    chain = next(st[0].get_chains())
    plddt, cb = {}, {}
    for res in chain:
        if res.id[0] != " ":
            continue
        i = res.id[1]
        plddt[i] = res["CA"].get_bfactor()
        cb[i] = res["CB"].get_coord() if "CB" in res else res["CA"].get_coord()
    return plddt, cb


def structural_features(plddt, cb, catalytic_ser=None, loop_range=None,
                        names=("d_s70", "d_omega"), cutoff=10.0):
    """pLDDT, Cbeta contact number (burial) and, when anchors are given,
    distance to the catalytic serine and to the functional loop."""
    idx = sorted(cb)
    coords = np.array([cb[i] for i in idx])
    dm = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    rows = {}
    for k, i in enumerate(idx):
        d = dict(pLDDT=plddt[i],
                 contact_num=int(((dm[k] > 0) & (dm[k] < cutoff)).sum()))
        if catalytic_ser is not None:
            d[names[0]] = float(dm[k, idx.index(catalytic_ser)])
        if loop_range is not None:
            d[names[1]] = float(min(dm[k, idx.index(j)]
                                    for j in range(*loop_range) if j in idx))
        rows[i] = d
    return rows


# ---------------------------------------------------------------- sequence
class ESM2Scorer:
    """Masked-marginal zero-shot scores + per-residue embeddings from
    ESM-2 (esm2_t6_8M_UR50D), single sequence context, no MSA."""

    def __init__(self, wt_seq, device="cpu"):
        import torch, esm
        self.torch = torch
        self.model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.model = self.model.eval().to(device)
        bc = self.alphabet.get_batch_converter()
        self.toks = bc([("wt", wt_seq)])[2].to(device)
        self.L = len(wt_seq)
        # cache full-sequence log-softmax once (marginals approx. used per site)
        with torch.no_grad():
            self.base = torch.log_softmax(
                self.model(self.toks)["logits"][0], -1)  # (L+2, vocab)

    def score(self, pos1, wt_aa, mut_aa):
        """logP(mut | masked pos) - logP(wt | masked pos), 1-based pos."""
        with self.torch.no_grad():
            t = self.toks.clone()
            t[0, pos1] = self.alphabet.mask_idx  # BOS offset already in pos1
            lp = self.torch.log_softmax(self.model(t)["logits"][0, pos1], -1)
            return float(lp[self.alphabet.get_idx(mut_aa)]
                         - lp[self.alphabet.get_idx(wt_aa)])

    def embeddings(self):
        """Per-residue 320-dim embeddings (final layer), shape (L, 320)."""
        with self.torch.no_grad():
            out = self.model(self.toks, repr_layers=[6])["representations"][6]
        return out[0, 1:self.L + 1].cpu().numpy()


def blosum62_score(wt_aa, mut_aa):
    try:
        return float(BLOSUM62[wt_aa, mut_aa])
    except KeyError:
        return float(BLOSUM62[mut_aa, wt_aa])


def build_table(dms_csv, wt_seq, struct_rows, scorer, out_csv,
                mutant_col="mutant", score_col="DMS_score",
                bin_col="DMS_score_bin"):
    df = pd.read_csv(dms_csv)
    rows = []
    for _, r in df.iterrows():
        mut = r[mutant_col]                    # e.g. 'A43G'
        wt_aa, pos, mut_aa = mut[0], int(mut[1:-1]), mut[-1]
        assert wt_seq[pos - 1] == wt_aa, f"WT mismatch at {mut}"
        mseq = wt_seq[:pos - 1] + mut_aa + wt_seq[pos:]
        rows.append(dict(
            mutant=mut, mutated_sequence=mseq,
            DMS_score=r[score_col],
            DMS_score_bin=r[bin_col] if bin_col in df.columns else np.nan,
            wt=wt_aa, pos=pos, mut=mut_aa,
            esm2=scorer.score(pos, wt_aa, mut_aa),
            blosum62=blosum62_score(wt_aa, mut_aa),
            d_hydro=HYDRO[mut_aa] - HYDRO[wt_aa],
            d_vol=VOL[mut_aa] - VOL[wt_aa],
            d_chg=CHG.get(mut_aa, 0) - CHG.get(wt_aa, 0),
            **struct_rows[pos],
        ))
    pd.DataFrame(rows).to_csv(out_csv, index=False)


if __name__ == "__main__":
    # ---- TEM-1 example (UniProt P62593, catalytic Ser70, omega-loop 163-178)
    download_af_structure("P62593", "AF-P62593-F1-model.pdb")
    plddt, cb = parse_structure("AF-P62593-F1-model.pdb")
    srows = structural_features(plddt, cb, catalytic_ser=70,
                                loop_range=(163, 179))
    wt_seq = "".join(open("P62593_TEM1.fasta").read().split("\n")[1:])
    scorer = ESM2Scorer(wt_seq)
    np.save("tem1_esm2_emb.npy", scorer.embeddings())
    build_table("DMS_subst/BLAT_ECOLX_Stiffler_2015.csv", wt_seq, srows, scorer,
                "tem1_stiffler_modeling_table.csv")
    # VIM-2: anchor residue 64; use names=('d_act','d_loop') and the VIM-2 loop range.
    # AAC(3)-I: no catalytic anchor -> catalytic_ser=None, loop_range=None.
