#!/usr/bin/env python3
"""
13_new_proteins_esm2_8m.py — ESM-2 8M masked-marginal zero-shot scores (Eq. 1)
for the two expansion proteins and assembly of their final modelling tables.

Model: facebook/esm2_t6_8M_UR50D (HuggingFace mirror of fair-esm
esm2_t6_8M_UR50D; identical weights). One masked forward pass per mutated
position, identical protocol to the original pipeline.

Inputs:  data/aph3ii_partial_table.csv, data/dhfr_partial_table.csv
         data/aph3ii_wt.fasta, data/dhfr_wt.fasta
         local model dir (env ESM2_8M_DIR, default ./models/esm2_8m)
Outputs: data/aph3ii_modeling_table.csv, data/dhfr_modeling_table.csv
"""
import os
import re

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.environ.get("ESM2_8M_DIR",
                           os.path.join(os.path.dirname(__file__), "..",
                                        "models", "esm2_8m"))


def score_protein(name):
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_DIR).eval()
    wt = open(os.path.join(DATA, name + "_wt.fasta")).read().splitlines()[1]
    df = pd.read_csv(os.path.join(DATA, name + "_partial_table.csv"))
    positions = sorted(df.pos.unique())
    pos_lp = {}
    with torch.no_grad():
        for i in positions:
            masked = wt[: i - 1] + "<mask>" + wt[i:]
            enc = tok(masked, return_tensors="pt")
            lp = F.log_softmax(model(**enc).logits[0, i], dim=-1)
            pos_lp[i] = lp
    esm2 = []
    for _, r in df.iterrows():
        lp = pos_lp[r.pos]
        esm2.append(float(lp[tok.convert_tokens_to_ids(r.mut)]
                          - lp[tok.convert_tokens_to_ids(r.wt)]))
    df["esm2"] = esm2
    # 12 writes generic d_anchor/d_loop names; the shipped tables (and every
    # downstream script: 14/15/21/22/23) use the TEM-1-style d_s70/d_omega
    # aliases, where d_s70 means "distance to this protein's catalytic anchor".
    df = df.rename(columns={"d_anchor": "d_s70", "d_loop": "d_omega"})
    cols = ["mutant", "wt", "pos", "mut", "esm2", "blosum62", "d_hydro",
            "d_vol", "d_chg", "pLDDT", "contact_num", "d_s70", "d_omega",
            "DMS_score", "DMS_score_bin"]
    out = df[cols]
    path = os.path.join(DATA, name + "_modeling_table.csv")
    out.to_csv(path, index=False)
    print(name, out.shape, "esm2 mean %.3f" % out.esm2.mean(), "->", path)


if __name__ == "__main__":
    score_protein("aph3ii")
    score_protein("dhfr")
