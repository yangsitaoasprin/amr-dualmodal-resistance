#!/usr/bin/env python3
"""
09c_esm2_3b_new_proteins.py — ESM-2 3B zero-shot scores for the two expansion
proteins (APH(3')-IIa, DHFR).  *** RUN LOCALLY ON GPU ***

Mirrors 09b_esm2_650m_new_proteins.py exactly, but with esm2_t36_3B_UR50D
(ledger-registered weight: C:\\Users\\Administrator\\.cache\\torch\\hub\\
checkpoints\\esm2_t36_3B_UR50D.pt; fair-esm 2.0.0, boltz_cuda env).
Token-level masking (mask_idx), one masked forward pass per mutated position,
identical protocol to the 8M/650M scoring.

Input : data/aph3ii_modeling_table.csv, data/dhfr_modeling_table.csv;
        data/aph3ii_wt.fasta, data/dhfr_wt.fasta
Output: results/esm2_3b_newproteins.csv (assay,mutant,wt,pos,mut,esm2_3b)
Then run code/21_expand_proteins_3b_eval.py (CPU) and add the 3B rows to
paper Table 3 / Supplementary Table S8a.
"""
import glob
import os

import pandas as pd
import torch
import torch.nn.functional as F

# PyTorch>=2.6 defaults torch.load(weights_only=True); the official fair-esm
# checkpoints contain an argparse.Namespace and need the legacy behaviour.
_orig_load = torch.load
def _load_legacy(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)
torch.load = _load_legacy

ASSAYS = {
    "APH3II_Melnikov2014": ("aph3ii_modeling_table.csv", "aph3ii_wt.fasta"),
    "DHFR_Nguyen2023": ("dhfr_modeling_table.csv", "dhfr_wt.fasta"),
}
TAG = "esm2_3b"
# Checkpoint lookup is environment-driven so the script is portable: put the
# ESM-2 3B weights in $TORCH_HOME/hub/checkpoints, or point ESM_CKPT_DIRS at
# one or more directories that contain them (os.pathsep-separated).
SEARCH_DIRS = [os.environ.get("TORCH_HOME", os.path.join(
    os.path.expanduser("~"), ".cache", "torch"))]
SEARCH_DIRS += [d for d in os.environ.get("ESM_CKPT_DIRS", "").split(os.pathsep) if d]

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RES = os.path.join(os.path.dirname(__file__), "..", "results")

CKPT_NAME = "esm2_t36_3B_UR50D.pt"


def find_weights():
    """Return the 3B checkpoint path from TORCH_HOME / ESM_CKPT_DIRS."""
    for d in SEARCH_DIRS:
        for cand in (os.path.join(d, "hub", "checkpoints", CKPT_NAME),
                     os.path.join(d, CKPT_NAME)):
            if os.path.isfile(cand):
                return cand
    pats = ["esm2_t36_3B_UR50D*.pt", "*3B*.pt"]
    for d in SEARCH_DIRS:
        for pat in pats:
            hits = glob.glob(os.path.join(d, "**", pat), recursive=True)
            hits = [h for h in hits if "regression" not in os.path.basename(h)
                    and not h.endswith(".partial")]
            if hits:
                return hits[0]
    raise FileNotFoundError("esm2_t36_3B_UR50D.pt not found; run "
                            "esm.pretrained.esm2_t36_3B_UR50D() once to download")


def read_wt(fn):
    return "".join(l.strip() for l in open(os.path.join(DATA, fn))
                   if not l.startswith(">"))


def main():
    import esm
    model, alphabet = esm.pretrained.load_model_and_alphabet(find_weights())
    model = model.cuda().eval()
    bc = alphabet.get_batch_converter()
    idx = alphabet.get_idx
    rows = []
    for assay, (table, fasta) in ASSAYS.items():
        wt = read_wt(fasta)
        m = pd.read_csv(os.path.join(DATA, table))
        # sanity: table wt column must match the fasta at every assayed position
        wtm = {}
        for _, r in m.iterrows():
            wtm.setdefault(r.pos, r.wt)
            assert wtm[r.pos] == r.wt
        for p, a in wtm.items():
            assert wt[p - 1] == a, (assay, p, wt[p - 1], a)
        _, _, wt_toks = bc([("p", wt)])
        assert wt_toks.shape[1] == len(wt) + 2, (assay, wt_toks.shape, len(wt))
        wt_toks = wt_toks.cuda()
        with torch.no_grad():
            for pos, g in m.groupby("pos"):
                toks = wt_toks.clone()
                toks[0, pos] = alphabet.mask_idx  # residue pos = token index pos
                logits = model(toks)["logits"]
                lp = F.log_softmax(logits[0, pos], dim=-1).cpu()
                for _, r in g.iterrows():
                    rows.append(dict(
                        assay=assay, mutant=r.mutant, wt=r.wt, pos=pos, mut=r.mut,
                        **{TAG: float(lp[idx(r.mut)] - lp[idx(r.wt)])}))
                if pos % 50 == 0:
                    print(f"{assay} pos {pos} done", flush=True)
    out = pd.DataFrame(rows)
    path = os.path.join(RES, "esm2_3b_newproteins.csv")
    out.to_csv(path, index=False)
    print("wrote", path, len(out), "variants")


if __name__ == "__main__":
    main()
