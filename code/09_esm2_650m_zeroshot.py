#!/usr/bin/env python3
"""
09_esm2_650m_zeroshot.py — Compute ESM-2 zero-shot masked-marginal scores with a
large model (default 650M; optionally 3B) for every mutant in the dual-modal
modeling tables. This provides the strong-baseline comparison requested in the
manuscript revision (paper baseline M0 uses the 8M model).

Run LOCALLY on a GPU machine:
  - conda env:   see environment.yml (fair-esm 2.0.0, torch; ~24 GB VRAM for 650M)
  - weights:     $TORCH_HOME/hub/checkpoints/esm2_t33_650M_UR50D.pt
                 (set TORCH_HOME to an existing checkpoint dir so nothing downloads)

Usage (from the repository code/ directory):
  export TORCH_HOME="$HOME/.cache/torch"
  conda activate amr-dual
  python 09_esm2_650m_zeroshot.py --data_dir ../data --out ../results/esm2_650m_zeroshot.csv
  # optional, the 3B model as an additional baseline:
  python 09_esm2_650m_zeroshot.py --model esm2_t36_3B_UR50D --tag 3b \
      --data_dir ../data --out ../results/esm2_3b_zeroshot.csv

Scoring protocol: identical to 01_build_features.py — for each mutated position,
a single forward pass with that position masked; score =
log P(mut | masked context) - log P(wt | masked context)  (masked marginal).
Output columns: assay, mutant, wt, pos, mut, <score_col>
"""
import argparse
import os

import pandas as pd
import torch
import esm

ASSAYS = {
    "TEM1_Stiffler2015": "tem1_stiffler_modeling_table.csv",
    "TEM1_Deng2012": "BLAT_ECOLX_Deng_2012_modeling_table.csv",
    "TEM1_Firnberg2014": "BLAT_ECOLX_Firnberg_2014_modeling_table.csv",
    "VIM2_Chen2020": "ampc_modeling_table.csv",
    "AAC3I_Dandage2018": "aac3i_modeling_table.csv",
}

MODELS = {
    "650m": "esm2_t33_650M_UR50D",
    "3b": "esm2_t36_3B_UR50D",
}


def rebuild_wt(df):
    """Reconstruct the wild-type sequence from mutated_sequence + wt/pos/mut.
    Verifies 1-based residue numbering on every row."""
    wt = None
    for r in df.itertuples():
        seq = list(r.mutated_sequence)
        i = r.pos - 1
        assert 0 <= i < len(seq), f"pos {r.pos} out of range for {r.mutant}"
        assert seq[i] == r.mut, (
            f"numbering mismatch at {r.mutant}: mutated_sequence[{i}]={seq[i]}, "
            f"expected {r.mut}")
        seq[i] = r.wt
        s = "".join(seq)
        if wt is None:
            wt = s
        else:
            assert s == wt, f"inconsistent WT reconstruction at {r.mutant}"
    return wt


def masked_logprobs(model, alphabet, wt_seq, device):
    """One masked forward pass per position -> (L, vocab) log-probability matrix."""
    bc = alphabet.get_batch_converter()
    L = len(wt_seq)
    V = len(alphabet.all_toks)
    out = torch.zeros(L, V)
    with torch.no_grad():
        for i in range(L):
            masked = list(wt_seq)
            masked[i] = "<mask>"
            _, _, toks = bc([("seq", "".join(masked))])
            toks = toks.to(device)
            res = model(toks, repr_layers=[], return_contacts=False)
            # +1: BOS token prepended by the batch converter
            out[i] = torch.log_softmax(res["logits"][0, i + 1].float(), dim=-1).cpu()
            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{L} positions")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esm2_t33_650M_UR50D")
    ap.add_argument("--tag", default="650m")
    ap.add_argument("--data_dir", default="../data")
    ap.add_argument("--out", default="../results/esm2_650m_zeroshot.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {args.model} on {device} ...")
    model, alphabet = getattr(esm.pretrained, MODELS.get(args.tag, args.model))()
    model = model.eval().to(device)
    tok_idx = {aa: alphabet.get_idx(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}

    frames = []
    col = f"esm2_{args.tag}"
    for assay, fn in ASSAYS.items():
        path = os.path.join(args.data_dir, fn)
        if not os.path.exists(path):
            print(f"  [skip] {fn} not found")
            continue
        df = pd.read_csv(path)
        wt_seq = rebuild_wt(df)
        print(f"[{assay}] {len(df)} mutants, WT length {len(wt_seq)}")
        lp = masked_logprobs(model, alphabet, wt_seq, device)
        scores = []
        for r in df.itertuples():
            i = r.pos - 1
            scores.append(float(lp[i, tok_idx[r.mut]] - lp[i, tok_idx[r.wt]]))
        sub = df[["mutant", "wt", "pos", "mut"]].copy()
        sub["assay"] = assay
        sub[col] = scores
        frames.append(sub)
        print(f"  done {assay}: score range {min(scores):.2f} .. {max(scores):.2f}")

    out = pd.concat(frames, ignore_index=True)[
        ["assay", "mutant", "wt", "pos", "mut", col]]
    out.to_csv(args.out, index=False)
    print(f"saved {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
