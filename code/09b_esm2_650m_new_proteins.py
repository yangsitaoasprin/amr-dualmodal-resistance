#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
09b_esm2_650m_new_proteins.py — ESM-2 650M masked-marginal zero-shot scores
for the two expansion proteins APH(3')-IIa and DHFR.

Run from the repository code/ directory (GPU, see requirements.txt):

  export TORCH_HOME="$HOME/.cache/torch"
  python 09b_esm2_650m_new_proteins.py

The script looks for esm2_t33_650M_UR50D*.pt under $TORCH_HOME and any extra
directories listed in $ESM_CKPT_DIRS; if none is found it prints the candidates
it saw. Input files (in ../data/):
  aph3ii_modeling_table.csv, dhfr_modeling_table.csv,
  aph3ii_wt.fasta, dhfr_wt.fasta
输出：esm2_650m_newproteins.csv（assay,mutant,wt,pos,mut,esm2_650m）
跑完把 esm2_650m_newproteins.csv 传回对话即可。
"""
import glob
import os
import sys

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
TAG = "esm2_650m"
# Checkpoint lookup is environment-driven so the script is portable: put the
# ESM-2 650M weights in $TORCH_HOME/hub/checkpoints, or point ESM_CKPT_DIRS at
# one or more directories that contain them (os.pathsep-separated).
SEARCH_DIRS = [os.environ.get("TORCH_HOME", os.path.join(
    os.path.expanduser("~"), ".cache", "torch"))]
SEARCH_DIRS += [d for d in os.environ.get("ESM_CKPT_DIRS", "").split(os.pathsep) if d]

CKPT_NAME = "esm2_t33_650M_UR50D.pt"


def find_weights():
    """Return the 650M checkpoint path from TORCH_HOME / ESM_CKPT_DIRS."""
    for d in SEARCH_DIRS:
        for cand in (os.path.join(d, "hub", "checkpoints", CKPT_NAME),
                     os.path.join(d, CKPT_NAME)):
            if os.path.isfile(cand):
                return cand
    pats = ["esm2_t33_650M_UR50D*.pt", "*650M*.pt"]
    seen = []
    for d in SEARCH_DIRS:
        for pat in pats:
            hits = glob.glob(os.path.join(d, "**", pat), recursive=True)
            hits = [h for h in hits if "regression" not in os.path.basename(h)]
            if hits:
                return hits[0]
            seen.extend(hits)
    cands = []
    for d in SEARCH_DIRS:
        cands.extend(glob.glob(os.path.join(d, "**", "*.pt"), recursive=True))
    print("未找到 650M 权重。搜索到的 .pt 文件：")
    for c in cands[:30]:
        print("  ", c)
    sys.exit("请把上面的某个 650M 权重路径直接写进脚本 find_weights() 的 return 语句。")


def read_wt(fn):
    return "".join(l.strip() for l in open(fn, encoding="utf-8")
                   if not l.startswith(">"))


def main():
    wpath = find_weights()
    print("权重文件:", wpath)
    import esm
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(wpath)
    model = model.eval().cuda()
    bc = alphabet.get_batch_converter()
    rows = []
    for assay, (table, fasta) in ASSAYS.items():
        wt = read_wt(fasta)
        df = pd.read_csv(table)
        # 校验 WT 与建模表一致
        wtm = {}
        for m in df.mutant:
            g = m[0], int(m[1:-1]), m[-1]
            wtm[g[1]] = g[0]
        for p, a in wtm.items():
            assert wt[p - 1] == a, (assay, p, wt[p - 1], a)
        # 只对 WT 做一次分词，之后直接在 token 上替换为 mask_idx。
        # fair-esm 的 batch_converter 版本间对 "<mask>" 字符串的处理不一致，
        # 直接改 token id 最稳妥（BOS 占 index 0，残基 pos 对应 token index pos）。
        _, _, wt_toks = bc([("p", wt)])
        assert wt_toks.shape[1] == len(wt) + 2, (assay, wt_toks.shape, len(wt))
        wt_toks = wt_toks.cuda()
        for pos in sorted(df.pos.unique()):
            toks = wt_toks.clone()
            toks[0, pos] = alphabet.mask_idx
            with torch.no_grad():
                logits = model(toks)["logits"]
            lp = F.log_softmax(logits[0, pos], dim=-1).cpu()  # BOS 偏移
            sub = df[df.pos == pos]
            for _, r in sub.iterrows():
                idx = alphabet.get_idx
                rows.append(dict(
                    assay=assay, mutant=r.mutant, wt=r.wt, pos=pos, mut=r.mut,
                    **{TAG: float(lp[idx(r.mut)] - lp[idx(r.wt)])}))
        print(assay, "完成,", len(df), "个突变")
    out = pd.DataFrame(rows)
    out.to_csv("esm2_650m_newproteins.csv", index=False)
    print("已保存 esm2_650m_newproteins.csv", out.shape)


if __name__ == "__main__":
    main()
