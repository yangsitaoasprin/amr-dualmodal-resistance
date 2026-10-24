# Alignment-free dual-modal prediction of antibiotic resistance mutation effects

Code and data accompanying the manuscript:

> Yang S.†, Ye Y.†, Wang G., Ma N., Guo X.\*, Jia Y.\* (2026). Alignment-free
> prediction of antibiotic resistance mutation effects by combining protein
> language model scores with predicted structural context. *Bioinformatics*
> (submitted).

Repository: https://github.com/yangsitaoasprin/amr-dualmodal-resistance

## Overview

We combine a zero-shot masked-marginal score from the ESM-2 protein language
model with four structural-context features derived from AlphaFold-predicted
structures (pLDDT, Cβ contact number, distance to catalytic serine, distance
to Ω-loop) in a 9-dimensional interpretable feature set, trained with
gradient-boosted trees (GBM) under stringent modulo 5-fold by-position
cross-validation. An end-to-end graph neural network branch (naive GAT →
extended-training GCN → redesigned GCNv2) is included with the full
diagnose–fix–validate analysis.

## Repository layout

```
code/
  01_build_features.py       # DMS + AlphaFold structure -> 9-dim feature table
  02_train_gbm.py            # dual-modal GBM, modulo CV, transfer matrix (Tables 1,3,4)
  03_gcnv2.py                # pure-PyTorch GCNv2 on the contact graph (Tables 5,6; Fig 3)
  04_external_validation.py  # cross-DMS-experiment validation (Table 2)
  05_gat_extended.py         # naive GAT budget ramp extended to 60 epochs (Fig 3a)
  06_extended_gcn_ramp.py    # extended-training 2-layer GCN, fold 0, 5-60 epochs (Fig 3a GCN ramp; GCN@15ep train/test diagnosis)
  07_mlp_ablation.py         # MLP ablation (no graph propagation), fold 0, 25 epochs (diagnosis experiment i)
  08_naive_gat_5fold.py      # naive GATv2, 3 ep x 1,200 samples, full 5-fold CV (naive-GAT number, Fig 3c)
  make_figs_sci.py               # all paper figures (Fig GA, 1-3), 300-dpi PNG + vector PDF
data/
  tem1_stiffler_modeling_table.csv   # TEM-1 (n=4,996) feature table
  vim2_modeling_table.csv            # VIM-2  (n=5,004) feature table
  aac3i_modeling_table.csv           # AAC(3)-I (n=1,793) feature table
  BLAT_ECOLX_{Deng_2012,Firnberg_2014,Jacquier_2013}_modeling_table.csv  # external assays
  tem1_structural_features.csv       # full-length (286 aa) TEM-1 structural features
  tem1_esm2_emb.npy, vim2_esm2_emb.npy   # per-residue ESM-2 embeddings (layer 6)
  AF-*.pdb, *.fasta                  # AlphaFold structures and WT sequences
results/
  gcnv2_5fold_cv.csv                 # per-fold GCNv2 vs GBM on TEM-1
  vim2_gcnv2_5fold_cv.csv            # per-fold GCNv2 vs GBM on VIM-2
  multiprotein_results.csv           # three-protein summary
  gat_extended_ramp.csv              # naive-GAT fold-0 rho at 3..60 epochs (Fig 3a)
  gcn_extended_ramp.csv              # extended GCN fold-0 train/test rho at 5..60 epochs (Fig 3a; GCN@15ep diagnosis)
  mlp_ablation.csv                   # MLP ablation fold-0 train/test rho at 25 epochs (diagnosis experiment i)
  naive_gat_5fold.csv                # naive GATv2 per-fold rho, 3 ep x 1,200 samples (naive-GAT 5-fold number)
figures/                             # 300-dpi manuscript figures (Fig GA, 1-3; Fig 3 is the combined GNN cycle figure)
```

## Data sources

- DMS assays: ProteinGym v1.3 substitution benchmark (Zenodo record
  15293562) — `BLAT_ECOLX_Stiffler_2015`, `BLAT_ECOLX_Deng_2012`,
  `BLAT_ECOLX_Firnberg_2014`, `BLAT_ECOLX_Jacquier_2013`,
  `A4GRB6_PSEAI_Chen_2020`, `AACC1_PSEAI_Dandage_2018`.
  NOTE: `A4GRB6_PSEAI_Chen_2020` is the VIM-2 metallo-beta-lactamase DMS assay
  (Chen, Fowler & Tokuriki, eLife 2020, e56707); the score column used is
  `0.031ug_mL_MEM_37C` (meropenem selection). Files named `ampc_*` in earlier
  revisions were renamed to `vim2_*`.
- Predicted structures: AlphaFold DB entries AF-P62593 (TEM-1),
  AF-P23181 (AAC(3)-I); VIM-2 structure sequence-verified residue-by-residue
  against the DMS target sequence (UniProt A4GRB6 entry deprecated).
- Protein language model: ESM-2 `esm2_t6_8M_UR50D` (fair-esm).

## Environment

```
python >= 3.10
pip install torch fair-esm biopython pandas numpy scipy scikit-learn matplotlib
```

All experiments are CPU-only (GBM: seconds; GCNv2: ~1–2 min per fold).

## Reproduce

```bash
cd code
python 01_build_features.py     # rebuild feature tables (downloads AF structure + runs ESM-2)
python 02_train_gbm.py          # Tables 1, 3, 4
python 03_gcnv2.py              # Tables 5-6, Figure 3 numbers
python 04_external_validation.py# Table 2
python 05_gat_extended.py       # Figure 3a naive-GAT extended ramp
python 06_extended_gcn_ramp.py  # Figure 3a extended-GCN ramp + GCN@15ep train/test
python 07_mlp_ablation.py       # MLP ablation (diagnosis, overfitting gap)
python 08_naive_gat_5fold.py    # naive-GAT 5-fold CV number

Reproducibility of the GNN storyline numbers (naive-GAT 5-fold mean,
extended-GCN ramp, MLP ablation): scripts 06-08 fix three random seeds
(0, 1, 2; every weight initialisation, shuffle and subsample is seeded) and
are pure-PyTorch CPU code, so the CSVs they write reproduce exactly in this
environment. The paper reports the mean +/- std across the three seeds from
these CSVs (gcn_extended_ramp.csv, mlp_ablation.csv, naive_gat_5fold.csv);
rerunning the scripts regenerates precisely those numbers.
```

## Key results

| Evaluation | Zero-shot ESM-2 | Dual-modal GBM | GCNv2 |
|---|---|---|---|
| TEM-1 Stiffler 2015 (5-fold) | 0.399 | 0.705 ± 0.023 (AUC 0.859) | 0.703 ± 0.029 (p = 0.854 vs GBM) |
| VIM-2 Chen 2020 (5-fold) | 0.431 | 0.728 ± 0.054 (AUC 0.901) | 0.736 ± 0.026 (p = 0.616 vs GBM) |
| AAC(3)-I Dandage 2018 (5-fold) | 0.204 | 0.251 ± 0.026 (AUC 0.639) | — |
| External Deng 2012 / Firnberg 2014 / Jacquier 2013 | — | 0.497 / 0.820 / 0.748 | — |

Cross-protein transfer (train → test, Spearman ρ): TEM-1 → VIM-2 0.713,
VIM-2 → TEM-1 0.631, AAC(3)-I → TEM-1 0.522, AAC(3)-I → VIM-2 0.635,
TEM-1 → AAC(3)-I 0.268, VIM-2 → AAC(3)-I 0.268.

## Contact

Sitao Yang <yangsitaoasprin@swpu.edu.cn>; Yongfeng Ye <yongfengyip@126.com>
