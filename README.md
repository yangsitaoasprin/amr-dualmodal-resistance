# Alignment-free dual-modal prediction of antibiotic resistance mutation effects

Code and data accompanying the manuscript:

> Yang S.†, Ye Y.†, Wang G., Ma N., Guo X.\*, Jia Y.\* (2026). Alignment-free
> prediction of antibiotic resistance mutation effects by combining protein
> language model scores with predicted structural context. *Bioinformatics*
> (submitted).

Repository: https://github.com/yangsitaoasprin/amr-dualmodal-resistance

> **Caliber notice (2026-11-13).** All numbers below are the **canonical,
> corrected-anchor** results. Earlier commits (up to 2026-10-24) reported the
> archived offset-anchor build; see [Caliber and reproducibility](#caliber-and-reproducibility)
> and the [Changelog](#changelog). The manuscript, figures and supplementary
> tables S20–S21 use this canonical caliber.

> **New to this repository?** Start with `00_START_HERE.md` (package root):
> it explains the script naming, maps every table and figure to the script
> that produces it, and gives a five-minute verification path.

## Overview

We combine a zero-shot masked-marginal score from ESM-2 with four
structural-context features derived from AlphaFold-predicted structures
(pLDDT, Cβ contact number, distance to a functional-site reference residue,
distance to the Ω-loop or an analogous active-site-flanking segment) in a
9-dimensional interpretable feature set, trained with gradient-boosted trees
(GBM) under modulo 5-fold **by-position** cross-validation. An end-to-end graph
neural network branch (naive GATv2 → extended-training GCN → redesigned GCNv2)
is included with the full diagnose–fix–validate analysis.

Beyond the core benchmark, the repository contains assay-agreement ceilings,
a corrected Ambler-to-precursor residue-numbering conversion, holo-conformation
and full-length dimer controls, a clinical TEM variant panel, and a
multi-substitution clinical MIC boundary test.

## Caliber and reproducibility

- **Residue numbering.** Ambler (ABL) numbering is not the same as precursor
  numbering for TEM-1: ABL = precursor + 2 through ABL 238 (ABL positions 239
  and 253 are gaps in TEM-1). The catalytic Ser70 (ABL) is **precursor residue
  68**; the Ω-loop (ABL 164–179) is **precursor 162–177**. The released feature
  tables use these corrected anchors. Verified against UniProt P62593 ACT_SITE
  annotations (68/71/128/164) and PDB 1BTL.
- **Full re-run.** All 13 result-producing scripts were re-executed on the
  corrected tables (13/13 OK); log: `results/canonical_rerun_log.txt`.
  Machine-readable number sets: `results/canonical_numbers.json`,
  `results/canonical_extra.json`, `results/canonical_report.md`.
- **Historical build.** The two-residue-shifted anchors (precursor 70 and
  163–178) gave 8M M3 0.704 ± 0.025 and 650M M3 0.805 ± 0.013 on identical
  folds (|Δρ| ≤ 0.01). Original tables are preserved as
  `data/_backup_anchor70_*.csv`. Supplementary Tables S1–S15 are archived
  per-run records of that build; **Table S20** is the authoritative canonical
  recomputation underlying every main-text table and figure, and Tables S21–S24
  add the position-shuffle, transfer-anchor, embedding-dimensionality and
  robustness controls.
- **Figures** were re-rendered from the canonical CSVs; result labels in
  `41_fig1_dualmodal.py`, `44_fig4_gnn_cycle.py` and `40_fig_ga.py` are computed
  dynamically from `results/canonical_numbers.json`.

## Repository layout

```
code/                           # pipeline, figures and build drivers (run from here)
  01_build_features.py         # DMS + AlphaFold structure -> 9-dim feature table
  02_train_gbm.py              # dual-modal GBM, modulo CV, transfer (Tables 1,3,4)
  03_gcnv2.py                  # pure-PyTorch GCNv2, per-fold CSV output (Tables 5,6)
  04_external_validation.py    # cross-DMS-experiment validation (Table 2)
  05_gat_extended.py           # naive GAT budget ramp to 60 epochs
  06_extended_gcn_ramp.py      # 2-layer GCN fold-0 ramp, 5-60 epochs
  07_mlp_ablation.py           # MLP ablation (no graph propagation)
  08_naive_gat_5fold.py        # naive GATv2 full 5-fold CV
  09_esm2_650m_zeroshot.py     # ESM-2 650M zero-shot scores (GPU)
  09b/09c_*                    # 650M / 3B zero-shot for APH(3')-IIa + DHFR (GPU)
  10_esm2_backbone_eval.py     # 650M + 3B evaluation (Tables 1-4, S7)
  11_expand_proteins.py        # parse APH(3')-IIa / DHFR ProteinGym assays
  12_new_proteins_struct_features.py  # structural features for expansion proteins
  13_new_proteins_esm2_8m.py   # ESM-2 8M scores for expansion proteins
  14/15/21_expand_proteins*_eval.py   # 8M / 650M / 3B evaluation + transfer (Tables 3,4, S8)
  16_anchor_counterfactual.py  # with/without anchor distances (Table S9)
  17_contiguous_cv.py          # contiguous-position CV, 650M (Table S10)
  18_proteingym_benchmark.py   # 18 published zero-shot predictors (Table S11)
  18b_fetch_proteingym_zs.py   # rebuild data/proteingym_zs/ from the Zenodo archive
  19_aac_anchor_experiment.py  # reverse CoA-pocket anchor, PDB 1BO4 (Table S12)
  20_random_anchor_null.py     # random anchors, collinearity, paired tests, permutation importance (S13-S15)
  22_verify_and_dump.py        # per-fold dump + oof predictions + per-position table
  23_build_delta_rho_table.py  # delta-rho table for Figures 2 and 5
  24_assay_ceiling.py          # assay-agreement ceilings (Table S16a)
  25_dimer_burial_boltz.py     # full-length dimer-aware burial (Table S18)
  26_multisite_mic.py          # multi-substitution clinical MIC (Table S19)
  27_audit_controls.py         # position-shuffle null + PCA AUC (Table S21)
  28_transfer_controls.py      # PCA dimensionality + transfer controls (Table S22)
  40_fig_ga.py                 # graphical abstract
  41_fig1_dualmodal.py         # Figures 1-5, one script per figure
  42_fig2_structural_gain.py
  43_fig3_external_benchmark.py
  44_fig4_gnn_cycle.py
  45_fig5_decision_map.py
  46_figs_s1_s6.py             # supplementary Figures S1-S6
  47_render_pymol_tem1.py      # Figure 1e structural render (PyMOL)
  49_render_all_figures.py     # render all 12 figures and sync figures/
  50_rerun_pipeline.py         # re-run the 13 result scripts -> results/
  51_collect_numbers.py        # manuscript numbers -> canonical_numbers.json
  52_canonical_report.py       # -> canonical_report.md
  53_extra_numbers.py          # -> canonical_extra.json
  54_stats_intervals.py        # clustered-bootstrap CIs, TOST -> stats_intervals.json
  55_build_docx.py             # md -> docx with caption metadata (pandoc)
  figstyle.py                  # shared style/palette/paths + load/save helpers
  figstyle_core.py             # theme, sizing, export, delivery gate
gates/                          # verification only; each gate is standalone
  91_verify_table1.py          # every Table 1 cell vs its source CSV
  92_audit_table3_sd.py        # Table 3 recomputation, SD caliber
  93_verify_manuscript.py      # stale-value guards, docx integrity
  94_audit_captions.py         # caption style (n / error bars / test+P)
  95_verify_links.py           # every path cited in the paper/SI resolves
  96_verify_docs.py            # MANIFEST / README numbers / cited script names
archive/                        # authors' working tree only, NOT distributed:
                                #   one-shot scripts that already ran
  _exp9_patch_canonical.py     # applied corrected anchors (rewrites data/ in place)
  _exp15/17/25/35, _patch_*.py # manuscript/SI patches, frozen at their round
  _archived_guard.py           # makes the above refuse to re-run (see below)
  _tmp_*, _crop_*, _mkpreview  # scratch utilities kept for provenance
MANIFEST.md                     # one row per script: purpose + outputs
REPRODUCE.md                    # step-by-step reproduction with expected outputs
requirements.txt, environment.yml
data/
  tem1_stiffler_modeling_table.csv, vim2_modeling_table.csv,
  aac3i_modeling_table.csv, aph3ii_modeling_table.csv, dhfr_modeling_table.csv
  BLAT_ECOLX_{Deng_2012,Firnberg_2014,Jacquier_2013}_modeling_table.csv
  tem1_structural_features.csv, tem1_esm2_emb.npy, vim2_esm2_emb.npy
  AF-*.pdb, *.fasta, 1RX1.pdb, 1BO4.pdb, 1BTL.pdb
  boltz_aac3i_dimer.yaml, boltz_out/ (full-length AAC(3)-I homodimer, Boltz-2)
  proteingym_zs/               # 5 ProteinGym v1.3 zero-shot tables (37 MB,
                               #   md5 in checksums.md5; see its README)
  _backup_anchor70_*.csv       # archived offset-anchor tables
results/
  canonical_numbers.json, canonical_extra.json, canonical_report.md,
  canonical_rerun_log.txt      # canonical caliber, single source of truth
  gcnv2_5fold_cv.csv, vim2_gcnv2_5fold_cv.csv, multiprotein_results.csv
  esm2_{650m,3b}_{zeroshot,evaluation,newproteins}.csv
  expanded_proteins{,_650m,_3b}_evaluation.csv
  anchor_counterfactual.csv, contiguous_cv_650m.csv,
  proteingym_zeroshot_benchmark.csv, aac_anchor_experiment*.csv,
  random_anchor_null.csv, paired_tests.csv, feature_collinearity.csv,
  permutation_importance.csv, delta_rho_table.csv, recomputed_folds*.csv,
  tem1_per_position_pred_vs_true.csv, oof_predictions_tem1_8m.npz
  assay_agreement_ceiling.csv, anchor_fix_cutoff_sensitivity.csv,
  clinical_tem_validation*.csv, multisite_clinical_mic.csv,
  multisite_mic_tests.csv, boltz_dimer_aac3i.csv, audit_position_shuffle.csv
figures/                        # 300-dpi PNG + vector PDF: GA, Figures 1-5, S1-S6
                                #   (synced by 49_render_all_figures.py)
  _superseded/                  # retired renders: fig1_results, fig2_multiprotein,
                                #   fig3_combined and the pre-revision GA. Kept for
                                #   provenance only -- they carry archived numbers
                                #   (p = 0.854, GBM 0.691) and are NOT used by the
                                #   manuscript or the SI.
```

The `archive/` directory is kept in the authors' working tree only. It is **frozen
history**: its scripts encode an old -> new text
mapping (or rewrite released tables), so re-running them against the current
tree would revert later corrections or duplicate appended tables. They import
`_archived_guard`, which exits with a message unless
`AMR_ALLOW_ARCHIVED_PATCH=1` is set. Nothing in `code/` or `gates/` imports them.

### Plotting helpers

`figstyle.py` takes its style primitives from `figstyle_core.py`, which is part
of this repository: a journal rc-theme, figure sizing, panel labels, PNG+PDF
export, colour tint/shade helpers and a delivery gate. The figure scripts import
`figstyle` only.

`figures/` holds the reference renders used in the manuscript. Re-running the
figure scripts reproduces the same data and panel layout; typesetting details
(font ladder, a few label positions) may differ slightly from those files. Set
`AMR_FIGDIR` to render somewhere other than the project root:

```bash
cd code && AMR_FIGDIR=/tmp/figs python 44_fig4_gnn_cycle.py
```

Style parameters, for anyone comparing a re-render with `figures/`:

| parameter | value |
|---|---|
| font sizes | note 5.5 pt · label 7 pt · tick 5 pt · panel 8 pt |
| page width | single 89 mm · double 183 mm (max 190.5 mm) |
| export | PNG 600 dpi + vector PDF, TrueType (Type 42) fonts |
| ticks / grid | inward, 2.6 pt; no grid; top/right spines off |

## Data sources

- **DMS assays:** ProteinGym v1.3 substitution benchmark (Zenodo record
  15293562) — `BLAT_ECOLX_Stiffler_2015` (TEM-1, ampicillin, n = 4,996),
  `BLAT_ECOLX_Deng_2012` and `BLAT_ECOLX_Firnberg_2014` (external validation),
  `BLAT_ECOLX_Jacquier_2013` (excluded from the main analysis),
  `A4GRB6_PSEAI_Chen_2020` (VIM-2 metallo-β-lactamase, meropenem, n = 5,004;
  score column `0.031ug_mL_MEM_37C`; Chen, Fowler & Tokuriki, eLife 2020,
  e56707), `AACC1_PSEAI_Dandage_2018` (AAC(3)-I, gentamicin, n = 1,793),
  `KKA2_KLEPN_Melnikov_2014` (APH(3′)-IIa, kanamycin, n = 4,960),
  `DYR_ECOLI_Nguyen_2023` (DHFR, growth selection, n = 2,916).
- **Published zero-shot predictors (Table S11):** the matching per-assay score
  tables from the same record (`zero_shot_substitutions_scores.zip`, 99 columns
  = `DMS_score` + 95 model scores) are shipped verbatim in `data/proteingym_zs/`
  so that the 18-predictor comparison runs without the 1.78 GB archive
  (`18b_fetch_proteingym_zs.py --fetch` rebuilds them over HTTP Range,
  `--verify` checks the md5 list; upstream is MIT-licensed).
- **Predicted structures:** AlphaFold DB AF-P62593-F1 (model v4, TEM-1),
  AF-A4GRB6-F1 (VIM-2), AF-P23181-F1 (AAC(3)-I), AF-P00552-F1 (APH(3′)-IIa),
  AF-P0ABQ4-F1 (DHFR). Experimental controls: PDB 1BTL (ABL numbering),
  1RX1 (DHFR holo), 1BO4 (AAC(3)-I homologue from *Serratia marcescens*,
  99.3% identity). The full-length AAC(3)-I homodimer was predicted locally
  with **Boltz-2** (`boltz2_conf.ckpt`, `boltz_cuda` environment, GPU).
- **Protein language models:** ESM-2 `esm2_t6_8M_UR50D` (CPU; default
  pipeline), `esm2_t33_650M_UR50D` (GPU; accuracy-oriented) and
  `esm2_t36_3B_UR50D` (GPU; scale-robustness check), all via `fair-esm`.

## Environment

CPU-only path (reproduces every manuscript number and figure from the shipped
feature tables):

```bash
conda env create -f environment.yml && conda activate amr-dual
# or, with an existing Python >= 3.10:
pip install -r requirements.txt
```

GBM, evaluation, figures and most validations are CPU-only (GBM: seconds;
GCNv2: ~1–2 min per fold). Only ESM-2 embedding/zero-shot scoring (scripts
`09*`, `13`) and the Boltz-2 dimer prediction need a GPU; the pre-computed
scores and the predicted dimer PDB are shipped, so a GPU is **not** required to
reproduce the manuscript. The GPU runs used torch 2.12 + fair-esm 2.0, listed
as commented extras in `requirements.txt`.

## Reproduce

`REPRODUCE.md` has the full step table with expected outputs and runtimes.
Shortest path to every number and figure in the manuscript:

```bash
cd code
# 1) result bundle from the shipped feature tables (CPU, ~6 min)
python 50_rerun_pipeline.py    # runs the 13 result scripts, log in results/
python 51_collect_numbers.py   # results/canonical_numbers.json
python 49_render_all_figures.py  # GA + Figures 1-5 + S1-S6, synced to figures/

# 2) verify against the manuscript (CPU, ~2 min)
cd ../gates
python 91_verify_table1.py       # every Table 1 cell vs its source
python 92_audit_table3_sd.py            # Table 3 recomputation, SD caliber
python 93_verify_manuscript.py        # stale-value guards + docx integrity
python 94_audit_captions.py            # caption style (n / error bars / test+P)
python 95_verify_links.py        # every cited path resolves
```

Individual stages, if you want them one at a time:

```bash
cd code
python 22_verify_and_dump.py    # per-fold dump, oof predictions, per-position table
python 10_esm2_backbone_eval.py # 650M / 3B CV, external validation, transfer
python 14_expand_proteins_eval.py
python 15_expand_proteins_650m_eval.py
python 21_expand_proteins_3b_eval.py
python 16_anchor_counterfactual.py
python 17_contiguous_cv.py
python 20_random_anchor_null.py
python 23_build_delta_rho_table.py
python 06_extended_gcn_ramp.py
python 07_mlp_ablation.py
python 08_naive_gat_5fold.py
python 03_gcnv2.py             # Tables 5-6, writes per-fold CSVs
python 02_train_gbm.py         # Tables 1, 3, 4
python 04_external_validation.py
python 18b_fetch_proteingym_zs.py --verify   # md5 of the shipped ProteinGym tables
python 18_proteingym_benchmark.py           # Table S11 published predictors

# added validations quoted in the SI
python 24_assay_ceiling.py   # assay-agreement ceilings
python 25_dimer_burial_boltz.py     # full-length dimer burial (Table S18)
python 26_multisite_mic.py   # multi-substitution clinical MIC (Table S19)
python 27_audit_controls.py # position-shuffle null + PCA AUC
python 28_transfer_controls.py    # PCA dimensionality + transfer controls (S22)
```

Corrected TEM-1 anchors (`d_s70` = precursor 68, `d_omega` = 162–177) are
**already applied** to the shipped tables in `data/`; the pre-correction copies
are kept as `data/_backup_anchor70_*.csv`. The one-shot script that applied them
lives in the authors' working tree only (it rewrites the tables in place) and is
not part of this distribution.

GNN diagnostics (scripts 06–08) fix three seeds (0/1/2); `03_gcnv2.py` uses
`random_state = 0` and writes per-fold CSVs. All manuscript numbers are dumped
to `results/canonical_numbers.json` and `results/canonical_report.md`.

## Key results (canonical caliber)

Spearman ρ, mean ± SD over the five modulo folds; AUC from pooled out-of-fold
predictions. Full tables in the manuscript (Tables 1–6) and Supplementary
Table S20.

| Assay | Zero-shot (8M) | Dual-modal GBM (8M) | Dual-modal GBM (650M) | GCNv2 |
|---|---|---|---|---|
| TEM-1 Stiffler 2015 | 0.399 | **0.696 ± 0.032** (AUC 0.855) | **0.814 ± 0.014** (AUC 0.925) | 0.701 ± 0.033 (p = 0.448 vs GBM) |
| VIM-2 Chen 2020 | 0.431 | 0.729 ± 0.053 (AUC 0.901) | 0.795 ± 0.035 (AUC 0.938) | 0.743 ± 0.032 (p = 0.294 vs GBM) |
| AAC(3)-I Dandage 2018 | 0.204 | 0.251 ± 0.026 (AUC 0.639) | 0.446 ± 0.043 (AUC 0.735) | — |
| APH(3′)-IIa Melnikov 2014 | 0.212 | 0.643 ± 0.024 (AUC 0.865) | 0.710 ± 0.026 (AUC 0.900) | — |
| DHFR Nguyen 2023 | 0.133 | 0.302 ± 0.064 (AUC 0.792) | 0.511 ± 0.129 (AUC 0.882) | — |

TEM-1 3B backbone: zero-shot 0.589, dual-modal GBM 0.735 ± 0.028 (AUC 0.873).
M3 with PCA(8) embeddings: 0.697 (AUC 0.858).

| External validation (trained on Stiffler 2015, no retraining) | 8M dual | 650M dual |
|---|---|---|
| Deng 2012 (ampicillin, n = 4,996) | 0.492 | 0.556 |
| Firnberg 2014 (ampicillin gradient, n = 4,783) | 0.820 | 0.869 |

Assay-agreement ceilings (Stiffler ↔ external assay, shared mutants):
Deng 0.611, Firnberg 0.937 (Deng–Firnberg 0.603). Jacquier 2013 is excluded
from the main analysis.

Cross-protein transfer is strong between the two β-lactamases (TEM-1 → VIM-2
0.714 / 0.773 at 8M / 650M; VIM-2 → TEM-1 0.629 / 0.769) and weak towards
AAC(3)-I. **Full 5 × 5 matrices for both backbones are given in Table 4 and
Tables S8b–c of the manuscript, and in
`results/expanded_proteins{,_650m}_evaluation.csv`.**

## Added validations (Supplementary Tables S16–S24)

- **Numbering:** ABL → precursor conversion verified against UniProt P62593
  and PDB 1BTL; corrected-anchor sensitivity |Δρ| ≤ 0.01 (S16b–c).
- **Assay ceilings:** external-validation values read against assay–assay
  agreement (S16a).
- **Structural controls:** DHFR NADPH-bound holo anchors (1RX1) roughly double
  its structural gain (+0.020 → +0.044); a Boltz-2 full-length AAC(3)-I
  homodimer and the 1BO4 crystallographic dimer both fail to rescue that
  boundary case (S16d, S18).
- **Clinical panels:** 14 clinically observed / engineered TEM variants (S17),
  and 16 combinatorially complete A42G/E104K/M182T/G238S alleles with
  cefotaxime MICs (Knies et al. 2017, *Mol. Biol. Evol.* 34:1040–1054).
  The ampicillin-trained score does **not** rank cefotaxime MIC
  (ρ = 0.183, P = 0.498, n = 16), defining a cross-drug, cross-genotype
  boundary (S19).
- **Transfer anchor controls (S22):** zero-anchoring all proteins leaves transfer unchanged or improved on average (mean Δρ +0.033 / +0.015 at 8M / 650M across all 20 directions); per-protein z-scoring retains most β-lactamase transfer (0.656–0.766).
- **Embedding dimensionality (S23):** PCA(8)/32/64/128 and all 320 raw embedding dims give Δρ = +0.001/+0.002/−0.013/−0.024/+0.007, so even the full embedding adds <0.01 ρ.
- **Robustness (S24):** five random by-position partitions ρ = 0.686 ± 0.009, five GBM seeds 0.696 ± 0.001, pooled oof bootstrap 95% CI 0.677–0.707 (variant-level; the position-clustered interval for the structural gain is wider, 0.110–0.214), Wilcoxon concordant with Tables 5–6; all intervals and the TOST equivalence tests are regenerated by `code/54_stats_intervals.py`.
- **Position-shuffle null (S21):** shuffling the position → structure mapping
  (20 permutations) removes the dual-modal gain (+0.159 → −0.097), while a
  bare position ID gives no gain (−0.010); the structural modality therefore
  carries genuine position-specific biological information.

## Changelog

- **2026-11-13 (this revision):** corrected TEM-1 anchors made canonical;
  full 13-script re-run; all main-text tables, figures and GA regenerated;
  added assay ceilings, holo/dimer controls, clinical variant and MIC panels,
  position-shuffle null; README Key results, layout and reproduce commands
  synchronised to the canonical caliber.
- **up to 2026-10-24:** archived offset-anchor build (8M 0.705 ± 0.023,
  650M 0.805 ± 0.013, GCNv2 p = 0.854). Those numbers are superseded; see
  `data/_backup_anchor70_*.csv` and Supplementary Table S16.

## Contact

Sitao Yang <yangsitaoasprin@swpu.edu.cn>; Yongfeng Ye <yongfengyip@126.com>
