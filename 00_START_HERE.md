# Start here

This is the **code-and-data companion** to the manuscript *"Alignment-free prediction of
antibiotic resistance mutation effects by combining protein language model scores with
predicted structural context"*. It contains the code and data needed to reproduce
**every number and figure** in the paper.

The manuscript itself is not duplicated here: it is submitted through the journal system,
and a second copy could drift out of sync with the submitted version. The Supplementary
Information is included, because its tables cite files that live in this package.

---

## 1. What to read, in what order

| step | file | what it gives you |
|---|---|---|
| 1 | `supplementary_data_S1.docx` | 6 supplementary figures, 24 supplementary tables |
| 2 | `repository/README.md` | what each directory holds, environment, how to run |
| 3 | `repository/MANIFEST.md` | **one row per script**: purpose + files it writes |
| 4 | `repository/REPRODUCE.md` | commands, expected output, runtime, troubleshooting |

If you only want to check that the numbers are real, jump to §4 — it takes about five
minutes and needs nothing but Python.

---

## 2. How the scripts are named

```
<NN>_<verb>_<object>.py

44_fig4_gnn_cycle.py
│  │   └── object: what it works on (fig4_gnn_cycle)
│  └────── verb:   what it does  (build, train, eval, fetch, render,
│  └──────          collect, rerun, audit, verify, dump)
└───────── NN:     the stage, i.e. the order things run in
```

The number is a **stage index, not a version**. It exists so the directory sorts in
execution order and so a filename tells you both "when" and "what":

| stage | meaning | examples |
|---|---|---|
| 01–09 | build inputs, fit the core models | `01_build_features.py`, `03_gcnv2.py`, `08_naive_gat_5fold.py` |
| 10–19 | evaluation, external validation, cross-protein | `10_esm2_backbone_eval.py`, `16_anchor_counterfactual.py` |
| 20–29 | derived tables, controls, robustness | `20_random_anchor_null.py`, `24_assay_ceiling.py` |
| 40–47 | figures — one script per figure | `41_fig1_dualmodal.py`, `44_fig4_gnn_cycle.py`, `46_figs_s1_s6.py` |
| 49–55 | assemble the result bundle, build the document | `49_render_all_figures.py`, `51_collect_numbers.py`, `55_build_docx.py` |
| 90–95 | verification gates (read-only, produce no results) | `91_verify_table1.py`, `93_verify_manuscript.py` |

Two modules carry plain names because they are imported rather than executed:
`figstyle.py` (shared style layer) and `figstyle_core.py` (theme, sizing, export).
The suffix `b`/`c` (e.g. `09b`, `09c`) marks a parallel variant of the same stage,
used where one model had to be run separately per backbone or per protein subset.

---

## 3. Where each table and figure comes from

Script names below are relative to `repository/code/`; outputs are in
`repository/results/`. "Gate" is the script in `repository/gates/` that asserts the
values match their source.

### Main tables

| paper item | produced by | output | gate |
|---|---|---|---|
| Table 1 | `02_train_gbm.py` (8M), `10_esm2_backbone_eval.py` (650M/3B), `08_naive_gat_5fold.py` (GATv2 row), `27_audit_controls.py` (PCA row) | `recomputed_folds_8m.csv`, `esm2_650m_evaluation.csv`, `esm2_3b_evaluation.csv`, `naive_gat_5fold.csv`, `p2_pca_dims.csv` | `91_verify_table1.py` |
| Table 2 | `04_external_validation.py`, `10_esm2_backbone_eval.py` | `esm2_650m_evaluation.csv`, `canonical_numbers.json` | `93_verify_manuscript.py` |
| Table 3 | `02_train_gbm.py`, `10_esm2_backbone_eval.py`, `14/15/21_expand_proteins_*_eval.py` | `multiprotein_results.csv`, `expanded_proteins*_evaluation.csv` | `92_audit_table3_sd.py` |
| Table 4 | `14/15/21_expand_proteins_*_eval.py` | same CSVs (transfer rows) | `93_verify_manuscript.py` |
| Tables 5–6 | `03_gcnv2.py` | `gcnv2_5fold_cv.csv`, `vim2_gcnv2_5fold_cv.csv` | `92_audit_table3_sd.py` |

### Main figures

| paper item | produced by | output |
|---|---|---|
| Graphical abstract | `40_fig_ga.py` | `figures/fig0_framework_300dpi.*` |
| Fig. 1a, 1d | `22_verify_and_dump.py` | `oof_predictions_tem1_8m.npz`, `tem1_per_position_pred_vs_true.csv` |
| Fig. 1b | `02_train_gbm.py`, `10_esm2_backbone_eval.py` | the Table 1 CSVs |
| Fig. 1c | `20_random_anchor_null.py` | `permutation_importance.csv` |
| Fig. 1e | `47_render_pymol_tem1.py` | structure render (PyMOL) |
| Fig. 2a, 2b | `23_build_delta_rho_table.py` | `delta_rho_table.csv` |
| Fig. 2c | `16_anchor_counterfactual.py` | `anchor_counterfactual.csv` |
| Fig. 2d | `20_random_anchor_null.py` | `random_anchor_null.csv` |
| Fig. 3a, 3b | `18_proteingym_benchmark.py` | `proteingym_zeroshot_benchmark.csv` |
| Fig. 3c | `14/15/21_expand_proteins_*_eval.py` | transfer matrices in the same CSVs |
| Fig. 3d | `04_external_validation.py`, `17_contiguous_cv.py` | `contiguous_cv_650m.csv` |
| Fig. 4a | `06_extended_gcn_ramp.py`, `05_gat_extended.py` | `gcn_extended_ramp.csv`, `gat_extended_ramp.csv` |
| Fig. 4b | `03_gcnv2.py` | `gcnv2_5fold_cv.csv` |
| Fig. 4c | `07_mlp_ablation.py` | `mlp_ablation.csv` |
| Fig. 4d | `08_naive_gat_5fold.py` | `naive_gat_5fold.csv` |
| Fig. 5a, 5b | `23_build_delta_rho_table.py`, Table 1/3 CSVs | `delta_rho_table.csv` |
| Fig. 5c | `19_aac_anchor_experiment.py` | `aac_anchor_experiment_verified.csv` |

### Supplementary tables

| table | produced by |
|---|---|
| S1 | `08_naive_gat_5fold.py` |
| S2 | `06_extended_gcn_ramp.py` |
| S3 | `05_gat_extended.py` |
| S4 | `07_mlp_ablation.py` |
| S5, S6 | `03_gcnv2.py` (archived per-fold records) |
| S7 | `10_esm2_backbone_eval.py` |
| S8 | `14/15/21_expand_proteins_*_eval.py` |
| S9 | `16_anchor_counterfactual.py` |
| S10 | `17_contiguous_cv.py` |
| S11 | `18_proteingym_benchmark.py` |
| S12 | `19_aac_anchor_experiment.py` |
| S13, S14, S15 | `20_random_anchor_null.py` |
| S16 | `24_assay_ceiling.py` – `28_transfer_controls.py` |
| S17 | `22_verify_and_dump.py`, `53_extra_numbers.py` |
| S18 | `25_dimer_burial_boltz.py` |
| S19 | `26_multisite_mic.py` |
| S20, S20e | `50_rerun_pipeline.py`, `51_collect_numbers.py` |
| S21 | `27_audit_controls.py` |
| S22, S23, S24 | `28_transfer_controls.py` |

Confidence intervals (position-clustered bootstrap) and the GCNv2-versus-GBM equivalence tests come from `54_stats_intervals.py`, which writes `results/stats_intervals.json`.

`repository/MANIFEST.md` lists every script with the files it writes, if you prefer to
search the other way round.

---

## 4. Verify in five minutes

Nothing below writes to the manuscript; the five gates are read-only and print a verdict.

```bash
cd repository/gates
python 91_verify_table1.py        # every Table 1 cell vs its source CSV -> ALL PASS
python 92_audit_table3_sd.py      # recompute Table 3, check the SD convention -> 0 mismatches
python 93_verify_manuscript.py    # headline values, transfer directions, .docx integrity
python 94_audit_captions.py       # caption completeness (n / spread / test and P)
python 95_verify_links.py         # every path cited in the paper and SI exists
python 97_verify_si.py            # supplementary tables trace to the released results
```

Requirements: Python ≥ 3.10 with numpy, pandas, scipy, scikit-learn, matplotlib
(`pip install -r repository/requirements.txt`). No GPU, no network access.

---

## 5. Recompute the results

See `repository/REPRODUCE.md`. The short version, from the shipped feature tables:

```bash
cd repository/code
python 50_rerun_pipeline.py       # 13 result scripts -> repository/results/   (~18 min, CPU)
python 51_collect_numbers.py      # -> canonical_numbers.json
python 49_render_all_figures.py   # 12 figures -> repository/figures/          (~40 s)
```

`figures/` holds the renders used in the manuscript. Re-running the figure scripts
reproduces the same data and panel layout; typographic details can differ slightly with
the matplotlib version. Set `AMR_FIGDIR` to write the renders elsewhere.

---

## 6. What is not in this package

**The manuscript** (it travels through the journal's submission system, so that there is
exactly one authoritative version).

The authors' one-shot working scripts (internal patch utilities that rewrote released
tables or the manuscript during revision) are not included: they are not needed to
reproduce any result, and several of them overwrite released files if re-run. Everything
required for reproduction is present and runs from the shipped data.

Pre-computed ESM-2 scores and the predicted protein structures are included, so no GPU
and no model download is needed for any number or figure in the paper.
