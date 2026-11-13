# Manifest -- what every file in this repository does

Generated from the released tree; one row per script. `Outputs` lists the artefacts the script writes, so the provenance of any table or figure can be traced in one hop. Regenerate with `python audit_20261113/_make_manifest.py`.

Totals: **87 scripts, 11313 lines**, grouped by pipeline stage.

## 1. Feature construction and core models

| script | purpose | outputs |
|---|---|---|
| `code/01_build_features.py` | Build the 9-dimensional alignment-free feature table | -- |
| `code/02_train_gbm.py` | Train/evaluate the dual-modal GBM | -- |
| `code/03_gcnv2.py` | Pure-PyTorch implementation of the GCNv2 graph model on the | `gcnv2_5fold_cv.csv`, `vim2_gcnv2_5fold_cv.csv` |
| `code/04_external_validation.py` | Cross-experiment external validation, 8M backbone: | `canonical_numbers.json`, `esm2_650m_evaluation.csv` |
| `code/05_gat_extended.py` | Continue the naive GAT (GATv2-style, pure PyTorch) budget | `gat_extended_ramp.csv` |
| `code/06_extended_gcn_ramp.py` | Extended-training 2-layer GCN on the TEM-1 Cbeta | `gcn_extended_ramp.csv` |
| `code/07_mlp_ablation.py` | MLP ablation of the extended 2-layer GCN on TEM-1 | `mlp_ablation.csv` |
| `code/08_naive_gat_5fold.py` | Naive GATv2 on the TEM-1 contact graph under the | `naive_gat_5fold.csv` |
| `code/09_esm2_650m_zeroshot.py` | Compute ESM-2 zero-shot masked-marginal scores with a | `esm2_650m_zeroshot.csv`, `esm2_3b_zeroshot.csv` |
| `code/09b_esm2_650m_new_proteins.py` | ESM-2 650M masked-marginal zero-shot scores | `esm2_650m_newproteins.csv` |
| `code/09c_esm2_3b_new_proteins.py` | ESM-2 3B zero-shot scores for the two expansion | `esm2_3b_newproteins.csv` |

## 2. Result bundle and build drivers

| script | purpose | outputs |
|---|---|---|
| `code/49_render_all_figures.py` | render all 12 figures and sync them into repository/figures/ | -- |
| `code/50_rerun_pipeline.py` | re-run the 13 result scripts and write results/canonical_rerun_log.txt | -- |
| `code/51_collect_numbers.py` | collect every manuscript number into results/canonical_numbers.json | `canonical_numbers.json` |
| `code/52_canonical_report.py` | dump the numbers needed to patch the manuscript into results/canonical_report.md | -- |
| `code/53_extra_numbers.py` | remaining canonical numbers (clinical panel, controls) into canonical_extra.json | `canonical_extra.json` |
| `code/54_stats_intervals.py` | intervals and equivalence tests cited in the manuscript | `stats_intervals.json` |
| `code/55_build_docx.py` | build the submission .docx files from the markdown sources | -- |
| `code/figstyle.py` | shared style layer for the manuscript figures (thin wrapper) | -- |
| `code/figstyle_core.py` | plotting primitives used by figstyle.py | -- |

## 3. Evaluations, controls and derived tables

| script | purpose | outputs |
|---|---|---|
| `code/10_esm2_backbone_eval.py` | Evaluate ESM-2 large-backbone configurations | `esm2_650m_zeroshot.csv`, `esm2_3b_zeroshot.csv`, `esm2_650m_evaluation.csv`, `esm2_3b_evaluation.csv` |
| `code/11_expand_proteins.py` | Build modelling tables for the two expansion proteins, | -- |
| `code/12_new_proteins_struct_features.py` | Structural features + physicochemical | -- |
| `code/13_new_proteins_esm2_8m.py` | ESM-2 8M masked-marginal zero-shot scores (Eq | -- |
| `code/14_expand_proteins_eval.py` | Evaluation of the two expansion proteins | `expanded_proteins_evaluation.csv` |
| `code/15_expand_proteins_650m_eval.py` | Evaluation of the two expansion proteins | `esm2_650m_zeroshot.csv`, `esm2_650m_newproteins.csv`, `expanded_proteins_650m_evaluation.csv` |
| `code/16_anchor_counterfactual.py` | Counterfactual anchor ablation | `anchor_counterfactual.csv` |
| `code/17_contiguous_cv.py` | Contiguous-position 5-fold CV control | `contiguous_cv_650m.csv` |
| `code/18_proteingym_benchmark.py` | Published zero-shot benchmark comparison | `proteingym_zeroshot_benchmark.csv`, `esm2_650m_evaluation.csv`, `esm2_650m_newproteins.csv` |
| `code/18b_fetch_proteingym_zs.py` | Rebuild data/proteingym_zs/ from the Zenodo archive | -- |
| `code/19_aac_anchor_experiment.py` | Reverse anchor ablation for AAC(3)-I | `aac_anchor_experiment.csv` |
| `code/20_random_anchor_null.py` | Random-anchor null model + collinearity diagnostics | `random_anchor_null.csv`, `feature_collinearity.csv`, `paired_tests.csv`, `permutation_importance.csv` |
| `code/21_expand_proteins_3b_eval.py` | 3B evaluation of the expansion proteins | `esm2_3b_newproteins.csv`, `expanded_proteins_3b_evaluation.csv` |
| `code/22_verify_and_dump.py` | 数据核对 + 补算 + 折外预测落盘 | -- |
| `code/23_build_delta_rho_table.py` | 统一构建 Δρ 全表（5 蛋白 × 3 backbone = 15 组合） | -- |
| `code/24_assay_ceiling.py` | Major 5: assay-agreement ceilings | `assay_agreement_ceiling.csv` |
| `code/25_dimer_burial_boltz.py` | Item 3a: full-length dimer-aware burial for AAC(3)-I | `boltz_dimer_aac3i.csv` |
| `code/26_multisite_mic.py` | Item 3b: unseen multi-substitution genotypes and | `multisite_clinical_mic.csv` |
| `code/27_audit_controls.py` | third-round expert-review controls | `audit_position_shuffle.csv` |
| `code/28_transfer_controls.py` | P2 methodological controls (third review round) | `p2_pca_dims.csv` |

## 4. Figures

| script | purpose | outputs |
|---|---|---|
| `code/40_fig_ga.py` | graphical abstract: the four-layer framework (Figure GA) plus the 5:2 submission graphic | -- |
| `code/41_fig1_dualmodal.py` | Figure 1: dual-modal gain, importances, per-position landscape, structure | -- |
| `code/42_fig2_structural_gain.py` | Figure 2: where the structural gain comes from and where it does not | -- |
| `code/43_fig3_external_benchmark.py` | Figure 3: published zero-shot predictors, transfer matrix, external validity | -- |
| `code/44_fig4_gnn_cycle.py` | Figure 4: the diagnose-fix-validate cycle (GCN ramp, GCNv2 vs GBM, GAT runs) | -- |
| `code/45_fig5_decision_map.py` | Figure 5: when dual-modal supervision pays off (size, backbone, anchor test) | -- |
| `code/46_figs_s1_s6.py` | Figures S1-S6: feature distributions, collinearity, agreement, calibration, GNN detail, 3B panel | -- |
| `code/47_render_pymol_tem1.py` | Figure 1e: TEM-1 structure render coloured by predicted deleteriousness | -- |

## 5. Verification gates

| script | purpose | outputs |
|---|---|---|
| `gates/91_verify_table1.py` | verify every Table 1 cell against its source CSV | -- |
| `gates/92_audit_table3_sd.py` | recompute Table 3 and audit the SD caliber | -- |
| `gates/93_verify_manuscript.py` | stale-value guards, transfer directions, docx integrity | -- |
| `gates/94_audit_captions.py` | caption gate: sample size, spread, test and P value | -- |
| `gates/95_verify_links.py` | every path cited in the paper, SI and README must resolve | `xxx.csv` |
| `gates/96_verify_docs.py` | keep the documentation in step with the code and the data | -- |

## 6. Archived one-shot scripts

| script | purpose | outputs |
|---|---|---|
| `archive/_archived_guard.py` | one-shot protection for the historical patch scripts | -- |
| `archive/_caption_stats.py` | 汇总 6 条新图注所需的样本量、离散度与检验统计，全部从归档文件现算。 | -- |
| `archive/_crop_preview.py` | 按整图归一化坐标 (x0,y0,x1,y1) 裁一块并放大，用于版式定点复核。 | -- |
| `archive/_exp15_patch_manuscript.py` | [ARCHIVED, one-shot] switch the manuscript | -- |
| `archive/_exp17_make_s21.py` | [ARCHIVED, one-shot] append Supplementary Table S21 | `canonical_numbers.json`, `canonical_extra.json`, `canonical_report.md`, `canonical_rerun_log.txt` |
| `archive/_exp19_final_consistency.py` | last consistency sweep on the canonical | -- |
| `archive/_exp21_readme_numbers.py` | collect every number the public README needs, | -- |
| `archive/_exp22_round3_patch.py` | apply the third-round P0 + P1 revisions | `audit_position_shuffle.csv` |
| `archive/_exp23_round3_finish.py` | final consistency fixes after the P0/P1 patch | `esm2_650m_evaluation.csv`, `canonical_extra.json` |
| `archive/_exp24_round3_verify.py` | final verification of the P0/P1 revision | -- |
| `archive/_exp25_supp_sync.py` | [ARCHIVED, one-shot] synchronise the few | `gcnv2_5fold_cv.csv` |
| `archive/_exp27_p2_writepaper.py` | write the P2 controls into the manuscript | `p2_transfer_controls.csv`, `p2_pca_dims.csv`, `p2_robustness.json`, `p2_robustness_report.md` |
| `archive/_exp28_p2_fix_claims.py` | replace representative-direction claims with | -- |
| `archive/_exp2_clinical_tem.py` | Major 6: validation on clinically observed TEM variants | `clinical_tem_validation.csv`, `oof_predictions_tem1_8m.npz` |
| `archive/_exp31_sd_unify.py` | unify mean +/- SD reporting across the manuscript | -- |
| `archive/_exp35_table1_patch.py` | [ARCHIVED, one-shot] make Table 1 self-explanatory | `naive_gat_5fold.csv`, `esm2_650m_evaluation.csv` |
| `archive/_exp3_anchor_fix_cutoff.py` | Task 20 + Task 16 | `anchor_fix_cutoff_sensitivity.csv` |
| `archive/_exp4_anchor_fix_650m.py` | Task 20 (650M backbone check) | `esm2_650m_zeroshot.csv` |
| `archive/_exp5_holo_oligomer.py` | Task 18: competing-explanation tests | -- |
| `archive/_exp6_vim2_gcnv2.py` | resume the interrupted unified-caliber GCNv2 rerun | `vim2_gcnv2_5fold_cv.csv` |
| `archive/_exp9_patch_canonical.py` | [ARCHIVED, one-shot] make the corrected TEM-1 | -- |
| `archive/_mkpreview.py` | 把 600 dpi 成图缩成可复核的预览 + 按版面行裁三张局部放大图。 | -- |
| `archive/_patch_fig1.py` | Figure 1 第二轮版式修订（一次性补丁，逐块精确替换 + 校验）。 | -- |
| `archive/_patch_fig1b.py` | Figure 1 第三轮定点修订：ROC 插图 / (b) 底注 / (c) pLDDT 记法。 | -- |
| `archive/_patch_fig4.py` | Figure 4 定点修订：口径统一 + 面板号避让 + 架构图文字溢出。 | -- |
| `archive/_patch_submission.py` | paper1_submission.md 修订：图号重映射 + top-10 数据订正 + 6 张新图与图注。 | -- |
| `archive/_patch_supp.py` | 补充图 S2/S5：imshow -> 矢量 Rectangle，去掉连续 colorbar 的栅格渐变。 | -- |
| `archive/_patch_supp2.py` | 补充图二轮修订：S2 颜色语义、S3 样本量口径、S1/S3/S4/S5 面板号统一。 | -- |
| `archive/_patch_supp_md.py` | 把 6 张补充图与图注插入 supplementary_data_S1.md，并生成中英对照图注文件。 | -- |
| `archive/_pymol_smoke.py` | PyMOL 渲染冒烟测试：无头环境下加载 TEM-1 AlphaFold 模型并出一张图。 | -- |
| `archive/_tmp_inventory.py` | Temporary READ-ONLY script (stage 8): what produced the archived GBM column? | -- |
| `archive/_tmp_inventory_layout.py` | 临时探针：检查 Figure 1(d) 底部留白与数据范围，以及 PyMOL 图的空白区域。 | -- |
| `archive/_verify_top10.py` | 核对正文 §3.4 与 Figure 1 图注的 top-10 位点集合与重合数。 | -- |
