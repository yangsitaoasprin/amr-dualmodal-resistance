# Canonical (corrected-anchor) numbers — full report

## GNN diagnostics
- naive GATv2 15 runs: 0.366 +/- 0.162
- extended GCN plateau (epoch 60): test 0.664, train 0.793
- naive GAT ramp at last epoch: 0.593; max 0.602
- MLP ablation: train 0.904, test 0.557

## Figure 1 / Section 3.5
- pooled oof rho = 0.692
- per-position rho = 0.730 (n=263)
- predicted most-deleterious top10 = [130, 129, 65, 68, 232, 132, 71, 73, 124, 167]
- experimental top10 = [71, 177, 164, 134, 155, 128, 68, 43, 129, 64]
- overlap = [68, 71, 129] (3/10)

## Anchor counterfactual (canonical)
              assay backbone          model      rho  rho_std      auc
  TEM1_Stiffler2015       8m            seq 0.536826 0.030331 0.778409
  TEM1_Stiffler2015       8m      dual_full 0.696263 0.031689 0.857748
  TEM1_Stiffler2015       8m dual_noanchors 0.695915 0.027349 0.859735
  TEM1_Stiffler2015     650m            seq 0.757658 0.019782 0.898974
  TEM1_Stiffler2015     650m      dual_full 0.814057 0.013774 0.925277
  TEM1_Stiffler2015     650m dual_noanchors 0.807800 0.016779 0.922367
      VIM2_Chen2020       8m            seq 0.506466 0.029156 0.779020
      VIM2_Chen2020       8m      dual_full 0.729260 0.052967 0.902967
      VIM2_Chen2020       8m dual_noanchors 0.740775 0.044131 0.907014
      VIM2_Chen2020     650m            seq 0.744389 0.022579 0.910256
      VIM2_Chen2020     650m      dual_full 0.795192 0.035009 0.937532
      VIM2_Chen2020     650m dual_noanchors 0.789425 0.039380 0.934160
APH3II_Melnikov2014       8m            seq 0.374797 0.077413 0.714440
APH3II_Melnikov2014       8m      dual_full 0.643180 0.024347 0.865477
APH3II_Melnikov2014       8m dual_noanchors 0.575856 0.068498 0.833369
APH3II_Melnikov2014     650m            seq 0.641818 0.048820 0.855812
APH3II_Melnikov2014     650m      dual_full 0.709821 0.026356 0.899619
APH3II_Melnikov2014     650m dual_noanchors 0.691483 0.034060 0.891658
    DHFR_Nguyen2023       8m            seq 0.281168 0.064472 0.725245
    DHFR_Nguyen2023       8m      dual_full 0.301569 0.063612 0.791757
    DHFR_Nguyen2023       8m dual_noanchors 0.346728 0.085015 0.792083
    DHFR_Nguyen2023     650m            seq 0.535541 0.078838 0.870827
    DHFR_Nguyen2023     650m      dual_full 0.510762 0.128922 0.882157
    DHFR_Nguyen2023     650m dual_noanchors 0.500609 0.157986 0.885862

## Random-anchor null (canonical)
            assay backbone  seed      rho  rho_std  true_anchor_rho
TEM1_Stiffler2015       8m     0 0.702158 0.030258         0.696263
TEM1_Stiffler2015       8m     1 0.688621 0.018479         0.696263
TEM1_Stiffler2015       8m     2 0.685682 0.029041         0.696263
TEM1_Stiffler2015       8m     3 0.682866 0.043909         0.696263
TEM1_Stiffler2015       8m     4 0.667263 0.035152         0.696263
TEM1_Stiffler2015       8m     5 0.700555 0.029654         0.696263
TEM1_Stiffler2015       8m     6 0.701441 0.023250         0.696263
TEM1_Stiffler2015       8m     7 0.693980 0.022868         0.696263
TEM1_Stiffler2015       8m     8 0.668898 0.044604         0.696263
TEM1_Stiffler2015       8m     9 0.678994 0.031322         0.696263
TEM1_Stiffler2015     650m     0 0.811744 0.017143         0.814057
TEM1_Stiffler2015     650m     1 0.801635 0.017144         0.814057
TEM1_Stiffler2015     650m     2 0.801785 0.019361         0.814057
TEM1_Stiffler2015     650m     3 0.793235 0.017011         0.814057
TEM1_Stiffler2015     650m     4 0.791538 0.021111         0.814057
TEM1_Stiffler2015     650m     5 0.809528 0.014635         0.814057

## Test-set permutation importance (canonical)
backbone  fold     feature  importance
      8m     0        esm2    0.118661
      8m     0    blosum62    0.179277
      8m     0     d_hydro    0.021311
      8m     0       d_vol    0.028585
      8m     0       d_chg    0.017576
      8m     0       pLDDT    0.037060
      8m     0 contact_num    0.581430
      8m     0       d_s70   -0.003803
      8m     0     d_omega    0.033740
      8m     1        esm2    0.155995
      8m     1    blosum62    0.218469
      8m     1     d_hydro    0.010555

## Feature collinearity (canonical)
              assay      feat_x      feat_y  spearman
  TEM1_Stiffler2015 contact_num       d_s70 -0.622915
  TEM1_Stiffler2015 contact_num     d_omega -0.301180
  TEM1_Stiffler2015       d_s70     d_omega  0.754260
  TEM1_Stiffler2015       pLDDT contact_num  0.660783
      VIM2_Chen2020 contact_num       d_s70 -0.425106
      VIM2_Chen2020 contact_num     d_omega -0.434821
      VIM2_Chen2020       d_s70     d_omega  0.303561
      VIM2_Chen2020       pLDDT contact_num  0.755118
APH3II_Melnikov2014 contact_num       d_s70 -0.663156
APH3II_Melnikov2014 contact_num     d_omega -0.103097
APH3II_Melnikov2014       d_s70     d_omega  0.452644
APH3II_Melnikov2014       pLDDT contact_num  0.556832

## Contiguous CV 650M (canonical)
            assay    model      rho  rho_std      auc
TEM1_Stiffler2015 zeroshot 0.731492      NaN 0.882732
TEM1_Stiffler2015      seq 0.762558 0.065014 0.900762
TEM1_Stiffler2015   struct 0.468855 0.028590 0.731724
TEM1_Stiffler2015     dual 0.800137 0.043780 0.920728
    VIM2_Chen2020 zeroshot 0.738172      NaN 0.904645
    VIM2_Chen2020      seq 0.694705 0.164627 0.900365
    VIM2_Chen2020   struct 0.516128 0.128676 0.801010
    VIM2_Chen2020     dual 0.743738 0.114328 0.926046

## Cross-protein transfer (canonical)
### expanded_proteins_evaluation.csv
                                 assay      protocol            model      rho  rho_std      auc                                note
                   APH3II_Melnikov2014    full_assay esm2_8m_zeroshot 0.211863      NaN 0.631849                                 NaN
                   APH3II_Melnikov2014    cv5_modulo       gbm_8m_seq 0.374797 0.077413 0.714440 folds=0.479,0.390,0.255,0.422,0.327
                   APH3II_Melnikov2014    cv5_modulo    gbm_8m_struct 0.562513 0.023664 0.817395 folds=0.593,0.582,0.559,0.554,0.525
                   APH3II_Melnikov2014    cv5_modulo      gbm_8m_dual 0.643180 0.024347 0.865477 folds=0.668,0.668,0.629,0.647,0.604
                       DHFR_Nguyen2023    full_assay esm2_8m_zeroshot 0.132777      NaN 0.564261                                 NaN
                       DHFR_Nguyen2023    cv5_modulo       gbm_8m_seq 0.281168 0.064472 0.725245 folds=0.218,0.376,0.209,0.271,0.331
                       DHFR_Nguyen2023    cv5_modulo    gbm_8m_struct 0.234190 0.079451 0.711956 folds=0.385,0.228,0.205,0.150,0.203
                       DHFR_Nguyen2023    cv5_modulo      gbm_8m_dual 0.301569 0.063612 0.791757 folds=0.362,0.342,0.344,0.192,0.268
      TEM1_Stiffler2015->VIM2_Chen2020 cross_protein      gbm_8m_dual 0.714163      NaN      NaN                                 NaN
  TEM1_Stiffler2015->AAC3I_Dandage2018 cross_protein      gbm_8m_dual 0.246536      NaN      NaN                                 NaN
TEM1_Stiffler2015->APH3II_Melnikov2014 cross_protein      gbm_8m_dual 0.592360      NaN      NaN                                 NaN
    TEM1_Stiffler2015->DHFR_Nguyen2023 cross_protein      gbm_8m_dual 0.386851      NaN      NaN                                 NaN
### expanded_proteins_650m_evaluation.csv
                                 assay      protocol              model      rho  rho_std      auc                                note
                   APH3II_Melnikov2014    full_assay esm2_650m_zeroshot 0.601436      NaN 0.844029                                 NaN
                   APH3II_Melnikov2014    cv5_modulo       gbm_650m_seq 0.641818 0.048820 0.855812 folds=0.642,0.698,0.596,0.694,0.579
                   APH3II_Melnikov2014    cv5_modulo    gbm_650m_struct 0.562513 0.023664 0.817395 folds=0.593,0.582,0.559,0.554,0.525
                   APH3II_Melnikov2014    cv5_modulo      gbm_650m_dual 0.709821 0.026356 0.899619 folds=0.733,0.728,0.693,0.729,0.666
                       DHFR_Nguyen2023    full_assay esm2_650m_zeroshot 0.544218      NaN 0.886307                                 NaN
                       DHFR_Nguyen2023    cv5_modulo       gbm_650m_seq 0.535541 0.078838 0.870827 folds=0.649,0.577,0.558,0.439,0.454
                       DHFR_Nguyen2023    cv5_modulo    gbm_650m_struct 0.234190 0.079451 0.711956 folds=0.385,0.228,0.205,0.150,0.203
                       DHFR_Nguyen2023    cv5_modulo      gbm_650m_dual 0.510762 0.128922 0.882157 folds=0.645,0.543,0.643,0.333,0.389
      TEM1_Stiffler2015->VIM2_Chen2020 cross_protein      gbm_650m_dual 0.772772      NaN      NaN                                 NaN
  TEM1_Stiffler2015->AAC3I_Dandage2018 cross_protein      gbm_650m_dual 0.446252      NaN      NaN                                 NaN
TEM1_Stiffler2015->APH3II_Melnikov2014 cross_protein      gbm_650m_dual 0.634003      NaN      NaN                                 NaN
    TEM1_Stiffler2015->DHFR_Nguyen2023 cross_protein      gbm_650m_dual 0.523510      NaN      NaN                                 NaN

## delta_rho_table (canonical, first 20 rows)
              assay backbone  rho_seq  rho_struct  rho_dual     delta  sd_delta        p    n  auc_seq  auc_struct  auc_dual       label    cls
  TEM1_Stiffler2015       3b 0.652062    0.495936  0.734573  0.082512  0.028833 0.004612 4996 0.837658    0.748065  0.873447       TEM-1   beta
  TEM1_Stiffler2015     650m 0.757658    0.495936  0.814057  0.056399  0.017150 0.002766 4996 0.898974    0.748065  0.925277       TEM-1   beta
  TEM1_Stiffler2015       8m 0.536826    0.495936  0.696263  0.159437  0.031214 0.000517 4996 0.778409    0.748065  0.857748       TEM-1   beta
      VIM2_Chen2020       3b 0.716017    0.614461  0.775472  0.059455  0.013705 0.000971 5004 0.896229    0.838784  0.928369       VIM-2   beta
      VIM2_Chen2020     650m 0.744389    0.614461  0.795192  0.050803  0.019222 0.006146 5004 0.910256    0.838784  0.937532       VIM-2   beta
      VIM2_Chen2020       8m 0.506466    0.614461  0.729260  0.222794  0.049346 0.000833 5004 0.779020    0.838784  0.902967       VIM-2   beta
  AAC3I_Dandage2018       3b 0.508620    0.146340  0.491408 -0.017212  0.035145 0.382800 1793 0.759839    0.582102  0.756357    AAC(3)-I  amino
  AAC3I_Dandage2018     650m 0.476748    0.146340  0.446468 -0.030279  0.058492 0.358997 1793 0.748613    0.582102  0.735246    AAC(3)-I  amino
  AAC3I_Dandage2018       8m 0.242895    0.146340  0.251394  0.008499  0.075844 0.833646 1793 0.638672    0.582102  0.639900    AAC(3)-I  amino
APH3II_Melnikov2014       3b 0.683952    0.562513  0.715806  0.031854  0.015908 0.016067 4960 0.878693    0.817395  0.903688 APH(3′)-IIa  amino
APH3II_Melnikov2014     650m 0.641818    0.562513  0.709821  0.068003  0.029101 0.009493 4960 0.855812    0.817395  0.899619 APH(3′)-IIa  amino
APH3II_Melnikov2014       8m 0.374797    0.562513  0.643180  0.268383  0.062423 0.001005 4960 0.714440    0.817395  0.865477 APH(3′)-IIa  amino
    DHFR_Nguyen2023       3b 0.525108    0.234190  0.495692 -0.029416  0.063240 0.404883 2916 0.872612    0.711956  0.884477        DHFR folate
    DHFR_Nguyen2023     650m 0.535541    0.234190  0.510762 -0.024778  0.064415 0.484609 2916 0.870827    0.711956  0.882157        DHFR folate
    DHFR_Nguyen2023       8m 0.281168    0.234190  0.301569  0.020401  0.098361 0.699550 2916 0.725245    0.711956  0.791757        DHFR folate