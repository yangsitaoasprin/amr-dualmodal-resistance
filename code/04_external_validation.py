#!/usr/bin/env python3
"""
04_external_validation.py — Cross-experiment external validation, 8M backbone:
train the dual-modal GBM on the Stiffler 2015 TEM-1 DMS assay and evaluate on
the independent Deng 2012 / Firnberg 2014 / Jacquier 2013 TEM-1 assays
without retraining.

Scope: this script reproduces the *dual-modal (8M)* column of Table 2
(Stiffler -> Deng 0.492, Stiffler -> Firnberg 0.820) and adds Jacquier 2013
(0.758), which is reported only here. The complete Table 2 — including the
ESM-2 8M zero-shot, sequence-only, 650M zero-shot and 650M dual-modal columns
— is produced by `51_collect_numbers.py` and stored in
`results/canonical_numbers.json` (key "T2"); the per-assay 650M rows are also
tabulated in `results/esm2_650m_evaluation.csv` (protocol
"external_from_Stiffler"). This script prints its values and writes no file.
"""
import os

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import spearmanr

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(BASE, "data")

FEATS_ALL = ["esm2", "blosum62", "d_hydro", "d_vol", "d_chg",
             "pLDDT", "contact_num", "d_s70", "d_omega"]

train = pd.read_csv(os.path.join(DATA, "tem1_stiffler_modeling_table.csv"))
model = GradientBoostingRegressor(random_state=0)
model.fit(train[FEATS_ALL], train["DMS_score"])

for assay in ["BLAT_ECOLX_Deng_2012", "BLAT_ECOLX_Firnberg_2014",
              "BLAT_ECOLX_Jacquier_2013"]:
    test = pd.read_csv(os.path.join(DATA, f"{assay}_modeling_table.csv"))
    rho = spearmanr(model.predict(test[FEATS_ALL]), test["DMS_score"]).statistic
    print(f"Stiffler 2015 -> {assay}: rho = {rho:.3f} (n = {len(test)})")
