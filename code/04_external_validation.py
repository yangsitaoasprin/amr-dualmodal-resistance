#!/usr/bin/env python3
"""
04_external_validation.py — Cross-experiment external validation:
train the dual-modal GBM on the Stiffler 2015 TEM-1 DMS assay and evaluate on
the independent Deng 2012 / Firnberg 2014 / Jacquier 2013 TEM-1 assays
(reproduces Table 2, rho up to 0.820 on Firnberg 2014).
"""
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import spearmanr

FEATS_ALL = ["esm2", "blosum62", "d_hydro", "d_vol", "d_chg",
             "pLDDT", "contact_num", "d_s70", "d_omega"]

train = pd.read_csv("../data/tem1_stiffler_modeling_table.csv")
model = GradientBoostingRegressor(random_state=0)
model.fit(train[FEATS_ALL], train["DMS_score"])

for assay in ["BLAT_ECOLX_Deng_2012", "BLAT_ECOLX_Firnberg_2014",
              "BLAT_ECOLX_Jacquier_2013"]:
    test = pd.read_csv(f"../data/{assay}_modeling_table.csv")
    rho = spearmanr(model.predict(test[FEATS_ALL]), test["DMS_score"]).statistic
    print(f"Stiffler 2015 -> {assay}: rho = {rho:.3f}")
