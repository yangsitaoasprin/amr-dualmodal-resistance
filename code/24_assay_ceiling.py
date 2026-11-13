#!/usr/bin/env python3
"""24_assay_ceiling.py — Major 5: assay-agreement ceilings.

Pairwise Spearman correlation between the three TEM-1 DMS assays
(Stiffler 2015, Deng 2012, Firnberg 2014) over shared single mutants.
These values are the practical ceilings for any cross-experiment
transfer model, and calibrate the interpretation of Table 2.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

D = "../data/"
stif = pd.read_csv(D + "tem1_stiffler_modeling_table.csv")
deng = pd.read_csv(D + "BLAT_ECOLX_Deng_2012_modeling_table.csv")
firn = pd.read_csv(D + "BLAT_ECOLX_Firnberg_2014_modeling_table.csv")

print(f"n stiffler={len(stif)}  deng={len(deng)}  firnberg={len(firn)}")

def pair(a, la, b, lb):
    m = a.merge(b, on="mutant", suffixes=("_a", "_b"))
    rho, p = spearmanr(m["DMS_score_a"], m["DMS_score_b"])
    # also on binary labels
    both = m.dropna(subset=["DMS_score_bin_a", "DMS_score_bin_b"])
    agree = (both["DMS_score_bin_a"] == both["DMS_score_bin_b"]).mean()
    print(f"{la:10s} vs {lb:10s}: n={len(m):5d}  rho={rho:.3f} (P={p:.1e})  "
          f"bin-agreement={agree:.3f} (n_bin={len(both)})")
    return rho

r1 = pair(stif, "Stiffler", deng, "Deng")
r2 = pair(stif, "Stiffler", firn, "Firnberg")
r3 = pair(deng, "Deng", firn, "Firnberg")

# For context: model performance on the same shared sets (from Table 2)
print("\n--- ceiling vs model transfer (8M dual-modal) ---")
print(f"Model on Deng     = 0.497  |  Stiffler-Deng ceiling = {r1:.3f}")
print(f"Model on Firnberg = 0.820  |  Stiffler-Firnberg ceiling = {r2:.3f}")

pd.DataFrame([
    dict(pair="Stiffler-Deng", n=4996, rho=r1),
    dict(pair="Stiffler-Firnberg", n=4783, rho=r2),
    dict(pair="Deng-Firnberg", n=4783, rho=r3),
]).to_csv("../results/assay_agreement_ceiling.csv", index=False)
print("\nsaved -> results/assay_agreement_ceiling.csv")
