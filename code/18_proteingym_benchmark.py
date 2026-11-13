#!/usr/bin/env python3
"""
18_proteingym_benchmark.py — Published zero-shot benchmark comparison.

Reproduces results/proteingym_zeroshot_benchmark.csv: Spearman rho of 18
published zero-shot variant-effect predictors on the five assays of this
study, computed from the official ProteinGym v1.3 zero-shot score files.

Input (shipped, 37 MB): data/proteingym_zs/*.csv — the five per-assay score
tables extracted from ProteinGym v1.3 (Zenodo record 15293562,
https://doi.org/10.5281/zenodo.15293562). They are byte-identical members of
the 1.78 GB archive zero_shot_substitutions_scores.zip; see
data/proteingym_zs/README.md for provenance and checksums.md5 for integrity.
Rebuild them with `code/18b_fetch_proteingym_zs.py --fetch` (HTTP Range, pulls
only these five entries) and check them with --verify. Set PG_ZS_DIR to point
at a different folder holding the same files.

The official ProteinGym ESM2_650M column reproduces our own masked-marginal
scores (scripts 09/09b) to within 0.003 rho on all five assays,
cross-validating our scoring pipeline.

Output: results/proteingym_zeroshot_benchmark.csv (the 18 x 5 zero-shot
matrix only; the "our ESM-2 650M zero-shot" and "our dual-modal 650M" columns
of Supplementary Table S11 come from results/esm2_650m_evaluation.csv /
results/esm2_650m_newproteins.csv and from Table 1/Table 3 respectively).
"""
import hashlib
import os

import pandas as pd
from scipy.stats import spearmanr

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCORES_DIR = os.environ.get("PG_ZS_DIR", os.path.join(BASE, "data",
                                                     "proteingym_zs"))
ASSAYS = {
    "TEM-1": "BLAT_ECOLX_Stiffler_2015.csv",
    "VIM-2": "A4GRB6_PSEAI_Chen_2020.csv",
    "AAC(3)-I": "AACC1_PSEAI_Dandage_2018.csv",
    "APH(3')-IIa": "KKA2_KLEPN_Melnikov_2014.csv",
    "DHFR": "DYR_ECOLI_Nguyen_2023.csv",
}
MODELS = ["ESM1v_ensemble", "GEMME", "VESPA", "VespaG", "TranceptEVE_L",
          "PoET", "MIF", "ESM-IF1", "SaProt_650M_AF2", "ProSST-2048",
          "ProtSSN_ensemble", "SiteRM", "ESM3", "ESMC-600M", "ESM2_650M",
          "ESM2_3B", "VenusREM", "RSALOR"]


def verify_scores_dir():
    """Check the shipped third-party score tables against checksums.md5.

    The five CSVs are extracted from the ProteinGym v1.3 archive and shipped so
    that this comparison runs without the 1.9 GB download. Checking their md5
    first turns "silently different inputs" into a clear failure.
    """
    manifest = os.path.join(SCORES_DIR, "checksums.md5")
    if not os.path.isfile(manifest):
        print("  checksums.md5 not found - skipping integrity check")
        return
    bad, checked = [], 0
    for line in open(manifest, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        want, name = parts[0], parts[-1].lstrip("*")
        path = os.path.join(SCORES_DIR, name)
        if not os.path.isfile(path):
            bad.append("%s (missing)" % name)
            continue
        checked += 1
        if hashlib.md5(open(path, "rb").read()).hexdigest() != want:
            bad.append("%s (md5 mismatch)" % name)
    if bad:
        raise SystemExit("proteingym_zs integrity check failed:\n  "
                         + "\n  ".join(bad)
                         + "\nRebuild with: python 18b_fetch_proteingym_zs.py --fetch")
    print("  %d score table(s) match checksums.md5" % checked)


def main():
    verify_scores_dir()
    res = {}
    for a, fn in ASSAYS.items():
        df = pd.read_csv(os.path.join(SCORES_DIR, fn))
        res[a] = {}
        for m in MODELS:
            if m in df.columns:
                d = df[["DMS_score", m]].dropna()
                res[a][m] = round(spearmanr(d.DMS_score, d[m]).statistic, 3)
    out = pd.DataFrame(res).T
    out.to_csv(os.path.join(BASE, "results",
                           "proteingym_zeroshot_benchmark.csv"))
    print(out.to_string())


if __name__ == "__main__":
    main()
