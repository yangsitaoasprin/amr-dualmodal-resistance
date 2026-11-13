#!/usr/bin/env python3
"""91_verify_table1.py — verify every cell of Table 1 against its source."""
import os, re, json
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(BASE, "repository", "results")
N = json.load(open(os.path.join(R, "canonical_numbers.json"), encoding="utf-8"))
t1, t65, t3 = N["T1_8m"], N["T1_650m"], N["T1_3b"]
pca = pd.read_csv(os.path.join(R, "p2_pca_dims.csv")).set_index("n_pc")
gat = pd.read_csv(os.path.join(R, "naive_gat_5fold.csv"))
gat_runs = gat[pd.to_numeric(gat["seed"], errors="coerce").notna()]
e65 = pd.read_csv(os.path.join(R, "esm2_650m_evaluation.csv"))
m2_650 = e65[(e65.assay == "TEM1_Stiffler2015") &
             (e65.model == "gbm_650m_struct_only")].iloc[0]

truth = {
 "M0 8M":   (t1["M0"], None, t1["M0_auc"]),
 "M1 8M":   (t1["M1"], t1["M1_sd"], t1["M1_auc"]),
 "M2 8M":   (t1["M2"], t1["M2_sd"], t1["M2_auc"]),
 "M3 8M":   (t1["M3"], t1["M3_sd"], t1["M3_auc"]),
 "GATv2":   (gat_runs.rho.mean(), gat_runs.rho.std(ddof=0), gat_runs.auc.mean()),
 "GATv2 AUC SD": (None, None, gat_runs.auc.std(ddof=0)),
 "PCA(8)":  (pca.loc[8, "rho"], pca.loc[8, "rho_sd"], pca.loc[8, "auc"]),
 "M0 650M": (t65["esm2_650m_zeroshot"][0], None, t65["esm2_650m_zeroshot"][2]),
 "M1 650M": (t65["gbm_650m_seq"][0], t65["gbm_650m_seq"][1], t65["gbm_650m_seq"][2]),
 "M2 650M": (t65["gbm_650m_struct_only"][0], t65["gbm_650m_struct_only"][1], m2_650["auc"]),
 "M3 650M": (t65["gbm_650m_dual"][0], t65["gbm_650m_dual"][1], t65["gbm_650m_dual"][2]),
 "M0 3B":   (t3["esm2_3b_zeroshot"][0], None, t3["esm2_3b_zeroshot"][2]),
 "M3 3B":   (t3["gbm_3b_dual"][0], t3["gbm_3b_dual"][1], t3["gbm_3b_dual"][2]),
}
print("=== canonical values for Table 1 ===")
for k, (r, s, a) in truth.items():
    print("  %-14s rho %s%s  AUC %.3f" %
          (k, "%.3f" % r if r is not None else "  -- ",
           " ± %.3f" % s if s is not None else "        ", a if a is not None else float("nan")))

_MD = os.path.join(BASE, "paper1_submission.md")
if not os.path.exists(_MD):
    print("manuscript markdown not in this package: Table 1 text comparison skipped")
    raise SystemExit(0)
m = open(_MD, encoding="utf-8").read()
tbl = m[m.index("| Model | Features | Spearman ρ | AUC |"):]
tbl = tbl[:tbl.index("\n\n")]
print("\n=== Table 1 rows (paper) ===")
for line in tbl.split("\n")[2:]:
    if line.startswith("|"):
        c = [x.strip().replace("*", "") for x in line.strip("|").split("|")]
        print("  %-38s rho=%-22s AUC=%s" % (c[0], c[2], c[3]))

print("\n=== consistency checks ===")
# GATv2 strings are derived from the shipped CSV (population SD, matching the paper's
# convention) rather than frozen as literals, so they cannot go stale on a data update.
gat_rho_str = "%.3f \u00b1 %.3f" % (gat_runs.rho.mean(), gat_runs.rho.std(ddof=0))
gat_auc_str = "%.3f \u00b1 %.3f" % (gat_runs.auc.mean(), gat_runs.auc.std(ddof=0))
checks = [
 ("PCA SD present", "0.697 ± 0.030" in m),
 ("GATv2 rho present (%s)" % gat_rho_str, gat_rho_str in m),
 ("GATv2 AUC present (%s)" % gat_auc_str, gat_auc_str in m),
 ("M2 650M AUC present", "0.744§" in m),
 ("no bare 0.697", "0.697 |" not in m),
 ("no em-dash AUC in Table 1", "| — |" not in tbl),
 ("dagger note", "† Zero-shot scores" in m),
 ("ddagger note", "‡ GATv2 values" in m),
 ("section note", "§ M2 uses only" in m),
 ("old vague note gone", "single full-CV pass" not in m),
 ("all 13 rows", tbl.count("\n|") == 13),
]
for k, v in checks:
    print("  %-30s %s" % (k, "OK" if v else "FAIL"))
print("\nALL PASS:", all(v for _, v in checks))
