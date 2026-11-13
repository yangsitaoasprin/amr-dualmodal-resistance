#!/usr/bin/env python3
"""92_audit_table3_sd.py — verify every mean +/- SD in Tables 1 and 3 against the
canonical result CSVs, and locate main-text values missing their SD.
"""
import os, re, json
import numpy as np, pandas as pd
SEQ = ["esm2", "blosum62", "d_hydro", "d_vol", "d_chg"]

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(BASE, "repository", "results")
F = lambda fn: pd.read_csv(os.path.join(R, fn))
N = json.load(open(os.path.join(R, "canonical_numbers.json"), encoding="utf-8"))

# ---------------------------------------------------- canonical rho, rho_std, auc
truth = {}

def put(protein, bb, rho, sd, auc=None):
    truth[(protein, bb)] = (rho, sd, auc)

t1 = N["T1_8m"]; t65 = N["T1_650m"]; t3 = N["T1_3b"]
put("TEM-1", "8M", t1["M3"], t1["M3_sd"], t1["M3_auc"])
put("TEM-1", "650M", t65["gbm_650m_dual"][0], t65["gbm_650m_dual"][1], t65["gbm_650m_dual"][2])
put("TEM-1", "3B", t3["gbm_3b_dual"][0], t3["gbm_3b_dual"][1], t3["gbm_3b_dual"][2])

# AAC(3)-I 8M: the anchor-free 7-feature dual model used by script 02
# (recomputed_folds_8m.csv only stores AAC's CoA-anchor ablation, not the main model)
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import spearmanr
aac = pd.read_csv(os.path.join(BASE, "repository", "data", "aac3i_modeling_table.csv"))
_fold = aac["pos"].values % 5
_rs = []
for _f in range(5):
    _tr, _te = _fold != _f, _fold == _f
    _m = GradientBoostingRegressor(random_state=0).fit(
        aac.loc[_tr, SEQ + ["pLDDT", "contact_num"]], aac.loc[_tr, "DMS_score"])
    _rs.append(spearmanr(_m.predict(aac.loc[_te, SEQ + ["pLDDT", "contact_num"]]),
                         aac.loc[_te, "DMS_score"]).statistic)
put("AAC(3)-I", "8M", float(np.mean(_rs)), float(np.std(_rs)))

# 8M other proteins from recomputed_folds_8m.csv (per-fold rho)
rf = F("recomputed_folds_8m.csv")
ASSAY = {"VIM-2": "VIM2_Chen2020", "AAC(3)-I": "AAC3I_Dandage2018"}
for p, a in ASSAY.items():
    sub = rf[(rf.assay == a) & (rf.model == "dual_full")]
    if len(sub): put(p, "8M", sub.rho.mean(), sub.rho.std(ddof=0))
for fn, tag in [("expanded_proteins_evaluation.csv", "8M"),
                ("expanded_proteins_650m_evaluation.csv", "650M"),
                ("expanded_proteins_3b_evaluation.csv", "3B")]:
    d = F(fn)
    if "protocol" not in d.columns:
        print("  (no protocol column in", fn, "- trying model column)")
        cv = d[d.model.astype(str).str.contains("dual")] if "model" in d.columns else d
    else:
        cv = d[(d.protocol.astype(str) == "cv5_modulo") &
               d.model.astype(str).str.contains("dual")]
    for _, r in cv.iterrows():
        key = ("APH(3')-IIa" if "APH3II" in str(r["assay"]) else
               "DHFR" if "DHFR" in str(r["assay"]) else None)
        if key: put(key, tag, float(r["rho"]), float(r["rho_std"]))
# 650M / 3B VIM-2 + AAC from backbone evaluation CSVs
for fn, tag in [("esm2_650m_evaluation.csv", "650M"), ("esm2_3b_evaluation.csv", "3B")]:
    d = F(fn)
    cv = d[d.protocol.astype(str) == "cv5_modulo"]
    for _, r in cv.iterrows():
        if "dual" not in str(r["model"]): continue
        key = ("VIM-2" if "VIM2" in str(r["assay"]) else
               "AAC(3)-I" if "AAC3I" in str(r["assay"]) else None)
        if key: put(key, tag, float(r["rho"]), float(r["rho_std"]))

print("=== canonical dual-modal values (rho +/- SD) ===")
for k in sorted(truth): print("  %-14s %-5s %.4f +/- %.4f" % (k[0], k[1], truth[k][0], truth[k][1]))

# ---------------------------------------------------- parse Table 3 from md
_MD = os.path.join(BASE, "paper1_submission.md")
if not os.path.exists(_MD):
    print("manuscript markdown not in this package: Table 3 text audit skipped")
    raise SystemExit(0)
m = open(_MD, encoding="utf-8").read()
t3txt = m[m.index("| Protein (drug class; anchors)"):]
t3txt = t3txt[:t3txt.index("\n\n")]
rows = []
for line in t3txt.split("\n"):
    if not line.startswith("|") or "---" in line or "Protein (drug" in line: continue
    c = [x.strip() for x in line.strip("|").split("|")]
    if len(c) < 6: continue
    prot = ("TEM-1" if "TEM-1" in c[0] else
            "VIM-2" if "VIM-2" in c[0] else
            "AAC(3)-I" if "AAC(3)-I" in c[0] else
            "APH(3')-IIa" if "APH(3" in c[0] else
            "DHFR" if "DHFR" in c[0] else None)
    if prot is None: continue
    bb = re.search(r"(8M|650M|3B)", c[0]).group(1)
    dual = re.search(r"([\d.]+)\s*\u00b1\s*([\d.]+)", c[5])   # Dual-modal GBM column
    auc = c[6]                                                  # Dual-modal AUC column
    if dual: rows.append((prot, bb, float(dual.group(1)), float(dual.group(2)), auc))
print("\n=== Table 3 verification ===")
bad = 0
for p, bb, rho, sd, auc in rows:
    key = (p, bb)
    if key not in truth:
        print(f"  {p:14s} {bb:5s} paper {rho:.3f} +/- {sd:.3f}  -> NO CANONICAL SOURCE"); continue
    tr, ts, ta = truth[key]
    ok = (abs(tr - rho) < 0.0016 and abs(ts - sd) < 0.0016)
    if not ok: bad += 1
    print(f"  {p:14s} {bb:5s} paper {rho:.3f} +/- {sd:.3f} | canonical {tr:.3f} +/- {ts:.3f} "
          + ("OK" if ok else "<-- MISMATCH") + ("  AUC paper " + auc))
print("  mismatches:", bad, "of", len(rows))

# ---------------------------------------------------- main-text values missing SD
print("\n=== main-text 5-fold values reported without SD ===")
for i, l in enumerate(m.split("\n"), 1):
    if 99 <= i <= 150:
        for pat in [r"reached \u03c1 = 0\.\d{3}(?! \u00b1)", r"reached \u03c1 = 0\.\d{3} \("]:
            for hit in re.finditer(pat, l):
                print(f"  line {i}: ...{l[max(0,hit.start()-60):hit.end()+90]}...")
