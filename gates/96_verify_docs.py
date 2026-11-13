#!/usr/bin/env python3
"""96_verify_docs.py -- keep the documentation in step with the code and the data.

Three checks, all read-only:

  (a) every script under code/, gates/ and archive/ has exactly one MANIFEST row,
      and every MANIFEST row names a script that exists;
  (b) the numbers in README's "Key results" table match the CSVs in results/,
      rounded to three decimals;
  (c) every script name cited in the documentation (README, START_HERE,
      REPRODUCE, MANIFEST) exists on disk.

Runs both in the authors' tree and in a reviewer package. The package does not
ship archive/, so archive rows are reported as "not distributed" there rather
than as failures; everything else must match in both places.

Exit code 0 on pass, 1 on any mismatch.
"""
import os
import re
import sys

import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPO = os.path.join(BASE, "repository")
RES = os.path.join(REPO, "results")

fails = []


def scripts_on_disk():
    found = set()
    for sub in ("code", "gates", "archive"):
        d = os.path.join(REPO, sub)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith(".py"):
                found.add("%s/%s" % (sub, fn))
    return found


# ---------------------------------------------------------------- (a) MANIFEST
print("=== (a) MANIFEST rows vs scripts on disk ===")
man_path = os.path.join(REPO, "MANIFEST.md")
man = open(man_path, encoding="utf-8").read()
rows = {"%s/%s" % (a, b) for a, b in
        re.findall(r"\| `(code|gates|archive)/([^`]+\.py)` \|", man)}
disk = scripts_on_disk()
has_archive = os.path.isdir(os.path.join(REPO, "archive"))

missing_rows = sorted(disk - rows)
orphan_rows = sorted(r for r in rows - disk)
if not has_archive:
    orphan_rows = [r for r in orphan_rows if not r.startswith("archive/")]
    print("  archive/ not present in this copy: its rows are not counted")

print("  scripts: %d | rows: %d" % (len(disk), len(rows)))
if missing_rows:
    fails.append("no MANIFEST row for: " + ", ".join(missing_rows))
if orphan_rows:
    fails.append("MANIFEST row without a script: " + ", ".join(orphan_rows))
print("  %s" % ("OK" if not missing_rows and not orphan_rows else "MISMATCH"))

# ------------------------------------------------------------- (b) README numbers
print("\n=== (b) README key-results table vs results/*.csv ===")

ASSAY_KEYS = {
    "TEM-1": "TEM1_Stiffler2015",
    "VIM-2": "VIM2_Chen2020",
    "AAC(3)-I": "AACC1_PSEAI_Dandage_2018",
    "APH(3\u2032)-IIa": "APH3II_Melnikov2014",
    "DHFR": "DHFR_Nguyen2023",
}


def load():
    """Return lookup[(assay, backbone, model)] -> dict(rho, sd, auc)."""
    out = {}
    mp = pd.read_csv(os.path.join(RES, "multiprotein_results.csv"))
    for _, r in mp.iterrows():
        key = {"TEM-1 (Amp)": "TEM1_Stiffler2015",
               "VIM-2 PSEAI (Amp)": "VIM2_Chen2020",
               "AAC(3)-I (Gent)": "AACC1_PSEAI_Dandage_2018"}.get(r["protein"])
        if key:
            out[(key, "8m", "dual")] = dict(rho=r["dual"], sd=None, auc=r["auc_dual"])
            out[(key, "8m", "zeroshot")] = dict(rho=r["zeroshot"], sd=None, auc=r["auc0"])
    for fn, bb in (("expanded_proteins_evaluation.csv", "8m"),
                   ("expanded_proteins_650m_evaluation.csv", "650m")):
        d = pd.read_csv(os.path.join(RES, fn))
        for _, r in d.iterrows():
            model = str(r["model"])
            if "zeroshot" in model:
                kind = "zeroshot"
            elif model.endswith("dual"):
                kind = "dual"
            elif model.endswith("_seq"):
                kind = "seq"
            else:
                continue          # structure-only rows are not in the README table
            out[(r["assay"], bb, kind)] = dict(rho=r["rho"], sd=r.get("rho_std"), auc=r.get("auc"))
    e6 = pd.read_csv(os.path.join(RES, "esm2_650m_evaluation.csv"))
    e6 = e6[(e6.protocol == "cv5_modulo") & (e6.model == "gbm_650m_dual")]
    for _, r in e6.iterrows():
        out[(r["assay"], "650m", "dual")] = dict(rho=r["rho"], sd=r["rho_std"], auc=r["auc"])
    for fn, assay in (("gcnv2_5fold_cv.csv", "TEM1_Stiffler2015"),
                      ("vim2_gcnv2_5fold_cv.csv", "VIM2_Chen2020")):
        d = pd.read_csv(os.path.join(RES, fn))
        out[(assay, "gcnv2", "gcnv2")] = dict(rho=d["GCNv2_rho"].mean(),
                                              sd=d["GCNv2_rho"].std(ddof=1), auc=None)
    return out


def num(cell, pattern=r"(\d\.\d+)"):
    m = re.search(pattern, cell)
    return float(m.group(1)) if m else None


lookup = load()
readme = open(os.path.join(REPO, "README.md"), encoding="utf-8").read()
checked = 0
for line in readme.splitlines():
    if not line.startswith("| "):
        continue
    cells = [c.strip() for c in line.strip("|").split("|")]
    if len(cells) < 5:
        continue
    label = cells[0].replace("*", "")
    assay = next((v for k, v in ASSAY_KEYS.items() if label.startswith(k)), None)
    if not assay:
        continue
    # zero-shot, 8M dual, 650M dual
    for cell, bb, col in ((cells[1], "8m", "zeroshot"),
                          (cells[2], "8m", "dual"),
                          (cells[3], "650m", "dual")):
        ref = lookup.get((assay, bb, col))
        if not ref:
            continue
        got = num(cell)
        if got is None:
            continue
        checked += 1
        if round(float(ref["rho"]), 3) != got:
            fails.append("README %s / %s %s rho: table %.3f vs csv %.3f"
                         % (label, bb, col, got, ref["rho"]))
        auc_txt = re.search(r"AUC (\d\.\d+)", cell)
        if auc_txt and ref.get("auc") is not None and pd.notna(ref["auc"]):
            checked += 1
            if round(float(ref["auc"]), 3) != float(auc_txt.group(1)):
                fails.append("README %s / %s auc: table %s vs csv %.3f"
                             % (label, bb, auc_txt.group(1), ref["auc"]))
print("  values compared: %d" % checked)
if checked == 0:
    fails.append("README table parsed to zero comparisons - the check itself is broken")
print("  %s" % ("OK" if not any(f.startswith("README") for f in fails) else "MISMATCH"))

# ------------------------------------------------------- (c) cited script names
print("\n=== (c) script names cited in the documentation ===")
DOCS = ["README.md", "00_START_HERE.md", "REPRODUCE.md", "MANIFEST.md"]
PAT = re.compile(r"`((?:code/|gates/|archive/)?[A-Za-z0-9_]+\.py)`")
known = {p.split("/")[-1] for p in scripts_on_disk()}
# names that live in archive/: absent from a reviewer package by design
archived = {r.split("/")[-1] for r in rows if r.startswith("archive/")}
cited = 0
for doc in DOCS:
    p = os.path.join(REPO, doc)
    if not os.path.exists(p):
        continue
    for name in set(PAT.findall(open(p, encoding="utf-8").read())):
        base = name.split("/")[-1]
        cited += 1
        if base in known:
            continue
        if not has_archive and base in archived:
            continue          # authored but not distributed; not a broken reference
        fails.append("%s cites a script that does not exist: %s" % (doc, name))
print("  names checked: %d" % cited)
print("  %s" % ("OK" if not any("cites" in f for f in fails) else "MISMATCH"))

# ------------------------------------------------------------------- verdict
print()
if fails:
    print("RESULT: FAIL (%d problem(s))" % len(fails))
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("RESULT: PASS")
