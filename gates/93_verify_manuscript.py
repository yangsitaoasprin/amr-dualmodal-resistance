#!/usr/bin/env python3
"""93_verify_manuscript.py — comprehensive verification after P0/P1/P2."""
import os, re, json, zipfile
import numpy as np, pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(BASE, "repository", "results")
_MD = os.path.join(BASE, "paper1_submission.md")
HAS_MANUSCRIPT = os.path.exists(_MD)
m = open(_MD, encoding="utf-8").read() if HAS_MANUSCRIPT else ""
s = open(os.path.join(BASE, "supplementary_data_S1.md"), encoding="utf-8").read()

print("=== 1. S22 'published' column vs official transfer CSVs (all 20 directions) ===")
T = pd.read_csv(os.path.join(R, "p2_transfer_controls.csv"))
map8 = {"TEM-1": "TEM1_Stiffler2015", "VIM-2": "VIM2_Chen2020",
        "AAC(3)-I": "AAC3I_Dandage2018", "APH(3')-IIa": "APH3II_Melnikov2014",
        "DHFR": "DHFR_Nguyen2023"}
src = pd.concat([pd.read_csv(os.path.join(R, "expanded_proteins_evaluation.csv")),
                 pd.read_csv(os.path.join(R, "expanded_proteins_650m_evaluation.csv"))],
                ignore_index=True)
src = src[src.protocol.astype(str) == "cross_protein"]
ref = {}
for _, r in src.iterrows():
    parts = r["assay"].split("->")
    key = ([k for k, v in map8.items() if v == parts[0]][0] + "->" +
           [k for k, v in map8.items() if v == parts[1]][0])
    bb = "650m" if "650m" in str(r["model"]) else "8m"
    ref[(bb, key)] = float(r["rho"])
bad = 0
for bb in ["8m", "650m"]:
    sub = T[(T.backbone == bb) & (T["mode"] == "as_published")]
    for _, r in sub.iterrows():
        exp = ref.get((bb, r["pair"]))
        if exp is None or abs(exp - r["rho"]) > 1e-6:
            bad += 1; print("  MISMATCH", bb, r["pair"], r["rho"], exp)
print("  directions checked:", 40, "| mismatches:", bad)

print("\n=== 2. canonical headline numbers ===")
N = json.load(open(os.path.join(R, "canonical_numbers.json"), encoding="utf-8"))
E = json.load(open(os.path.join(R, "canonical_extra.json"), encoding="utf-8"))
checks = [
    ("8M M3", "0.696 \u00b1 0.032", True), ("650M M3", "0.814 \u00b1 0.014", True),
    ("3B M3", "0.735 \u00b1 0.028", True), ("GCNv2 TEM1", "0.701 \u00b1 0.033", True),
    ("GCNv2 p", "p = 0.448", True), ("VIM2 p", "p = 0.294", True),
    ("PCA AUC", "0.855 \u2192 0.858", True), ("11 mutants", "11 multi-substitution", True),
    ("BH-FDR", "Benjamini\u2013Hochberg", True), ("S20e", "## Table S20e", False),
    ("S21", "## Table S21", False), ("S22", "## Table S22", False),
    ("S23", "## Table S23", False), ("S24", "## Table S24", False),
    ("Feng 2022", "Feng,S. *et al.* (2022)", True),
    ("1BO4 species", "*Serratia marcescens*", True),
    ("EUCAST version", "version 16.1 was published", True),
]
for label, key, in_main in checks:
    target = m if in_main else s
    print(f"  {label:18s}", "OK" if key in target else "MISS")

if not HAS_MANUSCRIPT:
    print("\n(manuscript markdown not in this package: text-level sections skipped; "
          "the CSV-level checks above still apply)")
    raise SystemExit(0)

print("\n=== 3. stale values in main text (must be 0) ===")
STALE = ["0.856", "0.369", "12 multi-substitution", "0.315/0.416",
         "three drug classes", "p = 0.854", "0.871", "correctly rank clinically",
         "by analogy with serine", "confirm the stability prior", "0.705 \u00b1 0.023",
         # round 12 (2026-11-13): values superseded by re-derivation from the shipped tables
         "0.366 \u00b1 0.162", "0.694 \u00b1 0.089",   # GATv2 SD was ddof=1
         "0.686 \u00b1 0.009",                        # random-partition SD
         "0.507 \u00b1 0.029", "AUC 0.937",           # VIM-2 8M seq / 650M AUC
         "plateaued at 0.59", "0.58\u20130.66",       # naive-GAT ramp mis-statement
         "one run with negative correlation",
         "\u22120.30 to \u22120.66"]                   # collinearity lower bound
for k in STALE:
    n = m.count(k)
    print(f"  {k:30s} {n} {'OK' if n == 0 else '<-- CHECK'}")

print("\n=== 4. figure / caption / reference inventory ===")
print("  main images:", len(re.findall(r"^!\[", m, re.M)),
      "| supp images:", len(re.findall(r"^!\[", s, re.M)))
print("  main captions:", len(re.findall(r"^\*\*Figure", m, re.M)),
      "| supp captions:", len(re.findall(r"^\*\*Figure S", s, re.M)))
refs = m[m.index("## References"):]
print("  reference entries:", len(re.findall(r"^[A-Z][A-Za-z\u00fc\u00f6\u00e4\u00e9'-]+,", refs, re.M)))

print("\n=== 5. deliverable docx ===")
# Accept the author's working copy name as well as the shipped one, so the gate
# runs both in the project tree and in a clean reviewer package.
_cands = ["paper1_submission_P2_latest.docx", "paper1_submission.docx",
          "supplementary_data_S1.docx"]
_present = [f for f in _cands if os.path.exists(os.path.join(BASE, f))]
if not _present:
    print("  no .docx found next to the markdown sources")
for f in _present:
    z = zipfile.ZipFile(os.path.join(BASE, f))
    t = re.sub("<[^>]+>", "", z.read("word/document.xml").decode("utf-8"))
    print(f"  {f:38s} media={len([i for i in z.infolist() if 'media' in i.filename])} "
          f"| S22={'Table S22' in t} | S24={'Table S24' in t} | 0.858={'0.858' in t} "
          f"| GATsd156={'0.156' in t} | stale162={'0.162' in t}")
