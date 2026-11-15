#!/usr/bin/env python3
"""97_verify_si.py -- keep the supplementary tables tied to the released results.

Read-only. Three kinds of check:

  (c) structure  -- Tables S1-S24 are all present and six supplementary figures
                    are embedded, so a truncated file cannot pass by default;
  (a) targeted   -- S7a/S7d are compared cell by cell against the backbone
                    evaluation CSVs, S17 row by row against the clinical table,
                    S4 against the MLP ablation, S16a against the assay-agreement
                    ceilings, and S24 against the interval in stats_intervals.json;
  (b) traceability -- every decimal in every other table must appear in the pool
                    of numbers held in results/*.csv and results/*.json, unless it
                    is listed in ALLOWED with a reason.

ALLOWED exists because some cells are differences of two neighbouring cells, or
were computed on the fly and never written to results/. Recording the reason keeps
each exception visible instead of turning it into an invisible tolerance.

Exit code 0 on pass, 1 on any mismatch.
"""
import glob
import json
import os
import re
import sys

import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(BASE, "repository", "results")
SI = os.path.join(BASE, "supplementary_data_S1.md")

ALLOWED = {
    ("Table S9", 0.045): "difference of two table cells (0.789 - 0.744)",
    ("Table S9", 0.201): "difference of two table cells (0.576 - 0.375)",
    ("Table S9", 0.066): "difference of two table cells (0.347 - 0.281)",
    ("Table S13", 0.045): "DHFR anchor contrast, computed ad hoc; not in results/",
    ("Table S13", 0.699): "mean of the three 650M random-anchor seeds",
    ("Table S13", 0.559): "mean of the three 650M random-anchor seeds",
    ("Table S16", 0.325): "DHFR holo-anchor rerun; recomputed ad hoc",
    ("Table S16", 0.213): "DHFR holo + crystal burial; recomputed ad hoc",
    ("Table S16", 0.776): "binary-label agreement, derived from the two assays",
    ("Table S16", 0.774): "binary-label agreement, derived from the two assays",
    ("Table S16", 0.960): "binary-label agreement, derived from the two assays",
    ("Table S20", 0.448): "paired t-test p computed from the five folds",
}

fails = []
si = open(SI, encoding="utf-8").read()


def numbers_in_tables(text):
    """(line, value, decimals) for every decimal inside a markdown table."""
    for line in text.split("\n"):
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        for cell in line.strip().strip("|").split("|"):
            for m in re.finditer(r"(?<![\d.])(\d+\.\d+)(?![\d])", cell):
                yield line, float(m.group(1)), len(m.group(1).split(".")[1])


def collect_pool():
    """Every number in results/*.csv (numeric columns) and results/*.json,
    pre-rounded to 2-7 decimals so a displayed value can be matched at its own
    precision."""
    pool = set()

    def add(f):
        for nd in range(2, 8):
            pool.add(round(f, nd))

    for path in glob.glob(os.path.join(RES, "*.csv")):
        try:
            d = pd.read_csv(path)
        except Exception:
            continue
        for c in d.columns:
            if pd.api.types.is_numeric_dtype(d[c]):
                for v in d[c].dropna():
                    add(float(v))

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            add(float(o))

    for path in glob.glob(os.path.join(RES, "*.json")):
        try:
            walk(json.load(open(path, encoding="utf-8")))
        except Exception:
            continue
    return pool


def section(start_marker, end_marker):
    i = si.index(start_marker)
    j = si.index(end_marker, i + len(start_marker)) if end_marker else len(si)
    return si[i:j]


# ------------------------------------------------------------------ (c) structure
print("=== (c) SI structure ===")
absent = [n for n in range(1, 25) if not re.search(r"## Table S%d\." % n, si)]
n_figs = len(re.findall(r"!\[", si))
print("  tables S1-S24: %d/24 | figures: %d" % (24 - len(absent), n_figs))
if absent:
    fails.append("SI is missing Table S%d" % absent[0])
if n_figs < 6:
    fails.append("SI has %d figures, expected 6" % n_figs)

# ------------------------------------------------------------------ (a) targeted
print("\n=== (a) targeted tables ===")

# S4 -- MLP ablation, three seeds plus mean and SD
mlp = pd.read_csv(os.path.join(RES, "mlp_ablation.csv"))
want = ([round(r.train_rho, 3) for r in mlp.itertuples()]
        + [round(r.test_rho, 3) for r in mlp.itertuples()]
        + [round(mlp.train_rho.mean(), 3), round(mlp.test_rho.mean(), 3),
           round(mlp.train_rho.std(ddof=0), 3), round(mlp.test_rho.std(ddof=0), 3)])
got = [v for _, v, _ in numbers_in_tables(section("## Table S4", "## Table S5"))]
missing = [v for v in want if v not in got]
print("  S4 vs mlp_ablation.csv: %d values, %d missing" % (len(want), len(missing)))
if missing:
    fails.append("Table S4 does not reproduce mlp_ablation.csv (%s)" % missing[:3])

# S7a / S7d -- backbone evaluation, cell by cell
ASSAY_LABEL = {
    "TEM-1 Stiffler 2015": "TEM1_Stiffler2015",
    "TEM-1 Deng 2012": "TEM1_Deng2012",
    "TEM-1 Firnberg 2014": "TEM1_Firnberg2014",
    "VIM-2 Chen 2020": "VIM2_Chen2020",
    "AAC(3)-I Dandage 2018": "AAC3I_Dandage2018",
}
for tag, csv_name, ender in (("S7a", "esm2_650m_evaluation.csv", "### S7b"),
                             ("S7d", "esm2_3b_evaluation.csv", "### S7e")):
    path = os.path.join(RES, csv_name)
    if not os.path.exists(path):
        continue
    d = pd.read_csv(path)   # zero-shot sits under full_assay, the GBM rows under cv5_modulo
    seg = section("### " + tag, ender)
    n_cells = bad = 0
    for line in seg.split("\n"):
        if not line.startswith("| "):
            continue
        label = line.strip("| ").split("|")[0].strip()
        assay = ASSAY_LABEL.get(label)
        if not assay:
            continue
        rows = d[d.assay == assay]
        if rows.empty:
            continue
        expect = []
        for model in ("zeroshot", "seq", "dual", "struct_only"):
            hit = rows[rows.model.astype(str).str.contains(model, regex=False)]
            if "protocol" in rows.columns and model != "zeroshot":
                hit = hit[hit.protocol == "cv5_modulo"]
            if hit.empty:
                continue
            r = hit.iloc[0]
            expect.append(float(r["rho"]))
            if model != "zeroshot":
                expect.append(float(r["rho_std"]))
            if model != "struct_only" and pd.notna(r.get("auc")):
                expect.append(float(r["auc"]))
        cells = [float(x) for x in re.findall(r"\d+\.\d+", line)]
        for w, g in zip(expect, cells):
            n_cells += 1
            if abs(round(w, 3) - g) > 1e-9:      # displayed value must be the 3-dp rounding
                bad += 1
    print("  %s vs %s: %d cells compared, %d off" % (tag, csv_name, n_cells, bad))
    if bad:
        fails.append("%s disagrees with %s in %d cell(s)" % (tag, csv_name, bad))

# S16a -- assay-agreement ceilings, rho and n
ceil = pd.read_csv(os.path.join(RES, "assay_agreement_ceiling.csv"))
seg = section("### S16a", "### S16b")
bad16 = 0
for r in ceil.itertuples():
    if ("%.3f" % r.rho) not in seg:
        bad16 += 1
    if ("{:,}".format(int(r.n))) not in seg:
        bad16 += 1
print("  S16a vs assay_agreement_ceiling.csv: %d checks, %d missing" % (2 * len(ceil), bad16))
if bad16:
    fails.append("Table S16a disagrees with assay_agreement_ceiling.csv (%d cells)" % bad16)

# S17 -- clinical variants, row by row
cli = pd.read_csv(os.path.join(RES, "clinical_tem_validation_canonical.csv")).set_index("ABL")
seg = section("## Table S17", "## Table S18")
rows = [l for l in seg.split("\n") if l.startswith("| ")
        and "---" not in l and "Substitution" not in l]
bad17 = 0
for line in rows:
    c = [x.strip() for x in line.strip("|").split("|")]
    if c[0] not in cli.index:
        bad17 += 1
        continue
    for cell, col in ((c[4], "pred_pct"), (c[5], "meas_pct")):
        if abs(round(cli.loc[c[0], col] * 100, 1) - float(cell)) > 0.051:
            bad17 += 1
print("  S17 vs clinical_tem_validation_canonical.csv: %d rows, %d mismatches"
      % (len(rows), bad17))
if bad17:
    fails.append("Table S17 disagrees with the clinical table in %d cell(s)" % bad17)

# S24 -- the structural-gain interval must be the position-clustered one
st = json.load(open(os.path.join(RES, "stats_intervals.json"), encoding="utf-8"))
lo, hi = st["structural_gain"]["position_cluster_bootstrap_95CI"]
want_ci = "(%.3f\u2013%.3f)" % (lo, hi)
print("  S24 interval %s present: %s" % (want_ci, want_ci in si))
if want_ci not in si:
    fails.append("Table S24 does not quote the position-clustered interval %s" % want_ci)

# ------------------------------------------------------------------- (b) general
print("\n=== (b) every SI decimal traces to results/ ===")
pool = collect_pool()
print("  pool: %d numbers" % len(pool))
unmatched = []
parts = re.split(r"\n## (Table S\d+)\.", si)
for i in range(1, len(parts), 2):
    tid, body = parts[i], parts[i + 1]
    if tid == "Table S17":           # covered row by row above
        continue
    for line, val, dp in numbers_in_tables(body):
        if 2 <= dp <= 7 and val in pool:
            continue
        if (tid, val) in ALLOWED:
            continue
        unmatched.append((tid, val, line.strip("| ")[:60]))
print("  unmatched: %d" % len(unmatched))
for tid, val, ctx in unmatched[:8]:
    print("    %s  %.3f  %s" % (tid, val, ctx))
if unmatched:
    fails.append("%d SI value(s) cannot be traced to results/ and are not allow-listed"
                 % len(unmatched))

# ---------------------------------------------------------------------- verdict
print()
if fails:
    print("RESULT: FAIL (%d problem(s))" % len(fails))
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("RESULT: PASS")
