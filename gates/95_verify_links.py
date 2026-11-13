#!/usr/bin/env python3
"""95_verify_links.py -- provenance-link gate for the submission package.

Every path the manuscript, the supplementary information and the README point
at must resolve to a real file in the released tree, otherwise a reviewer
cannot follow the provenance chain. This gate extracts path-like tokens
(code/..., gates/..., archive/..., results/..., data/..., figures/... plus
bare *.csv / *.py filenames in backticks) and reports the ones that do not
exist. Exit code 1 if anything dangles.
"""
import os
import re
import sys

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPO = os.path.join(BASE, "repository")

DOCS = ["paper1_submission.md", "supplementary_data_S1.md",
        os.path.join("repository", "README.md"),
        os.path.join("repository", "00_START_HERE.md")]
N_DOCS = sum(os.path.exists(os.path.join(BASE, d)) for d in DOCS)

# code/xxx.py, gates/xxx.py, archive/xxx.py, results/xxx.csv, data/xxx, figures/xxx
PATH_RE = re.compile(
    r"`((?:code|gates|archive|results|data|figures)/[A-Za-z0-9_.\-/{}*]+)`"
    r"|`([A-Za-z0-9_\-]+\.(?:py|csv|json|npz|pdb|png|pdf))`")

# directories a bare filename may live in
SEARCH = ["repository/code", "repository/gates", "repository/archive",
          "repository/results", "repository/data", "repository/figures", "."]

missing, checked = [], set()
for doc in DOCS:
    p = os.path.join(BASE, doc)
    if not os.path.exists(p):
        continue
    txt = open(p, encoding="utf-8").read()
    for m in PATH_RE.finditer(txt):
        token = m.group(1) or m.group(2)
        if token in checked or any(ch in token for ch in "{}*"):
            continue
        checked.add(token)
        # 1) literal path relative to repository/ or to the project root
        cands = [os.path.join(REPO, token), os.path.join(BASE, token)]
        # 2) bare filename anywhere in the released tree
        cands += [os.path.join(BASE, d, token) for d in SEARCH]
        if not any(os.path.exists(c) for c in cands):
            missing.append((doc, token))

print("documents scanned : %d" % N_DOCS)
print("distinct paths    : %d" % len(checked))
if missing:
    print("\nDANGLING REFERENCES (%d):" % len(missing))
    for doc, token in missing:
        print("  %-28s %s" % (os.path.basename(doc), token))
else:
    print("\nno dangling references")
print("\nRESULT:", "PASS" if not missing else "FAIL")
sys.exit(0 if not missing else 1)
