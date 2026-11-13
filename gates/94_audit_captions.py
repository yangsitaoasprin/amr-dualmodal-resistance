# -*- coding: utf-8 -*-
"""94_audit_captions.py -- caption style gate for the six main-text captions.

Extracts the Graphical-abstract caption and the Figure 1-5 captions from
paper1_submission.md and checks the three elements Bioinformatics requires of
every caption: sample size (n), what the error bars / +/- spread mean, and the
statistical test with its P value.

Scoring: uses the caption checker exposed by `figstyle` (see
`code/figstyle_core.py`), which tests the same three elements.
No data values are checked here -- that is what the numeric gates do.
"""
import io
import os
import re
import sys

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# The caption checker ships with the figure-style layer, so the gate uses
# whatever implementation that layer resolves to.
_CODE = os.path.join(BASE, "repository", "code")
sys.path.insert(0, _CODE)
import figstyle as _F                      # noqa: E402

C = _F.caption_style
_MD = os.path.join(BASE, "paper1_submission.md")
if not os.path.exists(_MD):
    print("manuscript markdown not in this package: caption audit skipped")
    raise SystemExit(0)
md = io.open(_MD, encoding="utf-8").read()

caps = []
# Graphical abstract
m = re.search(r"\*\*Figure GA\.[^\n]*", md, re.S)
if m:
    caps.append(("Figure GA", m.group(0).replace("**", ""), None))
# Figure 1-5
for m in re.finditer(r"\*\*(Figure [1-5]\..*?)\*\*(.*?)(?=\n\n|\Z)", md, re.S):
    caps.append(("Figure " + m.group(1).split(".")[0].split()[-1],
                 m.group(1) + m.group(2), None))

print("found %d captions\n" % len(caps))
allok = True
for name, text, kind in caps:
    kw = {"journal": "Bioinformatics"}
    if kind:
        kw["kind"] = kind
    r = C.audit_caption(text, **kw)
    miss = [x["name"] for x in r["missing"]]
    ok = r["ok"]
    allok &= ok
    print("%-12s %-5s style=%-12s n=%-5s spread=%-5s test=%-5s schematic=%s"
          % (name, "PASS" if ok else "FAIL", r["style"],
             r["hits"].get("n"), r["hits"].get("spread"), r["hits"].get("test"),
             r["is_schematic"]))
    if miss:
        print("            missing:", miss)
    if r["warn"]:
        print("            warn:", r["warn"][:2])
    print("            len=%d  head: %s..." % (len(text), text[:75]))
print("\nALL:", "PASS" if allok and len(caps) == 6 else "CHECK NEEDED")
