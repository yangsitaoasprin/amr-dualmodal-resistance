#!/usr/bin/env python3
"""50_rerun_pipeline.py — re-run the full result pipeline on the
canonical (corrected-anchor) TEM-1 feature tables.

Every script writes its CSVs into ../results/.  Failures are logged (not
fatal) so that a partial re-run is visible.  Run from repository/code/.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
LOG = os.path.join(HERE, "..", "results", "canonical_rerun_log.txt")

SCRIPTS = [
    "22_verify_and_dump.py",
    "10_esm2_backbone_eval.py",
    "16_anchor_counterfactual.py",
    "17_contiguous_cv.py",
    "20_random_anchor_null.py",
    "23_build_delta_rho_table.py",
    "14_expand_proteins_eval.py",
    "15_expand_proteins_650m_eval.py",
    "21_expand_proteins_3b_eval.py",
    "06_extended_gcn_ramp.py",
    "07_mlp_ablation.py",
    "08_naive_gat_5fold.py",
    "03_gcnv2.py",
]

def scrub(text):
    """Make the captured output relocatable before it lands in the log.

    Child scripts print absolute output paths; the log is a released artefact
    (cited from Supplementary Table S20), so the machine-specific prefix is
    stripped to keep it free of local paths.
    """
    root = os.path.abspath(os.path.join(HERE, "..", ".."))
    for prefix in (root + os.sep, root + "/", root):
        text = text.replace(prefix, "")
    return text.replace("\\", "/")


with open(LOG, "w", encoding="utf-8") as log:
    for s in SCRIPTS:
        t0 = time.time()
        log.write("\n" + "=" * 78 + "\nRUN " + s + "\n")
        log.flush()
        try:
            p = subprocess.run([PY, s], cwd=HERE, capture_output=True,
                               text=True, timeout=1800)
            log.write(scrub(p.stdout[-6000:]))
            if p.stderr:
                log.write("\n[stderr]\n" + scrub(p.stderr[-3000:]))
            status = "OK" if p.returncode == 0 else "EXIT=%d" % p.returncode
        except subprocess.TimeoutExpired as e:
            log.write("[TIMEOUT]\n" + str(e))
            status = "TIMEOUT"
        log.write("\nSTATUS %s  (%.1f s)\n" % (status, time.time() - t0))
        print("%-36s %-8s %6.1fs" % (s, status, time.time() - t0), flush=True)
        log.flush()
print("full log ->", LOG)
