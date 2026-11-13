#!/usr/bin/env python3
"""49_render_all_figures.py — re-render every figure on the canonical
result bundle, report the 9-gate audit result of each, and sync the
delivered copies into repository/figures/.

The sync step exists because repository/README.md advertises
`figures/ # 300-dpi PNG + vector PDF: GA, Figures 1-5, S1-S6`, but the
figure scripts all write to the project root. Without this copy the
shipped directory silently kept pre-revision renders (it still held an
Oct-22 framework figure while the manuscript used the Nov-13 one).
"""
import os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIGDIR = os.path.join(ROOT, "repository", "figures")

SCRIPTS = ["40_fig_ga.py", "41_fig1_dualmodal.py", "42_fig2_structural_gain.py", "43_fig3_external_benchmark.py",
           "44_fig4_gnn_cycle.py", "45_fig5_decision_map.py", "46_figs_s1_s6.py"]
for s in SCRIPTS:
    p = subprocess.run([sys.executable, s], cwd=HERE, capture_output=True,
                       text=True, timeout=900)
    tail = (p.stdout or "")[-900:].strip().replace("\n", " | ")
    print(f"{s:22s} exit={p.returncode}  {tail[:400]}")
    if p.returncode != 0:
        print("   STDERR:", (p.stderr or "")[-500:].replace("\n", " | "))

# ---- deliver exactly the 12 figures the manuscript and SI embed ----
STEMS = ["fig0_framework_300dpi",
         "fig1_dualmodal_300dpi", "fig2_structural_gain_300dpi",
         "fig3_external_benchmark_300dpi", "fig4_gnn_cycle_300dpi",
         "fig5_decision_map_300dpi",
         "figS1_feature_distributions_300dpi", "figS2_collinearity_300dpi",
         "figS3_zeroshot_scatter_300dpi", "figS4_calibration_dca_300dpi",
         "figS5_gnn_full_300dpi", "figS6_3b_panel_300dpi"]

print("\n=== sync -> repository/figures/ ===")
os.makedirs(FIGDIR, exist_ok=True)
missing, copied = [], 0
for stem in STEMS:
    for ext in (".png", ".pdf"):
        src = os.path.join(ROOT, stem + ext)
        if not os.path.exists(src):
            missing.append(stem + ext)
            continue
        shutil.copy2(src, os.path.join(FIGDIR, stem + ext))
        copied += 1
print(f"  copied {copied} file(s); missing {len(missing)}")
for m in missing:
    print("  MISSING:", m)
print("  SYNC:", "OK" if not missing else "INCOMPLETE")
