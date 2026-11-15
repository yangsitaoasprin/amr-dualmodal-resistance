# Reproduction guide

Two paths, depending on what you want to check.

* **Path A — numbers and figures (recommended).** Starts from the shipped
  feature tables in `data/`. CPU only, ~20 minutes end to end (measured: the
  driver below takes 18 min on a laptop, of which `03_gcnv2.py` is ~6.5 min),
  reproduces every value in the manuscript tables and every figure.
* **Path B — from raw DMS assays.** Rebuilds the feature tables from ProteinGym
  and AlphaFold structures. Needs the raw archives and, for the 650M/3B
  baselines, a GPU for a few hours.

Both paths are deterministic: scripts 06–08 fix seeds 0/1/2, `03_gcnv2.py` and
the GBMs use `random_state = 0`, and the modulo-5 by-position split is defined
by position index, not by a random draw.

---

## Path A — from the shipped feature tables (CPU)

```bash
conda env create -f environment.yml && conda activate amr-dual
cd code
```

| # | command | writes | expected | time |
|---|---|---|---|---|
| 1 | `python 50_rerun_pipeline.py` | the 13 per-result CSVs in `results/` | log ends `13/13 OK` in `results/canonical_rerun_log.txt` | ~18 min |
| 2 | `python 51_collect_numbers.py` | `results/canonical_numbers.json` | 8M M3 ρ = 0.696, 650M 0.814, 3B 0.735 | <5 s |
| 2b | `python 54_stats_intervals.py` | `results/stats_intervals.json` | cluster CI 0.110–0.214; TOST P = 0.0007 / 0.0152 | ~20 s |
| 3 | `python 49_render_all_figures.py` | GA, Figures 1–5, S1–S6 (PNG+PDF) | 7 scripts `exit=0`, each `2/2` delivery gate, `SYNC: OK` (24 files) | ~40 s |
| 4 | `cd ../gates && python 91_verify_table1.py` | — | `ALL PASS: True` | ~5 s |
| 5 | `python 92_audit_table3_sd.py` | — | `mismatches: 0 of 15` | ~40 s |
| 6 | `python 93_verify_manuscript.py` | — | no `CHECK` lines; stale-value guards all 0 | ~10 s |
| 7 | `python 94_audit_captions.py` | — | `ALL: PASS` (6/6 captions) | ~2 s |
| 8 | `python 97_verify_si.py` | — | `RESULT: PASS` (SI tables vs results/) | ~8 s |
| 9 | `python 96_verify_docs.py` | — | `RESULT: PASS` (MANIFEST, README numbers, cited names) | ~3 s |
| 10 | `python 95_verify_links.py` | — | `no dangling references` | ~1 s |
| 11 | `cd ../code && python 55_build_docx.py` | `paper1_submission.docx`, `supplementary_data_S1.docx` | `OK, 6 pictures, 12 descr slots, 6 captions mapped` | ~20 s |

Step 3 rewrites the PNG/PDF files in `figures/`. Bitwise comparison of the PNGs
before and after a re-run is the strongest signal that nothing drifted; the PDFs
carry a `/CreationDate`, so their md5 changes on every build even when the
content is identical. To render without touching `figures/`, set `AMR_FIGDIR`
(see below).

Figure rendering uses `figstyle_core.py` from the repository; the scripts export
PNG and PDF with no additional package. `figures/` holds the renders used in the
manuscript — re-running reproduces the same data and panel layout, though
typesetting details (font ladder, a few label positions) can differ slightly.
To keep those files untouched, render to a scratch directory:

```bash
AMR_FIGDIR=/tmp/figs python 44_fig4_gnn_cycle.py
```

---

## Path B — from raw assays (rebuild `data/`)

Only needed if you want to regenerate the feature tables themselves.

```bash
cd code
python 18b_fetch_proteingym_zs.py --fetch    # 1.9 GB ProteinGym archive -> data/proteingym_zs/
python 01_build_features.py                  # DMS + AlphaFold -> 9-dim feature table
python 11_expand_proteins.py                 # parse APH(3')-IIa / DHFR assays
python 12_new_proteins_struct_features.py
```

The ESM-2 embeddings and zero-shot scores are the only GPU steps:

```bash
export TORCH_HOME="$HOME/.cache/torch"       # point at existing checkpoints
python 09_esm2_650m_zeroshot.py               # 650M (and --model esm2_t36_3B_UR50D for 3B)
python 09b_esm2_650m_new_proteins.py
python 09c_esm2_3b_new_proteins.py
python 13_new_proteins_esm2_8m.py
```

They are **not** required for Path A: the resulting CSVs are shipped in
`results/`, so the manuscript numbers can be reproduced on a laptop.

---

## What not to run

`archive/` holds the one-shot scripts that produced intermediate states of the
manuscript. They are kept for provenance only. Each guarded script exits with an
explanation unless you set `AMR_ALLOW_ARCHIVED_PATCH=1`, because re-running them
would revert later corrections (manuscript patches) or duplicate appended
tables (`_exp17_make_s21.py`) or overwrite the released feature tables
(`_exp9_patch_canonical.py`).

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `ModuleNotFoundError: figstyle` | ran a figure script from the wrong directory | `cd code` first; the scripts import `figstyle` from their own folder |
| `FileNotFoundError: ../results/...` | ran a pipeline script from the repository root | `cd code` |
| figure PNG differs from the shipped one | different matplotlib version | compare numbers, not pixels; the 9/9 style gate still applies |
| `REFUSING TO RUN: ... archived one-shot patch` | you ran something from `archive/` | intended; see *What not to run* |
