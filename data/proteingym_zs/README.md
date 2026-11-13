# ProteinGym zero-shot score tables (shipped subset)

Five per-assay score tables extracted from the ProteinGym v1.3 release, shipped
here so that the 18-predictor comparison in Supplementary Table S12 can be
reproduced without downloading the full 1.78 GiB archive.

## Provenance

| field | value |
|---|---|
| Source | ProteinGym v1.3 (Notin et al., NeurIPS 2023 Datasets & Benchmarks) |
| Record | Zenodo [10.5281/zenodo.15293562](https://doi.org/10.5281/zenodo.15293562), published 2025-04-27 |
| Archive | `zero_shot_substitutions_scores.zip`, 1,911,045,703 bytes (1.78 GiB), md5 `7912b6a1f6c80b0b78c12a15c84d96fc` |
| Licence | MIT (as declared on the record) |
| Extraction | 2026-11-13, HTTP Range reads of the remote zip central directory; only the five files below were decompressed (38.1 MB fetched, 23 requests) |

The extracted CSVs are **byte-for-byte the archive members** — no reformatting,
no column selection, no row filtering. Each file carries all 99 columns of the
published table (`mutant`, `mutated_sequence`, `DMS_score`, `DMS_score_bin`,
plus the ~95 model score columns); `code/18_proteingym_benchmark.py` selects the
18 models it reports and drops rows with missing values per model.

## Files

| file | assay (this study) | rows | md5 |
|---|---|---|---|
| `BLAT_ECOLX_Stiffler_2015.csv` | TEM-1, ampicillin | 4,996 | `886b01253df9e84546102e2bec593286` |
| `A4GRB6_PSEAI_Chen_2020.csv` | VIM-2, meropenem | 5,004 | `3e19afc6960c4e4e04e06c2de358d52d` |
| `AACC1_PSEAI_Dandage_2018.csv` | AAC(3)-I, gentamicin | 1,801 | `b77b4f5b5911a6a6b3ea58d55d56ade7` |
| `KKA2_KLEPN_Melnikov_2014.csv` | APH(3′)-IIa, kanamycin | 4,960 | `ad5ed816386a955aa9cb81fbe1c9dac6` |
| `DYR_ECOLI_Nguyen_2023.csv` | DHFR, growth selection | 2,916 | `16ca8148457e109251199a73249dbb61` |

Row counts agree with the n reported in the paper except for AAC(3)-I, where the
modelling table has 1,793 rows (8 fewer, reflecting the modelling-table filters
described in Methods; n = 1,793 in Table 1 and Table S12). The score table itself
is left unfiltered.

`checksums.md5` is the machine-readable version of the md5 column. Verify with
`md5sum -c checksums.md5` from this directory.

## Regenerating

```bash
python ../code/18b_fetch_proteingym_zs.py --fetch      # re-download (≈2 min)
python ../code/18b_fetch_proteingym_zs.py --verify     # md5 against checksums.md5
```

The script resolves `zenodo.org` itself and falls back to a record-listed IP with
the original hostname as SNI, so it also works where plain DNS for the record
host is unavailable.

## Consumer

`code/18_proteingym_benchmark.py` → `results/proteingym_zeroshot_benchmark.csv`
(18 predictors × 5 assays), which supplies the published-predictor columns of
Table S12. Override the input location with `PG_ZS_DIR` if the tables live
elsewhere.

These tables are redistributed under the upstream MIT licence (see the record
above), which is the same licence this repository uses; the measurements
themselves remain the work of the original DMS studies cited in
`../../README.md`. Please cite ProteinGym (Notin et al., NeurIPS 2023) and those
primary assays if you reuse them.
