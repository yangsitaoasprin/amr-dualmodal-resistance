# Model weights used in this study — provenance and checksums

The model weights themselves are **not redistributed in this archive**. They are
third-party artefacts with their own licences and their own release channels, and
nothing in the paper needs them to be re-hosted: every reported number and figure
is reproduced from the tables in `data/` and `results/`, which ship with this
release.

This file records, for each weight file we used, its size, its MD5 and where it
came from, so that a reader who wants to re-run inference can obtain exactly the
same bytes and verify them.

## Weights

| file | model | bytes | MD5 | used for |
|---|---|---|---|---|
| `esm2_t6_8M_UR50D.pt` | ESM-2 8M | 30,099,493 | `8039fc9cee7f71cd2633b13b5a38ff50` | zero-shot scores for the CPU pipeline (Tables 1–4; all GNN comparisons) |
| `esm2_t33_650M_UR50D.pt` | ESM-2 650M | 2,604,537,549 | `ba6d997e29db07a2ad9dca20e024b102` | zero-shot scores, headline configuration (Tables 1 and 3) |
| `esm2_t36_3B_UR50D.pt` | ESM-2 3B | 5,678,116,398 | `d37a0d0dbe7431e48a72072b9180b16b` | zero-shot scores, scale check (Tables 1 and 3) |
| `esmfold_3B_v1.pt` | ESMFold | 2,771,653,574 | `07861d5df73124a876d7d7d7c6062372` | structure-prediction experiments during method development |
| `boltz2_conf.ckpt` | Boltz-2 confidence model | 2,286,561,469 | `2f0a1775bf8fc366a1a85e2019eca288` | dimer-aware burial control (Supplementary Table S18) |
| `boltz2_aff.ckpt` | Boltz-2 affinity module | 2,062,139,170 | `8e93dadedd6edb7a4d170f6051b99ec0` | shipped with Boltz-2; not used for any reported number |
| `boltz1_conf.ckpt` | Boltz-1 confidence model | 3,595,352,714 | `81c6339c7553bec981ca62c5e9e78db5` | early structure experiments during method development |
| `mols.tar` | Boltz chemical-component library | 1,855,662,080 | `5399454dcbbba695ab01f88deede577c` | Boltz runtime dependency |
| `ccd.pkl` | Boltz CCD cache | 345,859,128 | `aa43b8ba95608ce179033e990e39b4f3` | Boltz runtime dependency |

Total ≈ 21.2 GB. `weights_checksums.md5` holds the same MD5 values in `md5sum`
format.

## Where to obtain them

**ESM-2 and ESMFold.** Distributed by Meta via the `fair-esm` package. The loader
in `esm/pretrained.py` (fair-esm 2.0.0, MIT licence) fetches each file from

```
https://dl.fbaipublicfiles.com/fair-esm/models/<file>
```

so `esm2_t33_650M_UR50D.pt`, for example, is
`https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt`.
Installing `fair-esm` and calling the corresponding `esm.pretrained.*()` function
downloads the same bytes and caches them under `~/.cache/torch/hub/checkpoints/`.

**Boltz-1 and Boltz-2.** Distributed with the Boltz project
(https://github.com/jwohlwend/boltz) and cached under `~/.boltz/`. We do not
restate release URLs here because they change between Boltz releases; take them
from that repository and check the MD5 above, which identifies the exact bytes
this study used.

## Licences

The weights remain under the licences of their upstream releases (the `fair-esm`
code and weights are released under the MIT licence). This archive does not
relicense them and does not redistribute them; consult the upstream release
before any further use.

## Verifying a download

```bash
md5sum -c weights_checksums.md5
```

## Reproducing the paper

None of the above is required. The released feature tables, per-fold results and
figure scripts reproduce every number from `data/` and `results/` alone:

```bash
cd code && python 50_rerun_pipeline.py   # ~18 min, CPU
cd ../gates && python 91_verify_table1.py   # and 92–96
```
