# iDISK 2.0: published counts vs. public release

The iDISK 2.0 counts reported in Hou et al. (JMIR 2025, Table 1) do not match the
public release on GitHub/Google Drive. The release does not include content derived
from the Natural Medicines Comprehensive Database (NMCD).

- **Paper**: Hou Y, Bishop JR, Liu H, Zhang R. *J Med Internet Res* 2025;27:e67677, Table 1.
- **Repo README**: the "released" statistics table in `Data/Entity/README.md` and
  `Data/Relation/README.md` at github.com/houyurain/iDISK2.0. The paper's numbers are
  still in those files, inside HTML comments.
- **Downloaded files**: row counts of the CSVs in `datasets/iDISK2.0/Data/`, header
  excluded, downloaded 2026-08-26.

## Concepts (entities)

| Concept | Paper | Repo README | Downloaded files | File in release |
|---|---:|---:|---:|---|
| Dietary supplement ingredient (DSI) | 8,091 | 7,876 | 7,876 | `DSI.csv` |
| Dietary supplement product (DSP) | 163,806 | 163,806 | 163,806 | `DSP.csv` |
| Drug | 625 | 214 | 214 | `D.csv` |
| Disease | 786 | 172 | 172 | `Dis.csv` |
| Therapeutic class | 567 | — | — | not released |
| System organ class | 17 | — | — | not released |
| Sign / symptom | 425 | 392 | 424 | `SS.csv` |
| **Total** | **174,317** | **172,460** | **172,492** | |

## Relationships

| Relationship | Paper | Repo README | Downloaded files | Source in files |
|---|---:|---:|---:|---|
| is_effective_for (DSI–Disease) | 5,245 | 828 | 827 | MSKCC 100% |
| has_therapeutic_class (DSI–TC) | 4,435 | — | — | not released |
| has_adverse_effect_on (DSI–SOC) | 2,598 | — | — | not released |
| has_adverse_reaction (DSI–SS) | 1,342 | 1,274 | 1,342 | MSKCC 100% |
| has_ingredient (DSP–DSI) | 317,062 | 317,062 | 317,062 | NHP 316,161 · DSLD 899 · both 2 |
| interacts_with (DSI–Drug) | 3,583 | 536 | 535 | MSKCC 100% |
| **Total** | **334,265** | **319,700** | **319,766** | |

The README counts for `is_effective_for` and `interacts_with` are one higher than the
files. They appear to include the CSV header line. The README's 1,274 for
`has_adverse_reaction` (and 392 for sign/symptom) does not match the file, and the
reason is unknown.

## Attributes

| Attribute | Paper (iDISK 1.0 origin) | Downloaded files |
|---|---|---|
| Effectiveness rating (DSI–Disease) | NMCD | column present, 0 / 827 filled |
| Interaction rating (DSI–Drug) | NMCD | column present, 0 / 535 filled |
| Safety (DSI) | NMCD 1,150 · MSKCC 69 (iDISK 1.0) | column present, 0 / 7,876 filled |
| Background (DSI) | NMCD + MSKCC | 279 / 7,876 (MSKCC only) |
| Mechanism of action (DSI) | MSKCC | 277 / 7,876 |
| NM_Name (DSI, Drug, Disease) | NMCD | 0 filled |

## What differs, by source

| Source | Paper | Release |
|---|---|---|
| NMCD | Included (reused from iDISK 1.0, per the paper) | Not included |
| MSKCC | Included | Included; sole source of all DSI clinical relations (279 ingredients) |
| LNHPD (NHP) | Included | Included; 99.7% of `has_ingredient` |
| DSLD | Included | Products included (92,650 with DSLD_ID); 529 products have any `has_ingredient` edge |

## Release history

From the commit history of github.com/houyurain/iDISK2.0 and the modification dates of
the Google Drive folders:

| Date | Event |
|---|---|
| 2024-04 / 2024-05 | Original data uploaded to Google Drive |
| 2025-03-19 | Paper published with the full counts |
| 2025-05-29 | README edits: TC/SOC files, NMCD example row, and the paper's statistics commented out; reduced "released" tables added |
| 2025-08-14 | In the original Drive folders, `D.csv`, `Dis.csv`, `DSI.csv`, `dsi_d.csv`, `dsi_dis.csv` replaced; TC/SOC files removed. `dsi_ss.csv`, `dsp_dsi.csv`, `DSP.csv` unchanged since 2024 |

Neither the repository nor the paper states a reason for the change.
