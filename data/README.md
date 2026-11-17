# Data

**No patient-level data are distributed.** eMERGE-PGx data are controlled-access (dbGaP / eMERGE network
data-use agreements); PharmGKB and ClinVar annotations have their own terms. Run `make demo` for the synthetic demonstration.

## Schema for your approved extract (`data/raw/`, git-ignored)

`patients.csv` - one row per patient **per gene-drug pair**:

| column | description |
|---|---|
| `patient_id` | unique per row |
| `pair` | one of `CYP2C19_clopidogrel`, `CYP2C9_VKORC1_warfarin`, `SLCO1B1_statins`, `TPMT_thiopurines`, `DPYD_fluoropyrimidines` |
| `ancestry` | genetic-ancestry group: `EUR, SAS, AFR, EAS, AMR` (edit `ancestries` in the config if you use other labels) |
| `outcome` | 0/1 drug-response phenotype defined with the standardized eMERGE criteria |
| one column per variant | allele dosage 0/1/2 |
| `age`, `sex`, `comorbidity` | optional clinical covariates (used in the RWE-augmented condition) |

`variant_catalog.csv`: `pair, variant, gene, source` where `source` is `pharmgkb_1AB` (PharmGKB Level 1A/1B),
`clinvar` (ClinVar pharmacogene annotation) or `rwe` (variants only present in the real-world data).
**Curated-only** = `pharmgkb_1AB` + `clinvar`; **RWE-augmented** = all variants + covariates (see `conditions` in the config).
