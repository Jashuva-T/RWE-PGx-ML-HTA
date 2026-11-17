"""Load data, validate the schema and build condition-specific feature matrices."""
from __future__ import annotations
import pandas as pd

REQUIRED = ["patient_id", "pair", "ancestry", "outcome"]
CAT_REQUIRED = ["pair", "variant", "source"]
COVARIATES = ["age", "sex", "comorbidity"]

def validate(df, cat, cfg):
    miss = [c for c in REQUIRED if c not in df] + [c for c in CAT_REQUIRED if c not in cat]
    if miss: raise ValueError(f"missing columns: {miss}")
    if not set(df.outcome.unique()) <= {0, 1}: raise ValueError("outcome must be 0/1")
    bad = set(df.ancestry) - set(cfg["ancestries"])
    if bad: raise ValueError(f"unknown ancestry labels: {bad}")
    if df.patient_id.duplicated().any(): raise ValueError("duplicate patient_id")
    unknown = set(cat.variant) - set(df.columns)
    if unknown: raise ValueError(f"catalog variants absent from patients table: {sorted(unknown)[:5]}")

def features(df_pair, cat_pair, cond: dict):
    cols = cat_pair.loc[cat_pair.source.isin(cond["variant_sources"]), "variant"].tolist()
    if cond["covariates"]:
        cols += [c for c in COVARIATES if c in df_pair]
    return cols
