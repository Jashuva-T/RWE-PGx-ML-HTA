"""Model factories and fold-consistent cross-validation (identical folds across models/conditions)."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def make(name: str, params: dict, seed: int):
    if name == "xgboost":
        return XGBClassifier(**params, random_state=seed, eval_metric="logloss", verbosity=0)
    if name == "random_forest":
        return RandomForestClassifier(**params, random_state=seed)
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(**params))   # scaler fit inside each fold
    raise ValueError(name)

def folds(df: pd.DataFrame, k: int, seed: int):
    strata = df.ancestry.astype(str) + "_" + df.outcome.astype(str)
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(df, strata))

def oof_predict(df, cols, model_name, cfg, fold_idx):
    p = np.zeros(len(df)); fid = np.zeros(len(df), int)
    X, y = df[cols].to_numpy(float), df.outcome.to_numpy()
    for i, (tr, te) in enumerate(fold_idx):
        m = make(model_name, cfg["model_params"][model_name], cfg["seed"]).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]; fid[te] = i
    return p, fid
