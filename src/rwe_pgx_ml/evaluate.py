"""Discrimination, calibration, ancestry-stratified metrics, paired bootstrap, decision curves."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

def auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan

def calibration(y, p, bins=10):
    p = np.clip(p, 1e-6, 1 - 1e-6); lg = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(lg, y)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1)); idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    ece = sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())
    return {"brier": float(brier_score_loss(y, p)), "cal_slope": float(lr.coef_[0, 0]), "cal_intercept": float(lr.intercept_[0]), "ece": float(ece)}

def metrics_table(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall, by_anc = [], []
    for (pair, model, cond), g in oof.groupby(["pair", "model", "condition"]):
        y, p = g.y.to_numpy(), g.p.to_numpy()
        overall.append({"pair": pair, "model": model, "condition": cond, "n": len(g), "auc": auc(y, p), **calibration(y, p)})
        for a, h in g.groupby("ancestry"):
            by_anc.append({"pair": pair, "model": model, "condition": cond, "ancestry": a, "n": len(h), "n_events": int(h.y.sum()), "auc": auc(h.y.to_numpy(), h.p.to_numpy())})
    return pd.DataFrame(overall), pd.DataFrame(by_anc)

def _macro_auc(groups, idxs):
    return np.nanmean([auc(g.y.to_numpy()[i], g.p.to_numpy()[i]) for g, i in zip(groups, idxs)])

def paired_delta(oof, model, cfg, rng):
    """Macro-average (over gene-drug pairs) AUC: RWE-augmented minus curated-only, paired bootstrap CI."""
    pairs = cfg["pairs"]; B = cfg["bootstrap"]["n"]
    gs = {c: [oof[(oof.model == model) & (oof.condition == c) & (oof.pair == p)].sort_values("patient_id") for p in pairs] for c in ("rwe_augmented", "curated_only")}
    n = [len(g) for g in gs["rwe_augmented"]]
    point = _macro_auc(gs["rwe_augmented"], [np.arange(k) for k in n]) - _macro_auc(gs["curated_only"], [np.arange(k) for k in n])
    base = _macro_auc(gs["curated_only"], [np.arange(k) for k in n])
    boots = []
    for _ in range(B):
        ix = [rng.integers(0, k, k) for k in n]
        boots.append(_macro_auc(gs["rwe_augmented"], ix) - _macro_auc(gs["curated_only"], ix))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"model": model, "auc_curated": base, "auc_rwe": base + point, "delta_auc": point, "delta_lo": lo, "delta_hi": hi,
            "rel_improvement_pct": 100 * point / base, "rel_lo_pct": 100 * lo / base, "rel_hi_pct": 100 * hi / base}

def disparity(oof, model, cond, cfg, rng):
    """Relative AUC reduction (%) of each group vs the reference ancestry, macro over pairs, bootstrap CI."""
    ref, B = cfg["reference_ancestry"], cfg["bootstrap"]["n"]; rows = []
    sub = oof[(oof.model == model) & (oof.condition == cond)]
    for grp in cfg["disparity_groups"]:
        G = {a: [sub[(sub.pair == p) & (sub.ancestry == a)] for p in cfg["pairs"]] for a in (ref, grp)}
        def stat(ix_ref=None, ix_grp=None):
            r = _macro_auc(G[ref], ix_ref or [np.arange(len(g)) for g in G[ref]]); q = _macro_auc(G[grp], ix_grp or [np.arange(len(g)) for g in G[grp]])
            return r, q, 100 * (r - q) / r
        r, q, red = stat(); boots = []
        for _ in range(B):
            boots.append(stat([rng.integers(0, len(g), len(g)) for g in G[ref]], [rng.integers(0, len(g), len(g)) for g in G[grp]])[2])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"model": model, "condition": cond, "group": grp, "auc_ref": r, "auc_group": q, "rel_reduction_pct": red, "lo_pct": lo, "hi_pct": hi,
                     "n_group": int(sum(len(g) for g in G[grp]))})
    return rows

def decision_curve(y, p, thresholds):
    n = len(y); rows = []
    for t in thresholds:
        pred = p >= t; tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum()
        rows.append({"threshold": t, "net_benefit": tp / n - fp / n * t / (1 - t), "treat_all": y.mean() - (1 - y.mean()) * t / (1 - t)})
    return pd.DataFrame(rows)
