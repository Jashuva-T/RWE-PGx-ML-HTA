"""SYNTHETIC data generator (toy DGP) so the pipeline runs without controlled-access data.

Design (documented, not calibrated to any published result):
 * curated variants: PharmGKB 1A/1B / ClinVar-style variants with effects discovered in EUR;
   allele frequencies differ by ancestry.
 * rwe variants: additional variants only visible in real-world data, some enriched in SAS/AFR.
 * outcome = logistic(base + curated + rwe + clinical covariates).
Outputs MUST be labelled synthetic wherever shown.
"""
from __future__ import annotations
import numpy as np, pandas as pd

def generate(cfg: dict):
    s = cfg["synthetic"]; rng = np.random.default_rng(cfg["seed"])
    anc = list(s["ancestry_mix"]); w = np.array([s["ancestry_mix"][a] for a in anc]); w = w / w.sum()
    rows, cat = [], []
    for pair in cfg["pairs"]:
        n = s["n_per_pair"]; a = rng.choice(anc, n, p=w)
        nc, nr = s["n_curated_variants"], s["n_rwe_variants"]
        X, eff = {}, {}
        for j in range(nc + nr):
            kind = "curated" if j < nc else "rwe"
            name = f"{pair}_{kind}{j}"
            base = rng.uniform(0.03, 0.25)
            # ancestry-specific allele frequency multipliers; rwe variants enriched in SAS/AFR
            mult = {k: rng.uniform(0.5, 1.5) for k in anc}
            if kind == "rwe":
                mult["SAS"] *= rng.uniform(1.0, 2.5); mult["AFR"] *= rng.uniform(1.0, 2.5); mult["EUR"] *= rng.uniform(0.3, 1.0)
            maf = np.clip(np.array([base * mult[x] for x in a]), 0.005, 0.6)
            X[name] = rng.binomial(2, maf)
            eff[name] = rng.normal(0.7 if kind == "curated" else 0.5, 0.15)
            src = ("pharmgkb_1AB" if j < nc // 2 else "clinvar") if kind == "curated" else "rwe"
            cat.append({"pair": pair, "variant": name, "gene": pair.split("_")[0], "source": src})
        age = rng.normal(62, 12, n); sex = rng.binomial(1, 0.5, n); comorb = rng.poisson(2, n)
        logit = np.log(s["base_rate"] / (1 - s["base_rate"])) - 0.9
        logit = logit + sum(eff[k] * X[k] * 0.55 for k in X) + 0.015 * (age - 62) + 0.12 * comorb + rng.normal(0, 0.3, n)
        y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
        d = pd.DataFrame(X); d.insert(0, "patient_id", [f"{pair}_{i}" for i in range(n)])
        d["pair"] = pair; d["ancestry"] = a; d["age"] = age; d["sex"] = sex; d["comorbidity"] = comorb; d["outcome"] = y
        rows.append(d)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(cat)
