"""python -m rwe_pgx_ml.pipeline --config config/config.yaml --out results [--synthetic]"""
from __future__ import annotations
import argparse, json, os
import numpy as np, pandas as pd, yaml
from . import synth, data, models, evaluate as ev, figures as fg

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config/config.yaml"); ap.add_argument("--out", default="results")
    ap.add_argument("--synthetic", action="store_true"); a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config)); rng = np.random.default_rng(cfg["seed"])
    os.makedirs(f"{a.out}/figures", exist_ok=True); os.makedirs(f"{a.out}/tables", exist_ok=True)
    df, cat = synth.generate(cfg) if a.synthetic else (pd.read_csv(cfg["data"]["patients_csv"]), pd.read_csv(cfg["data"]["variant_catalog_csv"]))
    data.validate(df, cat, cfg)
    recs, imp = [], []
    for pair in cfg["pairs"]:
        d = df[df.pair == pair].reset_index(drop=True); c = cat[cat.pair == pair]
        fold_idx = models.folds(d, cfg["cv"]["folds"], cfg["seed"])            # same folds for every model/condition
        for cond, spec in cfg["conditions"].items():
            cols = data.features(d, c, spec)
            for m in cfg["models"]:
                p, fid = models.oof_predict(d, cols, m, cfg, fold_idx)
                recs.append(pd.DataFrame({"patient_id": d.patient_id, "pair": pair, "model": m, "condition": cond, "ancestry": d.ancestry, "y": d.outcome, "p": p, "fold": fid}))
        # feature importance (xgboost gain, RWE-augmented, full data) - interpretability / concordance check
        cols = data.features(d, c, cfg["conditions"]["rwe_augmented"])
        mdl = models.make("xgboost", cfg["model_params"]["xgboost"], cfg["seed"]).fit(d[cols].to_numpy(float), d.outcome)
        g = pd.Series(mdl.get_booster().get_score(importance_type="gain")); g.index = [cols[int(k[1:])] for k in g.index]
        src = c.set_index("variant").source
        for k, v in g.items(): imp.append({"pair": pair, "feature": k, "gain": v, "source": src.get(k, "covariate")})
    oof = pd.concat(recs, ignore_index=True); oof.to_csv(f"{a.out}/tables/oof_predictions.csv.gz", index=False, float_format="%.5g")
    overall, by_anc = ev.metrics_table(oof); overall.to_csv(f"{a.out}/tables/metrics_overall.csv", index=False); by_anc.to_csv(f"{a.out}/tables/metrics_by_ancestry.csv", index=False)
    imp = pd.DataFrame(imp); imp.to_csv(f"{a.out}/tables/feature_importance.csv", index=False)
    delta = pd.DataFrame([ev.paired_delta(oof, m, cfg, rng) for m in cfg["models"]]); delta.to_csv(f"{a.out}/tables/delta_rwe_vs_curated.csv", index=False)
    disp = pd.DataFrame([r for cond in cfg["conditions"] for r in ev.disparity(oof, "xgboost", cond, cfg, rng)]); disp.to_csv(f"{a.out}/tables/ancestry_disparity.csv", index=False)
    grp = {"EUR": [cfg["reference_ancestry"]], "SAS+AFR": cfg["disparity_groups"]}
    dc = pd.concat([ev.decision_curve(s.y.to_numpy(), s.p.to_numpy(), cfg["decision_curve_thresholds"]).assign(condition=c, group=gname)
                    for gname, an in grp.items() for c in cfg["conditions"]
                    for s in [oof[(oof.model == "xgboost") & (oof.condition == c) & oof.ancestry.isin(an)]]], ignore_index=True)
    dc.to_csv(f"{a.out}/tables/decision_curve.csv", index=False)
    share = imp[imp.source == "pharmgkb_1AB"].groupby("pair").gain.sum() / imp.groupby("pair").gain.sum()
    best = overall.groupby(["model", "condition"]).auc.mean().unstack()
    summary = {"synthetic": bool(a.synthetic), "macro_auc": best.round(4).to_dict("index"), "delta_rwe_vs_curated": delta.round(4).to_dict("records"),
               "ancestry_disparity_xgboost": disp.round(3).to_dict("records"), "importance_share_pharmgkb_1AB": share.round(3).to_dict(),
               "note": "Synthetic-demo numbers say nothing about eMERGE-PGx; compare only against results from real data." if a.synthetic else ""}
    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=2, sort_keys=True)
    s = a.synthetic
    fg.fig_auc_models(overall, a.out, s); fg.fig_auc_ancestry(by_anc, "xgboost", a.out, s, cfg["ancestries"]); fg.fig_disparity(disp, a.out, s)
    fg.fig_calibration(oof, "xgboost", a.out, s); fg.fig_decision_curve(dc, a.out, s)
    print(json.dumps({k: summary[k] for k in ("macro_auc", "ancestry_disparity_xgboost")}, indent=1, default=str)[:2500])

if __name__ == "__main__":
    main()
