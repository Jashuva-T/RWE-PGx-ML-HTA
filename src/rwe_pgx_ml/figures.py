from __future__ import annotations
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, BLUE, AMBER, GREY = "#0B2545", "#1F6FA8", "#F2A900", "#7A8794"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})
CFC = {"curated_only": BLUE, "rwe_augmented": AMBER}

def _save(fig, out, name, synthetic):
    note = ("SYNTHETIC DEMONSTRATION DATA - toy data-generating process, NOT eMERGE-PGx; not evidence about real populations"
            if synthetic else "Results from user-supplied data")
    fig.text(0.01, 0.005, note, fontsize=7, color=NAVY if synthetic else GREY, fontweight="bold" if synthetic else "normal")
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    for e in ("png", "pdf"): fig.savefig(f"{out}/figures/{name}.{e}", dpi=300 if e == "png" else None)
    plt.close(fig)

def fig_auc_models(overall, out, syn):
    m = overall.groupby(["model", "condition"]).auc.mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 4)); x = np.arange(len(m)); w = 0.38
    for i, c in enumerate(["curated_only", "rwe_augmented"]):
        b = ax.bar(x + (i - 0.5) * w, m[c], w, color=CFC[c], label=c.replace("_", "-")); ax.bar_label(b, fmt="%.2f", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(m.index); ax.set(ylim=(0.5, 1.0), ylabel="Macro-average AUC (5 gene-drug pairs)", title="Discrimination by model and condition")
    ax.legend(frameon=False); _save(fig, out, "fig1_auc_by_model_condition", syn)

def fig_auc_ancestry(by_anc, model, out, syn, order):
    d = by_anc[by_anc.model == model].groupby(["condition", "ancestry"]).auc.mean().unstack().reindex(columns=order)
    fig, ax = plt.subplots(figsize=(7.5, 4)); x = np.arange(len(order)); w = 0.38
    for i, c in enumerate(["curated_only", "rwe_augmented"]):
        b = ax.bar(x + (i - 0.5) * w, d.loc[c], w, color=CFC[c], label=c.replace("_", "-")); ax.bar_label(b, fmt="%.2f", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(order); ax.set(ylim=(0.5, 1.0), ylabel="Macro-average AUC", title=f"Ancestry-stratified AUC ({model})"); ax.legend(frameon=False)
    _save(fig, out, "fig2_auc_by_ancestry", syn)

def fig_disparity(disp, out, syn):
    fig, ax = plt.subplots(figsize=(6.5, 3.8)); y = 0
    for cond in ("curated_only", "rwe_augmented"):
        for r in disp[disp.condition == cond].itertuples():
            ax.errorbar(r.rel_reduction_pct, y, xerr=[[r.rel_reduction_pct - r.lo_pct], [r.hi_pct - r.rel_reduction_pct]], fmt="o", color=CFC[cond], capsize=3)
            ax.text(r.hi_pct + 0.3, y, f"{r.group}, {cond.replace('_','-')}", va="center", fontsize=8); y += 1
    ax.axvline(0, color=GREY, lw=1); ax.set_yticks([]); ax.set(xlabel="Relative AUC reduction vs EUR (%) with 95% bootstrap CI", title="Ancestry performance gap")
    _save(fig, out, "fig3_ancestry_disparity", syn)

def fig_calibration(oof, model, out, syn, bins=10):
    fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, grp in zip(axs, (["EUR"], ["SAS", "AFR"])):
        for c in ("curated_only", "rwe_augmented"):
            g = oof[(oof.model == model) & (oof.condition == c) & oof.ancestry.isin(grp)]
            q = pd.qcut(g.p, bins, duplicates="drop"); t = g.groupby(q, observed=True).agg(p=("p", "mean"), y=("y", "mean"))
            ax.plot(t.p, t.y, "o-", color=CFC[c], label=c.replace("_", "-"))
        ax.plot([0, 1], [0, 1], color=GREY, ls=":"); ax.set(xlabel="Predicted risk", title="+".join(grp)); ax.legend(frameon=False, fontsize=8)
    axs[0].set_ylabel("Observed event rate"); _save(fig, out, "fig4_calibration", syn)

def fig_decision_curve(dc, out, syn):
    fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, grp in zip(axs, dc.group.unique()):
        d = dc[dc.group == grp]
        for c in ("curated_only", "rwe_augmented"):
            e = d[d.condition == c]; ax.plot(e.threshold, e.net_benefit, color=CFC[c], lw=2, label=c.replace("_", "-"))
        ax.plot(e.threshold, e.treat_all, color=GREY, ls="--", label="treat all"); ax.axhline(0, color=GREY, lw=0.8)
        ax.set(xlabel="Threshold probability", title=grp); ax.legend(frameon=False, fontsize=8)
    axs[0].set_ylabel("Net benefit"); _save(fig, out, "fig5_decision_curve", syn)
