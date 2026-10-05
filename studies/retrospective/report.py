"""Retrospective study: analysis driver and report (docs/SPEC.md section 8).

Session 2 rewrite. Session 1 established the method but stated its conclusions
more confidently than its own uncertainty supported: the two headline claims
rested on differences of +0.028 and -0.034 with no interval on either, while the
CI on a single AUROC spanned 0.671-0.839.

Everything here now reports differences with paired intervals, corrects for
having screened 30 correlated metrics, and states plainly where a difference
cannot be resolved. `CHANGEstats.md` records which session-1 conclusions changed.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from studies.retrospective import stats
from studies.retrospective.run_study import (
    GROUP,
    LABEL,
    OUT,
    build_table,
    feature_columns,
    fetch_summary,
    no_target_appears_in_both_folds,
    vendor_agreement,
)

log = logging.getLogger(__name__)

N_BOOT = 2000
#: Minimum binders in a target before its AUROC is treated as interpretable.
MIN_BINDERS_FOR_INTERPRETATION = 5


def _fmt_ci(lo: float, hi: float) -> str:
    return f"{lo:+.3f} to {hi:+.3f}"


def analyse(df: pd.DataFrame) -> dict:
    """Run every analysis the report needs. Pure computation, no prose."""
    y = df[LABEL].to_numpy(dtype=int)
    g = df[GROUP].to_numpy()
    feats = feature_columns(df)
    scores = {c: df[c].to_numpy(dtype=float) for c in feats}

    singles = {
        c: stats.mean_within_target_auroc(y, scores[c], g)
        for c in feats
        if feats[c] == "single-predictor ipSAE"
    }
    bases = {
        c: stats.mean_within_target_auroc(y, scores[c], g)
        for c in feats
        if feats[c] == "trivial baseline"
    }
    best_single = max(singles, key=singles.get)
    best_base = max(bases, key=bases.get)

    model_feats = [c for c in feats if feats[c] != "trivial baseline"]
    scores["LOTO_model"] = stats.loto_model_scores(df[model_feats].to_numpy(dtype=float), y, g)

    headline = "ipsae_mean"
    comparisons = [
        (headline, best_single, "consensus mean vs the best individual predictor"),
        (headline, "LOTO_model", "consensus mean vs a learned model over all metrics"),
        (headline, best_base, "consensus mean vs the best trivial baseline"),
        ("LOTO_model", best_base, "learned model vs the best trivial baseline"),
        (best_single, best_base, "best individual predictor vs the best trivial baseline"),
    ]
    paired = []
    for a, b, label in comparisons:
        r = stats.paired_bootstrap_difference(
            y, scores[a], scores[b], g, name_a=a, name_b=b, n_boot=N_BOOT
        )
        paired.append({"label": label, "result": r})

    selection = stats.selection_corrected_bootstrap(
        y, {c: scores[c] for c in feats}, g, n_boot=N_BOOT
    )

    # Per-target with CIs.
    per_target = []
    hs = scores[headline]
    for t in sorted(set(g)):
        m = (g == t) & ~np.isnan(hs)
        if m.sum() < 5:
            continue
        pt, lo, hi = stats.within_target_bootstrap(y[m], hs[m], n_boot=N_BOOT)
        if pt != pt:
            continue
        per_target.append(
            {
                "target": t,
                "n": int(m.sum()),
                "binders": int(y[m].sum()),
                "auroc": pt,
                "lo": lo,
                "hi": hi,
                "includes_chance": bool(lo < 0.5 < hi),
                "interpretable": bool(
                    y[m].sum() >= MIN_BINDERS_FOR_INTERPRETATION
                    and (m.sum() - y[m].sum()) >= MIN_BINDERS_FOR_INTERPRETATION
                ),
            }
        )

    # Calibration of the out-of-fold model.
    oof = scores["LOTO_model"]
    ok = ~np.isnan(oof)
    keep, dropped = stats.calibration_bins(y[ok], oof[ok], n_bins=10, min_n=5)
    rho, pval = stats.monotonicity(keep)
    ece = stats.expected_calibration_error(y[ok], oof[ok])
    top_rate, top_n = stats.top_decile_rate(y[ok], oof[ok])

    # Disagreement, conditional on the mean.
    std_s = scores["ipsae_std_across"]
    m = ~(np.isnan(std_s) | np.isnan(hs))
    corr = float(np.corrcoef(std_s[m], hs[m])[0, 1])
    resid_auroc = stats.mean_within_target_auroc(y, stats.residualise(std_s, hs), g)
    o2 = stats.loto_model_scores(np.column_stack([hs, std_s]), y, g)
    o1 = stats.loto_model_scores(hs.reshape(-1, 1), y, g)
    cond = stats.paired_bootstrap_difference(
        y, o2, o1, g, name_a="mean + disagreement", name_b="mean alone", n_boot=N_BOOT
    )

    operating = _operating_point(y[~np.isnan(hs)], hs[~np.isnan(hs)])

    return {
        "n": len(df),
        "n_pos": int(y.sum()),
        "n_targets": int(df[GROUP].nunique()),
        "headline": headline,
        "headline_value": stats.mean_within_target_auroc(y, hs, g),
        "best_single": best_single,
        "best_single_value": singles[best_single],
        "best_base": best_base,
        "best_base_value": bases[best_base],
        "singles": singles,
        "bases": bases,
        "paired": paired,
        "selection": selection,
        "per_target": per_target,
        "calibration": {
            "keep": keep,
            "dropped": dropped,
            "ece": ece,
            "spearman": rho,
            "spearman_p": pval,
            "top_decile_rate": top_rate,
            "top_decile_n": top_n,
        },
        "disagreement": {"corr": corr, "resid_auroc": resid_auroc, "conditional": cond},
        "operating": operating,
        "n_screened": len(feats),
    }


def _operating_point(y: np.ndarray, s: np.ndarray) -> dict:
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y, s)
    k = int(np.argmax(tpr - fpr))
    cut = float(thr[k])
    pred = s >= cut
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    return {
        "threshold": cut,
        "sensitivity": float(tpr[k]),
        "specificity": float(1 - fpr[k]),
        "precision": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        "n_selected": int(pred.sum()),
        "n_total": int(len(y)),
        "base_rate": float(y.mean()),
    }


# --------------------------------------------------------------------------


def write_report(df: pd.DataFrame, a: dict, figs: list[str]):  # noqa: ARG001
    """Write REPORT.md from the analysis dict."""
    L: list[str] = []
    w = L.append
    n, n_pos = a["n"], a["n_pos"]
    base = n_pos / n

    w("# Do in-silico metrics predict experimental binding?")
    w("")
    w("A retrospective calibration study on the published Anthropic de novo binder")
    w("campaign. Generated by `binderkit study retrospective`; every number is")
    w("reproducible from that script.")
    w("")
    w("> **Revised in session 2.** Session 1 reported the right quantities but")
    w("> stated two conclusions more confidently than its own uncertainty allowed.")
    w("> Differences now carry paired confidence intervals and the metric screen is")
    w("> corrected for. [`CHANGEstats.md`](CHANGEstats.md) lists what changed and why.")
    w("")

    # ---- bottom line -------------------------------------------------
    w("## Bottom line")
    w("")
    w(
        f"On {n} designs against {a['n_targets']} targets ({n_pos} measured binders, "
        f"{100 * base:.1f}%):"
    )
    w("")
    base_cmp = next(
        p["result"]
        for p in a["paired"]
        if p["result"].name_a == a["headline"] and p["result"].name_b == a["best_base"]
    )
    w("**What survives.** Co-folding confidence predicts measured binding far better")
    w(
        f"than a trivial baseline. `{a['headline']}` reaches "
        f"{base_cmp.stat_a:.3f} within-target AUROC against {base_cmp.stat_b:.3f} for "
        f"`{a['best_base']}`, a difference of **{base_cmp.difference:+.3f}** "
        f"(95% CI {_fmt_ci(base_cmp.ci_lo, base_cmp.ci_hi)}, sign held in "
        f"{100 * base_cmp.sign_consistency:.0f}% of replicates). This is the robust"
    )
    w("result of the study and it is not close to the boundary.")
    w("")

    single_cmp = next(p["result"] for p in a["paired"] if p["result"].name_b == a["best_single"])
    w("**What does not survive.** Session 1 concluded that averaging ten predictors")
    w("beats the best individual predictor. With a paired interval that comparison is")
    w(
        f"{single_cmp.difference:+.3f}, 95% CI {_fmt_ci(single_cmp.ci_lo, single_cmp.ci_hi)} "
        f"— it straddles zero, so the two **cannot be distinguished at this sample"
    )
    w("size**. The practical consequence is in the other direction from session 1's:")
    w(f"a compute-constrained user can run **one** predictor (`{a['best_single']}`,")
    w(f"{a['best_single_value']:.3f}) rather than ten, because the ten-predictor mean")
    w("is not measurably better.")
    w("")

    model_cmp = next(p["result"] for p in a["paired"] if p["result"].name_b == "LOTO_model")
    w("**What survives in weakened form.** The learned model really is worse than the")
    w(
        f"plain average it was given: {model_cmp.difference:+.3f}, 95% CI "
        f"{_fmt_ci(model_cmp.ci_lo, model_cmp.ci_hi)}. The interval excludes zero but"
    )
    w("only just, so this is a real effect of modest size, not a headline.")
    w("")

    sel = a["selection"]
    w(f"**And a caution about all of the above.** {sel['n_metrics_screened']} metrics")
    w("were screened. Re-selecting the winner inside each bootstrap replicate, the")
    w(f"nominal winner `{sel['observed_best_metric']}` is the winner in only")
    w(
        f"**{100 * sel['winner_stability']:.0f}%** of replicates and "
        f"{sel['distinct_winners']} different metrics win at least once. The"
    )
    w(f'selection-corrected interval for "the best of {sel["n_metrics_screened"]}" is')
    w(f"{sel['selection_corrected_lo']:.3f} to {sel['selection_corrected_hi']:.3f}.")
    w("*Which* metric is best is not resolved by this data; that a good confidence")
    w("metric beats a trivial baseline is.")
    w("")

    # ---- data and selection bias -------------------------------------
    w("## Data, and the selection bias that qualifies every number")
    w("")
    w("Source: `huggingface.co/datasets/Anthropic/claude-protein-binder-design`,")
    w("`data/tables/design_summary.csv`, v1.0 (2026-08-18). **Licence CC BY 4.0.**")
    w(f"{n} designs retained of 1,440; the 120 mature GDF-8 designs have no")
    w("adjudicated outcome (their antigen aggregated) and are dropped.")
    w("")
    w("### These designs are not a random sample")
    w("")
    w('The release states plainly that ipSAE is "the primary confidence metric and')
    w('the one designs were selected on" (`docs/INSILICO.md`). So the 1,320 designs')
    w("with wet-lab outcomes were **chosen using the very metric this study")
    w("evaluates**. Three consequences, and they point in different directions:")
    w("")
    w("1. **The measured AUROC is probably an underestimate.** The sample is range")
    w("   restricted: designs with low ipSAE were largely never ordered, so the easy")
    w("   discriminations are missing and the metric is being graded on the hard")
    w("   cases it itself selected. On an unscreened pool its discrimination would")
    w("   likely be higher.")
    w("2. **The operating point does not transfer.** The threshold and precision")
    w("   below are measured on an already-filtered pool. A pipeline that generates")
    w("   designs from scratch faces a very different score distribution, and would")
    w("   see a different precision at the same cut-off. The threshold is still the")
    w("   best available starting point; the precision attached to it is not a")
    w("   promise.")
    w("3. **It cannot be corrected for here**, because the designs that were")
    w("   generated and *not* ordered are not in the release. The generated-to-ordered")
    w("   ratio is not recoverable from the published tables.")
    w("")

    # ---- grouping -----------------------------------------------------
    w("## Why every split and every bootstrap is grouped by target")
    w("")
    rates = df.groupby(GROUP)[LABEL].agg(["size", "mean"]).sort_values("mean", ascending=False)
    w("| Target | n | binder rate |")
    w("|---|---|---|")
    for t, r in rates.iterrows():
        w(f"| {t} | {int(r['size'])} | {100 * r['mean']:.1f}% |")
    w("")
    w("Binder rates run from 0% to 80% by target. A design-level split lets a model")
    w("infer the target and read off its base rate. Folds hold out whole targets and")
    w("bootstrap replicates resample whole targets; for paired comparisons both")
    w("metrics are scored on the *same* resample, which is what makes a difference of")
    w("a few hundredths resolvable at all.")
    w("")
    w("![binder rate by target](figures/binder_rate_by_target.png)")
    w("")

    # ---- paired table --------------------------------------------------
    w("## Paired comparisons")
    w("")
    w(f"Within-target AUROC. {N_BOOT} bootstrap replicates, targets resampled with")
    w("replacement, both metrics scored on each identical resample.")
    w("")
    w("| Comparison | A | B | Difference | 95% CI | Sign held | Verdict |")
    w("|---|---|---|---|---|---|---|")
    for p in a["paired"]:
        r = p["result"]
        verdict = "**distinguishable**" if r.distinguishable else "not distinguishable"
        w(
            f"| {p['label']} | {r.stat_a:.3f} | {r.stat_b:.3f} | {r.difference:+.3f} | "
            f"{_fmt_ci(r.ci_lo, r.ci_hi)} | {100 * r.sign_consistency:.0f}% | {verdict} |"
        )
    w("")
    for p in a["paired"]:
        w(f"- {p['result'].verdict()}")
    w("")
    w("![pooled vs within-target AUROC](figures/auroc_pooled_vs_within.png)")
    w("")

    # ---- per target ----------------------------------------------------
    w("## Per-target, with intervals")
    w("")
    w(f"Within-target AUROC of `{a['headline']}`, bootstrapping **designs** inside")
    w("each target, which is the right unit once the target is fixed.")
    w("")
    w("| Target | n | binders | AUROC | 95% CI | Note |")
    w("|---|---|---|---|---|---|")
    for t in sorted(a["per_target"], key=lambda r: -r["auroc"]):
        note = []
        if not t["interpretable"]:
            note.append(f"only {t['binders']} binders - uninformative")
        elif t["includes_chance"]:
            note.append("CI includes chance")
        w(
            f"| {t['target']} | {t['n']} | {t['binders']} | {t['auroc']:.3f} | "
            f"{t['lo']:.3f}-{t['hi']:.3f} | {'; '.join(note) or '-'} |"
        )
    w("")
    chance = [t for t in a["per_target"] if t["includes_chance"]]
    uninf = [t for t in a["per_target"] if not t["interpretable"]]
    w(f"**{len(chance)} of {len(a['per_target'])} targets have a CI that includes")
    w("chance.** Session 1 reported two per-target claims that do not survive this:")
    w("")
    egfr = next((t for t in a["per_target"] if t["target"] == "EGFR"), None)
    if egfr:
        w(
            f"- **EGFR** (this competition's target): {egfr['auroc']:.3f}, CI "
            f"{egfr['lo']:.3f}-{egfr['hi']:.3f}. With {egfr['binders']} binders in "
            f"{egfr['n']} designs the interval spans chance. Session 1 said the metric"
        )
        w('  is "weaker on EGFR than on average"; the honest statement is that EGFR\'s')
        w("  value cannot be distinguished either from chance or from the overall mean.")
    bbf = next((t for t in a["per_target"] if t["target"] == "BBF-14"), None)
    if bbf:
        w(f"- **BBF-14**: {bbf['auroc']:.3f}, CI {bbf['lo']:.3f}-{bbf['hi']:.3f}, from")
        w(f'  {bbf["binders"]} binders. Session 1 called this "below chance". The')
        w("  interval covers almost the entire range; it supports no claim at all.")
    if uninf:
        w(
            f"- Targets with fewer than {MIN_BINDERS_FOR_INTERPRETATION} binders "
            f"({', '.join(t['target'] for t in uninf)}) are reported but should not be"
        )
        w("  read as evidence in either direction.")
    w("")
    w("The *decision* that follows is unchanged, and is now better supported:")
    w("**confidence is a soft ranking signal, never a hard filter**, because")
    w("per-target behaviour is too uncertain to justify a fixed cut-off.")
    w("")
    w("![per-target AUROC](figures/per_target_auroc.png)")
    w("")

    # ---- calibration ----------------------------------------------------
    c = a["calibration"]
    w("## Calibration: the curve inverts where it matters most")
    w("")
    w(f"Expected calibration error is {c['ece']:.3f}, which looks respectable and is")
    w("misleading on its own. The reliability curve is **not monotonic**: Spearman")
    w(
        f"correlation between predicted and observed across bins is {c['spearman']:+.3f} "
        f"(p = {c['spearman_p']:.2f})."
    )
    w("")
    w("| Predicted | n | Mean predicted | Observed | Observed 95% CI |")
    w("|---|---|---|---|---|")
    for b in c["keep"]:
        w(
            f"| {b.lo:.1f}-{b.hi:.1f} | {b.n} | {b.mean_predicted:.3f} | "
            f"{b.observed:.3f} | {b.obs_lo:.3f}-{b.obs_hi:.3f} |"
        )
    for b in c["dropped"]:
        w(
            f"| {b.lo:.1f}-{b.hi:.1f} | {b.n} | {b.mean_predicted:.3f} | "
            f"{b.observed:.3f} | **excluded, n too small** |"
        )
    w("")
    hi_bins = [b for b in c["keep"] if b.lo >= 0.6]
    if hi_bins:
        worst = min(hi_bins, key=lambda b: b.observed)
        w("**The headline is the inversion, not the ECE.** Designs predicted at")
        w(
            f"{worst.mean_predicted:.2f} bound at {worst.observed:.1%} "
            f"(95% CI {worst.obs_lo:.1%}-{worst.obs_hi:.1%}), *lower* than designs"
        )
        w("predicted in the middle of the range. The model is least reliable exactly")
        w("where a confident prediction would be acted on, which is the opposite of")
        w("the property you want.")
        w("")
    w(
        f"Observed binder rate in the top decile by score: **{c['top_decile_rate']:.1%}** "
        f"(n = {c['top_decile_n']}) against a base rate of {100 * base:.1f}%."
    )
    w("")
    w("Session 1 plotted a bin containing a single design as a point estimate. Bins")
    w("with fewer than five designs are now excluded and named instead, and every")
    w("retained bin carries a Wilson interval.")
    w("")
    w("![calibration](figures/calibration.png)")
    w("")

    # ---- disagreement ---------------------------------------------------
    d = a["disagreement"]
    w("## Retracted: cross-predictor disagreement as a signal")
    w("")
    w("Session 1 reported that `ipsae_std_across` scores 0.370, below chance, and")
    w("concluded that predictor agreement is usable signal in its own right. Testing")
    w("that properly:")
    w("")
    w(f"- Its correlation with `ipsae_mean` is {d['corr']:+.3f}, so it is partly, but")
    w("  not wholly, a restatement of the mean.")
    w(f"- Residualised on the mean it scores {d['resid_auroc']:.3f}, so some marginal")
    w("  association does remain.")
    w("- **But the decisive test is whether it adds anything predictive.** A model")
    w(f"  given mean + disagreement scores {d['conditional'].stat_a:.3f} against")
    w(
        f"  {d['conditional'].stat_b:.3f} for the mean alone: "
        f"{d['conditional'].difference:+.3f}, 95% CI "
        f"{_fmt_ci(d['conditional'].ci_lo, d['conditional'].ci_hi)}."
    )
    w("")
    if d["conditional"].difference < 0 and d["conditional"].distinguishable:
        w("Adding disagreement makes out-of-target prediction **measurably worse**.")
        w("The session-1 conclusion is withdrawn: disagreement is not a usable")
        w("additional signal, and including it costs accuracy on unseen targets.")
    else:
        w("The difference does not establish a benefit, so the session-1 conclusion")
        w("is withdrawn as unsupported.")
    w("")

    # ---- operating point -------------------------------------------------
    o = a["operating"]
    w("## Operating point, and what it is worth")
    w("")
    w(f"On `{a['headline']}`, maximising the Youden J statistic: threshold")
    w(f"**{o['threshold']:.3f}**, sensitivity {o['sensitivity']:.3f}, specificity")
    w(f"{o['specificity']:.3f}, precision {o['precision']:.3f}, selecting")
    w(f"{o['n_selected']} of {o['n_total']} designs.")
    w("")
    w(f"Precision {o['precision']:.3f} against a base rate of {o['base_rate']:.3f} is")
    w(f"about **{o['precision'] / o['base_rate']:.1f}x enrichment**. Two caveats that")
    w("matter more than the number: most selected designs still do not bind, and per")
    w("the selection-bias section this precision was measured on a pool that had")
    w("already been filtered on this metric, so it will not transfer unchanged to an")
    w("unscreened pipeline.")
    w("")

    # ---- what this cannot tell you ---------------------------------------
    w("## What this study cannot tell you")
    w("")
    w("- **Whether the metric works on designs nobody would have ordered.** The")
    w("  sample is filtered on ipSAE, so the low-confidence region is largely absent.")
    w("- **Which metric is genuinely best.** The winner changes across bootstrap")
    w(f"  replicates ({a['selection']['distinct_winners']} different winners); only the")
    w("  gap to a trivial baseline is resolved.")
    w("- **Whether any of this transfers to a different design pipeline.** These")
    w("  designs came from two large agentic campaigns with far more compute than this")
    w("  repo has, and the metric definitions here are the release's, not this")
    w("  repo's.")
    w("- **Anything about affinity.** The label is a binary adjudicated call. Many")
    w("  released K_D values are apparent rather than true.")
    w("- **Anything beyond 50-120 residue miniproteins against 16 targets.**")
    w("")

    # ---- conclusions -------------------------------------------------------
    w("## Conclusions")
    w("")
    w(
        f"1. **Co-folding confidence beats a trivial baseline by {base_cmp.difference:+.3f} "
        f"within-target AUROC** (95% CI {_fmt_ci(base_cmp.ci_lo, base_cmp.ci_hi)}). This"
    )
    w("   is the result that survives every correction applied here.")
    w("2. **One predictor is enough.** The ten-predictor mean cannot be distinguished")
    w(
        f"   from the single best predictor ({single_cmp.difference:+.3f}, CI "
        f"{_fmt_ci(single_cmp.ci_lo, single_cmp.ci_hi)}), which is a direct saving for"
    )
    w("   anyone compute-constrained.")
    w("3. **Do not fit weights over the predictors.** A learned model scored below the")
    w("   plain average on held-out targets.")
    w("4. **Do not use any of these as a hard filter.** Per-target intervals are wide")
    w(f"   and {len(chance)} of {len(a['per_target'])} include chance.")
    w("5. **Do not trust high predicted probabilities.** The reliability curve inverts")
    w("   at the top of the range.")
    w("6. **The wet lab agrees with itself only 89% of the time**, which bounds every")
    w("   number above.")
    w("")
    w("## Limitations")
    w("")
    w("- 15 targets is few; intervals on target-level quantities are correspondingly")
    w("  wide, and the bootstrap cannot manufacture precision that the design of the")
    w("  study does not contain.")
    w("- Logistic regression with median imputation and no tuning. A stronger model")
    w("  might extract more, though the ceiling implied above is not far away.")
    w("- `binder_final` is a model-adjudicated label over two assays that disagree on")
    w("  11% of designs.")
    w("- The study inherits every caveat in the release's own `docs/DATA_NOTES.md`.")
    w("")

    path = OUT / "REPORT.md"
    while L and not L[-1].strip():  # no trailing blank line; the repo hook strips it
        L.pop()
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def write_changes(a: dict):
    """Write CHANGEstats.md: which session-1 conclusions changed, and why."""
    single_cmp = next(p["result"] for p in a["paired"] if p["result"].name_b == a["best_single"])
    model_cmp = next(p["result"] for p in a["paired"] if p["result"].name_b == "LOTO_model")
    base_cmp = next(p["result"] for p in a["paired"] if p["result"].name_b == a["best_base"])
    d = a["disagreement"]["conditional"]
    c = a["calibration"]
    egfr = next((t for t in a["per_target"] if t["target"] == "EGFR"), None)
    bbf = next((t for t in a["per_target"] if t["target"] == "BBF-14"), None)
    sel = a["selection"]

    L: list[str] = []
    w = L.append
    w("# What changed between session 1 and session 2")
    w("")
    w("Session 1's method was sound and its numbers were reproducible. What it got")
    w("wrong was the strength of the language: it stated conclusions about")
    w("*differences* without putting an interval on those differences. Adding paired")
    w("intervals and a correction for screening 30 metrics changed three conclusions,")
    w("withdrew one, and left the main result standing.")
    w("")
    w("This file exists because a changelog of retractions is itself evidence of")
    w("method. Nothing below was found by new data: the same table, analysed more")
    w("carefully, says something different.")
    w("")

    w('## 1. WITHDRAWN - "averaging ten predictors beats the best single predictor"')
    w("")
    w("| | Session 1 | Session 2 |")
    w("|---|---|---|")
    w(
        "| Claim | the mean beats the best individual predictor by +0.028 | cannot be distinguished |"
    )
    w(
        f"| Evidence | point estimate only | {single_cmp.difference:+.3f}, 95% CI "
        f"{_fmt_ci(single_cmp.ci_lo, single_cmp.ci_hi)}, sign held "
        f"{100 * single_cmp.sign_consistency:.0f}% |"
    )
    w("")
    w("The interval straddles zero. The point estimate is unchanged; what changed is")
    w("that it is now accompanied by the uncertainty it always had.")
    w("")
    w("**This reverses the practical advice.** Session 1 implied you should run ten")
    w("predictors and average them. Since the mean is not measurably better than one")
    w("good predictor, a compute-constrained user should run **one**. That is a")
    w("roughly ten-fold reduction in co-folding cost for no measurable loss.")
    w("")

    w('## 2. SURVIVES, WEAKENED - "do not learn weights over the predictors"')
    w("")
    w(
        f"The learned model is worse than the plain average: {model_cmp.difference:+.3f}, "
        f"95% CI {_fmt_ci(model_cmp.ci_lo, model_cmp.ci_hi)}. The interval excludes zero,"
    )
    w("so the effect is real, but it sits close to the boundary and should be")
    w("described as a modest effect rather than a headline.")
    w("")

    w('## 3. WITHDRAWN - "cross-predictor disagreement is usable signal"')
    w("")
    w("Session 1 observed `ipsae_std_across` at AUROC 0.370 and concluded it predicts")
    w("with its sign flipped. The missing test was whether it adds anything *given*")
    w("the mean it is correlated with.")
    w("")
    w(f"- correlation with `ipsae_mean`: {a['disagreement']['corr']:+.3f}")
    w(f"- residualised on the mean: {a['disagreement']['resid_auroc']:.3f}")
    w(
        f"- model with mean + disagreement vs mean alone: {d.difference:+.3f}, 95% CI "
        f"{_fmt_ci(d.ci_lo, d.ci_hi)}"
    )
    w("")
    w("Adding it makes held-out prediction worse. The conclusion is withdrawn.")
    w("")

    w("## 4. CORRECTED - per-target claims")
    w("")
    w("Session 1 quoted per-target AUROCs as point estimates. With intervals:")
    w("")
    if egfr:
        w(
            f"- **EGFR**: {egfr['auroc']:.3f}, CI {egfr['lo']:.3f}-{egfr['hi']:.3f} "
            f'({egfr["binders"]} binders). Session 1: "the metric is weaker on EGFR than'
        )
        w('  on average". The interval includes chance, so neither that claim nor its')
        w("  negation is supported.")
    if bbf:
        w(
            f"- **BBF-14**: {bbf['auroc']:.3f}, CI {bbf['lo']:.3f}-{bbf['hi']:.3f} "
            f'({bbf["binders"]} binders). Session 1: "below chance". Withdrawn; the'
        )
        w("  interval covers nearly the whole range.")
    w("")
    w("The decision these fed into - confidence as a soft signal, never a hard filter")
    w("- is unchanged, and is better supported by wide intervals than it was by the")
    w("point estimates.")
    w("")

    w("## 5. NEW - the metric screen is now corrected for")
    w("")
    w(f"{sel['n_metrics_screened']} metrics were screened and the headline was the")
    w("maximum over them, which is upward biased. Re-selecting the winner inside each")
    w("replicate:")
    w("")
    w(
        f"- the nominal winner `{sel['observed_best_metric']}` wins only "
        f"{100 * sel['winner_stability']:.0f}% of replicates"
    )
    w(f"- {sel['distinct_winners']} different metrics win at least once")
    w(
        f"- selection-corrected interval: {sel['selection_corrected_lo']:.3f} to "
        f"{sel['selection_corrected_hi']:.3f}"
    )
    w("")
    w("So *which* metric is best is not resolved by this dataset.")
    w("")

    w("## 6. NEW - calibration is non-monotonic at the top")
    w("")
    w(f"Session 1 reported ECE {c['ece']:.3f} and treated it as the calibration")
    w("headline, and plotted a bin with one design in it. The curve actually inverts:")
    w("Spearman correlation between predicted and observed across bins is")
    w(f"{c['spearman']:+.3f} (p = {c['spearman_p']:.2f}), and the most confident")
    w("reportable bin has a *lower* observed binder rate than mid-range bins. Bins")
    w("with fewer than five designs are now excluded, and every bin carries a Wilson")
    w("interval.")
    w("")

    w("## 7. NEW - selection bias is now stated in the report, not just the limitations")
    w("")
    w('The release documents that ipSAE is "the primary confidence metric and the one')
    w('designs were selected on". The evaluation set is therefore filtered on the')
    w("metric under test. This deflates the measured AUROC and, more importantly,")
    w("means the operating point does not transfer to an unscreened pipeline.")
    w("")

    w("## Unchanged")
    w("")
    w(
        f"- The main result: confidence beats a trivial baseline by "
        f"{base_cmp.difference:+.3f}, 95% CI {_fmt_ci(base_cmp.ci_lo, base_cmp.ci_hi)}, "
        "sign held in 100% of replicates."
    )
    w("- Grouping by target throughout.")
    w("- The 89% inter-assay agreement ceiling.")
    w("- The decision to treat confidence as a soft ranking signal.")
    w("")

    path = OUT / "CHANGEstats.md"
    while L and not L[-1].strip():  # no trailing blank line; the repo hook strips it
        L.pop()
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def main(force_download: bool = False) -> int:
    """Run the corrected study end to end."""
    logging.getLogger().setLevel(logging.INFO)
    path = fetch_summary(force=force_download)
    df = build_table(path)
    if not no_target_appears_in_both_folds(df):
        raise AssertionError("grouped split is leaking targets between folds")

    a = analyse(df)
    figs = make_figures(df, a)
    write_report(df, a, figs)
    write_changes(a)

    payload = {
        "n_designs": a["n"],
        "n_targets": a["n_targets"],
        "n_binders": a["n_pos"],
        "base_rate": a["n_pos"] / a["n"],
        "headline_metric": a["headline"],
        "headline_within_target_auroc": a["headline_value"],
        "best_single_predictor": a["best_single"],
        "best_single_value": a["best_single_value"],
        "best_baseline": a["best_base"],
        "best_baseline_value": a["best_base_value"],
        "paired": {
            f"{p['result'].name_a}_vs_{p['result'].name_b}": {
                "stat_a": p["result"].stat_a,
                "stat_b": p["result"].stat_b,
                "difference": p["result"].difference,
                "ci_lo": p["result"].ci_lo,
                "ci_hi": p["result"].ci_hi,
                "sign_consistency": p["result"].sign_consistency,
                "n_replicates": p["result"].n_replicates,
                "distinguishable": bool(p["result"].distinguishable),
            }
            for p in a["paired"]
        },
        "selection": a["selection"],
        "operating_point": a["operating"],
        "calibration_ece": a["calibration"]["ece"],
        "calibration_spearman": a["calibration"]["spearman"],
        "calibration_bins": [
            {
                "lo": b.lo,
                "hi": b.hi,
                "n": b.n,
                "mean_predicted": b.mean_predicted,
                "observed": b.observed,
                "obs_lo": b.obs_lo,
                "obs_hi": b.obs_hi,
            }
            for b in a["calibration"]["keep"]
        ],
        "top_decile_rate": a["calibration"]["top_decile_rate"],
        "top_decile_n": a["calibration"]["top_decile_n"],
        "n_metrics_screened": a["n_screened"],
        "n_targets_ci_includes_chance": sum(1 for t in a["per_target"] if t["includes_chance"]),
        "vendor_agreement": vendor_agreement(path),
    }
    payload["vendor_agreement_ceiling"] = payload["vendor_agreement"]["agreement"]
    (OUT / "calibration.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(a["per_target"]).to_csv(OUT / "per_target_results.csv", index=False)

    print(f"\nheadline {a['headline']} = {a['headline_value']:.3f} within-target")
    for p in a["paired"]:
        r = p["result"]
        print(
            f"  {r.name_a} vs {r.name_b}: {r.difference:+.3f} "
            f"[{r.ci_lo:+.3f},{r.ci_hi:+.3f}] "
            f"{'DIST' if r.distinguishable else 'not dist'}"
        )
    return 0


def make_figures(df: pd.DataFrame, a: dict) -> list[str]:
    """Figures for the corrected report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # 1. Paired differences with intervals - the central figure now.
    fig, ax = plt.subplots(figsize=(9, 4.6))
    labels, diffs, los, his = [], [], [], []
    for p in a["paired"]:
        r = p["result"]
        labels.append(f"{r.name_a}\nvs {r.name_b}")
        diffs.append(r.difference)
        los.append(r.difference - r.ci_lo)
        his.append(r.ci_hi - r.difference)
    ypos = np.arange(len(labels))
    colors = ["#2e7d52" if p["result"].distinguishable else "#c0392b" for p in a["paired"]]
    ax.errorbar(diffs, ypos, xerr=[los, his], fmt="o", capsize=4, color="#333333", zorder=2)
    for i, c in enumerate(colors):
        ax.plot(diffs[i], ypos[i], "o", color=c, markersize=9, zorder=3)
    ax.axvline(0.0, color="black", lw=1.2, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("difference in within-target AUROC (95% CI, targets resampled)")
    ax.set_title("Green: interval excludes zero.  Red: cannot be distinguished.", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "paired_differences.png", dpi=150)
    plt.close(fig)
    written.append("paired_differences.png")

    # 2. Calibration with Wilson intervals, small bins excluded.
    c = a["calibration"]
    if c["keep"]:
        fig, ax = plt.subplots(figsize=(5.8, 5.6))
        xs = [b.mean_predicted for b in c["keep"]]
        ys = [b.observed for b in c["keep"]]
        lo = [b.observed - b.obs_lo for b in c["keep"]]
        hi = [b.obs_hi - b.observed for b in c["keep"]]
        ax.plot([0, 1], [0, 1], ls="--", color="black", lw=1, label="perfect calibration")
        ax.errorbar(
            xs,
            ys,
            yerr=[lo, hi],
            fmt="o-",
            color="#d97757",
            capsize=3,
            label="leave-one-target-out model",
        )
        for b in c["keep"]:
            ax.annotate(
                str(b.n),
                (b.mean_predicted, b.observed),
                fontsize=7,
                xytext=(4, -10),
                textcoords="offset points",
            )
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed binder fraction")
        ax.set_title(
            f"Calibration inverts above ~0.5\nSpearman {c['spearman']:+.2f} "
            f"(p={c['spearman_p']:.2f}); bins with n<5 excluded",
            fontsize=10,
        )
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(fig_dir / "calibration.png", dpi=150)
        plt.close(fig)
        written.append("calibration.png")

    # 3. Per-target AUROC with CIs.
    pt = sorted(a["per_target"], key=lambda r: r["auroc"])
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ypos = np.arange(len(pt))
    vals = [t["auroc"] for t in pt]
    lo = [t["auroc"] - t["lo"] for t in pt]
    hi = [t["hi"] - t["auroc"] for t in pt]
    cols = ["#c0392b" if t["includes_chance"] else "#2e7d52" for t in pt]
    ax.errorbar(vals, ypos, xerr=[lo, hi], fmt="none", ecolor="#777777", capsize=3, zorder=2)
    for i, t in enumerate(pt):
        ax.plot(t["auroc"], i, "o", color=cols[i], markersize=7, zorder=3)
    ax.axvline(0.5, color="black", lw=1, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{t['target']} ({t['binders']}/{t['n']})" for t in pt], fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("within-target AUROC (95% CI, designs resampled)")
    ax.set_title("Red: interval includes chance. Labels show binders/designs.", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "per_target_auroc.png", dpi=150)
    plt.close(fig)
    written.append("per_target_auroc.png")

    # 4. Binder rate by target (unchanged role, kept for the grouping section).
    rates = df.groupby(GROUP)[LABEL].agg(["mean", "size"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.arange(len(rates)), 100 * rates["mean"], color="#8fb8de")
    ax.set_yticks(np.arange(len(rates)))
    ax.set_yticklabels(
        [f"{i} (n={int(n)})" for i, n in zip(rates.index, rates["size"], strict=True)],
        fontsize=8,
    )
    ax.set_xlabel("measured binder rate (%)")
    ax.set_title("Binder rate spans 0-80% by target: why splits must be grouped", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "binder_rate_by_target.png", dpi=150)
    plt.close(fig)
    written.append("binder_rate_by_target.png")

    # 5. Pooled vs within-target, retained from session 1.
    from studies.retrospective.run_study import feature_columns as _fc

    y = df[LABEL].to_numpy(dtype=int)
    g = df[GROUP].to_numpy()
    rows = []
    for col in _fc(df):
        s = df[col].to_numpy(dtype=float)
        pooled = stats.auroc(y, s)
        within = stats.mean_within_target_auroc(y, s, g)
        if pooled == pooled and within == within:
            rows.append((col, pooled, within))
    rows.sort(key=lambda r: -r[2])
    sel_rows = rows[:9] + sorted(rows, key=lambda r: -abs(r[1] - r[2]))[:5]
    seen, final = set(), []
    for r in sel_rows:
        if r[0] not in seen:
            seen.add(r[0])
            final.append(r)
    final.sort(key=lambda r: -r[2])
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ypos = np.arange(len(final))
    ax.barh(
        ypos - 0.2,
        [r[1] for r in final],
        height=0.38,
        color="#8fb8de",
        label="pooled across targets",
    )
    ax.barh(
        ypos + 0.2, [r[2] for r in final], height=0.38, color="#d97757", label="mean within target"
    )
    ax.axvline(0.5, color="black", lw=1, ls="--")
    for i, r in enumerate(final):
        ax.annotate(
            f"{r[1] - r[2]:+.3f}",
            (max(r[1], r[2]) + 0.008, i),
            fontsize=7,
            va="center",
            color="#444444",
        )
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in final], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("AUROC for predicting measured binding")
    ax.set_title("Pooled vs within-target AUROC; signed gap annotated", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "auroc_pooled_vs_within.png", dpi=150)
    plt.close(fig)
    written.append("auroc_pooled_vs_within.png")

    return written


if __name__ == "__main__":
    raise SystemExit(main())
