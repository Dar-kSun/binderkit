# What changed between session 1 and session 2

Session 1's method was sound and its numbers were reproducible. What it got
wrong was the strength of the language: it stated conclusions about
*differences* without putting an interval on those differences. Adding paired
intervals and a correction for screening 30 metrics changed three conclusions,
withdrew one, and left the main result standing.

This file exists because a changelog of retractions is itself evidence of
method. Nothing below was found by new data: the same table, analysed more
carefully, says something different.

## 1. WITHDRAWN - "averaging ten predictors beats the best single predictor"

| | Session 1 | Session 2 |
|---|---|---|
| Claim | the mean beats the best individual predictor by +0.028 | cannot be distinguished |
| Evidence | point estimate only | +0.028, 95% CI -0.031 to +0.081, sign held 84% |

The interval straddles zero. The point estimate is unchanged; what changed is
that it is now accompanied by the uncertainty it always had.

**This reverses the practical advice.** Session 1 implied you should run ten
predictors and average them. Since the mean is not measurably better than one
good predictor, a compute-constrained user should run **one**. That is a
roughly ten-fold reduction in co-folding cost for no measurable loss.

## 2. SURVIVES, WEAKENED - "do not learn weights over the predictors"

The learned model is worse than the plain average: +0.034, 95% CI +0.004 to +0.063. The interval excludes zero,
so the effect is real, but it sits close to the boundary and should be
described as a modest effect rather than a headline.

## 3. WITHDRAWN - "cross-predictor disagreement is usable signal"

Session 1 observed `ipsae_std_across` at AUROC 0.370 and concluded it predicts
with its sign flipped. The missing test was whether it adds anything *given*
the mean it is correlated with.

- correlation with `ipsae_mean`: -0.220
- residualised on the mean: 0.399
- model with mean + disagreement vs mean alone: -0.023, 95% CI -0.042 to -0.008

Adding it makes held-out prediction worse. The conclusion is withdrawn.

## 4. CORRECTED - per-target claims

Session 1 quoted per-target AUROCs as point estimates. With intervals:

- **EGFR**: 0.669, CI 0.466-0.857 (10 binders). Session 1: "the metric is weaker on EGFR than
  on average". The interval includes chance, so neither that claim nor its
  negation is supported.
- **BBF-14**: 0.398, CI 0.000-0.764 (3 binders). Session 1: "below chance". Withdrawn; the
  interval covers nearly the whole range.

The decision these fed into - confidence as a soft signal, never a hard filter
- is unchanged, and is better supported by wide intervals than it was by the
point estimates.

## 5. NEW - the metric screen is now corrected for

30 metrics were screened and the headline was the
maximum over them, which is upward biased. Re-selecting the winner inside each
replicate:

- the nominal winner `ipsae_mean` wins only 64% of replicates
- 8 different metrics win at least once
- selection-corrected interval: 0.689 to 0.847

So *which* metric is best is not resolved by this dataset.

## 6. NEW - calibration is non-monotonic at the top

Session 1 reported ECE 0.078 and treated it as the calibration
headline, and plotted a bin with one design in it. The curve actually inverts:
Spearman correlation between predicted and observed across bins is
+0.524 (p = 0.18), and the most confident
reportable bin has a *lower* observed binder rate than mid-range bins. Bins
with fewer than five designs are now excluded, and every bin carries a Wilson
interval.

## 7. NEW - selection bias is now stated in the report, not just the limitations

The release documents that ipSAE is "the primary confidence metric and the one
designs were selected on". The evaluation set is therefore filtered on the
metric under test. This deflates the measured AUROC and, more importantly,
means the operating point does not transfer to an unscreened pipeline.

## Unchanged

- The main result: confidence beats a trivial baseline by +0.172, 95% CI +0.077 to +0.273, sign held in 100% of replicates.
- Grouping by target throughout.
- The 89% inter-assay agreement ceiling.
- The decision to treat confidence as a soft ranking signal.
