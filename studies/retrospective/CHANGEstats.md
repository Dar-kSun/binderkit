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


---

# What changed in session 4

Two corrections and one narrowing, all of them found by reading the shipped
reports against their own tables, or against the literature. None came from
new data. docs/SPEC.md section 8.7 specifies them; this is the record.

## 8. WITHDRAWN - "designs that bound had fewer contacts and smaller buried surface"

`studies/interface_geometry/REPORT.md` stated this and read it as evidence
that a large contact patch marks an implausible pose. The comparison was
computed **pooled across targets**. It is Simpson's paradox: the targets with
the largest interfaces (TNFa, MBP) are the ones almost nobody could bind, so
the pooled difference measures which targets are hard, not which designs are
good.

Recomputed within target on the same file:

| Metric | pooled difference | within-target difference | targets positive |
|---|---|---|---|
| `n_atom_contacts` | -51.13 | **+31.18** | 9/14 |
| `bsa_binder` | -70.04 | **+49.28** | 10/14 |
| `bsa_total` | -136.61 | **+72.84** | 9/14 |
| `contact_density` | -0.55 | **+0.49** | 8/14 |

Every sign reverses. Binders have **more** contacts and **more** buried
surface within a target, which is exactly what the report's own within-target
AUROC column said three paragraphs earlier (`n_atom_contacts` 0.530,
`bsa_binder` 0.536, both "higher is better"). The report contradicted itself
on the same page.

**The study's conclusion is unaffected** - the conditional test was
within-target throughout, and the numbers in it have not moved - but the most
quotable sentence in the report was false, and it had been copied into
`docs/WRITEUP.md`.

The guard: `stats.within_group_mean_difference` now returns the pooled and the
within-group difference together, with the pooled field documented as
do-not-quote, and `tests/test_pooled_difference_guard.py` asserts that no line
of the report generator formats a pooled difference without the within-target
one beside it. The whole of section 8.3 exists to prevent this error and it
still reached a published file; a test is worth more than another instruction.

## 9. WITHDRAWN - "no published precedent" / "almost nobody has checked"

Claimed in `studies/interface_geometry/REPORT.md`, `docs/CLUSTER_REQUEST.md`,
`docs/WRITEUP.md` and `NEXT_SESSION.md`. It does not survive a literature
search:

> Overath MD, Rygaard ASH, Jacobsen CP, Brasas V, Morell O, Sormanni P,
> Jenkins TP. *Predicting Experimental Success in De Novo Binder Design: A
> Meta-Analysis of 3,766 Experimentally Characterised Binders.* bioRxiv
> 2025.08.14.670059v2, 17 Sep 2025.

The accurate claim, which does survive, is **no prior analysis of this
dataset**. Their corpus is three times larger; ours is better instrumented
(one campaign, one adjudication rubric, two independent assays, generator
recorded per design).

Worth stating plainly: their work **agrees** with ours on the two results that
matter most. ipSAE-family confidence is the best single predictor in both, and
they also find that pooling features across structure predictors does not
improve performance - which is this repo's conclusion 2 and 3, replicated
externally on 3,766 designs. That is the strongest external support the repo
has and it was sitting unclaimed while the reports instead claimed novelty
that a reviewer could disprove in one search.

## 10. NARROWED - "interface geometry is redundant"

Their best combinations are confidence **times** geometry
(`ipSAE_min x interface_dG/dSASA`, `LIS x shape_complementarity`), and both
beat either component alone. Study 2 concluded no geometry metric adds
anything. That is not yet a real disagreement, for two reasons, and both are
limits of study 2:

1. **Different features.** Interface dG/dSASA and Lawrence-Colman shape
   complementarity were not among the thirteen tested. The two the literature
   says work are the two that were not measured.
2. **Different functional form.** Theirs is a *product*; study 2 added
   geometry as an additional *linear* term to a logistic model that already
   had confidence. A linear model in (confidence, geometry) cannot represent
   confidence x geometry without an explicit interaction term. **Study 2 did
   not test their claim** - it tested a weaker one, and a null additive effect
   is fully compatible with a real multiplicative effect.

Every statement of the geometry result is now in its narrow form: *no additive
effect was found for thirteen geometry metrics; a multiplicative interaction
with interface energetics was not tested and is reported in the literature to
work.* Study 3 (`binderkit study replication`, docs/SPEC.md section 8.6) is the
experiment that settles it, and its decision rules are pre-registered there
before any of its output is seen.

## Unchanged in session 4

- Every number in both studies. No analysis was re-run except the within-target
  group differences above, and the regenerated report reproduces all prior
  figures to the digit.
- The validation: 981/981 exact agreement on epitope, paratope and atom
  contacts.
- Study 1's headline and all of its intervals.
