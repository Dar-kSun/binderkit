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

# Study 3: the replication, and what it does to study 2

Run to try to break study 2, with the decision rules fixed in docs/SPEC.md
section 8.6 before any output was seen. Full numbers in
`studies/replication/REPORT.md`.

## 11. SURVIVES, STRENGTHENED - "no geometry metric adds to confidence"

The standing objection to study 2 was that it added geometry as a **linear**
term, while the published positive result uses a **product**, and a logistic
model in (confidence, geometry) cannot represent an interaction unless it is
handed one. That objection is now answered directly for shape complementarity,
on 1,153 designs across 15 targets:

| Test | Confidence | Confidence x Sc | Difference | 95% CI |
|---|---|---|---|---|
| (a) mean within-target AP, their measure | 0.582 | 0.521 | **-0.061** | -0.108 to -0.017 |
| (b) mean within-target AUROC, ours | 0.746 | 0.698 | **-0.048** | -0.091 to -0.008 |
| (c) LOTO model, + Sc linearly | 0.746 | 0.746 | -0.001 | -0.005 to +0.005 |
| (c) LOTO model, + explicit interaction | 0.746 | 0.744 | -0.002 | -0.019 to +0.014 |

(a) and (b) agree, so the AP-versus-AUROC choice is not what decides this; and
the model handed the interaction explicitly still does not improve.

One contrast is worth reading carefully. The **raw product measurably hurts**
while the **fitted interaction is merely inert**. An unweighted product forces
the geometry in at full strength, so Sc's noise goes straight into the
ranking; a logistic model given the same interaction can set its coefficient
near zero, and does. Both readings say the same thing: there is no signal here
for the model to find.

Precision@20 - the number this competition actually faces - is 0.433 for
confidence and 0.400 for the product, against a 27.0% base rate.

**The pre-registered rule that applies is "cannot reproduce their result", and
its instruction is explicit: do not claim they are wrong.** Three times the
data beats one campaign. The confounds are real and are listed in the study
report: their 11.6% binder rate against our 27.0%, geometry on complexes they
re-predicted against our design models, side chains on only 191 of 1,153
binder models here, their heterogeneous binding definitions against our single
rubric, and an Sc implementation that is independent rather than Rosetta's.

## 12. NOT ATTEMPTED - the half of their result that matters more

`interface_dG` and `interface_dSASA` come from Rosetta's
`InterfaceAnalyzerMover`. PyRosetta is free for academic use but is
distributed under a per-user licence requiring credentials this environment
does not have, and section 8.6 forbids substituting a different energy
function and calling it a replication.

So `ipSAE_min x interface_dG/dSASA`, **their strongest reported combination,
remains untested here**, and every statement about study 3 says so. This is
the single most valuable thing the author could unblock: it needs a licence
request and a re-run, not new data or new compute.

## 13. NEW - the Sc implementation, and what validating it caught

No per-design reference value for shape complementarity exists in the release,
so the implementation could not be checked the way the contact counts were
(981/981 exact). It was checked against an external published result instead,
which is the direction section 8.9 calls the cheapest credibility available.

It found a real defect. Without Lawrence and Colman's 1.5 A peripheral trim,
the implementation returned **0.457** on a crystallographic antibody-antigen
interface whose published band is 0.64-0.68 - 0.2 low. With the trim it
returns **0.612**, and a sweep over the trim width peaks at exactly the
published 1.5 A, which is independent confirmation that the trim was the
missing piece rather than a convenient fudge.

A 1.4 A probe would put it inside the published band. It was **not** adopted:
choosing a parameter because it reproduces the expected answer is fitting the
method to the result. The residual gap of about 0.05, attributable to not
reconstructing the re-entrant surface, is recorded instead, and absolute Sc
values from this repo must not be compared with published thresholds. Only
within-target ranking is used downstream, which a constant offset cannot
change.

## What study 3 changes, and what it does not

- **Changes:** the geometry conclusion may now be stated as surviving a direct
  test of the product form, for shape complementarity. The repo is no longer
  an unreplicated negative standing against a published positive on that half.
- **Does not change:** any number in study 1 or study 2. Re-running the
  geometry metrics with Sc added left every other column bit-identical and the
  contact validation at 981/981.
- **Still outstanding:** the energy half, and the design-model-versus-co-fold
  question, which remains the most plausible single explanation of any
  residual disagreement and needs GPU time.

## 14. CORRECTED - "PyRosetta requires licence credentials"

Study 3 shipped with this reason for not testing the interface-energy half of
Overath et al.'s result:

> PyRosetta requires licence credentials not present in this environment

**That was wrong on both counts, and it was never checked.** It was inferred
from `pip install pyrosetta` returning *No matching distribution found*, which
only means the package is absent from PyPI. The facts:

| Claimed | Actual |
|---|---|
| licence issued per user on request | **free for academic, non-profit and government use with no form, no account and no credentials** -- the non-commercial licence now ships with the download, and the paid UW CoMotion licence applies to commercial users only |
| distributed through a credentialed channel | an ordinary `--find-links` index: `pip install pyrosetta --find-links https://west.rosettacommons.org/pyrosetta/quarterly/release` |
| blocked by licensing | blocked by **platform**: the index carries 36 artifacts, 12 `linux_x86_64` and 24 `macosx`, and **zero** `win_amd64`. There is no Windows build. The documented Windows route is WSL. |

The conclusion -- that the dG arm could not be run on this machine as
configured -- happened to be right. The reason given for it was not, and a
reason that is wrong in a published file is a defect whether or not it
changes the verdict. It also made the problem look permanent when it is not:
a licence this project cannot obtain is a wall, a missing Windows wheel is an
afternoon.

**The lesson is the one section 8.9 keeps paying out.** A failed command was
read as evidence for a claim about licensing that the command could not
possibly support. The check that would have caught it -- reading the
distributor's own download page -- took one fetch, and it is the same class
of error as "no published precedent" in correction 9: an assumption about the
outside world, stated in a shipped document, that one lookup disproves.

Corrected in `docs/TOOLS.md`, `docs/LIMITATIONS.md`,
`studies/replication/run.py` and its report, and in `portfolio/`.
