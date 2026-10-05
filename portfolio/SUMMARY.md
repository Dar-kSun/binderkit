# Do the scores everyone ranks on actually predict binding?

Generative models made candidate binders cheap: ten thousand overnight. What
did not get cheaper is finding out whether any of them works. Each one has to
be synthesised and measured, which costs real money and several weeks, so a
team builds perhaps twenty of the ten thousand.

That makes ranking the whole game, and almost everyone ranks the same way:
predict the structure of the binder stuck to its target, read off the model's
confidence, build the top of the list. Whether that predicts what the lab
measures is hard to check, because you need designs whose scores *and* real
outcomes are public. In August 2026 such a dataset appeared: 1,440 designed
binders, openly licensed, with both.

## Confidence works, and three things about it are uncomfortable

Against a deliberately stupid baseline — rank by hydrophobic content alone —
confidence wins by **+0.172 AUROC** (95% CI +0.077 to +0.273). A real signal,
worth using.

**Averaging ten structure predictors is not measurably better than using
one** (+0.028, interval -0.031 to +0.081). The campaign ran ten predictors at
five seeds each; one at five seeds is a tenfold cut in the most expensive
stage for no measurable loss. **Training a model on all the score columns is
worse** than averaging them (+0.034 in the average's favour, CI +0.004 to
+0.063).

**The reliability curve inverts exactly where decisions are made.** Designs
predicted at 0.729 bound only **28.2%** of the time, worse than mid-range
predictions. Expected calibration error is 0.078, which looks respectable and
conceals this entirely. Taking the top twenty by score spends the whole budget
in the least trustworthy part of the metric.

## Interface geometry adds nothing — and I tried hard to break that

On 1,189 designs, none of thirteen interface geometry metrics improved a model
that already had the confidence score. Two made it measurably worse; all
thirteen together scored **-0.049** lower (CI -0.091 to -0.006).

That result had a standing objection, and a good one. Overath et al. (2025),
pooling 3,766 characterised binders, report that confidence *multiplied by*
interface geometry beats either alone. Their combination is a product where my
test added geometry as a linear term — and a model given the two separately
cannot represent their product unless the interaction is handed to it. So my
test had not addressed their claim at all.

So I ran the replication, with the decision rules written down first, and
both of their features ended up computed. Shape complementarity has a
licence-free definition, so I reimplemented it with the published parameters
and tested the product both their way and mine. Interface energy needs
Rosetta, which has no Windows build and runs here under WSL.

It does not help. By average precision, their measure: 0.582 for confidence
alone against 0.521 for the product (-0.061, CI -0.108 to -0.017). By
within-target AUROC: 0.746 against 0.698 (-0.048, CI -0.091 to -0.008). A
model handed the interaction explicitly gains nothing either (-0.002, CI
-0.019 to +0.014). At precision@20 — what a twenty-design submission actually
faces — confidence gets 0.433 and the product 0.400.

**Their stronger combination still cannot be tested here.** A Rosetta energy
means nothing on a backbone-only model, so that arm runs on just 173 labelled
designs across 12 targets, at a 16.2% binder rate against 26.8% overall.
Average precision goes 0.515 to 0.410 and within-target AUROC moves -0.082,
but the intervals are -0.269 to +0.044 and -0.340 to +0.224. Everything
straddles zero by a wide margin. The point estimates lean the same way as the
other arm and are worth nothing on their own.

The setup itself was checked before any of that was believed: Rosetta's
buried-surface-area number agrees with the one my own code computes at 0.947
correlation over 223 designs, a median ratio of 1.040 - two implementations
written independently. What fails is not the measurement but the material.
Only 51.6% of these deposited models reach even a negative interface energy,
where a real binding interface sits tens of units below zero, and Rosetta
segfaults outright on 21 of the 244 structures.

**This does not make them wrong**, and the rule I fixed in advance says so.
Their corpus is three times larger, their binder rate is far below the 26.8%
here, they computed geometry on complexes they re-predicted while mine comes
from the designers' own models, and only 191 of these binder models carry side
chains at all. The claim I can make is narrower: on this dataset the product
form does not rescue geometry, and the obvious objection to the earlier result
is now closed by direct test rather than by argument.

## Why the intervals are wide

Designs against the same target are not independent, so every split holds out
whole targets and every score is computed inside a target first. Metrics are
compared on identical resampled sets of targets — the only way a 0.03 gap
resolves at this sample size, and what showed two such gaps could not be. The
headline metric was the best of 30 screened, so the selection is re-run inside
every bootstrap replicate: the interval widens to 0.689–0.847 and the nominal
winner wins only 64% of them.

## Corrections, kept in the record

An early version stated two conclusions its own uncertainty did not support;
both were withdrawn once the differences carried intervals. A later version
claimed binders had *fewer* contacts and *less* buried surface — pooled across
targets, where the largest interfaces belong to the targets nobody could bind.
Within target, every sign reverses.

The most useful bug surfaced while reimplementing someone else's statistic.
Checked against a crystallographic interface with a published value, my first
version read 0.457 — it was missing a 1.5 Å peripheral trim the original
specifies. With it: 0.612. A smaller probe would have landed it inside the
published band; I did not use one, because choosing a parameter *because* it
reproduces the expected answer is fitting the method to the result.

## What this is not

The interface-energy half settles nothing. No design here has been made or tested,
and nothing has been submitted anywhere. Of 1,235 designs measured by two
independent laboratories, the two agree only 89.0% of the time, which bounds
how well any score could appear to do. At the measured operating point 46.3%
of selected designs bound against a 26.8% base rate — about 1.73x — measured
on a pool already filtered with the same metric. And most de novo designs
fail: on EGFR the campaign's own binder rate was 11.1%.
