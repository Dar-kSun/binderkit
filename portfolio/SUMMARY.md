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
binders, openly licensed, with both. This is what two studies on 1,320 and
1,189 of them found.

## Confidence works, and three things about it are uncomfortable

Against a deliberately stupid baseline — rank by hydrophobic content alone —
confidence wins by **+0.172 AUROC** (95% CI +0.077 to +0.273). A real signal,
worth using.

**Averaging ten structure predictors is not measurably better than using one.**
The difference is +0.028, with an interval of -0.031 to +0.081 that comfortably
contains zero. The published campaign ran ten predictors at five seeds each —
fifty folds per design. One predictor at five seeds is five. That is a tenfold
cut in the most expensive stage of the pipeline for no measurable loss.

**Training a model on all the score columns is worse** than averaging them, by
+0.034 in the average's favour (CI +0.004 to +0.063). More machinery, less
performance.

**The reliability curve inverts exactly where decisions are made.** Designs
predicted at 0.729 bound only **28.2%** of the time, worse than mid-range
predictions. Expected calibration error is 0.078, which looks respectable and
conceals this entirely. Taking the top twenty by score spends the whole budget
in the least trustworthy part of the metric.

## Interface geometry added nothing — in the form tested

Pipelines also compute interface geometry: buried surface, atom contacts, the
arrangement of charges and hydrophobic patches. On 1,189 designs with released
coordinates, none of thirteen such metrics improved a model that already had
the confidence score. Two made it measurably worse; all thirteen together
scored **-0.049** lower (CI -0.091 to -0.006).

That claim must stay narrow, because the literature disagrees with a stronger
version of it. Overath et al. (2025), pooling 3,766 characterised binders,
report that confidence *multiplied by* interface energy per buried area beats
either alone. Two reasons that may not be a real conflict, both limits of this
study rather than theirs: that feature was not among the thirteen, and their
combination is a product where this test added geometry as a linear term. A
model given confidence and geometry separately cannot represent their product
unless the interaction is supplied explicitly. So this study tested a weaker
claim than theirs, and a null for the weaker one says little. The experiment
that settles it is specified, with its decision rules fixed in advance.

The same paper *agrees* with the first study on both of its transferable
conclusions: ipSAE-family confidence is the best single predictor, and pooling
features across predictors does not help. Independent corroboration on three
times the data is worth more than a claim of novelty.

## Why the intervals are wide

Most of the work went into not being fooled. Designs against the same target
are not independent — targets range from nearly impossible to nearly trivial —
so every split holds out whole targets and every score is computed inside a
target before averaging. Metrics are compared on identical resampled sets of
targets, the only way a 0.03 gap resolves at this sample size, and that is what
showed two such gaps could not be. The headline metric was the best of 30
screened, so the selection is re-run inside every bootstrap replicate: the
interval widens to 0.689–0.847 and the nominal winner wins only 64% of them.

The geometry code was checked against contact counts the release publishes for
the same files: 981 of 981 exact. Reaching that caught a real bug — several
targets are not protein only, and counting binder-to-RNA and binder-to-ion
pairs as contacts had inflated some designs by a factor of eighty.

## Three corrections, kept in the record

An early version stated two conclusions its own uncertainty did not support;
both were withdrawn once the differences carried intervals. A later version
claimed binders had *fewer* contacts and *less* buried surface — a comparison
pooled across targets, where the largest interfaces belong to the targets
nobody could bind. Within target, every sign reverses. And two documents
claimed nobody had looked at this question before, which one literature search
disproves.

The third is the one worth dwelling on: the first two were caught by better
statistics, the third only by reading what other people had already published.

## What this is not

No design here has been made or tested, and nothing has been submitted
anywhere. Of 1,235 designs measured by two independent laboratories, the two
agree only 89.0% of the time, which bounds how well any score could appear to
do. At the measured operating point 46.3% of selected designs bound against a
26.8% base rate — about 1.73x — measured on a pool already filtered with the
same metric, so it will not transfer unchanged. And most de novo designs fail:
on EGFR the campaign's own binder rate was 11.1%, with far more compute than
this analysis had.
