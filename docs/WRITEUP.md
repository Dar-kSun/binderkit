# Does anyone check whether protein design scores actually work?

I measured two things on 1,320 designed proteins whose real-world results are
public. One confirmed a method people already use. The other says a step most
pipelines run is not worth running.

---

## The problem

A **binder** is a small protein designed to stick to a chosen target — a
receptor on a cancer cell, say. Designing one used to take months. Models now
generate candidates in minutes, by the thousand.

So the bottleneck moved. It is no longer *making* candidates but **choosing
which handful to physically build**: each lab test costs real money and several
weeks, so you might build twenty out of ten thousand.

Everyone chooses using scores from structure-prediction models — software that
guesses the 3D shape of the binder stuck to its target and reports how confident
it is. How well those scores predict what the lab measures has been asked
before — most thoroughly by Overath et al. in 2025, who pooled 3,766 binders
from many published campaigns — but it is hard to answer well, because pooled
campaigns disagree about what counts as "binding" and rarely record how each
design was made. In August 2026 Anthropic published 1,440 designed binders with
their scores **and** their wet-lab outcomes, openly licensed: one campaign, one
rubric for calling a binder, and two independent labs measuring the same
molecules. That makes a cleaner version of the check possible, and nobody had
run it on this dataset.

---

## Study 1: the confidence scores do work, but less neatly than advertised

I compared each score against the measured outcome — did this protein actually
stick? — across 15 different targets.

The honest measure is **within-target AUROC**: given one target, how well does
the score rank that target's designs? (0.5 is a coin flip, 1.0 is perfect.)
Pooling targets inflates it, because some targets are simply easier and a score
gets credit for noticing that rather than for ranking designs.

| What | Within-target AUROC |
|---|---|
| Co-folding confidence (ipSAE, averaged over ten predictors) | **0.761** |
| Best single predictor on its own | 0.733 |
| A model trained on all 27 score columns | 0.726 |
| A deliberately useless baseline (how greasy the protein is) | 0.589 |

The gap that matters is confidence versus the useless baseline: **+0.172, with a
95% confidence interval of +0.077 to +0.273**. That held in every one of 2,000
resamples. The scores carry real signal.

Three things I expected to find, and did not:

- **Averaging ten predictors is not measurably better than using one.** The gap
  is +0.028 with an interval of −0.031 to +0.081 — it includes zero. I had
  concluded otherwise before putting an interval on it. One predictor is enough,
  which is roughly a tenfold saving.
- **Training a model on all the scores made it worse** than simply averaging
  them (−0.034, interval −0.063 to −0.004). With only 15 targets, a model learns
  which scores to trust on the targets it saw and that does not carry over.
- **The confident predictions are the least trustworthy.** Designs the model
  rated 70–80% likely to bind actually bound 28% of the time — *worse* than
  designs it rated 50–60%. The reliability curve inverts exactly where a
  confident call would be acted on.

![pooled vs within-target AUROC](../studies/retrospective/figures/paired_differences.png)

---

## Study 2: the structural analysis adds nothing — in the form I tested it

Beyond the confidence score, pipelines compute the **geometry of the
interface** — how much surface is buried, how many atoms touch, how many
hydrogen bonds form. It is slower and more elaborate. Does it add anything?

The released data includes 3D models of the complexes, so I computed thirteen
geometry measurements on real coordinates for 1,189 designs with known
outcomes, without running any structure prediction myself.

**Added one at a time, none of them helped.** Not one of the thirteen improved
a model that already had the confidence score. Two made it measurably worse.
All thirteen together scored **0.049 lower** than confidence alone (interval
−0.091 to −0.006).

![conditional effect of geometry](../studies/interface_geometry/figures/geometry_conditional.png)

**Here is where my result argues with the literature, and I think the
literature may win.** Overath et al. report that confidence *multiplied by*
interface energy per buried area beats either on its own. I never computed that
feature, and more importantly my test adds geometry as an extra *linear* term,
which cannot express a product. So I have not tested their claim — I have
tested a weaker one, and a null for mine is perfectly compatible with a real
effect for theirs. The replication that would settle it is specified and needs
no GPU. Until it runs, the honest statement is: *no additive effect for
thirteen metrics*, not *geometry is useless*.

An earlier version of this write-up claimed designs that bound had **fewer**
contacts and **less** buried surface, and read that as a big contact patch
marking an implausible pose. That was wrong, and wrong in an instructive way:
the comparison was pooled across targets, and the targets with the biggest
interfaces are the ones nobody could bind. Recomputed inside each target, every
one of those differences reverses — binders have **more** contacts and **more**
buried surface, in 9 or 10 targets out of 14. The study's actual conclusion was
never affected, because that was within-target throughout, but the most
quotable sentence in it was false. It is withdrawn.

I trust these numbers because the code is checked, not assumed correct. The
release publishes its own contact counts for the same files, and my
implementation reproduces them **exactly on all 981 comparable designs**.
Getting there caught a genuine bug: some targets include RNA or metal ions, and
I had been counting protein-to-RNA contacts as part of the interface, inflating
some designs eightyfold.

---

## What this means if you do this work

1. **Rank on one co-folding confidence score.** Averaging several is not
   measurably better, and training a model on them is worse.
2. **Adding interface geometry to a confidence score did not help my ranking.**
   Thirteen metrics, none of them an improvement, added linearly. I would not
   yet generalise that to "skip the geometry stage": the literature reports a
   *multiplicative* combination that works and I have not tested it. Geometry
   remains useful for diagnosing *why* a design looks wrong either way.
3. **Do not use any score as a hard cut-off.** Performance varies widely by
   target, and on 4 of 14 targets the interval includes chance.
4. **Expect modest enrichment.** At the best threshold, about 46% of selected
   designs bound, against a 27% baseline — roughly 1.7× better than picking at
   random. That is worth having and it still means most selected designs fail.
5. **Treat high confidence with suspicion**, because that is where calibration
   breaks down.

---

## Why more compute is the next step, specifically

Not "more compute is better". Here is the actual gap.

Everything above was measured on **someone else's designs**. My own pipeline
runs end to end — target preparation, filtering, a novelty check against 1.1
million known protein sequences, ranking, packaging, 96 automated tests — but
the three stages that *create* designs are stubs, because they need hardware
this machine does not have.

Measured on this laptop (RTX 4060, 8.19 GB of graphics memory):

| Stage | Status here | Why |
|---|---|---|
| Scoring, novelty search, geometry | **Runs.** 0.32 s per design for the PDB search, 0.32 s for geometry | CPU only |
| Protein language model (ESM-2, 35M) | **Runs on the GPU**, 2.2 ms per sequence | uses 195 MB of 8,188 |
| Backbone generation (RFdiffusion) | **Cannot run** | needs ~12–24 GB |
| Co-folding (Boltz-2) | **Cannot run** | needs ~16–24 GB |

The complex in question is ~290 residues, which is small. The blocker is
memory, not difficulty: 8.19 GB against a 12 GB floor.

What compute would buy is a **test of a stated hypothesis**, not a promise of
results. Both findings rest on designs chosen by someone else's process, which
is their main weakness; generating designs and checking whether the same
relationships hold is the experiment. Study 2 may also remove a stage from that
pipeline, so hours go to generation rather than scoring — pending the
replication described above.

---

## What would prove this wrong

This section matters more than the results.

- **The designs were not randomly chosen.** The release states that ipSAE — one
  of the scores I evaluated — is what they used to pick which designs to build.
  I am grading a score on a set it already filtered. That probably *understates*
  its discrimination, but it means the 46% will not transfer to a pipeline
  generating designs from scratch.
- **Fifteen targets is not many.** Every per-target number has an interval wide
  enough to include chance for several of them.
- **The lab disagrees with itself.** Two contract labs tested the same designs
  and agreed only 89% of the time. No score can be expected to beat that.
- **Geometry on a *predicted* complex might behave differently** from geometry
  on the design model I used. That is the obvious next test and it needs no new
  lab work.
- **These were 50–120 residue mini-proteins against 16 targets.** Nothing here
  establishes that it generalises past that.

If someone repeated this on an unfiltered set of designs and found the
confidence scores did much better, or that geometry did add signal, I would not
be surprised. That is the point of writing the numbers down first.

---

## Status and reproducing it

Everything is in one repository. `binderkit study retrospective` regenerates
study 1, `binderkit study geometry` regenerates study 2, and every number in
this write-up comes from one of those two commands.

Honest status: the analysis is real and validated, the pipeline runs end to end,
and the design-generation stages are stubs that fail loudly rather than
pretending to work. `docs/LIMITATIONS.md` lists every one of them.
`studies/retrospective/CHANGES.md` lists the conclusions I had to withdraw when
I put confidence intervals on them, which is the part I would read first.
