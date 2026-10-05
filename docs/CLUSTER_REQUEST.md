# Compute request: de novo binder design, validated scoring

For someone who controls compute and has not read anything else in this
repository. Two pages.

Every figure below is labelled **measured** (recorded on this machine, in
`docs/benchmarks.json`, reproducible with `binderkit compute` and the benchmark
suite) or **cited** (taken from a published source, with the source named). None
is a guess, and the two are never combined into a single number without saying
so.

---

## 1. The result

I analysed 1,320 de novo designed proteins whose laboratory outcomes are public,
across 15 target proteins, to ask whether the in-silico scores the field ranks
designs on actually predict which ones bind.

They do, by a clear margin over a deliberately useless baseline: within-target
AUROC 0.761 against 0.589, a difference of **+0.172 with a 95% confidence
interval of +0.077 to +0.273**. Two things I expected did not survive contact
with a confidence interval: averaging ten structure predictors is **not**
measurably better than using one (+0.028, interval −0.031 to +0.081), and a
model trained on all 27 score columns did **worse** than their plain average.
Every per-target number has an interval wide enough that four of fourteen
include chance, so no score is safe as a hard filter.

The second study has no published precedent. Pipelines routinely compute
interface geometry — buried surface area, contact counts, hydrogen bonds — and
rank on it. Computing thirteen such measurements on real coordinates for 1,189
designs with known outcomes, **none of them improves a model that already has
the confidence score**, two make it measurably worse, and all thirteen together
score 0.049 lower than confidence alone (interval −0.091 to −0.006). The
expensive structural stage is redundant for ranking. That is a direct reduction
in what any scaled pipeline needs to compute.

---

## 2. What is already built and verified without GPUs

The point of this section: **the scoring and the plumbing are done**, so GPU
hours would go to generating designs rather than to debugging a pipeline.

- **96 automated tests**, passing offline with no GPU and no network, in 4
  seconds (**measured**).
- **Metric code validated against published values.** The interface geometry
  implementation reproduces the release's own epitope, paratope and atom-contact
  counts **exactly on all 981 comparable designs**, correlation 1.0000
  (**measured**). It can be pointed at new designs without re-litigating whether
  it is correct.
- **A real novelty search**, MMseqs2 against 1,103,516 PDB protein sequences,
  with a positive control confirming a known protein is found at e = 6 × 10⁻³⁶
  (**measured**).
- End-to-end pipeline: target preparation from PDB and UniProt with a verified
  residue-numbering map, filtering, multi-criteria ranking, submission packaging
  and a format validator.

Per-design cost of everything that is not generation (**measured** on 20 CPU
cores):

| Stage | Cost per design |
|---|---|
| Novelty search against 1.1M PDB sequences | 0.34 s |
| Interface geometry on real coordinates | 0.32 s |
| Sequence-level metrics | 0.001 s |
| ESM-2 35M embedding (GPU) | 0.002 s |

At those rates, scoring 10,000 designs costs under two CPU-hours. **Scoring is
not the expensive part and does not need the cluster.**

---

## 3. What this machine can and cannot do

**Measured** on the development machine (RTX 4060 Laptop, 20 CPU cores, 23.7 GB
RAM):

| Property | Value |
|---|---|
| Total VRAM | 8,188 MiB (7,106 MiB free in practice) |
| fp16 matrix throughput | 17–20 TFLOP/s |
| ESM-2 35M protein language model | runs on GPU: 2.15 ms/sequence, peak 195 MiB |
| Free disk | 36 GB |

A caution for anyone sizing models by trial allocation on Windows: a naive
allocate-until-failure probe reported **18.25 GiB of "usable" VRAM on an 8.19
GiB card**, because the driver silently spills into system RAM. The real
ceiling is 8.19 GiB.

**Cannot run here**, and the reason is memory rather than difficulty — the
complex of interest is only ~290 residues:

| Stage | Typical requirement (**cited**) | Shortfall |
|---|---|---|
| Backbone generation (RFdiffusion) | ~12–24 GB VRAM | 4–16 GB short |
| Co-folding (Boltz-2) | ~16–24 GB VRAM | 8–16 GB short |

---

## 4. The ask

**Reference point (cited).** The published campaign that produced the dataset
analysed above used up to 12,500 H100-hours for a multi-target campaign and
about 2,500 H100-hours per target for single-target campaigns. The tiers below
are expressed as fractions of that per-target figure, so the request is anchored
to a number someone else has already spent rather than to my estimate.

**Assumption, stated explicitly:** co-folding dominates the cost, and generation
plus sequence design together are a minority of it. This follows from the
published campaign's own description rather than from my measurement, since I
cannot run either stage. If it is wrong, the design counts below move and the
hour totals do not.

| Tier | H100-hours | Fraction of one published campaign-target | Designs generated | What it is for |
|---|---|---|---|---|
| **Minimum viable** | 250 | 10% | ~500 | Prove the pipeline end to end on one target and produce a submittable, honestly-scored set |
| **Target** | 800 | 32% | ~2,000 | Enough designs to re-test both findings on designs this pipeline generated, rather than on someone else's |
| **Stretch** | 2,500 | 100% | ~6,000 | A full single-target campaign comparable to the published one |

Hardware: any GPU with **≥24 GB VRAM** removes both blockers. An A100 40 GB or
H100 is ideal; a single 24 GB card is sufficient for the minimum tier. The work
parallelises across designs trivially, so N GPUs give close to N× throughput.

**The request can be granted partially.** Each tier is independently useful and
the lower ones are prefixes of the higher ones.

---

## 5. What each tier yields

- **Minimum (250 h).** One target, end to end, with real generated designs
  instead of stubs. Delivers a validated pipeline and a package a human can
  review. Does **not** deliver statistical power to re-test the studies.
- **Target (800 h).** Enough designs to ask the question that matters: both
  published findings were measured on designs selected by someone else's
  process, using the very score being evaluated. On designs generated here, the
  selection bias is removed and the finding can be confirmed or refuted. This is
  the tier I would argue for.
- **Stretch (2,500 h).** A campaign comparable to the published one, with the
  scoring stage already validated and the redundant structural stage removed.

---

## 6. Risk, and what happens if this does not work

- **Most de novo designs do not bind.** In the published campaign the measured
  rate was **26.8%** across all targets and **11.1%** on EGFR specifically — one
  of the harder targets in the set — with far more compute than any tier above.
  No tier here should be read as promising a hit.
- **The two wet-lab assays in that campaign agreed with each other only 89% of
  the time.** That bounds what any in-silico method can be asked to achieve.
- **Enrichment is modest.** At the best threshold, 46% of selected designs bound
  against a 27% base rate, roughly 1.7×. And that figure comes from a pool
  already filtered on the same score, so it is likely optimistic for a pipeline
  generating designs from scratch.
- **Graceful degradation.** If the allocation is smaller than the minimum tier,
  the pipeline still runs with generation stubbed: scoring, novelty and ranking
  are unaffected, and the stubs fail loudly rather than emitting fabricated
  numbers. Nothing silently degrades into a mock.
- **The honest failure mode** is that designs generated here have a lower hit
  rate than the published campaign, which had more compute and more mature
  tooling. That would still be a publishable measurement of how method quality
  and compute trade off.

---

## 7. Storage and data

| Item | Size |
|---|---|
| Model weights (RFdiffusion, ProteinMPNN, Boltz-2) | ~30–60 GB (**cited**) |
| PDB sequence database for novelty search | 0.8 GB (**measured**) |
| Released reference dataset, tables and structures | 0.3 GB (**measured**) |
| Generated designs, structures and scores | ~1 GB per 1,000 designs (**measured**, extrapolated from 231 MB per 1,309 structures) |

Local free disk is **36 GB (measured)**, which is the second reason the
generation stages were never installed here rather than merely slow. A cluster
with ordinary scratch space removes that constraint entirely.

---

## 8. What I would want judged

Not the design results, which do not exist yet. The two measurements in section
1, the fact that the scoring code is validated against published values rather
than assumed correct, and `studies/retrospective/CHANGES.md`, which lists the
conclusions I withdrew once I put confidence intervals on them.
