# Project specification

The numbered requirements this repository is built against. Code, tests and
reports cite these section numbers directly, so the numbering is stable: a
section is never renumbered, only amended or marked superseded.

The spec is deliberately opinionated about *discipline* rather than about
tooling. Most of it exists because a specific mistake was made once and the
rule is what stops it happening again.

---

## 0. Operating principles

### 0.2 Deliverables

In order; each is finished before the next is started.

1. `docs/COMPUTE.md` — what hardware this machine actually has (§1).
2. `docs/BACKGROUND.md` — competition rules and prior art, with citations (§2).
3. Repo scaffold, packaging, tests, lint, CI, all passing (§3).
4. The pipeline, running end to end on a CPU fixture (§4–§7).
5. **The retrospective calibration study (§8) — the headline deliverable.**
6. `docs/TOOLS.md`, `README.md`, `docs/LIMITATIONS.md` (§9).
7. A live-challenge run if one is open, else a dress rehearsal (§10).

Deliverable 5 must be reached even with no GPU. If time runs short, scope is
cut from the build (fewer backends, simpler metrics), never from §8.

### 0.5 Error-handling ladder

When something fails, work down this ladder, and only move down a rung after
genuinely trying the one above.

1. Read the actual error and fix it if it is ours.
2. Check the pinned version, its docs, and its issue tracker for a known fix.
3. Downgrade the component: smaller model, fewer seeds, CPU path.
4. Substitute a different tool behind the same interface.
5. Mock it behind the interface, marked `# MOCK:` in code and recorded in
   `docs/LIMITATIONS.md`.
6. Skip the stage, record it, and keep the pipeline running without it.

Never retry the same failing command more than three times. Never leave a
`# MOCK:` out of `docs/LIMITATIONS.md`. **Never silently produce numbers from a
mock**: any artefact derived from a mocked stage carries `mocked: true` in its
metadata and a banner in its report.

Hard timeouts on every external call: 10 min for a download, 30 min for a
single model run on the fixture, 2 h for any one batch stage. On timeout, go to
rung 3.

---

## 1. Know the machine (`docs/COMPUTE.md`)

Record GPU model and VRAM, CUDA and driver versions, CPU cores, RAM, **free
disk in GB**, Python version, and whether network access works. Then set the
tier, which every later phase branches on.

| Tier | Condition | What the pipeline does |
|---|---|---|
| **A** | ≥24 GB VRAM | RFdiffusion backbones → ProteinMPNN → Boltz-2 co-folding, multiple seeds |
| **B** | 8–24 GB VRAM | Hallucination path or RFdiffusion at reduced length; smallest capable co-folding model; fewer seeds |
| **C** | <8 GB VRAM or no GPU | No generation. ProteinMPNN on supplied backbones, small ESM, MMseqs2, Foldseek, and **all of §8**. Everything heavy is mocked behind interfaces. |

On a cluster, record the **GPU architecture** as well as its VRAM. Volta (V100)
lacks bfloat16 and FlashAttention-2, which much of the current stack assumes;
that single fact constrains tool choice more than VRAM does.

Disk guard: below 50 GB free, do not attempt tier A/B weight downloads; drop a
tier and record why. Abort any download that would leave under 10 GB.

**Tier C is a perfectly good configuration** — §8 is the headline deliverable
and needs no GPU.

---

## 2. Know the competition (`docs/BACKGROUND.md`)

Fetch and summarise, with URLs and access dates: the competition overview and
terms, the current challenge page, the exact required submission columns, the
dataset release, and the method literature. Extract explicitly: the per-challenge
design cap, required CSV columns and allowed `molecule_class` values, length
bounds, the ranked-order requirement, how selection works, and what "de novo and
zero-shot" excludes.

Every claim carries its source. Anything unverifiable goes in
`docs/LIMITATIONS.md` as an assumption, never into `README.md` as fact.

---

## 3. Scaffold and conventions

`src/` layout, Python ≥3.10, ruff and pytest configured, work artefacts under a
gitignored `work/`.

- Type hints on every public function; NumPy-style docstrings.
- Every stage reads and writes **pandas DataFrames with declared schemas**
  (§6.5). No ad-hoc dicts between stages.
- Every stage is **resumable**: it writes `work/<run_id>/<stage>.parquet` and
  skips work whose output exists, unless `--force`.
- All randomness takes an explicit seed, recorded per design.
- Logging via `logging`, not `print`.
- No network calls at import time.

---

## 4. Target preparation (`target.py`)

Fetch the structure and sequence and cache them. Clean: select chain, strip
waters and ligands unless kept by config, renumber consistently, and record the
mapping from UniProt to PDB numbering. **Numbering mismatches are the commonest
silent bug here — the mapping has its own test.**

Epitope selection, in order of preference: residues the challenge recommends;
residues a cited therapeutic antibody contacts in a complex structure; computed
surface hotspots, reported with the method used. For ortholog objectives, fetch
the ortholog and compute per-residue conservation across the epitope.

---

## 5. Generation and sequence design

### 5.2 Sequences
ProteinMPNN over each backbone: sweep sampling temperature, K sequences per
backbone per temperature, fixed positions where configured. Temperature and seed
recorded per sequence. Runs on CPU.

### 5.3 Co-folding
Fold each designed sequence **in complex with the target**, and also alone.
Multiple seeds per design where affordable, and **report the spread across
seeds, not just the best** — low seed agreement is itself a signal. Cache by
sequence hash; never refold an unchanged sequence.

---

## 6. Evidence: metrics, novelty, objectives, ranking

### 6.1 Metrics
Computed per design and never discarded: interface confidence (ipTM, ipSAE,
mean interface PAE, interface pLDDT); monomer quality; self-consistency; interface
geometry (buried surface area, contact count, hydrogen bonds, salt bridges);
developability liabilities (exposed hydrophobic patch, net charge and pI,
unpaired cysteines, N-glycosylation sequons, deamidation and isomerisation
motifs, low-complexity runs); and seed agreement.

Each metric is a function whose docstring states what it means, its range, and
whether a published threshold exists. **If a threshold is a guess, the docstring
and `docs/TOOLS.md` say so.**

### 6.2 Novelty gate
Sequence search against UniProt and PDB seqres; structure search against the
PDB; within-batch all-pairs diversity with clustering; and an explicit screen
against known binders of the target. Anything resembling a known binder is
**rejected**, not down-weighted. Thresholds live in config and are recorded in
the methods package **with the actual numbers achieved**, not just pass/fail.

### 6.3 Objectives
Each objective module exposes `score(design, context) -> ObjectiveScore` with a
value, an uncertainty and a human-readable reason.

- **pH selectivity** — standard co-folding metrics are **pH-blind**, so this
  must be designed for rather than hoped for: count and position histidines at
  the interface, model the protonated state where tooling allows, and penalise
  interfaces whose predicted contacts are unaffected by protonation. Report the
  predicted effect **with an explicit statement that it is weakly validated**.
- **Ortholog cross-reactivity** — fold against each species and score the
  **minimum** across species, not the mean, so a design that fails one species
  cannot hide behind the other. Report both.

### 6.4 Ranking
**Never rank on a single metric.** Hard filters first (novelty gate, liability
caps, self-consistency floor), then rank by the challenge's stated priority
order, breaking ties with interface confidence, then diversity-aware selection
so the final set is not many variants of one backbone.

Emit a one-line plain-English reason per design, built from the actual numbers,
and the **full ranking table including rejected designs and why**.

### 6.5 Schemas
Declared in `config.py` and asserted in tests: `backbones`, `designs`, `folds`,
`metrics`, `novelty`, `objectives`, `ranking`.

---

## 7. Packaging and validation

### 7.1 Package contents
Into `challenges/<id>/submission/`: `submission.csv` (ranked best-first, exactly
the required columns, within the cap, unique sequences, lengths in bounds);
`METHODS.md`; `metrics.csv`; `ranking_full.csv` including rejected designs;
`novelty.csv`; predicted structures with a hash manifest; and `provenance.json`
recording tool versions, commit hash, config, seeds, timings, hardware, and
which stages were mocked.

### 7.2 Validation
Runs automatically at the end of packaging and in tests: row count, column names
and order, uniqueness, length bounds, allowed `molecule_class` values, no empty
fields, no non-standard amino acid characters, ranked order preserved, and **a
scan of every text field for anything that reads as an instruction to a model**.
Any hit is a hard failure (§12.1 rule 1).

---

## 8. The retrospective calibration study

**The headline deliverable. It runs on any tier.**

### 8.1 The question
Do the in-silico metrics everyone ranks on actually predict experimental
binding?

### 8.2 Method requirements
Build one row per design with its available metrics and its measured outcome,
then evaluate with discipline:

- AUROC and average precision for each metric, with bootstrap intervals;
- **grouped splits — hold out by target, never by design**, because designs
  against the same target are not independent;
- a **deliberately trivial baseline**, so improvements are measured against
  something;
- **calibration**: reliability curve and expected calibration error for any
  metric used as a probability;
- a per-target breakdown: where does the metric work, where does it fail?

Any difference whose interval straddles zero is reported as **"cannot be
distinguished at this sample size"**. Where many candidate metrics are screened,
the selection itself is re-run inside every bootstrap replicate, because a
maximum over correlated candidates is biased upward.

### 8.3 How it feeds the pipeline
The study's output sets the default thresholds and ranking weights in
`config.py`, with a comment citing it. The ranking is then justified by measured
data rather than folklore.

### 8.5 Prior art

> Overath MD, Rygaard ASH, Jacobsen CP, Brasas V, Morell O, Sormanni P,
> Jenkins TP. *Predicting Experimental Success in De Novo Binder Design: A
> Meta-Analysis of 3,766 Experimentally Characterised Binders.* bioRxiv
> 2025.08.14.670059v2, 17 Sep 2025. doi:10.1101/2025.08.14.670059

They agree with study 1 that ipSAE-family confidence is the best single
predictor, and that pooling features across structure predictors does not
improve performance. They disagree with study 2 on geometry, reporting that
confidence **multiplied by** interface dG/dSASA beats either component alone.

That conflict is not yet a real disagreement: study 2 never tested those two
features, and theirs is a **product** where study 2 added a **linear** term,
which a logistic model cannot use to represent an interaction. **Until §8.6
runs, the geometry conclusion is stated only in its narrow form:** no additive
effect was found for thirteen geometry metrics; a multiplicative interaction
with interface energetics was not tested and is reported in the literature to
work.

### 8.6 Study 3 — the replication

Run to try to break study 2, not to decorate it. Add `interface_dG`,
`interface_dSASA`, `interface_dG_dSASA` and Lawrence-Colman shape
complementarity, then run three tests in order:

1. **Their test, their way.** Average precision, leave-one-target-out, of the
   **product** `confidence x feature` against confidence alone, plus
   precision@k for k = 10, 20, 50.
2. **Their test, our way.** Within-target AUROC, paired target bootstrap, same
   product features. A disagreement between 1 and 2 is about the *metric*, not
   the feature, and must not be reported as a biology result.
3. **Our test, their features.** Add the two features linearly as the other
   thirteen were added, then add the **explicit interaction term**. If the
   linear addition shows nothing while the product helps, that contrast
   explains the entire apparent conflict.

Validation gate before any result is believed: the existing contact validation
must still pass exactly; `interface_dSASA` must correlate above r = 0.9 with the
independently computed buried surface area; the protein-chains-only target
restriction stays; and **energies on a backbone-plus-C-beta model are not
meaningful**, so `interface_dG` is reported as missing, not zero, wherever side
chains are absent, with the subset size stated next to every number.

Decision rules are written down before the output is seen:

| Outcome | What to do |
|---|---|
| Product helps on our data | **Narrow** study 2, do not keep it. Update the report, the README and the compute request. |
| Product does not help, and the corpora differ identifiably | Report a dataset-level disagreement and name the confounds. |
| Cannot reproduce their result at all | **Do not claim they are wrong.** Report "not reproduced on this dataset" with the confounds listed. |
| dG untestable for lack of side chains | Say so, and state which of their two combinations therefore remains untested. |

### 8.7 Corrections owed, and discharged

1. **Withdrawn**: that designs which bound had *fewer* interface contacts and
   *smaller* buried surface. That comparison was pooled across targets and is
   Simpson's paradox; within target every sign reverses. A regression guard now
   computes group differences within target and a test asserts no report text is
   generated from a pooled difference.
2. **Withdrawn**: "no published precedent". Replaced with the accurate claim,
   **no prior analysis of this dataset** (§8.5).
3. **Narrowed**: every statement of the geometry result now names the
   multiplicative test that was not run.

### 8.9 Validating in both directions
The comparison with prior work runs both ways: their features on our data
(§8.6), and **our code on their published data**, checking that we reproduce
their reported average precision. Two independent implementations agreeing on
their corpus means neither is buggy. This needs no GPU and either validates
everything downstream or finds a real defect.

---

## 9. Documentation

`README.md` with the honest status and the headline number; `docs/TOOLS.md` with
every tool, version pin, licence and VRAM need; `docs/LIMITATIONS.md` with every
mock, skip, assumption and untested threshold — that file is a feature, not an
apology; `docs/COMPUTE.md` and `docs/BACKGROUND.md`; and `docs/LESSONS.md`,
appended as work proceeds.

---

## 10. Live challenge handling

If a challenge is open: fetch the page, write `challenges/<id>/BRIEF.md` with
the target, recommended epitope, objectives **in the stated ranking order**,
format rules, cap and deadline, run the pipeline at whatever scale the tier
affords, and package. **Submission is always a human action** (§12.2).

If no challenge is open: do a full dress rehearsal on a past target with
published results, and compare the pipeline's ranking against the published
experimental outcomes.

---

## 12. Hard rules

### 12.1 Competition integrity
1. **Never write anything aimed at the selection model** — not in a design
   name, methods text, file name, structure metadata or README. Embedded
   instructions may be grounds for disqualification. `validate.py` enforces
   this (§7.2); if the check fires, it is a real failure, fixed and logged.
2. **De novo and zero-shot only.** No starting from an existing binder and no
   modifying one. Known binders may be used only to calibrate or evaluate
   filters, and the novelty gate must then reject anything resembling them.
3. **Honest methods.** Report the real workflow including abandoned branches.
   Never claim a model, metric or validation that was not run. **Every number
   in any document must be reproducible from a script in this repository.**
4. **Licences:** only tools that may legally be used here; each recorded in
   `docs/TOOLS.md`.
5. **Respect the caps:** within the per-challenge design limit, unique, within
   length bounds, ranked. Re-verified from the live page each week.

### 12.2 Safety
6. **Only the published competition targets.** This pipeline designs binders
   against the therapeutic and benchmark proteins named on the challenge pages.
   It is not pointed at toxins, virulence factors, or anything intended to
   increase a pathogen's transmissibility, host range or immune escape, and such
   targets are not added as test cases or examples.
7. **These designs would be physically synthesised.** Every output is treated as
   a real molecule: liabilities and anomalies are surfaced in the report rather
   than quietly filtered away.
8. Nothing is ever submitted, no account created and no message sent
   automatically. A human reviews every design before submission.

### 12.3 Repo hygiene
9. Terms listed in `.private/banned_terms.txt` (gitignored) must not appear in
   any tracked file. A pre-commit hook greps case-insensitively for each and
   fails the commit on a hit.
10. **Never backdate commits. Never rewrite history that has been pushed.**
    Rewriting history that has *not* been pushed is permitted and sometimes
    required (§12.4). Check `git remote -v` and `git log @{u}..` before touching
    anything.
11. No large binaries, no lab data, no unpublished images, no credentials in
    git. Check `git status` before each commit.
12. Commit at every phase boundary with a message describing what changed.

### 12.4 Rewriting unpushed history to remove large files
Work artefacts were committed before `work/` was gitignored: 543 MB of search
databases and tool archives against 1.7 MB of tracked source. Untracking them
does not remove them — the blobs remain in every earlier commit.

**This is the one case where rewriting history is correct**, and only while the
history is private. Back up with `git bundle create` first, confirm no remote
holds the commits, strip the path from every commit with `git filter-repo
--path work/ --invert-paths`, then verify: `.git` size, a large-blob scan, the
commit count, a clean worktree, a passing test run, and that the cached data the
studies are reproduced from still exists on disk.

Every commit SHA changes, so any document citing one must be updated. Afterwards
`work/` is never added back; a file under `work/` that is genuinely needed in
git is in the wrong directory.

---

## 13. Testing expectations

- **Fixture-first:** the whole pipeline runs end to end on a committed tiny
  fixture with no GPU and no network, in under two minutes. This is the primary
  regression test and runs in CI.
- Heavy backends are mocked behind their interfaces, returning deterministic
  values.
- `validate.py` has its own suite including deliberately malformed CSVs and a
  text field containing an instruction-like string, which must fail.
- The novelty gate is tested with a planted known binder, which must be
  rejected.
- Numbering-map tests for `target.py` (§4).
- Metric functions are tested against at least one complex whose expected value
  is known or recomputable.
- Grouped-split logic has a test asserting no target appears in both train and
  test.
- GPU and network tests are marked `slow`; `pytest -m "not slow"` must pass
  offline.

---

## 14. Commands

```bash
pip install -e ".[dev]"
pytest -q -m "not slow"
ruff check . && ruff format .

binderkit compute                      # writes docs/COMPUTE.md, sets the tier
binderkit brief --challenge <id>       # fetch and summarise a challenge page
binderkit run --challenge <id> --tier auto --resume
binderkit package --challenge <id>
binderkit validate challenges/<id>/submission/submission.csv
binderkit study retrospective          # §8
binderkit study geometry               # §8 study 2
binderkit study replication            # §8.6
binderkit benchmark                    # measure before requesting compute
```

---

## 15. Cluster scale-up

The available allocation is many 16 GB V100 nodes, not one large-VRAM node: by
§1's table that is **tier B per GPU**, and it should be planned that way.
Volta lacks bfloat16 and FlashAttention-2, so **every backend is smoke-tested on
one GPU before any large allocation is spent**, and recorded in `docs/TOOLS.md`
as *runs / runs degraded / does not run*. Compute nodes have no internet, so
weights are staged first and the package must run fully offline — which §13
already requires.

Work is structured as array jobs: many short independent resumable tasks, never
one long process. Heavy per-design scoring goes on CPU partitions; GPUs are
reserved for backbone generation and co-folding.

The studies pay for themselves here. One predictor instead of ten, at no
measurable loss of ranking quality, is roughly a tenfold increase in designs
folded per GPU-hour. **Spend the saving on breadth, not depth** — more
backbones and more sequences, not more scoring per design.

Before requesting an allocation, measure one design end to end on one GPU and
multiply. An allocation request is built from measured numbers only.

---

## 16. How the studies become a submission

**The studies answer a different question from the one the challenge asks.**
They are about *binding*; a challenge may rank conditional properties above
affinity. A ranking rule validated for binding must not be presented as a
ranking rule for the stated objectives.

1. **Binding is the gate, not the objective.** Use the validated confidence
   signal to reach a pool that plausibly binds at all, then rank *within* that
   pool on the challenge's stated objectives. Never present the confidence score
   as evidence about a conditional property.
2. **Do not take the top N by score.** The reliability curve inverts at the top,
   so ranking purely by score concentrates the budget in the least reliable
   region of the metric. Prefer the band where the **observed** rate is highest.
3. **Spend the slots on diversity** — epitope sub-site, scaffold topology,
   length, mechanism. Expected yield is maximised by decorrelating failure
   modes, not by stacking variations on one pose.
4. **State the expectation out loud** in `README.md`: the measured base rate,
   the modest enrichment, that the objectives compound, and that most de novo
   designs fail.

The edge here is not compute. It is that **the ranking rule is measured rather
than assumed**, including the parts that say a stage everyone runs may not be
worth running. That edge is conditional on §8.6: an unreplicated negative result
standing against a published positive one is a liability, not an asset.
