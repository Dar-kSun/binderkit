# Lessons

Appended each session. Things worth not relearning.

---

## Session 1 — 2026-10-04/05

### Scientific

- **PDB and UniProt numbering differ by the signal peptide, and nothing warns
  you.** 6ARU numbers mature EGFR, P00533 numbers the precursor: a constant +24.
  Derive the offset by alignment and assert the identity agreement (607/609
  here). A pipeline that assumes zero offset produces a Domain III epitope two
  helical turns off, with hotspots that look completely plausible.
- **Check whether the epitope can even support the objective before designing
  for it.** Mouse cross-reactivity is bounded by epitope conservation, which is
  a property of the epitope choice, not of any binder. Domain III is 90.6%
  identical human-to-mouse, so the objective is reachable. Had it been 40%, no
  amount of design effort would have helped and that would have been worth
  knowing on day one.
- **Standard co-folding metrics are pH-blind.** ipTM and PAE carry no
  protonation state, so a pH-selectivity objective cannot be satisfied by
  folding a design well. Histidine is the only standard residue with a pKa
  between 7.4 and 6.5, which makes the charge change across that window exactly
  computable from sequence — a rare case where the honest proxy is also cheap.

### Statistical

- **Pooled AUROC is biased in both directions and the correction is not
  constant.** Expected pooling to inflate everything; measured DockQ inflated by
  up to +0.084 and several ipSAE metrics *deflated* by up to 0.049. Always
  compute both pooled and within-group, per metric.
- **Averaging an ensemble beat learning weights over it.** The plain mean of ten
  predictors scored 0.761 within-target; a leave-one-target-out logistic
  regression over all 27 columns scored 0.726. With 15 independent groups,
  fitting relative reliability overfits while the average transfers. Try the
  dumb aggregate before the model.
- **An AUROC well below 0.5 is a finding, not a bug.** Cross-predictor spread
  scored 0.370: predictive with its sign flipped.
- **Group by the unit of non-independence in the bootstrap too, not just the
  CV.** Resampling designs would have understated the CI; resampling targets is
  the correct unit, and there is a test asserting the grouped interval is not
  narrower than the design-level one.
- **Write the report generator to compute its own narrative.** The first version
  hardcoded the conclusion "pooled AUROC overstates every metric" into both the
  prose and a figure title. The data said otherwise and the report shipped a
  claim its own tables contradicted until it was caught. Conditional text driven
  by the measured values costs little and cannot go stale.

### Engineering

- **pandas 3.0 infers `StringDtype`, not `object`.** A security scan guarded by
  `if df[col].dtype != object: continue` silently skipped every text column, so
  a planted prompt injection validated clean. It would have worked on pandas
  2.x. Guard on the value, not the dtype — and write the negative test, because
  this one failed open and nothing else would have caught it.
- **Make unavailable backends raise, never silently mock.** `BackendUnavailable`
  for a GPU backend at Tier C means a degraded run cannot masquerade as a real
  one. There is a test for this.
- **An empty result must fail loudly.** With production liability caps, every
  fixture design was rejected and the submission came out empty. Shipping an
  empty CSV that "validates" would have been the worst outcome; validation
  refuses it.
- **Long quoted heredocs break in Git Bash when the body has many
  apostrophes.** Use a file-writing tool for long prose instead of fighting the
  shell.
- **The Hugging Face API caps `siblings` at 100,000 files.** A dataset with more
  files silently loses the tail of the listing, which made `tables/` and `docs/`
  look absent when they were simply truncated out. Request documented paths
  directly rather than trusting an index that can be cut off.
- **`rest.uniprot.org` answers HEAD with 403 but GET normally.** A health check
  that only sends HEAD will wrongly report UniProt as down.

### Process

- **Two independent rules agreeing is worth more than one rule deciding.** The
  tier came out C on VRAM by a 4 MiB margin, which is far too thin to hang a
  night on — but the disk guard forced C independently. Reporting both removed
  the judgement call entirely.
- **Verify the spec against the live source before building to it.** docs/SPEC.md
  listed deadlines starting at Challenge 2; Challenge 1 was open and closing in
  two days. It also asserted a Track 3 cap of 20 that no live page confirms.
- **When a cap is unverifiable, pick the value that satisfies every reading.**
  20 designs is valid under both a 20-cap and a 40-cap; 40 is not.

---

## Session 2 — 2026-10-05

### Statistical

- **Put an interval on the difference, not just on each number.** Session 1
  reported CIs on individual AUROCs but stated its conclusions in terms of gaps
  between them. Two of those gaps turned out to straddle zero. A paired
  bootstrap, scoring both metrics on the same resampled targets, is what makes a
  difference of a few hundredths resolvable at all - the marginal intervals are
  far wider than the paired one because both metrics rise and fall together as
  easy targets enter the sample.
- **Correct for the screen.** Reporting the maximum over 30 correlated metrics
  and attaching that metric's naive CI understates the uncertainty twice over.
  Re-selecting the winner inside each replicate showed the nominal winner holds
  only 64 pct of the time and 8 metrics win at least once.
- **Test "does it add anything" conditionally, not marginally.** Cross-predictor
  disagreement looked informative alone. Given the mean it is correlated with,
  it makes held-out prediction worse. The marginal test answers a question
  nobody is asking.
- **A below-chance AUROC with a wide CI is not a finding.** BBF-14 at 0.398
  looked like "worse than chance" until the interval came out as 0.000-0.764.
  Three binders cannot support a claim.
- **ECE hides non-monotonicity.** 0.078 looked respectable while the reliability
  curve inverted above 0.6. Report the Spearman correlation across bins and the
  top-decile rate, and drop bins with almost nothing in them instead of plotting
  them as points.

### Engineering

- **Validate against a published number whenever one exists.** The release
  publishes its own contact counts, which turned a plausible-looking 95.5 pct
  agreement into a found bug: non-protein chains (Cas9 sgRNA, RBX1 zinc) were
  being counted as interface. Agreement went to 100 pct. Nothing else would have
  caught it.
- **Absence is not zero.** 975 of 1309 design models have no binder side chains,
  so hydrogen bonds there are unobservable. Emitting 0 would have described the
  model, not the design. NaN, and say why.
- **Check what a score is computed over.** MMseqs identity is over the aligned
  region only, so a 9-residue match reads as 77 pct identity. Novelty must use
  identity x coverage, and a positive control is mandatory: a silently-empty
  search looks exactly like a set of perfectly novel designs.
- **Profile before optimising, then optimise the inner loop.** sklearn's
  roc_auc_score spends most of its time validating input. An exact rank-sum
  AUROC is 37x faster and made a bootstrap that ran for 25 minutes finish in
  100 seconds.
- **An exact shortcut beats an approximation.** Only atoms near the partner
  chain change SASA on binding, so restricting the computation to that zone is
  ~50x faster and provably identical (verified to 0.000 A^2).
- **A naive VRAM probe lies on Windows.** Allocate-until-OOM reported 18.25 GiB
  on an 8.19 GiB card because the driver oversubscribes into system RAM.
- **Don't claim what you didn't push hard enough to observe.** I measured no
  slowdown up to 6000 MiB, which is inside physical VRAM, so I did not report a
  spill penalty.

### Process

- **A changelog of retractions is a deliverable.** CHANGES.md is the most
  informative file in the study directory: it shows the method working on its
  own earlier output.
- **Keep the downstream documents consistent with the corrected result.** The
  README still carried session 1's withdrawn claims until it was explicitly
  updated. A report that contradicts its own repo is worse than either version.
- **Label every number measured or cited, and never average the two.** It makes
  a compute request auditable instead of persuasive.

## Session 4

### Git

- **A `.gitignore` miss is cheap to make and expensive to carry.** 543 MB of
  MMseqs2 databases, tool archives and cached structures sat in 17 commits of
  history against 1.7 MB of tracked source. Untracking them (session 2) does
  not remove them; only a history rewrite does, and only while the history is
  private. `git filter-repo --path work/ --invert-paths` took **1.46 seconds**
  and brought `.git` from 544 MB to 1.2 MB. The expensive part was never the
  rewrite -- it was that nobody could do it from a bridged session, so it
  waited two sessions and nearly got pushed.
- **Do the rewrite before the first push, not after.** The window is exactly
  as long as the history stays private. Session 3 correctly ordered the
  remote *after* the rewrite for this reason.
- **A rewrite invalidates every SHA anyone wrote down.** Three citations went
  stale: two in docs/SPEC.md and one in a submission `provenance.json`, which is
  the one that matters -- a provenance record pointing at a commit that no
  longer exists is worse than no provenance record. `filter-repo` leaves
  `.git/filter-repo/commit-map`; use it rather than re-deriving by hand.
