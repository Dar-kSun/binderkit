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
