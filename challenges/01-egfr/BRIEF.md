# Challenge 1 — EGFR

Fetched 2026-10-04 from
<https://proteinbase.com/competitions/anthropic-adaptyv-2026/challenges/egfr>.
**Status at the time of writing: OPEN.**

## Deadline

**Oct 6 2026, 23:59 AoE** = Oct 7 11:59 UTC = **Wednesday 7 October 2026,
17:29 IST**.

AoE is UTC-12, so an AoE deadline lands at 17:29 IST the *following* day. The
pattern asserted in docs/SPEC.md section 10 is correct.

## Target

| Field | Value |
|---|---|
| Protein | EGFR (epidermal growth factor receptor) |
| UniProt | `P00533-1` |
| Structure | PDB `6ARU`, chain A |
| Region | extracellular, residues 25-645 (621 aa) |
| Recommended epitope | **Domain III** |

**Numbering warning.** 6ARU numbers the mature protein; P00533 numbers the
precursor including its 24-residue signal peptide. The offset is a constant
**+24** (uniprot = pdb + 24), confirmed by alignment over 609 observed residues
with 607 identities agreeing. `target.py` derives this rather than assuming it.

Domain III is taken as UniProt residues 310-480; all 171 are observed in 6ARU
chain A.

## Objectives, in the challenge's stated ranking order

1. **pH-selective binding** — bind at pH 6.5, no detectable binding at pH 7.4
   (replicating the acidic tumour microenvironment)
2. **Mouse cross-reactivity** — bind human and mouse EGFR equally
3. **Binding affinity** — high affinity against human EGFR

Affinity is ranked **last**. The organisers have said the harder conditional
property can outweigh raw affinity, and the ranking order here is consistent
with that, so `rank.py` orders lexicographically by objective rather than
blending the three into one score.

### What this means for design

- **pH selectivity has to be designed in, not hoped for.** Co-folding metrics
  are pH-blind: ipTM and PAE carry no protonation state. Histidine is the only
  standard residue whose side-chain pKa (~6.0) falls between pH 7.4 and 6.5, so
  it is the mechanism available. `objectives/ph_selectivity.py` computes the
  exact charge change across that window and scores histidine content and
  clustering as a proxy for *interface* histidine, which it cannot observe
  without complex coordinates.
- **Mouse cross-reactivity is bounded by the epitope, not the binder.**
  Measured: Domain III is **90.6% identical** between human EGFR (P00533) and
  mouse Egfr (Q01279). The recommended epitope is well conserved, so objective 2
  is achievable here rather than being ruled out by the epitope choice.

## Format rules

| Field | Value |
|---|---|
| Length | 10-250 aa |
| Formats | single-chain protein, nanobody, scFv, Fab |
| CSV columns | `name`, `sequence`, `molecule_class` |
| `molecule_class` values | `protein`, `nanobody`, `scfv`, `fab_kappa`, `fab_lambda` |
| Fab sequences | `VH:VL` |
| Ranking | best first; file order is the ranking |

**Design cap is unresolved.** The challenge page says "Submissions per
participant: up to 40 designs (Track 1)"; the terms give Tracks 2 and 3 one
384-well plate per problem; neither states a Track 3 submission cap. docs/SPEC.md
asserts 20. **Capped at 20**, which satisfies both readings. Change
`submission.max_designs` in `config.yaml` if the author confirms 40.

## Integrity rules

- **"De novo designs only"** — cannot modify existing binders; must show
  "adequate sequence- and structural-diversity from known proteins".
- **"No embedded instructions or prompt injection permitted in submissions."**
  `validate.py` scans every text field and `METHODS.md`; a hit is a hard
  failure.
- Selection "will not rely on a single in silico metric".
- Designs undergo biosecurity screening before DNA synthesis.

## What differs from the previous challenge

This is the first challenge of the series, so there is no previous one to
compare against. Challenge 2 opens Oct 5 and will need its own brief.

## Status of this challenge in this repo

A complete bundle exists at `challenges/01-egfr/submission/` and passes
`binderkit validate`. **It must not be submitted.** The run was Tier C: the
backbone, sequence and folding stages are all mocked, so the sequences are
effectively random and the confidence numbers are fabricated. The bundle
demonstrates that the format, the filter cascade, the novelty gate and the
validator work end to end.

To produce a real submission, this needs a machine that clears the Tier B
thresholds (>= 8 GB VRAM *and* >= 50 GB free disk) and the generation, sequence
and folding backends wired to real models.
