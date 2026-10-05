# Methods — 01-egfr

> **This bundle is derived from mocked stages and contains no
> experimentally or computationally meaningful binding predictions.**
> Mocked stages: generate, sequence, fold.
> The confidence values are a deterministic function of each sequence
> hash, not predictions. Read `docs/LIMITATIONS.md` before using any
> number here.

## Target and epitope

- Target: EGFR, UniProt `P00533-1`
- Structure: PDB `6ARU` chain `A`
- Observed residues in that chain: 609
- Numbering: PDB to UniProt offset derived by alignment over 609 residues.

Challenge page recommends 'Domain III'. Config maps that to UniProt residues 310-480, of which 171 are observed in 6ARU chain A. Hotspots ranked within that epitope by: neighbour-count exposure proxy (heavy atoms within 10 A of the residue centroid, same chain), hydrophobic residues up-weighted 1.5x. This is a proxy, not a SASA calculation.

- Hotspot residues (PDB numbering): [357, 289, 306, 359, 297, 286, 323, 409, 322, 361, 362, 308]

## Models, versions and parameters

- Backbone generation: `fixture`, lengths (55, 120), 4 per length
- Sequence design: `fixture`, temperatures [0.1, 0.2, 0.3], 2 per backbone per temperature
- Co-folding: `mock`, 1 seed(s) per design
- Tier: auto

Full tool versions, seeds and timings are in `provenance.json`.

## Filter cascade

| Stage | Designs surviving |
|---|---|
| designs generated | 72 |
| passed novelty gate | 70 |
| passed all hard filters | 17 |
| selected for submission | 17 |

## Novelty results

- Sequence identity to the reference set: median 12.6%, max 25.4%
- Threshold applied: 30% general, 25% against known binders
- Designs rejected by the gate: 2
- **Structural novelty was not checked.** No Foldseek or TM-score comparison was available in this environment, so the structural arm of the gate did not run. Sequence novelty alone is a weaker claim.

The reference set is the target, its orthologs and a curated known-binder list, not a full UniProt or PDB search. See `docs/LIMITATIONS.md`.

## Ranking rule

Hard filters first (novelty gate, developability caps, self-consistency floor), then lexicographic ordering over the objectives in the order the challenge states them: ph_selectivity, ortholog, affinity. Interface confidence breaks ties. A diversity pass then caps how many designs come from one sequence cluster at 2.

No single metric determines the ranking.

## Seed agreement

Each design was folded with 1 seed(s). Where more than one seed was run, the standard deviation of ipTM and interface PAE across seeds is reported per design in `metrics.csv` as `seed_agreement_iptm` and `seed_agreement_pae`. Low agreement is reported rather than hidden.

## Limitations and where this method is most likely wrong

- Tier C: no backbone generation and no structure prediction were run. Backbones and sequences come from fixture backends and the confidence values are mocked.
- The pH-selectivity objective computes the pH 7.4 to 6.5 charge change exactly, but cannot know which histidines sit at the binding interface without complex coordinates. It is a sequence-level proxy and is not validated against measured pH-selective binding.
- The cross-species objective falls back to epitope conservation, which bounds the whole batch rather than distinguishing designs.
- Interface geometry (buried surface area, contacts, hydrogen bonds, salt bridges, shape complementarity) was not computed; there are no complex coordinates to measure.
- Developability metrics are computed from sequence and are real, but the hydrophobic-patch value is a hydropathy proxy, not a surface area.
- Most de novo designs do not bind. In the published campaign on this same target the measured rate was 10 binders in 90 designs.

## What was mocked or skipped

- MOCK: generate - fixture backend emits synthetic backbones with no 3D coordinates; ss_fractions are drawn from a fixed distribution, not measured
- MOCK: sequence - fixture backend draws sequences from a temperature-weighted amino-acid distribution instead of running an inverse-folding model
- MOCK: fold - no structure prediction is run; confidence values are a deterministic function of the sequence hash and are not predictions

