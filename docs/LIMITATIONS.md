# Limitations

Every mock, skip, assumption and untested threshold in this repo. This file is
a feature, not an apology: a number whose provenance is not in here should not
be trusted, and a reviewer should be able to find the weak points without
reading the source.

Status as of 2026-10-05, Tier C.

---

## 1. Mocked stages

These produce values that are **not predictions**. Anything downstream carries
`mocked: true` in its metadata and a banner in `METHODS.md`.

| Stage | What it actually does | Where |
|---|---|---|
| `generate` | Emits synthetic backbone records with **no 3D coordinates**. `ss_fractions` are drawn from a fixed distribution, not measured. | `stages.generate`, `backend="fixture"` |
| `sequence` | Draws residues from a temperature-weighted amino-acid distribution. **No inverse-folding model runs.** | `stages.design_sequences`, `backend="fixture"` |
| `fold` | Returns confidence values that are a deterministic function of the sequence hash. **No structure prediction runs.** ipTM, ipSAE, PAE and pLDDT are all fabricated. | `stages.fold`, `backend="mock"` |

**Consequence: this repo cannot currently produce a submittable design.** The
packaged bundle in `challenges/01-egfr/submission/` is a format and plumbing
demonstration. Its sequences are effectively random and its confidence numbers
are meaningless. Do not submit it.

The GPU backends raise `BackendUnavailableError` rather than silently falling
back to a mock, so a Tier A/B run cannot quietly degrade into a fake one.

## 2. Why Tier C

Two independent rules in docs/SPEC.md section 1 both force it, which is why the
tier does not rest on a judgement call:

- **VRAM:** the RTX 4060 Laptop reports 8188 MiB = 7.996 GiB, which is 4 MiB
  below the 8 GB Tier B floor.
- **Disk:** 44.8 GB free is below the 50 GB weight-download guard, which drops
  a tier on its own.

The disk guard is the binding constraint in practice. RFdiffusion, ProteinMPNN
and Boltz-2 weights together will not fit in 44.8 GB while leaving the 10 GB
abort floor the same section requires.

`binderkit run --tier B` overrides this without a code change if disk is freed.

## 3. Skipped entirely

| Thing | Why | Impact |
|---|---|---|
| **Structural novelty (Foldseek / TM-score)** | Foldseek is not installed and there are no design structures to compare. | The novelty gate runs on **sequence only**. `novelty.csv` records `best_tm` as NaN and every reason string says `structural novelty NOT CHECKED`. A design could in principle recapitulate a known fold with a novel sequence and pass. |
| **MMseqs2 / BLAST database search** | Not installed; no full UniProt or PDB index locally. | Novelty is screened against an explicit reference set (the target, its orthologs, three known EGFR ligands), not against all known proteins. This is much weaker than a database search. |
| **Interface geometry** | Requires complex coordinates, which Tier C does not produce. | `bsa`, `n_contacts`, `n_hbonds`, `n_salt_bridges`, `shape_complementarity` are all NaN in `metrics.csv`. They are carried as columns so the schema is stable when a real fold backend is wired in. |
| **ProteinMPNN self-recovery** | No inverse-folding model installed. | `mpnn_recovery` is NaN, not an invented number. |
| **Real SASA for target hotspots** | No SASA implementation available without extra dependencies. | Hotspots are ranked by a neighbour-count exposure proxy (heavy atoms within 10 A of the residue centroid), hydrophobics up-weighted 1.5x. This is a proxy and is labelled as one in `TargetSpec.hotspot_method`. |
| **Per-species fold scores** | Tier C. | The `ortholog` objective falls back to epitope conservation, which bounds the whole batch rather than distinguishing designs. |

## 4. Proxies that are not the thing they are named after

- **`max_hydrophobic_patch`** is the largest mean Kyte-Doolittle hydropathy
  over a 5-residue window. A real exposed hydrophobic patch is a **surface
  area** on a structure. `LiabilityCaps.max_hydrophobic_patch` is expressed in
  A^2, so the two are not comparable and `liability_flags` deliberately does
  **not** apply that cap. Left in place so the units mismatch is visible rather
  than silently papered over.
- **`aggregation_propensity`** is the fraction of V/I/L/F/W/Y/M residues. It is
  not TANGO, AGGRESCAN or any validated predictor, and must never be quoted as
  a probability of aggregation.
- **`radius_of_gyration`** is the Rg = 2.2 * N^0.38 scaling for a compact
  globular chain, not a measurement.
- **`n_unpaired_cys`** is cysteine-count parity. It cannot tell whether an even
  number of cysteines actually forms disulfides; that needs the structure.
- **pH-selectivity histidine clustering** is a sequence-window proxy for
  interface histidine. **Which histidines sit at the binding interface is not
  known** without complex coordinates. The charge-change part of that objective
  is exact; the clustering and content parts are not.

## 5. Thresholds, and which ones are guesses

**Backed by measured data** (the section 8 study, `studies/retrospective/`):

| Constant | Value | Provenance |
|---|---|---|
| `CALIBRATED_IPSAE_THRESHOLD` | 0.635 | Youden-optimal cut on `ipsae_mean` over 1,320 designs with wet-lab outcomes |
| `CALIBRATED_PRECISION` | 0.463 | measured precision at that cut, base rate 0.268 |
| `PREFER_UNWEIGHTED_PREDICTOR_MEAN` | True | the plain mean beat the best single predictor by +0.028; a learned model did 0.034 worse |

**Guesses** (convention or judgement, not fitted to anything):

| Threshold | Value | Status |
|---|---|---|
| `NoveltyConfig.max_seq_identity` | 0.30 | Our choice. The challenge requires "adequate" diversity without giving a number. |
| `NoveltyConfig.known_binder_identity_cutoff` | 0.25 | Our choice, deliberately stricter than the general cut. |
| `NoveltyConfig.max_tm_score` | 0.60 | Conventional "same fold" threshold. **Never exercised**, since structural novelty is not checked. |
| `NoveltyConfig.batch_identity_cutoff` | 0.70 | Our choice. |
| `LiabilityCaps.*` | various | Conventional biologics developability rules of thumb. None fitted to this data. |
| `RankConfig.min_self_consistency_rmsd` | 2.0 A | Conventional. Applied against a **mocked** RMSD in Tier C, so it currently filters noise. |
| `ph_selectivity` saturation constants | 0.08, 1.0 | Guesses. Not fitted to any measured pH-selectivity data, because none was available. |

## 6. Unverified facts carried from docs/SPEC.md

- **Track 3 submission cap.** docs/SPEC.md asserts 20 per challenge. The live
  challenge page says "up to 40 designs (Track 1)" and the terms give Tracks 2
  and 3 one 384-well plate per problem. **No page fetched states a Track 3
  submission cap.** Capped at 20, which satisfies both readings. See
  `docs/BACKGROUND.md` section 2.
- **Exact submission CSV column order.** The `/submit` page returned only
  navigation and footer to the fetcher, so the column order could not be read
  from it. Using `name, sequence, molecule_class` from the challenge page, which
  docs/SPEC.md section 7.1 accepts as the minimum. **Re-verify before submitting.**
- **Historical ~10-15% hit rate.** Attributed to the Adaptyv EGFR post-analysis
  in docs/SPEC.md section 2. That post was not fetched, so the figure is unverified
  here. The 26.8% measured in the Anthropic release *is* verified, by direct
  computation.

## 7. Limits of the section 8 study

Repeated from `studies/retrospective/REPORT.md` because they qualify every
number this repo quotes:

- One campaign, two CROs, 16 targets, miniproteins of 50-120 residues.
- The designs studied were **not** produced by this pipeline, so the thresholds
  transfer only as far as the metric definitions match. This repo's mocked
  `ipsae` is not the release's `ipsae_min_<predictor>`.
- `binder_final` is a model-adjudicated label over two assays that disagree on
  11% of designs. It is not a direct measurement, and ~0.89 is a soft ceiling
  on achievable accuracy.
- Logistic regression with median imputation and no hyperparameter tuning.
- **The headline metric reaches only 0.669 within-target AUROC on EGFR**, below
  its 0.761 cross-target mean, and is below chance on BBF-14 (0.398). It is
  therefore wired in as a soft ranking signal, never a hard filter.
- 15 targets is a small number of independent units. The grouped bootstrap CI
  on the headline metric is correspondingly wide: 0.671 to 0.839.

## 8. Environment and process

- **No sandbox.** Claude Code runs Bash unsandboxed on native Windows, and this
  session ran with `bypassPermissions`. Nothing here was isolated from the
  filesystem.
- **`.private/banned_terms.txt` is empty**, so the pre-commit banned-term hook
  currently passes everything. It must be populated before this repo is pushed
  anywhere public (docs/SPEC.md section 12.3 rule 9).
- **CI has never run.** The GitHub Actions workflow is written but there is no
  remote, so it is untested. The same commands pass locally.
- The committed `.claude/settings.json` grants a bare `Bash` allow rule to
  anyone who clones this repo and accepts the workspace-trust dialog.
- Tests run on Python 3.13 locally; the CI matrix also lists 3.11, which has not
  been exercised.
