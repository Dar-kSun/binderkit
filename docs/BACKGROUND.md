# Background — competition rules and prior art

All pages fetched **2026-10-04** (access date for every citation below).
Where this file and `docs/SPEC.md` disagree, the live page wins and the conflict is
flagged. Unverifiable claims are in `docs/LIMITATIONS.md`, not here.

---

## 1. Competition structure

Anthropic x Adaptyv Protein Design Competition, **28 Sep – 1 Nov 2026**.
$1M+ Claude credits, $1M+ wet-lab validation, five weekly challenges, results
published openly on Proteinbase.
Source: <https://proteinbase.com/competitions/anthropic-adaptyv-2026>

| Track | Who | Testing allocation |
|---|---|---|
| 1 | Labs/companies with protein-design expertise (~30 teams) | "up to approximately 15 designs tested per problem, per team" (reserved) |
| 2 | <=3 researchers with institutional affiliation | no guaranteed allocation |
| 3 | **Open track — "any design method"** | no guaranteed allocation |

Tracks 2 and 3 share **"one 384-well plate (including controls) of designs per
problem"**, selected by "a pre-defined and jointly agreed upon Claude-based
workflow". Applications for Tracks 1/2 closed 24 Sep 2026, so **this repo is a
Track 3 entry** by default.
Source: <https://proteinbase.com/competitions/anthropic-adaptyv-2026/terms>

### Schedule (verify weekly — §12.3 rule 5)

| Challenge | Window | Status on 2026-10-04 |
|---|---|---|
| 1 — EGFR | Sep 28 – **Oct 6** | **OPEN** |
| 2 | Oct 5 – 11 | opens tomorrow |
| 3 | Oct 12 – 18 | |
| 4 | Oct 19 – 25 | |
| 5 | Oct 26 – Nov 1 | |

Experimental validation by **30 Nov 2026**; results published **15 Dec 2026**.

> **Conflict with docs/SPEC.md §14.** §14 lists "Challenge 2 closes Oct 11,
> Challenge 3 Oct 18, Challenge 4 Oct 25, Challenge 5 Nov 1" and does not
> mention Challenge 1. Challenge 1 (EGFR) is in fact open and closes **Oct 6**,
> two days from now. The §14 dates for challenges 2-5 are correct. Acted on the
> live page: the pipeline targets Challenge 1.

---

## 2. Challenge 1 — EGFR (the open challenge)

Source: <https://proteinbase.com/competitions/anthropic-adaptyv-2026/challenges/egfr>

| Field | Value |
|---|---|
| Target | EGFR, **UniProt P00533-1** |
| Structure | **PDB 6ARU, chain A** |
| Domain | extracellular region, residues **25-645** (621 aa) |
| Recommended epitope | **Domain III** |
| Length bounds | **10-250 aa** |
| Formats | single-chain protein, nanobody, scFv, Fab |
| CSV columns | `name`, `sequence`, `molecule_class` |
| `molecule_class` values | `protein`, `nanobody`, `scfv`, `fab_kappa`, `fab_lambda` |
| Fab sequence format | `VH:VL` |
| Deadline | **Oct 6 2026 23:59 AoE** |

### Objectives, in the challenge's stated ranking order

1. **pH-selective binding** — bind at pH 6.5, no detectable binding at pH 7.4
2. **Mouse cross-reactivity** — bind human and mouse EGFR equally
3. **Binding affinity** — high affinity against human EGFR

These map exactly onto the two objective modules docs/SPEC.md §6.3 specifies
(`ph_selectivity.py`, `ortholog.py`), with affinity third. Note that affinity is
ranked **last** of the three.

### Deadline in IST

AoE is UTC-12, so Oct 6 23:59 AoE = Oct 7 11:59 UTC = **Wednesday 7 Oct 2026,
17:29 IST**. About 2.8 days from this session. The pattern in docs/SPEC.md §10
("Sunday 23:59 AoE = Monday 17:29 IST", i.e. next day 17:29 IST) is correct.

### Design cap — unresolved on the live pages

- Challenge page: "Submissions per participant: **Up to 40 designs (Track 1)**"
- Terms: Track 1 gets ~15 *tested*; Tracks 2/3 share one 384-well plate
- **Neither page states a Track 3 submission cap.**

docs/SPEC.md §7.1 and §12.3 rule 5 both assert "<=20 designs per challenge
(Track 3)" and instruct re-verification. Re-verification did not confirm 20.
**Decision: cap at 20.** 20 satisfies a 20-cap and a 40-cap simultaneously,
whereas 40 would violate a 20-cap. The cap is a config value
(`submission.max_designs`), so it is a one-line change if the author confirms 40.
Recorded in `docs/LIMITATIONS.md`.

### Integrity rules

- **"De novo designs only"** — "cannot modify existing binders"; must show
  "adequate sequence- and structural-diversity from known proteins". This is the
  novelty gate of §6.2, and it is a published rule, not an inference.
- **"No embedded instructions or prompt injection permitted in submissions."**
  Enforced by `validate.py` (§7.2). The terms page is silent on this; the
  challenge page is explicit, so the challenge page governs.
- Selection "will not rely on a single in silico metric" — hence §6.4 ban on
  single-metric ranking.
- Designs undergo "biosecurity screening before DNA synthesis".
- **IP:** submitting grants sponsors a non-exclusive, worldwide, royalty-free,
  perpetual, irrevocable publication licence; "publication may affect
  patentability, and Participant is responsible for making any patent filings
  before submitting a design." Recorded in `docs/LIMITATIONS.md` —
  it is a decision only they can make, and it is irreversible after submission.

---

## 3. The Anthropic binder dataset — the basis of §8

`huggingface.co/datasets/Anthropic/claude-protein-binder-design`, v1.0,
last modified 2026-08-18. **CC BY 4.0** data/docs, MIT scripts.

**1,440 de novo miniprotein binders, 50-120 residues, against 16 targets**,
designed by two models acting as autonomous agents (Mythos Preview 900,
Opus 4.8 540) and characterised at **two independent CROs**:

- **Adaptyv Bio** — cell-free expression, design immobilised as monovalent
  ligand, antigen as analyte; SPR single-cycle kinetics (BLI some replicates).
- **Twist Bioscience** — design as human IgG1 Fc fusion (Expi293), captured on
  anti-Fc (bivalent ligand), six-point antigen titration; 1:1 kinetic and
  steady-state fits, plus expression titer and analytical SEC.

Co-folds from **ten predictors**: `ptxv2` (Protenix v2), `afm3`
(AlphaFold-Multimer v3 / ColabFold), `boltz2` (Boltz-2), `chai1` (Chai-1),
`of3` (OpenFold3), `odde` (OpenDDE), `ef2fast` / `ef2full` (ESMFold2),
`rf3` (RoseTTAFold3), `af3of3` (AF3-architecture code on OpenFold3 weights).
Five seeds each; binders always folded single-sequence without MSA.
Source: `data/docs/LOOKUP_TABLES.md`.

### Why this is a genuine labelled dataset

`data/tables/design_summary.csv` — **1,440 rows x 65 columns**, one row per
design, verified locally:

- **Features:** `ipsae_min_<pred>` and `sc_dockq_<pred>` for all 10 predictors
  (20 columns), `binder_length`, `sequence`, `epitope_residues`, `generator`,
  `sequence_design_method`.
- **Labels:** `binder_final` (bool, 1,320 labelled), `kd_nM_final` (354),
  `adaptyv_binding`, `twist_binding`, `vendor_agreement`,
  **`mouse_binding_final`** (154 mouse binders — directly relevant to
  Challenge 1 objective 2).
- **Grouping variable:** `target`, 15 targets with wet-lab data.

Final binder calls were adjudicated by Claude against a fixed rubric
(`docs/WETLAB.md` §6) over both vendors' evidence, because the assays disagree:
of 1,235 designs measured by both, **253 bind at both, 846 at neither, 69
Adaptyv-only, 67 Twist-only — 89.0% agreement**. That 11% vendor disagreement
is a floor on how well *any* in-silico metric can do, and §8 must report it.

### Measured binder rates (computed locally, not quoted)

**Overall: 354/1,320 = 26.8%.** Consistent with the 22-35% range docs/SPEC.md §2
attributes to Anthropic's large-compute campaign, and well above the ~10-15%
historical figure.

| Target | n | binders | rate |
|---|---|---|---|
| TREM2 | 90 | 72 | 80.0% |
| VEGF-A | 90 | 54 | 60.0% |
| IL-7Ra | 90 | 49 | 54.4% |
| PD-L1 | 90 | 39 | 43.3% |
| RBX1 | 90 | 28 | 31.1% |
| BHRF1 | 90 | 23 | 25.6% |
| Latent GDF-8 | 60 | 14 | 23.3% |
| TrkA | 90 | 20 | 22.2% |
| Nipah-G | 90 | 19 | 21.1% |
| Cas9 | 90 | 10 | 11.1% |
| **EGFR** | **90** | **10** | **11.1%** |
| TNFa | 150 | 12 | 8.0% |
| BBF-14 | 90 | 3 | 3.3% |
| 15-PGDH | 30 | 1 | 3.3% |
| MBP | 90 | 0 | 0.0% |

Two consequences that shape everything downstream:

1. **EGFR is among the hardest targets in the set — 11.1%**, a quarter of the
   TREM2 rate. The README must not promise a hit.
2. **Per-target rates span 0%-80%.** A design-level train/test split would let
   any model score well by inferring the target, so §8.3 grouped-by-target
   split is not a nicety — it is the difference between a real result and a
   leak. 14 of 15 targets have >=1 binder, so grouped CV is well posed.

The 120 mature GDF-8 designs are unlabelled (antigen aggregated, assay
inconclusive) and are excluded from the study.

### Prior-probe of signal (sanity check only; §8 does this properly)

Difference in means, binders vs non-binders:

| Column | binders | non-binders | delta |
|---|---|---|---|
| `ipsae_min_afm3` | 0.663 | 0.419 | **+0.243** |
| `sc_dockq_boltz2` | 0.724 | 0.523 | +0.201 |
| `ipsae_min_boltz2` | 0.759 | 0.561 | +0.198 |
| `binder_length` | 82.6 | 82.1 | **+0.535** |

Real separation in the confidence metrics; `binder_length` separates by half a
residue, which makes it the honest near-useless baseline §8.3 requires.

Caveat from the dataset's own `docs/DATA_NOTES.md`: antigen forms differ for
RBX1, Cas9 and GDF-8 and several analytes are oligomeric, so many K_D values are
**apparent**. Read that file before quoting any affinity.

---

## 4. Method background

Pinned/cited for `docs/TOOLS.md`:

- Anthropic, "How Claude is accelerating protein design and analytical
  chemistry", 18 Aug 2026, with technical report — the campaign behind the
  dataset above.
- `github.com/anthropics/uplifting-biomolecular-modeling` — inference kits.
  **Status: not maintained, not accepting contributions.** Pin a commit if used.
- Generators actually used in the campaign, by share of its 1,440 designs:
  PXDesign 387, RFdiffusion3 298, Genie3 213, BoltzGen 158, FreeBindCraft
  (BindCraft) 142, RFdiffusion 120, Proteina-Complexa 104, FoldCraft 14,
  BoltzDesign1 2, Protein Hunter 2.
- Sequence design: **SolubleMPNN 1,243**, Caliby/SolubleCaliby 114, native
  co-design 62, ProteinMPNN 21. Note docs/SPEC.md §5.2 specifies plain ProteinMPNN,
  which was the *least*-used method in the campaign; SolubleMPNN dominated.
  Recorded as an open question rather than a silent substitution.
- Adaptyv: novelty post ("What does de novo actually mean?"), EGFR competition
  post-analysis, BenchBB — **not yet fetched**, see Limitations.
- BindCraft, RFdiffusion, ProteinMPNN, Boltz-2 primary papers — to cite in
  `docs/TOOLS.md` for any backend actually invoked. Under Tier C none are
  invoked, so they are cited as interface contracts only.

### The field's prior art on the §8 question — missed until session 3

§2 fetches the *competition's* prior art. Nobody had fetched the **field's**,
and the result was that two shipped reports claimed novelty that one literature
search disproves. The paper that asks a version of §8's question:

> Overath MD, Rygaard ASH, Jacobsen CP, Brasas V, Morell O, Sormanni P,
> Jenkins TP. *Predicting Experimental Success in De Novo Binder Design: A
> Meta-Analysis of 3,766 Experimentally Characterised Binders.* bioRxiv
> 2025.08.14.670059v2, 17 Sep 2025. doi:10.1101/2025.08.14.670059

| | Overath et al. | This repo |
|---|---|---|
| Designs | 3,766 | 1,320 (study 1), 1,189 (study 2) |
| Targets | 15 | 15 (14 informative) |
| Binder rate | 11.6% (436) | 26.8% (354) |
| Source | meta-analysis over many campaigns, non-standardised binding definitions | one campaign, one adjudication rubric, two CROs |
| Predictors | AF2 (initial-guess + ColabFold), AF3, Boltz-1 | the release's 10, as published |
| Coordinates | complexes **re-predicted** by them | the release's **design models** |
| Headline metric | average precision (chosen for class imbalance) | within-target AUROC |
| Split | leave-one-group-out by target, plus precision@k | leave-one-target-out, plus paired target bootstrap |
| Best single feature | AF3 `ipSAE_min` | `ipsae_mean` / `ipsae_min_ptxv2` |

**They agree with us on the two conclusions that matter**, which is support,
not competition: ipSAE-family confidence is the best single in-silico
predictor, and pooling features across structure predictors does not improve
performance ("did not improve median AP"). The second is this repo's
conclusions 2 and 3, replicated externally on three times the data — and ours
is the stronger form, since we tested ten predictors against their three to
four.

**They disagree with study 2 on geometry**, reporting that
`ipSAE_min × interface_dG/dSASA` and `LIS × shape_complementarity` each beat
their components. That conflict may be an artefact of study 2's scope: neither
feature was tested here, and theirs is a product where study 2 added a linear
term. See docs/SPEC.md §8.5–§8.6 and `studies/retrospective/CHANGEstats.md` §10.

Accessed: not re-fetched this session; cited from docs/SPEC.md §8.5, which
records the full reference. Verify the DOI before quoting the numbers
elsewhere.

## 5. Published hit-rate expectations

- Historical de novo competition hit rates ~10-15% (docs/SPEC.md §2; the Adaptyv
  post-analysis is the primary source and is not yet fetched — treat as
  unverified).
- Anthropic's campaign: **26.8% measured here** over 1,320 labelled designs,
  within the 22-35% claim.
- **Most designs fail.** At EGFR specifically the campaign's own rate was 11.1%
  with far more compute than Tier C has. The README says so plainly.
