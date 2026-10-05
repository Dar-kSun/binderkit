# Tools

Every tool, its version or commit pin, licence, VRAM need, and whether an
optimised kit was used. Tools that are *declared but never invoked* at Tier C
are listed as such rather than implied to have run.

## Actually used in this repo

| Tool | Version | Licence | VRAM | Role |
|---|---|---|---|---|
| Python | 3.13.14 | PSF | — | runtime |
| NumPy | 2.4.1 | BSD-3 | — | numerics |
| pandas | 3.0.6 | BSD-3 | — | all stage tables |
| PyArrow | (pinned by pandas) | Apache-2.0 | — | parquet stage outputs |
| SciPy | 1.17.0 | BSD-3 | — | numerics |
| scikit-learn | 1.9.1 | BSD-3 | — | AUROC/AUPRC, logistic regression, grouped CV |
| matplotlib | 3.10.8 | PSF-based | — | study figures |
| Biopython | 1.8x | Biopython (BSD-like) | — | `Bio.Align.PairwiseAligner` for novelty and conservation |
| PyYAML | 6.x | MIT | — | config serialisation |
| pytest | 7.4+ | MIT | — | tests |
| ruff | 0.14 | MIT | — | lint and format |

No optimised inference kit was used, because no inference was run.

## Data sources

| Source | Identifier | Licence | Use |
|---|---|---|---|
| RCSB PDB | `6ARU` chain A | public domain | EGFR ectodomain structure |
| UniProt | `P00533-1` (human EGFR) | CC BY 4.0 | canonical sequence, numbering reference |
| UniProt | `Q01279` (mouse Egfr) | CC BY 4.0 | ortholog conservation for objective 2 |
| UniProt | `P01133`, `P01135`, `O14944` | CC BY 4.0 | EGF, TGF-alpha, epiregulin — known EGFR ligands, used **only** to calibrate and evaluate the novelty filter (docs/SPEC.md 12.1 rule 2) |
| Hugging Face | `Anthropic/claude-protein-binder-design` v1.0 (2026-08-18) | **CC BY 4.0** data and docs, MIT scripts | the entire section 8 study |
| Proteinbase | competition overview, EGFR challenge, terms pages | site terms; published results released ODC-BY | rules, target spec, objectives |

The Anthropic release is cited as its licence requires; see
`studies/retrospective/REPORT.md`.

## Declared but NOT invoked at Tier C

These have a concrete interface in `stages.py` and raise
`BackendUnavailableError` if selected. Versions and VRAM figures are the ones
that would apply, recorded now so a Tier A/B run has a starting point. **None of
these ran, so no result in this repo depends on them.**

| Tool | Would-be pin | Licence | VRAM | Interface |
|---|---|---|---|---|
| RFdiffusion | pin a commit; weights ~2 GB | BSD-3 | ~12-24 GB | `generate(backend="rfdiffusion")` |
| BindCraft / FreeBindCraft | pin a commit | MIT | ~8-16 GB | `generate(backend="bindcraft")` |
| ProteinMPNN | pin a commit; weights ~100 MB | MIT | CPU-capable | `design_sequences(backend="proteinmpnn")` |
| SolubleMPNN | ships with ProteinMPNN forks | MIT | CPU-capable | `design_sequences(backend="solublempnn")` |
| Boltz-2 | `boltz` 2.2.1, `boltz2_conf.ckpt` | MIT | ~16-24 GB | `fold(backend="boltz2")` |
| Foldseek | — | GPLv3 | — | structural novelty (not wired) |
| MMseqs2 | — | GPLv3 | — | sequence database search (not wired) |

Licence note: Foldseek and MMseqs2 are **GPLv3**. Linking them into a
distributed pipeline has licence consequences this repo has not analysed.
Invoking them as external binaries is the ordinary way to avoid that question.

### The predictors behind the section 8 numbers

The study's features come from the Anthropic release, which ran ten co-folding
predictors itself. This repo did **not** run them; it consumes their published
scores. Their configurations are documented in the release's
`docs/LOOKUP_TABLES.md`:

`ptxv2` (Protenix v2), `afm3` (AlphaFold-Multimer v3 via ColabFold 1.5.5),
`boltz2` (Boltz-2 2.2.1), `chai1` (Chai-1 0.6.1), `of3` (OpenFold3 0.4.1),
`odde` (OpenDDE 1.0.0), `ef2fast` / `ef2full` (ESMFold2), `rf3` (RoseTTAFold3),
`af3of3` (an Apache-2.0 AlphaFold-3-architecture fork run with **OpenFold3
weights**; no DeepMind AlphaFold 3 parameters).

All ten folded the binder as a single sequence with no MSA, five seeds each.

## Licence compliance

- Everything invoked is BSD, MIT, Apache-2.0, PSF or Biopython-licensed.
- **No commercially-licensed software is used.** No PyRosetta, no Schrodinger,
  no commercial MSA service.
- The one GPL dependency class (Foldseek, MMseqs2) is not installed and not
  invoked.
- `uplifting-biomolecular-modeling` is noted in docs/SPEC.md as not maintained and
  not accepting contributions. It was not used; if it ever is, pin a commit.

## Considered and not used

### PyRosetta

Study 3 (docs/SPEC.md section 8.6) needs `interface_dG` and `interface_dSASA`
from Rosetta's `InterfaceAnalyzerMover`, because those are the features the
result being replicated uses.

| | |
|---|---|
| Licence | PyRosetta Software Non-Commercial License Agreement. **Free for academic, non-profit and government use, with no form, no account and no credentials** — the non-commercial licence now ships with the download. A paid licence through UW CoMotion applies to commercial users only. |
| Distribution | not on PyPI. `pip install pyrosetta --find-links https://west.rosettacommons.org/pyrosetta/quarterly/release`, or the RosettaCommons conda channel |
| Platforms | Linux x86-64 and macOS wheels only. **There is no Windows build**; the documented Windows route is WSL. Verified by listing the index: 36 artifacts, 12 `linux_x86_64` and 24 `macosx`, zero `win_amd64`. |

**An earlier version of this file said the licence required credentials this
environment did not have. That was wrong**, and it was an inference from
`pip install pyrosetta` failing rather than a checked fact — the package is
simply absent from PyPI. The licensing model also changed: non-commercial
users no longer request a licence at all. The real constraint was always the
missing Windows build. Corrected here and in
`studies/retrospective/CHANGEstats.md`.

### Shape complementarity — reimplemented licence-free

| | |
|---|---|
| Definition | Lawrence MC, Colman PM, *J Mol Biol* 234:946-950 (1993) |
| Implementation | `binderkit.geometry.shape_complementarity`, NumPy and SciPy only |
| Parameters | the published ones: probe 1.7 A, 15 dots/A^2, w = 0.5, 1.5 A peripheral trim. **None tuned.** |
| Validation | `studies/interface_geometry/validate_sc.py` against crystallographic interfaces with published bands |
| Known bias | reads **about 0.05-0.08 low** in absolute terms, because the re-entrant surface is not reconstructed. Not comparable with published Sc thresholds; used only for within-dataset ranking, which a constant offset cannot change. |

## Environment notes

- Native Windows 11, Python 3.13, no conda. OpenMM is awkward or unavailable
  on 3.13, which is one reason no structure-manipulation library is in the
  dependency list; PyRosetta's absence is a licence question, above, not a
  packaging one.
- The Bash sandbox does not exist on native Windows, so all commands ran
  unsandboxed.
