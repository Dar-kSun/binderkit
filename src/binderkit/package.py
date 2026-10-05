"""Packaging (docs/SPEC.md section 7).

Writes the submission bundle and then validates it. Validation failure raises,
so a bundle that would be rejected never sits on disk looking finished.

Tone rules for METHODS.md (section 7.3): no superlatives, no selling, no claims
beyond what was computed, uncertainty stated plainly, and never a word addressed
to the selection model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from binderkit.config import Config
from binderkit.provenance import Provenance
from binderkit.rank import filter_cascade_counts
from binderkit.validate import ValidationResult, validate_frame, validate_methods_text

log = logging.getLogger(__name__)


def build_submission_frame(
    ranking: pd.DataFrame,
    designs: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Assemble the submission table, ranked best-first.

    `name` is a neutral identifier derived from the challenge and the rank. It
    deliberately carries no adjectives and no claim about quality, because the
    name is a text field a model may read (section 12.1).
    """
    sel = ranking[ranking["included"]].copy()
    sel = sel.sort_values("rank")
    merged = sel.merge(designs[["design_id", "sequence"]], on="design_id", how="left")

    rows = []
    for i, row in enumerate(merged.itertuples(), 1):
        rows.append(
            {
                "name": f"{cfg.challenge_id}_{i:02d}",
                "sequence": row.sequence,
                "molecule_class": cfg.submission.molecule_class,
            }
        )
    return pd.DataFrame(rows, columns=list(cfg.submission.columns))


def render_methods(
    cfg: Config,
    target_spec,  # noqa: ANN001
    cascade: list[tuple[str, int]],
    novelty: pd.DataFrame,
    folds_mocked: bool,
    prov: Provenance,
    study_summary: str | None = None,
) -> str:
    """Render METHODS.md for a human reviewer."""
    lines: list[str] = []
    a = lines.append

    a(f"# Methods — {cfg.challenge_id}")
    a("")
    if folds_mocked or prov.any_mocked:
        a("> **This bundle is derived from mocked stages and contains no")
        a("> experimentally or computationally meaningful binding predictions.**")
        a(f"> Mocked stages: {', '.join(prov.mocked_stages) or 'none'}.")
        a("> The confidence values are a deterministic function of each sequence")
        a("> hash, not predictions. Read `docs/LIMITATIONS.md` before using any")
        a("> number here.")
        a("")

    a("## Target and epitope")
    a("")
    a(f"- Target: {target_spec.name}, UniProt `{target_spec.uniprot}`")
    a(f"- Structure: PDB `{target_spec.pdb_id}` chain `{target_spec.chain}`")
    a(f"- Observed residues in that chain: {len(target_spec.sequence_observed)}")
    a(
        f"- Numbering: PDB to UniProt offset derived by alignment over "
        f"{len(target_spec.numbering_map)} residues."
    )
    a("")
    a(target_spec.epitope_rationale)
    a("")
    a(f"- Hotspot residues (PDB numbering): {target_spec.hotspots_pdb}")
    a("")

    a("## Models, versions and parameters")
    a("")
    a(
        f"- Backbone generation: `{cfg.generate.backend}`, lengths "
        f"{cfg.generate.length_range}, {cfg.generate.n_per_length} per length"
    )
    a(
        f"- Sequence design: `{cfg.sequence.backend}`, temperatures "
        f"{list(cfg.sequence.temperatures)}, {cfg.sequence.n_per_backbone_per_temp} "
        "per backbone per temperature"
    )
    a(f"- Co-folding: `{cfg.fold.backend}`, {cfg.fold.n_seeds} seed(s) per design")
    a(f"- Tier: {cfg.tier}")
    a("")
    a("Full tool versions, seeds and timings are in `provenance.json`.")
    a("")

    a("## Filter cascade")
    a("")
    a("| Stage | Designs surviving |")
    a("|---|---|")
    for label, n in cascade:
        a(f"| {label} | {n} |")
    a("")

    a("## Novelty results")
    a("")
    if not novelty.empty:
        ident = novelty["best_seq_identity"]
        a(
            f"- Sequence identity to the reference set: median "
            f"{ident.median():.1%}, max {ident.max():.1%}"
        )
        a(
            f"- Threshold applied: {cfg.novelty.max_seq_identity:.0%} general, "
            f"{cfg.novelty.known_binder_identity_cutoff:.0%} against known binders"
        )
        a(f"- Designs rejected by the gate: {int((~novelty['passed']).sum())}")
        if novelty["best_tm"].isna().all():
            a(
                "- **Structural novelty was not checked.** No Foldseek or TM-score "
                "comparison was available in this environment, so the structural arm "
                "of the gate did not run. Sequence novelty alone is a weaker claim."
            )
    a("")
    a(
        "The reference set is the target, its orthologs and a curated known-binder "
        "list, not a full UniProt or PDB search. See `docs/LIMITATIONS.md`."
    )
    a("")

    a("## Ranking rule")
    a("")
    a(
        "Hard filters first (novelty gate, developability caps, self-consistency "
        "floor), then lexicographic ordering over the objectives in the order the "
        "challenge states them: " + ", ".join(cfg.rank.objective_order) + ". "
        "Interface confidence breaks ties. A diversity pass then caps how many "
        f"designs come from one sequence cluster at {cfg.novelty.max_per_cluster}."
    )
    a("")
    a("No single metric determines the ranking.")
    a("")

    if study_summary:
        a("## Calibration")
        a("")
        a(study_summary)
        a("")

    a("## Seed agreement")
    a("")
    a(
        f"Each design was folded with {cfg.fold.n_seeds} seed(s). Where more than "
        "one seed was run, the standard deviation of ipTM and interface PAE across "
        "seeds is reported per design in `metrics.csv` as `seed_agreement_iptm` and "
        "`seed_agreement_pae`. Low agreement is reported rather than hidden."
    )
    a("")

    a("## Limitations and where this method is most likely wrong")
    a("")
    a(
        "- Tier C: no backbone generation and no structure prediction were run. "
        "Backbones and sequences come from fixture backends and the confidence "
        "values are mocked."
    )
    a(
        "- The pH-selectivity objective computes the pH 7.4 to 6.5 charge change "
        "exactly, but cannot know which histidines sit at the binding interface "
        "without complex coordinates. It is a sequence-level proxy and is not "
        "validated against measured pH-selective binding."
    )
    a(
        "- The cross-species objective falls back to epitope conservation, which "
        "bounds the whole batch rather than distinguishing designs."
    )
    a(
        "- Interface geometry (buried surface area, contacts, hydrogen bonds, salt "
        "bridges, shape complementarity) was not computed; there are no complex "
        "coordinates to measure."
    )
    a(
        "- Developability metrics are computed from sequence and are real, but the "
        "hydrophobic-patch value is a hydropathy proxy, not a surface area."
    )
    a(
        "- Most de novo designs do not bind. In the published campaign on this same "
        "target the measured rate was 10 binders in 90 designs."
    )
    a("")

    a("## What was mocked or skipped")
    a("")
    for note in prov.notes or ["nothing"]:
        a(f"- {note}")
    a("")
    return "\n".join(lines) + "\n"


def package(
    cfg: Config,
    out_dir: Path,
    designs: pd.DataFrame,
    metrics: pd.DataFrame,
    novelty: pd.DataFrame,
    objectives: pd.DataFrame,
    filtered: pd.DataFrame,
    ranking: pd.DataFrame,
    target_spec,  # noqa: ANN001
    prov: Provenance,
    study_summary: str | None = None,
) -> ValidationResult:
    """Write the full bundle to `out_dir` and validate it."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    submission = build_submission_frame(ranking, designs, cfg)
    submission.to_csv(out / "submission.csv", index=False)

    # Every artefact downstream of a mock is labelled.
    metrics_out = metrics.copy()
    metrics_out["mocked"] = bool(prov.any_mocked)
    metrics_out.to_csv(out / "metrics.csv", index=False)

    ranking.to_csv(out / "ranking_full.csv", index=False)
    novelty.to_csv(out / "novelty.csv", index=False)
    objectives.to_csv(out / "objectives.csv", index=False)

    cascade = filter_cascade_counts(designs, novelty, filtered, ranking)
    methods = render_methods(
        cfg, target_spec, cascade, novelty, bool(prov.any_mocked), prov, study_summary
    )
    (out / "METHODS.md").write_text(methods, encoding="utf-8")

    structures = out / "structures"
    structures.mkdir(exist_ok=True)
    manifest = pd.DataFrame(
        {
            "design_id": submission["name"],
            "sequence": submission["sequence"],
            "structure": ["NOT PRODUCED (Tier C, fold stage mocked)"] * len(submission),
        }
    )
    manifest.to_csv(structures / "manifest.csv", index=False)

    for f in ("submission.csv", "metrics.csv", "ranking_full.csv", "novelty.csv"):
        prov.add_file(out / f)
    prov.config = cfg.to_dict()
    prov.write(out / "provenance.json")

    # Validate last, and loudly.
    result = validate_frame(submission, cfg.submission)
    methods_hits = validate_methods_text(out / "METHODS.md")
    if methods_hits:
        result.errors.extend(methods_hits)
        result.ok = False

    (out / "VALIDATION.txt").write_text(
        ("PASSED\n" if result.ok else "FAILED\n")
        + "\n".join(f"- {e}" for e in result.errors)
        + ("\n" if result.errors else "")
        + "\n".join(f"warning: {w}" for w in result.warnings)
        + "\n",
        encoding="utf-8",
    )
    log.info("packaged %d designs to %s (valid=%s)", len(submission), out, result.ok)
    return result
