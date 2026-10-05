"""Cross-species binding: bind human and mouse EGFR equally (objective 2).

Scored as the **minimum** across species, never the mean, so a design that
fails one species cannot hide behind the other (docs/SPEC.md section 6.3). Both
the minimum and the per-species values are reported.

What is real here and what is not:

* **Epitope conservation between human and mouse is real and decisive.** It is
  computed by aligning the two orthologs and measuring identity across the
  epitope residues specifically. If the chosen epitope is poorly conserved,
  equal binding to both species is close to impossible no matter what the
  binder looks like, so this is a property of the *epitope choice* and it
  bounds every design in the batch.
* Per-design per-species fold scores are **mocked** in Tier C. Where they are
  absent the objective falls back to the conservation bound alone and says so.
"""

from __future__ import annotations

import logging
from typing import Any

from binderkit.objectives.base import ObjectiveScore

log = logging.getLogger(__name__)

name = "ortholog"


def epitope_conservation(
    human_seq: str,
    ortholog_seq: str,
    epitope_uniprot: list[int],
) -> tuple[float, dict[int, bool]]:
    """Identity across the epitope residues between human and an ortholog.

    Aligns the two full sequences, then reports identity restricted to the
    epitope positions, which is the number that matters: global identity can
    look healthy while the epitope itself diverges.

    Returns (fraction identical across the epitope, per-residue map).
    """
    from Bio import Align  # local import keeps module import cheap

    al = Align.PairwiseAligner()
    al.mode = "global"
    al.open_gap_score = -11
    al.extend_gap_score = -1
    al.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")

    try:
        alignment = al.align(human_seq, ortholog_seq)[0]
    except (ValueError, IndexError):
        return float("nan"), {}

    # Map human position -> ortholog residue via the aligned blocks.
    human_to_orth: dict[int, str] = {}
    for (h0, h1), (o0, o1) in zip(*alignment.aligned, strict=True):
        for hi, oi in zip(range(h0, h1), range(o0, o1), strict=True):
            human_to_orth[hi + 1] = ortholog_seq[oi]

    per_residue: dict[int, bool] = {}
    for pos in epitope_uniprot:
        if pos < 1 or pos > len(human_seq):
            continue
        orth = human_to_orth.get(pos)
        per_residue[pos] = bool(orth is not None and orth == human_seq[pos - 1])

    if not per_residue:
        return float("nan"), {}
    frac = sum(per_residue.values()) / len(per_residue)
    return round(frac, 4), per_residue


class Ortholog:
    """Score a design for equal binding across species."""

    name = name

    def score(self, design: Any, context: dict[str, Any]) -> ObjectiveScore:  # noqa: ARG002
        # `design` is required by the Objective protocol. This objective scores a
        # property of the epitope and the species pair, which is why it reads only
        # `context`: the value bounds the whole batch rather than varying per design.
        per_species: dict[str, float] = context.get("per_species_scores", {})
        conservation: float | None = context.get("epitope_conservation")
        species = context.get("species", [])

        if per_species:
            worst_species = min(per_species, key=lambda s: per_species[s])
            worst = per_species[worst_species]
            spread = max(per_species.values()) - worst
            value = round(max(0.0, min(1.0, worst)), 4)
            return ObjectiveScore(
                value=value,
                uncertainty=round(0.2 + 0.5 * spread, 4),
                reason=(
                    "scored as the minimum across species, not the mean: "
                    + ", ".join(f"{s}={per_species[s]:.3f}" for s in sorted(per_species))
                    + f". Worst species is {worst_species} at {worst:.3f}; spread "
                    f"across species is {spread:.3f}. Per-species fold scores are "
                    "MOCKED in Tier C and are not predictions."
                ),
            )

        if conservation is not None and conservation == conservation:
            # No per-design species scores; fall back to the epitope-level bound.
            value = round(max(0.0, min(1.0, conservation)), 4)
            return ObjectiveScore(
                value=value,
                uncertainty=0.35,
                reason=(
                    f"no per-species fold scores available, so this falls back to the "
                    f"epitope conservation bound: {conservation:.1%} of the chosen "
                    f"epitope residues are identical between {' and '.join(species) or 'the species'}. "
                    "This bounds the whole batch rather than distinguishing designs, "
                    "because it is a property of the epitope choice, not of the binder."
                ),
            )

        return ObjectiveScore(
            value=0.0,
            uncertainty=1.0,
            reason=(
                "NOT SCORED: neither per-species fold scores nor an epitope "
                "conservation value was available."
            ),
        )


OBJECTIVE = Ortholog()
