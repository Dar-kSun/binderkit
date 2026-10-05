"""pH-selective binding: bind at pH 6.5, not at pH 7.4 (Challenge 1 objective 1).

Standard co-folding metrics are **pH-blind**: ipTM and PAE carry no notion of
protonation state, so a design cannot acquire pH selectivity by being folded
well. It has to be designed in, which is what this module scores.

Mechanism. Histidine is the only standard residue whose side-chain pKa (~6.0)
sits inside the window between pH 7.4 and pH 6.5, so essentially all of the
protonation change across that window is histidine. A binder whose interface
gains positive charge as the pH falls is the canonical route to acid-switched
binding, and a binder with no histidine cannot be pH-switchable by this
mechanism at all.

What is real here and what is not:

* The charge calculation is **real**. `delta_charge` is computed from
  Henderson-Hasselbalch over the actual sequence at both pH values, and it is
  the quantity the mechanism turns on.
* Which histidines sit *at the interface* is **not known** in Tier C, because
  there are no complex coordinates. The score therefore rewards histidine
  content and clustering as a proxy for interface histidine, which is weaker.
* Nothing here has been validated against measured pH-selective binding. The
  reason string says so on every design, and so does METHODS.md.
"""

from __future__ import annotations

from typing import Any

from binderkit.metrics import net_charge
from binderkit.objectives.base import ObjectiveScore

name = "ph_selectivity"

#: The two pH values the challenge names.
PH_BINDING = 6.5
PH_NON_BINDING = 7.4

#: Histidine fraction that saturates the score. Above this, more histidine is
#: not obviously better and starts to look like a composition artefact.
#: GUESS: not fitted to any measured pH-selectivity data.
HIS_FRACTION_SATURATION = 0.08

#: Charge gain across the window that counts as a full signal, in proton units.
#: GUESS: chosen so that roughly two fully-responsive histidines saturate it.
DELTA_CHARGE_SATURATION = 1.0


def delta_charge(sequence: str) -> float:
    """Charge gained on going from pH 7.4 to pH 6.5.

    Range: 0.0 up to about +0.5 per histidine. Positive means the design gains
    positive charge as the environment acidifies, which is the direction a
    pH-switched binder needs. Computed, not estimated.
    """
    return round(net_charge(sequence, PH_BINDING) - net_charge(sequence, PH_NON_BINDING), 4)


def his_positions(sequence: str) -> list[int]:
    """1-based positions of every histidine."""
    return [i + 1 for i, a in enumerate(sequence) if a == "H"]


def his_clustering(sequence: str, window: int = 12) -> int:
    """Largest number of histidines inside any `window`-residue stretch.

    Range: 0 upward. A cluster matters because a single buried histidine rarely
    flips an interface, whereas several histidines in one patch can. This is a
    sequence-level proxy for interface clustering, since the interface itself is
    unknown in Tier C.
    """
    pos = his_positions(sequence)
    if not pos:
        return 0
    best = 0
    for p in pos:
        best = max(best, sum(1 for q in pos if p <= q < p + window))
    return best


class PhSelectivity:
    """Score a design for acid-switched binding."""

    name = name

    def score(self, design: Any, context: dict[str, Any]) -> ObjectiveScore:  # noqa: ARG002
        seq = design.sequence if hasattr(design, "sequence") else design["sequence"]
        n = max(len(seq), 1)
        n_his = seq.count("H")
        frac = n_his / n
        dq = delta_charge(seq)
        cluster = his_clustering(seq)

        if n_his == 0:
            return ObjectiveScore(
                value=0.0,
                uncertainty=0.05,
                reason=(
                    "no histidine, so the design cannot be pH-switchable by the "
                    "histidine-protonation mechanism; charge change from pH 7.4 to "
                    f"6.5 is {dq:+.3f}. Weakly validated: no measured pH-selectivity "
                    "data was used."
                ),
            )

        charge_term = min(1.0, max(0.0, dq / DELTA_CHARGE_SATURATION))
        content_term = min(1.0, frac / HIS_FRACTION_SATURATION)
        cluster_term = min(1.0, cluster / 3.0)
        value = round(0.5 * charge_term + 0.3 * content_term + 0.2 * cluster_term, 4)

        # Uncertainty is deliberately large: the interface is unknown, so the
        # clustering and content terms are proxies.
        uncertainty = round(0.25 + 0.1 * (1.0 - charge_term), 4)

        return ObjectiveScore(
            value=value,
            uncertainty=uncertainty,
            reason=(
                f"{n_his} histidine(s) ({frac:.1%} of sequence), largest cluster "
                f"{cluster} within 12 residues, charge change pH 7.4 -> 6.5 of "
                f"{dq:+.3f} proton units. The charge change is computed exactly; "
                "whether those histidines sit at the binding interface is NOT known "
                "without complex coordinates. Weakly validated: no measured "
                "pH-selectivity data informed this score."
            ),
        )


OBJECTIVE = PhSelectivity()
