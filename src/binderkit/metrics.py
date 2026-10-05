"""Per-design metrics (docs/SPEC.md section 6.1).

Every metric function states what it means, its range, and whether a published
threshold exists. Where a threshold is a guess, the docstring says so in those
words, and docs/TOOLS.md repeats it.

Nothing here is discarded: the pipeline carries every metric through to the
packaged `metrics.csv`, including ones that did not influence the ranking.

The developability metrics in this module are computed from sequence alone and
are therefore **real**, not mocked, even in Tier C. The interface and monomer
confidence metrics come from the fold stage and are mocked in Tier C; they
arrive already marked.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

import pandas as pd

log = logging.getLogger(__name__)

#: Kyte-Doolittle hydropathy. Kyte J, Doolittle RF, J Mol Biol 157:105-132 (1982).
KYTE_DOOLITTLE = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

#: Side-chain pKa values used for the charge/pI calculation, plus termini.
PKA_SIDE = {"D": 3.65, "E": 4.25, "C": 8.18, "Y": 10.07, "H": 6.00, "K": 10.53, "R": 12.48}
PKA_N_TERM = 9.69
PKA_C_TERM = 2.34
POSITIVE = set("KRH")
NEGATIVE = set("DECY")

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


# --------------------------------------------------------------------------
# Developability liabilities - computed from sequence, genuinely measured
# --------------------------------------------------------------------------


def net_charge(sequence: str, ph: float = 7.4) -> float:
    """Net charge at `ph` from Henderson-Hasselbalch over ionisable groups.

    Range: roughly -N to +N for a sequence of length N; in practice -20..+20.
    No published threshold. The cap in `LiabilityCaps.net_charge_range` is a
    conventional rule of thumb, not a fitted value.
    """
    charge = 1.0 / (1.0 + 10 ** (ph - PKA_N_TERM))
    charge -= 1.0 / (1.0 + 10 ** (PKA_C_TERM - ph))
    for aa in sequence:
        pka = PKA_SIDE.get(aa)
        if pka is None:
            continue
        if aa in POSITIVE:
            charge += 1.0 / (1.0 + 10 ** (ph - pka))
        else:
            charge -= 1.0 / (1.0 + 10 ** (pka - ph))
    return round(charge, 3)


def isoelectric_point(sequence: str) -> float:
    """pI by bisection on `net_charge`.

    Range: ~3.0-12.0. No published threshold; a pI near the formulation pH is
    an aggregation risk, which is a qualitative heuristic, not a number.
    """
    lo, hi = 2.0, 13.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if net_charge(sequence, mid) > 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 2)


def count_cys(sequence: str) -> int:
    """Number of cysteines. Range: 0-N. No threshold."""
    return sequence.count("C")


def count_unpaired_cys(sequence: str) -> int:
    """Cysteines that cannot all be paired, i.e. 1 if the count is odd else 0.

    Range: {0, 1}. This is a sequence-level proxy: it cannot tell whether an
    even number of cysteines actually forms disulfides, which needs the
    structure. A free thiol is a real manufacturability liability, so the
    default cap is 0 (`LiabilityCaps.max_unpaired_cys`) - a conventional
    biologics rule, not a fitted threshold.
    """
    return count_cys(sequence) % 2


def count_glyc_sequons(sequence: str) -> int:
    """N-linked glycosylation sequons, N-X-S/T with X not proline.

    Range: 0 upward. This is the standard sequon definition and is a published
    motif, not a guess. The cap of 0 is a choice: these designs are expressed
    in E. coli cell-free (Adaptyv) and Expi293 (Twist), and only the latter
    glycosylates, so a sequon is a cross-vendor inconsistency risk rather than
    an outright failure.
    """
    return len(re.findall(r"N[^P][ST]", sequence))


def count_deamidation_motifs(sequence: str) -> int:
    """NG and NS motifs, the fast asparagine deamidation hotspots.

    Range: 0 upward. The motif identity is published; no threshold count is.
    """
    return len(re.findall(r"N[GS]", sequence))


def count_isomerisation_motifs(sequence: str) -> int:
    """DG motifs, the aspartate isomerisation hotspot.

    Range: 0 upward. Motif published; no threshold count.
    """
    return len(re.findall(r"DG", sequence))


def max_lowcomplexity_run(sequence: str) -> int:
    """Longest run of a single residue.

    Range: 1-N. No published threshold; the default cap of 5 is a guess chosen
    to catch poly-Q / poly-K artefacts without flagging ordinary helices.
    """
    if not sequence:
        return 0
    best = run = 1
    for a, b in zip(sequence, sequence[1:], strict=False):  # deliberately offset by one
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def max_hydrophobic_patch(sequence: str, window: int = 5) -> float:
    """Largest mean Kyte-Doolittle hydropathy over a sliding window.

    Range: about -4.5 to +4.5. **Sequence-level proxy.** A real exposed
    hydrophobic patch is a surface-area measurement on a structure; without
    coordinates this reports the most hydrophobic stretch instead, which
    correlates with but is not equal to patch area. `LiabilityCaps` expresses
    its cap in A^2, so `liability_flags` compares against this proxy only after
    the caller has opted in; see docs/LIMITATIONS.md.
    """
    if len(sequence) < window:
        vals = [KYTE_DOOLITTLE.get(a, 0.0) for a in sequence]
        return round(sum(vals) / max(len(vals), 1), 3)
    best = -9.9
    for i in range(len(sequence) - window + 1):
        chunk = sequence[i : i + window]
        best = max(best, sum(KYTE_DOOLITTLE.get(a, 0.0) for a in chunk) / window)
    return round(best, 3)


def aggregation_propensity(sequence: str) -> float:
    """Fraction of residues in hydrophobic or beta-prone classes.

    Range: 0.0-1.0. A crude composition score (V, I, L, F, W, Y, M), not a
    validated aggregation predictor such as TANGO or AGGRESCAN. **The value is
    a guess in the sense of section 6.1** and must not be quoted as a
    probability of aggregation.
    """
    if not sequence:
        return 0.0
    prone = sum(1 for a in sequence if a in "VILFWYM")
    return round(prone / len(sequence), 4)


def fraction_nonstandard(sequence: str) -> float:
    """Fraction of characters outside the 20 standard amino acids.

    Range: 0.0-1.0. Must be 0.0 for a submittable design; `validate.py` makes
    that a hard failure.
    """
    if not sequence:
        return 1.0
    bad = sum(1 for a in sequence.upper() if a not in STANDARD_AA)
    return round(bad / len(sequence), 4)


def composition(sequence: str) -> dict[str, float]:
    """Amino-acid composition as fractions. Range per entry: 0.0-1.0."""
    n = max(len(sequence), 1)
    counts = Counter(sequence)
    return {aa: round(counts.get(aa, 0) / n, 4) for aa in sorted(STANDARD_AA)}


# --------------------------------------------------------------------------
# Self-consistency
# --------------------------------------------------------------------------


def mpnn_recovery(sequence: str, redesigned: str | None) -> float:
    """Fraction of positions where a redesign reproduces the original residue.

    Range: 0.0-1.0. Published designs typically sit around 0.3-0.5 for
    ProteinMPNN self-recovery; higher is more self-consistent. Returns NaN when
    no redesign is available, which is the Tier C case, rather than inventing a
    number.
    """
    if not redesigned or len(redesigned) != len(sequence):
        return float("nan")
    same = sum(1 for a, b in zip(sequence, redesigned, strict=True) if a == b)
    return round(same / max(len(sequence), 1), 4)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def compute_all(
    designs: pd.DataFrame,
    folded: pd.DataFrame,
    ss_by_backbone: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build the `metrics` table for every design.

    `folded` is the per-design collapse of the fold stage (mean across seeds
    plus the seed spread). Confidence columns are passed through untouched so
    that a mocked fold stays visibly mocked.
    """
    from binderkit.stages import gyration_proxy  # local import avoids a cycle

    ss_by_backbone = ss_by_backbone or {}
    rows: list[dict[str, object]] = []
    for d in designs.itertuples():
        seq = d.sequence
        ss = ss_by_backbone.get(d.backbone_id, "0:0:0")
        try:
            helix, sheet, loop = (float(x) for x in ss.split(":"))
        except ValueError:
            helix = sheet = loop = float("nan")
        rows.append(
            {
                "design_id": d.design_id,
                "radius_of_gyration": gyration_proxy(len(seq)),
                "frac_helix": helix,
                "frac_sheet": sheet,
                "frac_loop": loop,
                "mpnn_recovery": mpnn_recovery(seq, None),
                # interface geometry needs coordinates; see docs/LIMITATIONS.md
                "bsa": float("nan"),
                "n_contacts": float("nan"),
                "n_hbonds": float("nan"),
                "n_salt_bridges": float("nan"),
                "shape_complementarity": float("nan"),
                # developability - genuinely computed
                "max_hydrophobic_patch": max_hydrophobic_patch(seq),
                "net_charge": net_charge(seq),
                "pi": isoelectric_point(seq),
                "n_cys": count_cys(seq),
                "n_unpaired_cys": count_unpaired_cys(seq),
                "n_glyc_sequons": count_glyc_sequons(seq),
                "n_deamidation_motifs": count_deamidation_motifs(seq),
                "n_isomerisation_motifs": count_isomerisation_motifs(seq),
                "max_lowcomplexity_run": max_lowcomplexity_run(seq),
                "aggregation_propensity": aggregation_propensity(seq),
                "frac_nonstandard": fraction_nonstandard(seq),
            }
        )
    base = pd.DataFrame(rows)
    out = base.merge(folded, on="design_id", how="left")
    log.info("metrics: %d designs x %d columns", len(out), out.shape[1])
    return out


def liability_flags(metrics: pd.DataFrame, caps) -> pd.Series:  # noqa: ANN001
    """Boolean Series: True where a design breaches a hard liability cap.

    The hydrophobic-patch cap is skipped deliberately: `LiabilityCaps` states
    it in A^2 and the available metric is a unitless hydropathy proxy, so
    comparing them would be a category error. Recorded in docs/LIMITATIONS.md.
    """
    bad = pd.Series(False, index=metrics.index)
    bad |= metrics["n_unpaired_cys"] > caps.max_unpaired_cys
    bad |= metrics["n_glyc_sequons"] > caps.max_glyc_sequons
    bad |= metrics["max_lowcomplexity_run"] > caps.max_lowcomplexity_run
    lo, hi = caps.net_charge_range
    bad |= (metrics["net_charge"] < lo) | (metrics["net_charge"] > hi)
    bad |= metrics["frac_nonstandard"] > 0.0
    return bad
