"""Generation, sequence design and co-folding behind one interface each.

Scope note (docs/SPEC.md section 0.2 budget cut): only the `fixture` and `mock`
backends are implemented. The GPU backends are declared with the exact
signature they must satisfy and raise `BackendUnavailableError`, so swapping one in
is a localised change and nothing downstream has to move. Tier C mandates the
mock fold path anyway, so this cut removes less than it appears to.

Everything mocked here is marked with `# MOCK:` and listed in
docs/LIMITATIONS.md, and every row it produces carries `mocked=True`
(docs/SPEC.md section 0.5).
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from binderkit.config import SCHEMAS, Config
from binderkit.provenance import Provenance, sequence_hash
from binderkit.target import TargetSpec

log = logging.getLogger(__name__)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
#: Residues a soluble miniprotein surface is usually built from.
SURFACE_FAVOURED = "AEKQRSTDNLIV"


class BackendUnavailableError(RuntimeError):
    """A backend was requested that this install cannot run."""


@dataclass
class Backbone:
    """One generated backbone."""

    backbone_id: str
    method: str
    length: int
    seed: int
    path: str
    ss_fractions: str  # "helix:sheet:loop" so it survives a CSV round-trip


# --------------------------------------------------------------------------
# Stage 1 - backbones
# --------------------------------------------------------------------------


def generate(target: TargetSpec, cfg: Config, prov: Provenance) -> pd.DataFrame:  # noqa: ARG001
    """Produce backbones for `target`. Returns the `backbones` schema."""
    backend = cfg.generate.backend
    if backend in ("rfdiffusion", "bindcraft"):
        raise BackendUnavailableError(
            f"{backend} needs a GPU and model weights; this run is Tier C. "
            "Use backend='fixture' or raise the tier."
        )
    if backend != "fixture":
        raise BackendUnavailableError(f"unknown generate backend {backend!r}")

    # MOCK: fixture backbones are synthetic. No structure generation happens.
    prov.mark_mocked(
        "generate",
        "fixture backend emits synthetic backbones with no 3D coordinates; "
        "ss_fractions are drawn from a fixed distribution, not measured",
    )
    rng = random.Random(cfg.generate.seed)
    lo, hi = cfg.generate.length_range
    lengths = sorted({lo, (lo + hi) // 2, hi})

    rows: list[dict[str, object]] = []
    for length in lengths:
        for i in range(cfg.generate.n_per_length):
            seed = rng.randrange(1, 10**6)
            # Diversity across folds matters more than many near-identical
            # backbones (section 5.1), so alternate an all-helix bundle with a
            # sheet-containing fold rather than letting helices dominate.
            sheet_rich = i % 2 == 1
            if sheet_rich:
                helix, sheet = 0.25, 0.40
            else:
                helix, sheet = 0.70, 0.0
            loop = round(1.0 - helix - sheet, 3)
            bid = f"bb_{length}_{i:02d}"
            rows.append(
                {
                    "backbone_id": bid,
                    "method": "fixture",
                    "length": length,
                    "seed": seed,
                    "path": f"fixture://{bid}",
                    "ss_fractions": f"{helix}:{sheet}:{loop}",
                }
            )
    df = pd.DataFrame(rows, columns=SCHEMAS["backbones"])
    log.info("generate: %d backbones across lengths %s", len(df), lengths)
    return df


# --------------------------------------------------------------------------
# Stage 2 - sequences
# --------------------------------------------------------------------------


def _design_sequence(length: int, temperature: float, rng: random.Random) -> str:
    """Draw one plausible soluble miniprotein sequence.

    MOCK: this is a temperature-weighted draw from an amino-acid alphabet, not
    an inverse-folding model. Higher temperature widens the alphabet, which
    mimics the one real effect of the ProteinMPNN temperature sweep that
    downstream code depends on: diversity rises and conservatism falls.
    """
    pool = SURFACE_FAVOURED if temperature <= 0.15 else AMINO_ACIDS
    seq = "".join(rng.choice(pool) for _ in range(length))
    # Avoid a terminal Met/Cys artefact that would skew liability counts.
    return "M" + seq[1:]


def design_sequences(backbones: pd.DataFrame, cfg: Config, prov: Provenance) -> pd.DataFrame:
    """Sequence design over each backbone. Returns the `designs` schema."""
    backend = cfg.sequence.backend
    if backend in ("proteinmpnn", "solublempnn"):
        raise BackendUnavailableError(
            f"{backend} is not installed in this environment; this run is Tier C."
        )
    if backend != "fixture":
        raise BackendUnavailableError(f"unknown sequence backend {backend!r}")

    prov.mark_mocked(
        "sequence",
        "fixture backend draws sequences from a temperature-weighted amino-acid "
        "distribution instead of running an inverse-folding model",
    )
    rows: list[dict[str, object]] = []
    for bb in backbones.itertuples():
        for temp in cfg.sequence.temperatures:
            for k in range(cfg.sequence.n_per_backbone_per_temp):
                seed = (hash((bb.backbone_id, temp, k)) & 0x7FFFFFFF) ^ cfg.sequence.seed
                rng = random.Random(seed)
                seq = _design_sequence(int(bb.length), temp, rng)
                rows.append(
                    {
                        "design_id": f"{bb.backbone_id}_t{temp}_{k}",
                        "backbone_id": bb.backbone_id,
                        "sequence": seq,
                        "length": len(seq),
                        "mpnn_temp": temp,
                        "seed": seed,
                    }
                )
    df = pd.DataFrame(rows, columns=SCHEMAS["designs"])
    log.info("sequence: %d designs from %d backbones", len(df), len(backbones))
    return df


# --------------------------------------------------------------------------
# Stage 3 - co-folding
# --------------------------------------------------------------------------


def _mock_fold_scores(sequence: str, seed: int) -> dict[str, float]:
    """Deterministic pseudo-confidence for one design and seed.

    MOCK: these are NOT predictions. They are a deterministic function of the
    sequence hash, so tests are reproducible and the ranking code has something
    with realistic shape to consume. Any number derived from these is
    meaningless as biology and every row carries mocked=True.
    """
    h = int(sequence_hash(sequence), 16)
    rng = random.Random(h ^ seed)
    # Centre ipTM near the campaign-wide mean so downstream thresholds exercise.
    iptm = min(0.98, max(0.05, rng.gauss(0.60, 0.18)))
    ipsae = min(0.98, max(0.05, iptm - abs(rng.gauss(0.05, 0.04))))
    pae_int = max(0.5, min(30.0, rng.gauss(12.0, 4.0) * (1.2 - iptm)))
    return {
        "iptm": round(iptm, 4),
        "ipsae": round(ipsae, 4),
        "pae_int": round(pae_int, 3),
        "plddt_iface": round(min(98.0, max(20.0, 100 * iptm - rng.gauss(5, 3))), 2),
        "plddt_mono": round(min(98.0, max(20.0, 100 * iptm + rng.gauss(4, 4))), 2),
        "rmsd_sc": round(abs(rng.gauss(1.4, 0.8)), 3),
    }


def fold(designs: pd.DataFrame, cfg: Config, prov: Provenance) -> pd.DataFrame:
    """Co-fold each design against the target, and alone. `folds` schema.

    Caches by sequence hash: an unchanged sequence is never refolded
    (docs/SPEC.md section 5.3).
    """
    backend = cfg.fold.backend
    if backend in ("boltz2", "af3"):
        raise BackendUnavailableError(
            f"{backend} needs a GPU and weights that do not fit the Tier C disk "
            "guard; this run uses backend='mock'."
        )
    if backend != "mock":
        raise BackendUnavailableError(f"unknown fold backend {backend!r}")

    prov.mark_mocked(
        "fold",
        "no structure prediction is run; confidence values are a deterministic "
        "function of the sequence hash and are not predictions",
    )

    cache: dict[tuple[str, int], dict[str, float]] = {}
    rows: list[dict[str, object]] = []
    for d in designs.itertuples():
        shash = sequence_hash(d.sequence)
        for seed in range(cfg.fold.n_seeds):
            key = (shash, seed)
            if cfg.fold.cache_by_sequence_hash and key in cache:
                scores = cache[key]
            else:
                scores = _mock_fold_scores(d.sequence, seed)
                cache[key] = scores
            rows.append(
                {
                    "design_id": d.design_id,
                    "seed": seed,
                    **scores,
                    "path": f"mock://{shash}/seed{seed}",
                    "mocked": True,
                }
            )
    df = pd.DataFrame(rows, columns=SCHEMAS["folds"])
    log.info(
        "fold: %d rows (%d designs x %d seeds), %d unique sequence/seed folds",
        len(df),
        len(designs),
        cfg.fold.n_seeds,
        len(cache),
    )
    return df


def write_stage(df: pd.DataFrame, work_dir: Path, run_id: str, stage: str) -> Path:
    """Write a stage output so the stage is resumable (docs/SPEC.md section 3)."""
    out = Path(work_dir) / run_id / f"{stage}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def read_stage(work_dir: Path, run_id: str, stage: str) -> pd.DataFrame | None:
    """Read a previous stage output, or None if it does not exist."""
    out = Path(work_dir) / run_id / f"{stage}.parquet"
    if out.is_file():
        log.info("resume: reusing %s", out)
        return pd.read_parquet(out)
    return None


def seed_agreement(folds: pd.DataFrame) -> pd.DataFrame:
    """Per-design spread across folding seeds.

    Low seed agreement is itself a signal (docs/SPEC.md section 5.3), so the spread
    is reported rather than collapsed into a best-of.
    """
    g = folds.groupby("design_id")
    out = pd.DataFrame(
        {
            "seed_agreement_iptm": g["iptm"].std(ddof=0).fillna(0.0),
            "seed_agreement_pae": g["pae_int"].std(ddof=0).fillna(0.0),
        }
    )
    return out.reset_index()


def collapse_folds(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean across seeds per design, keeping the spread alongside."""
    numeric = ["iptm", "ipsae", "pae_int", "plddt_iface", "plddt_mono", "rmsd_sc"]
    mean = folds.groupby("design_id")[numeric].mean().reset_index()
    mocked = folds.groupby("design_id")["mocked"].any().reset_index()
    return mean.merge(mocked, on="design_id").merge(seed_agreement(folds), on="design_id")


def gyration_proxy(length: int) -> float:
    """Radius of gyration estimate for a compact globular chain, in Angstrom.

    Uses the standard Rg = 2.2 * N^0.38 scaling for folded monomers. A proxy,
    not a measurement, because Tier C has no 3D coordinates to measure.
    """
    return round(2.2 * math.pow(max(length, 1), 0.38), 2)
