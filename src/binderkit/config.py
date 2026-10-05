"""Configuration objects and dataframe schemas.

Every run is fully described by one config (docs/SPEC.md section 3), so a run can be
reproduced from its serialised config plus the recorded seeds. Thresholds that
came out of the retrospective calibration study (section 8) carry a citation in
their field comment; thresholds that are guesses say so, per section 6.1.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

Tier = Literal["A", "B", "C"]

# --------------------------------------------------------------------------
# Dataframe schemas (docs/SPEC.md section 6.5). Asserted in tests.
# --------------------------------------------------------------------------

SCHEMAS: dict[str, list[str]] = {
    "backbones": [
        "backbone_id",
        "method",
        "length",
        "seed",
        "path",
        "ss_fractions",
    ],
    "designs": [
        "design_id",
        "backbone_id",
        "sequence",
        "length",
        "mpnn_temp",
        "seed",
    ],
    "folds": [
        "design_id",
        "seed",
        "iptm",
        "ipsae",
        "pae_int",
        "plddt_iface",
        "plddt_mono",
        "rmsd_sc",
        "path",
        "mocked",
    ],
    "metrics": [
        "design_id",
        # interface confidence
        "iptm",
        "ipsae",
        "pae_int",
        "plddt_iface",
        # monomer quality
        "plddt_mono",
        "radius_of_gyration",
        "frac_helix",
        "frac_sheet",
        "frac_loop",
        # self-consistency
        "rmsd_sc",
        "mpnn_recovery",
        # interface geometry
        "bsa",
        "n_contacts",
        "n_hbonds",
        "n_salt_bridges",
        "shape_complementarity",
        # developability liabilities
        "max_hydrophobic_patch",
        "net_charge",
        "pi",
        "n_cys",
        "n_unpaired_cys",
        "n_glyc_sequons",
        "n_deamidation_motifs",
        "n_isomerisation_motifs",
        "max_lowcomplexity_run",
        "aggregation_propensity",
        # seed agreement
        "seed_agreement_iptm",
        "seed_agreement_pae",
        "mocked",
    ],
    "novelty": [
        "design_id",
        "best_seq_identity",
        "seq_hit",
        "best_tm",
        "struct_hit",
        "batch_cluster",
        "passed",
        "reason",
    ],
    "objectives": [
        "design_id",
        "objective",
        "value",
        "uncertainty",
        "reason",
    ],
    "ranking": [
        "design_id",
        "rank",
        "included",
        "reason",
    ],
}


# --------------------------------------------------------------------------
# Config sections
# --------------------------------------------------------------------------


@dataclass
class TargetConfig:
    """Which protein to design against, and where its epitope is."""

    name: str = "EGFR"
    uniprot: str = "P00533-1"
    pdb_id: str = "6ARU"
    chain: str = "A"
    residue_range: tuple[int, int] = (25, 645)
    #: Epitope guidance straight from the challenge page, not inferred.
    recommended_epitope: str = "Domain III"
    #: EGFR domain III in UniProt numbering. Domain boundaries from the
    #: canonical four-domain architecture of the EGFR ectodomain.
    epitope_residue_range: tuple[int, int] = (310, 480)
    #: Orthologs that objectives may need to score against.
    orthologs: dict[str, str] = field(default_factory=lambda: {"mouse": "Q01279"})
    keep_ligands: bool = False


@dataclass
class GenerateConfig:
    """Backbone generation. Backend is chosen by tier unless overridden."""

    backend: Literal["rfdiffusion", "bindcraft", "fixture"] = "fixture"
    length_range: tuple[int, int] = (55, 120)
    n_per_length: int = 4
    #: Challenge 1 allows 10-250 aa. The campaign used 50-120, so the default
    #: range sits inside both.
    seed: int = 0


@dataclass
class SequenceConfig:
    """Sequence design over generated backbones."""

    #: SolubleMPNN designed 1243 of the campaign 1440 designs; plain ProteinMPNN
    #: only 21. docs/SPEC.md section 5.2 names ProteinMPNN, so that stays the
    #: documented default and the substitution is left to the author.
    backend: Literal["proteinmpnn", "solublempnn", "fixture"] = "fixture"
    temperatures: tuple[float, ...] = (0.1, 0.2, 0.3)
    n_per_backbone_per_temp: int = 2
    fixed_positions: dict[str, list[int]] = field(default_factory=dict)
    seed: int = 0


@dataclass
class FoldConfig:
    """Co-folding of each design against the target."""

    backend: Literal["boltz2", "af3", "mock"] = "mock"
    n_seeds: int = 3
    cache_by_sequence_hash: bool = True


@dataclass
class NoveltyConfig:
    """The gate every design must pass (docs/SPEC.md section 6.2).

    Defaults are deliberately conservative. The challenge requires "adequate
    sequence- and structural-diversity from known proteins" without giving
    numbers, so these are *our* thresholds and are reported as such.
    """

    max_seq_identity: float = 0.30
    max_tm_score: float = 0.60
    #: Within-batch clustering: designs closer than this are one cluster.
    batch_identity_cutoff: float = 0.70
    max_per_cluster: int = 2
    #: Known binders of the target are rejected outright, not down-weighted.
    known_binder_identity_cutoff: float = 0.25


@dataclass
class LiabilityCaps:
    """Hard caps on developability liabilities.

    THRESHOLD PROVENANCE: these are conventional biologics-developability
    rules of thumb, not values fitted to the section 8 study. They are guesses
    in the sense of docs/SPEC.md section 6.1 and are recorded as such in
    docs/LIMITATIONS.md.
    """

    max_unpaired_cys: int = 0
    max_glyc_sequons: int = 0
    max_hydrophobic_patch: float = 400.0  # A^2
    net_charge_range: tuple[float, float] = (-8.0, 8.0)
    max_lowcomplexity_run: int = 5


@dataclass
class RankConfig:
    """Transparent, multi-criteria ranking (docs/SPEC.md section 6.4)."""

    #: Objective priority order. For Challenge 1 the challenge page ranks
    #: pH selectivity first, mouse cross-reactivity second, affinity third.
    objective_order: tuple[str, ...] = ("ph_selectivity", "ortholog", "affinity")
    #: Floor on self-consistency before a design is eligible at all.
    min_self_consistency_rmsd: float = 2.0
    #: Weight on the calibrated confidence score. Set by the section 8 study;
    #: see studies/retrospective/REPORT.md. Zero until the study has run.
    calibrated_confidence_weight: float = 0.0
    diversity_aware: bool = True


@dataclass
class SubmissionConfig:
    """Submission format. Verified against the live challenge page 2026-10-04."""

    #: Track 3 cap is unverified on the live pages: the challenge page says
    #: "up to 40 designs (Track 1)" and the terms give Tracks 2/3 one 384-well
    #: plate. docs/SPEC.md section 7.1 asserts 20. 20 satisfies both readings.
    max_designs: int = 20
    columns: tuple[str, ...] = ("name", "sequence", "molecule_class")
    allowed_molecule_class: tuple[str, ...] = (
        "protein",
        "nanobody",
        "scfv",
        "fab_kappa",
        "fab_lambda",
    )
    molecule_class: str = "protein"
    length_bounds: tuple[int, int] = (10, 250)
    require_unique_sequences: bool = True


@dataclass
class Config:
    """The whole run."""

    challenge_id: str = "01-egfr"
    tier: Tier | Literal["auto"] = "auto"
    run_id: str = "dev"
    work_dir: Path = Path("work")
    seed: int = 0

    target: TargetConfig = field(default_factory=TargetConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    fold: FoldConfig = field(default_factory=FoldConfig)
    novelty: NoveltyConfig = field(default_factory=NoveltyConfig)
    liabilities: LiabilityCaps = field(default_factory=LiabilityCaps)
    rank: RankConfig = field(default_factory=RankConfig)
    submission: SubmissionConfig = field(default_factory=SubmissionConfig)

    # ---- tier handling -------------------------------------------------

    def resolve_tier(self, detected: Tier) -> Tier:
        """Return the tier to run at, honouring an explicit override.

        Tier is a config value rather than a hardcoded branch so that a
        borderline machine can be overridden without a code change.
        """
        if self.tier == "auto":
            return detected
        return self.tier

    def apply_tier(self, tier: Tier) -> Config:
        """Return a copy with backends switched to what `tier` can actually run."""
        new = dataclasses.replace(self)
        if tier == "C":
            new.generate = dataclasses.replace(new.generate, backend="fixture")
            new.sequence = dataclasses.replace(new.sequence, backend="fixture")
            new.fold = dataclasses.replace(new.fold, backend="mock", n_seeds=1)
        elif tier == "B":
            new.generate = dataclasses.replace(new.generate, backend="bindcraft")
            new.sequence = dataclasses.replace(new.sequence, backend="proteinmpnn")
            new.fold = dataclasses.replace(new.fold, backend="boltz2", n_seeds=1)
        else:  # tier A
            new.generate = dataclasses.replace(new.generate, backend="rfdiffusion")
            new.sequence = dataclasses.replace(new.sequence, backend="proteinmpnn")
            new.fold = dataclasses.replace(new.fold, backend="boltz2", n_seeds=3)
        return new

    # ---- serialisation -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        def _conv(o: Any) -> Any:
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, tuple):
                return list(o)
            if isinstance(o, dict):
                return {k: _conv(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_conv(v) for v in o]
            return o

        return {k: _conv(v) for k, v in dataclasses.asdict(self).items()}

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        """Build a Config from a plain dict, coercing nested sections."""
        sections = {
            "target": TargetConfig,
            "generate": GenerateConfig,
            "sequence": SequenceConfig,
            "fold": FoldConfig,
            "novelty": NoveltyConfig,
            "liabilities": LiabilityCaps,
            "rank": RankConfig,
            "submission": SubmissionConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if key in sections and isinstance(value, dict):
                kwargs[key] = _build_section(sections[key], value)
            elif key == "work_dir":
                kwargs[key] = Path(value)
            else:
                kwargs[key] = value
        return cls(**kwargs)


def _build_section(klass: type, value: dict[str, Any]) -> Any:
    """Instantiate a config section, converting lists back to tuples."""
    fields = {f.name: f for f in dataclasses.fields(klass)}
    coerced: dict[str, Any] = {}
    for k, v in value.items():
        if k not in fields:
            continue
        ftype = str(fields[k].type)
        coerced[k] = tuple(v) if isinstance(v, list) and "tuple" in ftype else v
    return klass(**coerced)
