"""Fixture-first regression test: the whole pipeline, no GPU, no network.

docs/SPEC.md section 13 makes this the primary regression test. It must run
offline in under two minutes, so it uses the committed fixture target rather
than fetching 6ARU, and the fixture/mock backends throughout.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from binderkit import stages
from binderkit.config import SCHEMAS, Config
from binderkit.metrics import compute_all
from binderkit.novelty import Reference, gate, read_fasta
from binderkit.package import package
from binderkit.provenance import Provenance
from binderkit.rank import apply_hard_filters, filter_cascade_counts, rank, score_objectives
from binderkit.target import TargetSpec, build_numbering_map, parse_chain

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_spec() -> TargetSpec:
    """A TargetSpec built entirely from committed files - no network."""
    pdb = FIXTURES / "mini_target.pdb"
    uniprot = "".join(
        ln.strip()
        for ln in (FIXTURES / "mini_target.fasta").read_text(encoding="utf-8").splitlines()
        if not ln.startswith(">")
    )
    observed = parse_chain(pdb, "A")
    mapping, _ = build_numbering_map(observed, uniprot)
    return TargetSpec(
        name="MINI",
        uniprot="FIXTURE",
        pdb_id="MINI",
        chain="A",
        structure_path=str(pdb),
        sequence_uniprot=uniprot,
        sequence_observed="".join(r.one_letter for r in observed),
        numbering_map=mapping,
        epitope_residues_uniprot=sorted(mapping.values())[:8],
        epitope_residues_pdb=sorted(mapping.keys())[:8],
        hotspots_pdb=sorted(mapping.keys())[:4],
        hotspot_method="fixture",
        epitope_rationale="fixture epitope: first 8 observed residues",
    )


@pytest.fixture
def tiny_config(tmp_path: Path) -> Config:
    """Smallest config that still exercises every stage: 2 backbones, 1 seed."""
    cfg = Config(challenge_id="test", tier="C", run_id="t", work_dir=tmp_path / "work")
    cfg = cfg.apply_tier("C")
    cfg.generate.length_range = (40, 40)
    cfg.generate.n_per_length = 2
    cfg.sequence.temperatures = (0.2,)
    cfg.sequence.n_per_backbone_per_temp = 3
    cfg.fold.n_seeds = 1
    # The fixture sequence generator draws essentially random sequences, so with
    # production caps nearly all of them breach a developability limit and the
    # submission comes out empty. This test is about plumbing, so the caps are
    # relaxed here; `test_strict_filters_can_empty_the_submission` covers the
    # production-cap behaviour instead.
    cfg.liabilities.max_unpaired_cys = 1
    cfg.liabilities.max_glyc_sequons = 10
    cfg.liabilities.max_lowcomplexity_run = 20
    cfg.liabilities.net_charge_range = (-50.0, 50.0)
    cfg.rank.min_self_consistency_rmsd = 99.0
    return cfg


def test_pipeline_end_to_end(mini_spec: TargetSpec, tiny_config: Config, tmp_path: Path) -> None:
    cfg = tiny_config
    prov = Provenance(run_id=cfg.run_id)

    backbones = stages.generate(mini_spec, cfg, prov)
    assert list(backbones.columns) == SCHEMAS["backbones"]
    assert len(backbones) == 2

    designs = stages.design_sequences(backbones, cfg, prov)
    assert list(designs.columns) == SCHEMAS["designs"]
    assert len(designs) == 6
    assert designs["sequence"].str.len().eq(40).all()

    folds = stages.fold(designs, cfg, prov)
    assert list(folds.columns) == SCHEMAS["folds"]
    assert folds["mocked"].all(), "every mocked fold row must be marked mocked"

    folded = stages.collapse_folds(folds)
    metrics = compute_all(
        designs, folded, dict(zip(backbones.backbone_id, backbones.ss_fractions, strict=True))
    )
    for col in SCHEMAS["metrics"]:
        if col in ("iptm", "ipsae", "pae_int", "plddt_iface", "plddt_mono", "rmsd_sc"):
            continue  # supplied by the fold collapse
        assert col in metrics.columns, f"metrics schema missing {col}"

    refs = [Reference("MINI_target", mini_spec.sequence_uniprot)]
    refs += read_fasta(FIXTURES / "known_binders.fasta", is_known_binder=True)
    nov = gate(designs, cfg.novelty, refs)
    assert list(nov.columns) == SCHEMAS["novelty"]

    objectives = score_objectives(
        designs,
        {"epitope_conservation": 0.9, "species": ["human", "mouse"], "per_species_scores": {}},
        cfg.rank.objective_order,
    )
    assert list(objectives.columns) == SCHEMAS["objectives"]
    assert set(objectives["objective"]) <= set(cfg.rank.objective_order)

    filtered = apply_hard_filters(metrics, nov, cfg)
    ranking = rank(filtered, objectives, cfg)
    assert list(ranking.columns) == SCHEMAS["ranking"]
    assert len(ranking) == len(designs), "every design must appear, rejected ones included"

    # Ranked rows must be a prefix 1..n with no gaps.
    ranks = ranking.loc[ranking["included"], "rank"].tolist()
    assert ranks == list(range(1, len(ranks) + 1))

    result = package(
        cfg,
        tmp_path / "sub",
        designs,
        metrics,
        nov,
        objectives,
        filtered,
        ranking,
        mini_spec,
        prov,
    )
    assert result.ok, result.errors
    for name in (
        "submission.csv",
        "METHODS.md",
        "metrics.csv",
        "ranking_full.csv",
        "novelty.csv",
        "provenance.json",
        "VALIDATION.txt",
    ):
        assert (tmp_path / "sub" / name).is_file(), f"{name} not written"

    # A mocked run must say so in the methods document.
    methods = (tmp_path / "sub" / "METHODS.md").read_text(encoding="utf-8")
    assert "mocked" in methods.lower()
    assert prov.any_mocked

    cascade = filter_cascade_counts(designs, nov, filtered, ranking)
    assert cascade[0][1] == len(designs)
    assert cascade[-1][1] == int(ranking["included"].sum())


def test_resume_skips_recomputation(mini_spec: TargetSpec, tiny_config: Config) -> None:
    """A stage whose output exists is reused rather than recomputed."""
    cfg = tiny_config
    prov = Provenance(run_id=cfg.run_id)
    backbones = stages.generate(mini_spec, cfg, prov)
    stages.write_stage(backbones, cfg.work_dir, cfg.run_id, "backbones")
    again = stages.read_stage(cfg.work_dir, cfg.run_id, "backbones")
    assert again is not None
    pd.testing.assert_frame_equal(backbones, again)
    assert stages.read_stage(cfg.work_dir, cfg.run_id, "nonexistent") is None


def test_fold_cache_does_not_refold(mini_spec: TargetSpec, tiny_config: Config) -> None:
    """An unchanged sequence is folded once, not once per occurrence."""
    cfg = tiny_config
    prov = Provenance(run_id=cfg.run_id)
    designs = pd.DataFrame(
        {
            "design_id": ["a", "b"],
            "backbone_id": ["bb", "bb"],
            "sequence": ["MKTAYIAKQRQ"] * 2,  # identical
            "length": [11, 11],
            "mpnn_temp": [0.2, 0.2],
            "seed": [1, 2],
        }
    )
    folds = stages.fold(designs, cfg, prov)
    # Same sequence and seed must give identical scores.
    assert folds.loc[0, "iptm"] == folds.loc[1, "iptm"]


def test_gpu_backends_raise_rather_than_silently_mock(
    mini_spec: TargetSpec, tiny_config: Config
) -> None:
    """Requesting a GPU backend in Tier C must fail loudly, not quietly mock."""
    cfg = tiny_config
    cfg.generate.backend = "rfdiffusion"
    with pytest.raises(stages.BackendUnavailableError):
        stages.generate(mini_spec, cfg, Provenance(run_id="x"))


def test_strict_filters_can_empty_the_submission(
    mini_spec: TargetSpec, tiny_config: Config, tmp_path: Path
) -> None:
    """With production caps, random fixture sequences are all rejected, and an
    empty submission must fail validation loudly rather than ship as "done".

    This is the real Tier C situation: the fixture backend cannot produce a
    submittable batch, and the pipeline has to say so instead of writing an
    empty CSV that looks finished.
    """
    cfg = tiny_config
    cfg.liabilities.max_unpaired_cys = 0
    cfg.liabilities.max_glyc_sequons = 0
    cfg.liabilities.max_lowcomplexity_run = 5
    cfg.liabilities.net_charge_range = (-8.0, 8.0)
    cfg.rank.min_self_consistency_rmsd = 2.0

    prov = Provenance(run_id=cfg.run_id)
    backbones = stages.generate(mini_spec, cfg, prov)
    designs = stages.design_sequences(backbones, cfg, prov)
    folds = stages.fold(designs, cfg, prov)
    folded = stages.collapse_folds(folds)
    metrics = compute_all(
        designs, folded, dict(zip(backbones.backbone_id, backbones.ss_fractions, strict=True))
    )
    nov = gate(designs, cfg.novelty, [Reference("t", mini_spec.sequence_uniprot)])
    objectives = score_objectives(designs, {"epitope_conservation": 0.9}, cfg.rank.objective_order)
    filtered = apply_hard_filters(metrics, nov, cfg)
    ranking = rank(filtered, objectives, cfg)

    # Rejections must still be fully explained, which is the evidence of judgement.
    assert len(ranking) == len(designs)
    assert ranking["reason"].str.len().gt(0).all()

    result = package(
        cfg,
        tmp_path / "strict",
        designs,
        metrics,
        nov,
        objectives,
        filtered,
        ranking,
        mini_spec,
        prov,
    )
    if int(ranking["included"].sum()) == 0:
        assert not result.ok
        assert any("empty" in e for e in result.errors)
