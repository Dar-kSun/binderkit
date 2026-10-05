"""Novelty gate tests, including the planted known binder that must be
rejected (docs/SPEC.md section 13).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from binderkit.config import NoveltyConfig
from binderkit.novelty import Reference, gate, read_fasta, within_batch_clusters

FIXTURES = Path(__file__).parent / "fixtures"
KNOWN = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANNGSTLDWNGSCC"
NOVEL = "MEEKLRQATAKVDELTRQSNELKAQVDRLTAQNEALKAQVDRLTAQ"


def designs_frame(**seqs: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "design_id": list(seqs),
            "backbone_id": ["bb"] * len(seqs),
            "sequence": list(seqs.values()),
        }
    )


def test_planted_known_binder_is_rejected() -> None:
    """The headline requirement: a design identical to a known binder must be
    rejected outright, not merely down-weighted.
    """
    refs = [Reference("KNOWN_X", KNOWN, is_known_binder=True)]
    out = gate(designs_frame(planted=KNOWN, novel=NOVEL), NoveltyConfig(), refs)
    row = out.set_index("design_id").loc["planted"]
    assert not row["passed"]
    assert row["best_seq_identity"] == 1.0
    assert "known binder" in row["reason"].lower()
    assert "REJECTED" in row["reason"]
    assert out.set_index("design_id").loc["novel", "passed"]


def test_known_binder_fixture_file_is_used() -> None:
    refs = read_fasta(FIXTURES / "known_binders.fasta", is_known_binder=True)
    assert refs and refs[0].is_known_binder
    out = gate(designs_frame(planted=refs[0].sequence), NoveltyConfig(), refs)
    assert not out.loc[0, "passed"]


def test_near_identical_binder_also_rejected() -> None:
    """A single point mutation must not be enough to escape the gate."""
    mutated = "A" + KNOWN[1:]
    refs = [Reference("KNOWN_X", KNOWN, is_known_binder=True)]
    out = gate(designs_frame(mutant=mutated), NoveltyConfig(), refs)
    assert not out.loc[0, "passed"]


def test_novel_design_passes_and_reports_its_number() -> None:
    refs = [Reference("KNOWN_X", KNOWN, is_known_binder=True)]
    out = gate(designs_frame(novel=NOVEL), NoveltyConfig(), refs)
    assert out.loc[0, "passed"]
    assert "passed" in out.loc[0, "reason"]
    assert 0.0 <= out.loc[0, "best_seq_identity"] <= 1.0


def test_structural_arm_absence_is_stated_not_hidden() -> None:
    """With no TM comparison the gate must say so rather than silently pass."""
    out = gate(designs_frame(novel=NOVEL), NoveltyConfig(), [])
    assert "NOT CHECKED" in out.loc[0, "reason"]
    assert out.loc[0, "best_tm"] != out.loc[0, "best_tm"]  # NaN


def test_structural_hit_above_cutoff_rejects() -> None:
    out = gate(
        designs_frame(novel=NOVEL),
        NoveltyConfig(),
        [],
        structure_tm={"novel": (0.95, "1ABC_A")},
    )
    assert not out.loc[0, "passed"]
    assert "structural TM" in out.loc[0, "reason"]


def test_within_batch_clustering_groups_identical_sequences() -> None:
    frame = designs_frame(a=NOVEL, b=NOVEL, c=KNOWN)
    clusters = within_batch_clusters(frame, identity_cutoff=0.7)
    assert clusters["a"] == clusters["b"], "identical designs belong to one cluster"
    assert clusters["c"] != clusters["a"]


def test_cluster_cap_limits_representatives() -> None:
    """Only `max_per_cluster` members of one cluster may pass."""
    frame = designs_frame(a=NOVEL, b=NOVEL, c=NOVEL, d=NOVEL)
    cfg = NoveltyConfig(max_per_cluster=2)
    out = gate(frame, cfg, []).set_index("design_id")
    assert int(out["passed"].sum()) == 2
    assert not out.loc["c", "passed"]
    assert "cluster" in out.loc["c", "reason"]


def test_gate_reports_actual_identities_not_just_pass_fail() -> None:
    """Section 6.2 requires the achieved numbers, not a boolean."""
    refs = [Reference("KNOWN_X", KNOWN, is_known_binder=True)]
    out = gate(designs_frame(novel=NOVEL, planted=KNOWN), NoveltyConfig(), refs)
    assert out["best_seq_identity"].notna().all()
    assert (out["seq_hit"].astype(str).str.len() > 0).all()
