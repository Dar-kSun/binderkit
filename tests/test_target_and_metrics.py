"""Numbering-map tests (docs/SPEC.md section 4) and metric value tests (section 13).

The numbering map is singled out in section 4 as the commonest silent bug in
target preparation, so it is tested against both a committed fixture and a
synthetic case with a known offset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from binderkit import metrics as M
from binderkit.novelty import pairwise_identity
from binderkit.objectives.ortholog import epitope_conservation
from binderkit.objectives.ph_selectivity import PhSelectivity, delta_charge, his_clustering
from binderkit.target import Residue, build_numbering_map, epitope_from_range, parse_chain

FIXTURES = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------
# Numbering
# --------------------------------------------------------------------------


def _residues(seq: str, start: int) -> list[Residue]:
    """Observed residues numbered from `start`, for offset tests."""
    return [Residue(start + i, "", "XXX", aa) for i, aa in enumerate(seq)]


def test_numbering_map_recovers_a_known_positive_offset() -> None:
    """A construct numbered from 1 against a precursor with a 24-residue leader
    must come out with offset +24, which is the real EGFR/6ARU case.
    """
    mature = "AEKQRSTDNLIVAEKQRSTDNLIV"
    precursor = "M" * 24 + mature
    observed = _residues(mature, start=1)
    mapping, agree = build_numbering_map(observed, precursor)
    assert agree == len(mature), "every residue should match at the correct offset"
    offsets = {u - p for p, u in mapping.items()}
    assert offsets == {24}, f"expected a single +24 offset, got {offsets}"
    assert mapping[1] == 25


def test_numbering_map_handles_zero_offset() -> None:
    seq = "AEKQRSTDNLIV"
    observed = _residues(seq, start=1)
    mapping, agree = build_numbering_map(observed, seq)
    assert {u - p for p, u in mapping.items()} == {0}
    assert agree == len(seq)


def test_numbering_map_on_committed_fixture() -> None:
    observed = parse_chain(FIXTURES / "mini_target.pdb", "A")
    uniprot = "".join(
        ln.strip()
        for ln in (FIXTURES / "mini_target.fasta").read_text(encoding="utf-8").splitlines()
        if not ln.startswith(">")
    )
    assert len(observed) == 24, "fixture should expose 24 residues"
    mapping, agree = build_numbering_map(observed, uniprot)
    # fixture is numbered from 5 and the uniprot has a 4-residue leader
    assert {u - p for p, u in mapping.items()} == {0}
    assert agree == 24


def test_numbering_map_empty_input() -> None:
    mapping, agree = build_numbering_map([], "AEKQR")
    assert mapping == {} and agree == 0


def test_epitope_from_range_uses_the_inverse_map() -> None:
    mapping = {1: 25, 2: 26, 3: 27, 4: 28}
    uni, pdb = epitope_from_range((26, 27), mapping)
    assert uni == [26, 27]
    assert pdb == [2, 3]


def test_parse_chain_skips_waters_and_other_chains() -> None:
    observed = parse_chain(FIXTURES / "mini_target.pdb", "B")
    assert observed == [], "there is no chain B in the fixture"


# --------------------------------------------------------------------------
# Metrics with known expected values
# --------------------------------------------------------------------------


def test_glycosylation_sequons_counted_correctly() -> None:
    # NGS and NAT are sequons; NPT is not (proline at X).
    assert M.count_glyc_sequons("AANGSAA") == 1
    assert M.count_glyc_sequons("AANATAA") == 1
    assert M.count_glyc_sequons("AANPTAA") == 0
    assert M.count_glyc_sequons("NGSNAT") == 2


def test_deamidation_and_isomerisation_motifs() -> None:
    assert M.count_deamidation_motifs("ANGA") == 1
    assert M.count_deamidation_motifs("ANSANGA") == 2
    assert M.count_isomerisation_motifs("ADGA") == 1
    assert M.count_isomerisation_motifs("AAAA") == 0


def test_cysteine_pairing_parity() -> None:
    assert M.count_unpaired_cys("ACA") == 1
    assert M.count_unpaired_cys("ACAC") == 0
    assert M.count_unpaired_cys("AAA") == 0


def test_charge_ordering_is_physical() -> None:
    """A poly-lysine peptide must be strongly positive and poly-glutamate negative."""
    assert M.net_charge("KKKKKKKKKK") > 8
    assert M.net_charge("EEEEEEEEEE") < -8
    assert M.isoelectric_point("KKKKKKKKKK") > M.isoelectric_point("EEEEEEEEEE")


def test_charge_falls_as_ph_rises() -> None:
    seq = "MKTAYIAKQRQISFVKSHFSRHH"
    assert M.net_charge(seq, 5.0) > M.net_charge(seq, 7.4) > M.net_charge(seq, 10.0)


def test_hydrophobic_patch_ranks_as_expected() -> None:
    assert M.max_hydrophobic_patch("IIIIIIIIII") > M.max_hydrophobic_patch("DDDDDDDDDD")


def test_lowcomplexity_run() -> None:
    assert M.max_lowcomplexity_run("AAABB") == 3
    assert M.max_lowcomplexity_run("ABABAB") == 1
    assert M.max_lowcomplexity_run("") == 0


def test_nonstandard_detection() -> None:
    assert M.fraction_nonstandard("ACDEFG") == 0.0
    assert M.fraction_nonstandard("ACDEFX") > 0.0


def test_mpnn_recovery_is_nan_without_a_redesign() -> None:
    val = M.mpnn_recovery("ACDEF", None)
    assert val != val  # NaN
    assert M.mpnn_recovery("ACDEF", "ACDEF") == 1.0
    assert M.mpnn_recovery("ACDEF", "ACDEG") == 0.8


# --------------------------------------------------------------------------
# Objectives
# --------------------------------------------------------------------------


def test_delta_charge_is_driven_by_histidine() -> None:
    """Histidine is the only residue whose pKa sits between pH 7.4 and 6.5, so a
    sequence with histidine must gain charge on acidification and one without
    must barely move.
    """
    with_his = delta_charge("AAAHAAAHAAA")
    without = delta_charge("AAAKAAAEAAA")
    assert with_his > 0.3, with_his
    assert abs(without) < 0.05, without
    assert with_his > without


def test_more_histidine_gives_more_delta_charge() -> None:
    assert delta_charge("AHAHAHA") > delta_charge("AHA") > delta_charge("AAA")


def test_his_clustering() -> None:
    assert his_clustering("HHH" + "A" * 30) == 3
    assert his_clustering("H" + "A" * 30 + "H") == 1
    assert his_clustering("A" * 20) == 0


def test_ph_objective_scores_zero_without_histidine() -> None:
    class D:
        sequence = "AEKQRSTDNLIVAEKQRSTDNLIV"

    score = PhSelectivity().score(D(), {})
    assert score.value == 0.0
    assert "no histidine" in score.reason


def test_ph_objective_rewards_histidine_and_states_uncertainty() -> None:
    class D:
        sequence = "AEKHRSTDNHIVAEKHRSTDNLIV"

    score = PhSelectivity().score(D(), {})
    assert score.value > 0.0
    assert score.uncertainty > 0.2, "a proxy objective must carry real uncertainty"
    assert "weakly validated" in score.reason.lower()


def test_epitope_conservation_identical_and_divergent() -> None:
    human = "AEKQRSTDNLIVAEKQRSTDNLIV"
    same = human
    frac, per_res = epitope_conservation(human, same, [1, 2, 3, 4])
    assert frac == 1.0
    assert all(per_res.values())

    divergent = "WWWWWWWWWWWWWWWWWWWWWWWW"
    frac2, _ = epitope_conservation(human, divergent, [1, 2, 3, 4])
    assert frac2 < 1.0


def test_pairwise_identity_bounds() -> None:
    assert pairwise_identity("ACDEFGHIKL", "ACDEFGHIKL") == 1.0
    assert pairwise_identity("", "ACDEF") == 0.0
    assert 0.0 <= pairwise_identity("ACDEFGHIKL", "WWWWWWWWWW") <= 1.0


@pytest.mark.parametrize("ph", [5.0, 6.5, 7.4, 8.0])
def test_net_charge_is_finite_everywhere(ph: float) -> None:
    val = M.net_charge("MKTAYIAKQRQISFVKSHFSRHHDDEECC", ph)
    assert val == val and abs(val) < 100
