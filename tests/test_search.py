"""Tests for the MMseqs2 novelty search (NEXT_SESSION.md section 4.1).

The search itself needs a 1.1M-sequence database, so those tests are marked
slow. The coverage logic, which is where the real trap is, is tested offline.
"""

from __future__ import annotations

import pytest

from binderkit.search import (
    MIN_QUERY_COVERAGE,
    SIGNIFICANT_EVALUE,
    Hit,
    SearchUnavailableError,
    available,
    search,
)


def test_effective_identity_discounts_short_alignments() -> None:
    """The trap this guards against: MMseqs reports identity over the aligned
    region only, so a 9-residue local match looks like 77% identity.
    """
    noise = Hit(
        "d1", "1abc_A", identity=0.77, alignment_length=9, evalue=3000.0, query_coverage=0.16
    )
    real = Hit("d2", "3egf_A", identity=0.77, alignment_length=52, evalue=1e-30, query_coverage=1.0)
    assert noise.effective_identity == pytest.approx(0.1232)
    assert real.effective_identity == pytest.approx(0.77)
    assert noise.effective_identity < 0.30 < real.effective_identity, (
        "a 30% novelty cutoff must reject the real homolog and keep the noise hit"
    )


def test_empty_hit_is_not_significant() -> None:
    h = Hit("d")
    assert not h.significant
    assert h.effective_identity == 0.0


def test_thresholds_are_what_the_docstring_claims() -> None:
    assert SIGNIFICANT_EVALUE == 1e-3
    assert MIN_QUERY_COVERAGE == 0.50


def test_search_raises_clearly_when_unavailable(tmp_path) -> None:
    with pytest.raises(SearchUnavailableError):
        search({"a": "ACDEFGHIKL"}, binary=tmp_path / "nope.exe", db=tmp_path / "nodb")


@pytest.mark.slow
def test_real_search_finds_a_real_protein() -> None:
    """Positive control: a genuine EGF fragment must find itself in the PDB.

    Without this, a search that silently returned nothing would look exactly
    like a set of perfectly novel designs.
    """
    if not available():
        pytest.skip("mmseqs database not built")
    frag = "NSYPGCPSSYDGYCLNGGVCMHIESLDSYTCNCVIGYSGDRCQTRDLRWWELR"
    hits = search({"egf": frag})
    h = hits["egf"]
    assert h.significant, f"a real EGF fragment should hit the PDB, got {h}"
    assert h.identity > 0.9
    assert h.query_coverage > 0.9
    assert h.evalue < 1e-10
