"""Regression guard for the Simpson's-paradox error of docs/SPEC.md section 8.7.

A shipped version of `studies/interface_geometry/REPORT.md` stated that designs
which bound had *fewer* interface contacts and *smaller* buried surface area.
That comparison was pooled across targets. Targets with large interfaces (TNFa,
MBP) are the ones nobody could bind, so the pooled difference measured target
difficulty. Within target, every one of those differences reverses sign, which
is what the report's own within-target AUROC column had said all along.

These tests exist so the claim cannot come back: one checks the helper reports
the reversal rather than hiding it, and one checks the generated report text is
built from the within-target number and never from the pooled one.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pytest
from studies.retrospective.stats import GroupDifference, within_group_mean_difference

REPORT_PY = Path("studies/interface_geometry/report.py")
REPORT_MD = Path("studies/interface_geometry/REPORT.md")


def simpson() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two targets where pooling reverses the within-target effect.

    Target "easy" is mostly binders and has a small interface; target "hard" is
    mostly non-binders and has a large one. Inside each target binders have the
    larger value, but pooled the hard target's non-binders dominate the mean.
    """
    y, x, g = [], [], []
    for _ in range(40):  # easy target: small interface, high binder rate
        y.append(1), x.append(12.0), g.append("easy")
    for _ in range(5):
        y.append(0), x.append(10.0), g.append("easy")
    for _ in range(5):  # hard target: large interface, low binder rate
        y.append(1), x.append(102.0), g.append("hard")
    for _ in range(40):
        y.append(0), x.append(100.0), g.append("hard")
    return np.array(y), np.array(x, dtype=float), np.array(g)


def test_pooled_and_within_group_differences_disagree_in_sign() -> None:
    y, x, g = simpson()
    d = within_group_mean_difference(y, x, g, name="interface_size")
    assert d.pooled < 0, "the pooled comparison should be negative here"
    assert d.within > 0, "inside each target binders have the larger value"
    assert d.sign_flips, "the helper must flag that pooling reverses the sign"
    assert d.n_groups_positive == 2
    assert d.n_groups == 2


def test_within_group_difference_ignores_group_size() -> None:
    """One huge group must not outvote the others, which is the pooled failure."""
    y, x, g = simpson()
    y2 = np.concatenate([y, np.array([0] * 400)])
    x2 = np.concatenate([x, np.full(400, 100.0)])
    g2 = np.concatenate([g, np.array(["hard"] * 400)])
    assert within_group_mean_difference(y2, x2, g2).within > 0


def test_groups_without_both_classes_are_dropped_not_scored_zero() -> None:
    y = np.array([1, 1, 1, 0, 1])
    x = np.array([1.0, 2.0, 3.0, 1.0, 5.0])
    g = np.array(["all_pos", "all_pos", "all_pos", "mixed", "mixed"])
    d = within_group_mean_difference(y, x, g)
    assert d.n_groups == 1, "a single-class target contributes nothing"


def test_nan_values_are_dropped_per_group() -> None:
    y = np.array([1, 0, 1, 0])
    x = np.array([5.0, 1.0, np.nan, np.nan])
    g = np.array(["a", "a", "b", "b"])
    d = within_group_mean_difference(y, x, g)
    assert d.n_groups == 1
    assert d.within == pytest.approx(4.0)


def test_sentence_quotes_the_within_group_value_only() -> None:
    y, x, g = simpson()
    d = within_group_mean_difference(y, x, g, name="interface_size")
    text = d.sentence()
    assert "within target" in text
    assert "higher" in text, "the within-target direction is the one reported"
    assert f"{abs(d.pooled):.3g}" not in text


# ---------------------------------------------------------------------------
# The report text itself
# ---------------------------------------------------------------------------


def test_report_generator_uses_the_within_group_helper() -> None:
    """The report must obtain group differences from the audited helper."""
    src = REPORT_PY.read_text(encoding="utf-8")
    assert "within_group_mean_difference" in src


def test_report_generator_never_formats_a_pooled_difference_as_a_claim() -> None:
    """`.pooled` may be shown beside `.within`, never on its own.

    Every line of the generator that interpolates ``.pooled`` must interpolate
    ``.within`` too, so a pooled number cannot reach the reader unaccompanied by
    the honest one.
    """
    for line in REPORT_PY.read_text(encoding="utf-8").splitlines():
        if ".pooled" in line:
            assert ".within" in line, f"pooled difference quoted alone: {line.strip()}"


def test_withdrawn_sentences_are_absent_from_the_generator_and_the_report() -> None:
    """The exact claims section 8.7 withdrew, as substrings."""
    withdrawn = (
        "fewer** interface contacts",
        "smaller** buried surface",
        "implausible pose rather than a strong one",
        "Nobody had checked",
        "no published precedent",
    )
    for path in (REPORT_PY, REPORT_MD):
        text = path.read_text(encoding="utf-8")
        for phrase in withdrawn:
            assert phrase not in text, f"{path}: withdrawn claim present: {phrase!r}"


def test_report_states_the_geometry_result_in_its_narrow_form() -> None:
    """Section 8.5: not 'geometry is redundant' until the replication runs."""
    text = REPORT_MD.read_text(encoding="utf-8")
    assert "Overath" in text, "the conflicting published result must be named"
    assert "multiplicative" in text or "multiplied by" in text
    assert re.search(r"not\s+additively|additive", text, re.I), (
        "the finding must be qualified as additive-only"
    )


def test_report_shows_both_differences_side_by_side() -> None:
    """The reversal is evidence of the method working; it stays in the report."""
    text = REPORT_MD.read_text(encoding="utf-8")
    assert "pooled difference" in text
    assert "within-target difference" in text
    assert "Simpson" in text


def test_group_difference_fields_are_documented() -> None:
    """`pooled` carries a do-not-quote warning in the source, not just in a report."""
    doc = inspect.getdoc(GroupDifference) or ""
    assert "pooled" in doc.lower()
    assert "within" in doc.lower()
