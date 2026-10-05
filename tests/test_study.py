"""Tests for the retrospective study (docs/SPEC.md sections 8 and 13).

Section 13 specifically requires a test asserting that no target appears in both
train and test. The statistical helpers are tested on synthetic data so the
suite stays offline; the study itself is marked slow because it needs the
dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from studies.retrospective.run_study import (
    GROUP,
    LABEL,
    expected_calibration_error,
    grouped_bootstrap_ci,
    no_target_appears_in_both_folds,
    within_target_auroc,
)


def synthetic(n_per_target: int = 40, n_targets: int = 6, seed: int = 0) -> pd.DataFrame:
    """Designs across targets with deliberately different base rates.

    Mirrors the real dataset's defining feature: per-target binder rates differ
    a lot, so a design-level split would leak target difficulty.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_targets):
        rate = 0.05 + 0.85 * t / max(n_targets - 1, 1)
        for _ in range(n_per_target):
            y = int(rng.random() < rate)
            # score correlates with y, plus a target-level offset
            score = rng.normal(0.6 if y else 0.4, 0.15) + 0.1 * t
            rows.append({"target": f"T{t}", LABEL: bool(y), "score": score})
    return pd.DataFrame(rows)


def test_no_target_appears_in_both_folds() -> None:
    """The requirement from section 13, asserted directly."""
    df = synthetic()
    assert no_target_appears_in_both_folds(df)


def test_leave_one_group_out_actually_holds_out_whole_targets() -> None:
    from sklearn.model_selection import LeaveOneGroupOut

    df = synthetic()
    groups = df[GROUP].to_numpy()
    seen_test: set[str] = set()
    for train_idx, test_idx in LeaveOneGroupOut().split(
        np.zeros((len(df), 1)), df[LABEL].to_numpy(dtype=int), groups
    ):
        test_groups = set(groups[test_idx])
        assert len(test_groups) == 1, "each fold must hold out exactly one target"
        assert not (test_groups & set(groups[train_idx])), "target leaked into train"
        seen_test |= test_groups
    assert seen_test == set(groups), "every target must be held out exactly once"


def test_within_target_auroc_excludes_single_class_targets() -> None:
    """A target with one outcome class contributes nothing and is not scored 0.5."""
    df = pd.DataFrame(
        {
            "target": ["A"] * 4 + ["B"] * 4,
            LABEL: [True, False, True, False] + [False] * 4,
            "score": [0.9, 0.1, 0.8, 0.2, 0.5, 0.5, 0.5, 0.5],
        }
    )
    y = df[LABEL].to_numpy(dtype=int)
    s = df["score"].to_numpy(dtype=float)
    g = df[GROUP].to_numpy()
    mean, per, n_by = within_target_auroc(y, s, g)
    assert "A" in per and "B" not in per
    assert set(n_by) == {"A", "B"}, "excluded targets must still be reported"
    assert mean == per["A"]


def test_within_target_auroc_detects_a_perfect_ranker() -> None:
    df = pd.DataFrame(
        {
            "target": ["A"] * 4 + ["B"] * 4,
            LABEL: [True, True, False, False] * 2,
            "score": [0.9, 0.8, 0.2, 0.1] * 2,
        }
    )
    mean, per, _ = within_target_auroc(
        df[LABEL].to_numpy(dtype=int),
        df["score"].to_numpy(dtype=float),
        df[GROUP].to_numpy(),
    )
    assert mean == pytest.approx(1.0)
    assert set(per) == {"A", "B"}


def test_grouped_bootstrap_ci_brackets_the_point_estimate() -> None:
    df = synthetic(seed=1)
    y = df[LABEL].to_numpy(dtype=int)
    s = df["score"].to_numpy(dtype=float)
    g = df[GROUP].to_numpy()
    point, lo, hi = grouped_bootstrap_ci(y, s, g, n_boot=300)
    assert 0.0 <= lo <= point <= hi <= 1.0


def test_grouped_bootstrap_is_wider_than_naive_design_level() -> None:
    """Resampling targets must not understate uncertainty the way designs would.

    The grouped interval should be at least as wide as one built by resampling
    individual designs, because designs within a target are not independent.
    """
    df = synthetic(n_per_target=40, n_targets=6, seed=2)
    y = df[LABEL].to_numpy(dtype=int)
    s = df["score"].to_numpy(dtype=float)
    g = df[GROUP].to_numpy()
    _, glo, ghi = grouped_bootstrap_ci(y, s, g, n_boot=600, seed=3)

    # Naive: treat every design as its own group.
    naive_groups = np.arange(len(df))
    _, nlo, nhi = grouped_bootstrap_ci(y, s, naive_groups, n_boot=600, seed=3)
    assert (ghi - glo) >= (nhi - nlo), "grouped CI should not be narrower than design-level"


def test_expected_calibration_error_of_a_perfect_model_is_zero() -> None:
    p = np.array([0.0, 0.0, 1.0, 1.0])
    y = np.array([0, 0, 1, 1])
    ece, rows = expected_calibration_error(y, p, n_bins=4)
    assert ece == pytest.approx(0.0, abs=1e-9)
    assert rows


def test_expected_calibration_error_of_a_badly_calibrated_model_is_large() -> None:
    """Confidently wrong predictions must produce a large ECE."""
    p = np.full(100, 0.95)
    y = np.zeros(100, dtype=int)
    ece, _ = expected_calibration_error(y, p)
    assert ece > 0.9


def test_expected_calibration_error_bins_cover_the_endpoint() -> None:
    """A probability of exactly 1.0 must land in the last bin, not be dropped."""
    p = np.array([1.0, 1.0])
    y = np.array([1, 1])
    _, rows = expected_calibration_error(y, p, n_bins=5)
    assert sum(r["n"] for r in rows) == 2


@pytest.mark.slow
def test_study_runs_end_to_end() -> None:
    """The real study. Needs the cached dataset, so it is marked slow."""
    from pathlib import Path

    from studies.retrospective.report import main

    if not Path("work/hf_cache/tables/design_summary.csv").is_file():
        pytest.skip("dataset not cached")
    assert main() == 0
    assert Path("studies/retrospective/REPORT.md").is_file()
    assert Path("studies/retrospective/calibration.json").is_file()
