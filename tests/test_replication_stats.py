"""Tests for the ranking statistics study 3 introduces.

Study 1 and 2 reported within-target AUROC. Overath et al. report average
precision and precision@k, so study 3 has to compute both, and a comparison of
our AUROC against their AP would mean nothing if our AP were wrong.

`average_precision` is checked against scikit-learn's implementation, which is
an independent one, on random problems — the same kind of external check that
took the geometry code to exact agreement. The rest are checked against cases
whose answer can be worked out by hand.
"""

from __future__ import annotations

import numpy as np
import pytest
from studies.retrospective.stats import (
    average_precision,
    mean_within_target_ap,
    mean_within_target_precision_at_k,
    precision_at_k,
)


def test_average_precision_matches_sklearn_on_random_problems() -> None:
    """An independent implementation must agree to floating-point noise."""
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(20261005)
    for _ in range(25):
        n = int(rng.integers(20, 300))
        y = (rng.random(n) < rng.uniform(0.05, 0.6)).astype(int)
        if y.sum() == 0 or y.sum() == n:
            continue
        s = rng.random(n) + rng.uniform(0.0, 1.0) * y
        assert average_precision(y, s) == pytest.approx(average_precision_score(y, s), abs=1e-12)


def test_a_perfect_ranker_scores_one() -> None:
    y = np.array([1, 1, 1, 0, 0, 0])
    assert average_precision(y, np.array([6.0, 5, 4, 3, 2, 1])) == pytest.approx(1.0)


def test_a_constant_score_returns_the_base_rate() -> None:
    """Ties must be resolved pessimistically, not in the metric's favour.

    An optimistic tie-break would let a score that carries no information at
    all beat the base rate, which is exactly the illusion AP exists to avoid
    under class imbalance.
    """
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    assert average_precision(y, np.ones(10)) == pytest.approx(0.2)


def test_the_worst_possible_ranking_scores_poorly() -> None:
    y = np.array([1, 1, 0, 0, 0, 0])
    worst = average_precision(y, np.array([1.0, 2, 3, 4, 5, 6]))
    best = average_precision(y, np.array([6.0, 5, 4, 3, 2, 1]))
    assert worst < y.mean() < best


def test_average_precision_is_nan_without_positives() -> None:
    assert np.isnan(average_precision(np.zeros(5, dtype=int), np.arange(5.0)))


def test_nan_scores_are_dropped_not_ranked_last() -> None:
    """A design with no score is unmeasured, not a predicted negative."""
    y = np.array([1, 1, 0, 0])
    s = np.array([2.0, 1.0, np.nan, np.nan])
    assert average_precision(y, s) == pytest.approx(1.0)


def test_mean_within_target_ap_skips_targets_with_no_positives() -> None:
    """A target nobody bound cannot score a ranker, so it contributes nothing."""
    y = np.array([1, 0, 1, 0, 0, 0])
    s = np.array([2.0, 1.0, 2.0, 1.0, 5.0, 4.0])
    g = np.array(["a", "a", "b", "b", "c", "c"])
    both = mean_within_target_ap(y, s, g)
    assert both == pytest.approx(1.0), "targets a and b are ranked perfectly; c is dropped"


def test_mean_within_target_ap_is_not_the_pooled_value() -> None:
    """Pooling across targets of different difficulty is the study's standing error."""
    rng = np.random.default_rng(7)
    y, s, g = [], [], []
    for t, rate in enumerate([0.05, 0.8]):
        for _ in range(200):
            label = int(rng.random() < rate)
            y.append(label)
            s.append(rng.normal(0.5, 0.2) + 0.6 * t)  # target offset, no real signal
            g.append(f"T{t}")
    y, s, g = np.array(y), np.array(s), np.array(g)
    within = mean_within_target_ap(y, s, g)
    pooled = average_precision(y, s)
    assert pooled > within + 0.1, "the pooled value is inflated by target difficulty"


def test_precision_at_k_counts_the_top_k() -> None:
    y = np.array([1, 0, 1, 0, 0])
    s = np.array([5.0, 4, 3, 2, 1])
    assert precision_at_k(y, s, 1) == pytest.approx(1.0)
    assert precision_at_k(y, s, 2) == pytest.approx(0.5)
    assert precision_at_k(y, s, 4) == pytest.approx(0.5)


def test_precision_at_k_is_nan_when_there_are_fewer_than_k() -> None:
    """Reporting precision over a shorter list would not be precision@k."""
    assert np.isnan(precision_at_k(np.array([1, 0]), np.array([2.0, 1.0]), 5))


def test_mean_within_target_precision_at_k_reports_how_many_targets_qualified() -> None:
    """precision@50 over three targets is a different claim from over fourteen."""
    y = np.concatenate([np.ones(5, dtype=int), np.zeros(5, dtype=int), np.array([1, 0])])
    s = np.concatenate([np.arange(10.0)[::-1], np.array([1.0, 0.0])])
    g = np.array(["big"] * 10 + ["small"] * 2)
    value, n_targets = mean_within_target_precision_at_k(y, s, g, k=5)
    assert n_targets == 1, "the two-design target cannot support precision@5"
    assert value == pytest.approx(1.0)


def test_precision_at_k_and_ap_disagree_about_where_a_ranker_is_good() -> None:
    """The reason study 3 reports both: they answer different questions.

    `top_heavy` puts every positive in the first few places and then ranks the
    remainder badly; `even` spreads them. Precision@k at small k prefers the
    first, and that is the quantity a twenty-design submission cares about.
    """
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
    top_heavy = np.array([10.0, 9, 8, 1, 2, 3, 4, 5, 6, 7])
    even = np.array([10.0, 7, 4, 9, 8, 6, 5, 3, 2, 1])
    assert precision_at_k(y, top_heavy, 3) > precision_at_k(y, even, 3)
