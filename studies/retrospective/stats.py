"""Statistics for the retrospective studies (NEXT_SESSION.md section 2).

Session 1 reported confidence intervals on individual AUROCs but stated its
headline conclusions in terms of *differences* between metrics (+0.028, -0.034)
with no interval on those differences. With a single-AUROC CI spanning
0.671-0.839, a 0.028 gap could easily be noise. Everything here exists to fix
that:

* `paired_bootstrap_difference` resamples targets **once per replicate** and
  scores both metrics on the *same* resample, which removes the shared
  target-draw variance and is the only way a small difference can be resolved
  at this sample size.
* `selection_corrected_bootstrap` re-runs the metric *selection* inside every
  replicate, so the reported interval covers the fact that the headline number
  was the maximum over 30 correlated candidates.
* `within_target_bootstrap` resamples designs inside one target, which is the
  right unit once the target is fixed.

The guiding rule from the brief: any difference whose CI straddles zero is
reported as "cannot be distinguished at this sample size", not hedged into
something that still implies an effect.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

N_BOOT = 2000
RNG_SEED = 0


# --------------------------------------------------------------------------
# Scoring primitives
# --------------------------------------------------------------------------


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC, NaN when it cannot be computed.

    Implemented as the rank-sum (Mann-Whitney U) identity rather than by
    calling `sklearn.metrics.roc_auc_score`. The result is identical, including
    the mid-rank treatment of ties, but it is roughly fifty times faster
    because it skips sklearn's input validation. That matters here: the
    selection-corrected bootstrap alone needs on the order of a million AUROC
    evaluations, which is minutes rather than hours at this speed.

    Designs with a missing score are dropped rather than imputed: several
    predictors failed on a subset of designs (`ipsae_min_afm3` is absent for 90
    of them), and filling those with a constant would invent ties that change
    the ranking.
    """
    ok = ~np.isnan(s)
    if ok.sum() < 2:
        return float("nan")
    y, s = y[ok], s[ok]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Average ranks, so tied scores contribute 0.5 as they should.
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # Resolve ties to their mean rank.
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1

    rank_sum_pos = ranks[y == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def pooled_auroc(y: np.ndarray, s: np.ndarray, groups: np.ndarray) -> float:  # noqa: ARG001
    """AUROC over all designs regardless of target."""
    return auroc(y, s)


def mean_within_target_auroc(y: np.ndarray, s: np.ndarray, groups: np.ndarray) -> float:
    """Mean AUROC computed inside each target.

    Targets with a single outcome class contribute nothing rather than being
    scored 0.5, which would drag the mean toward chance for a reason that has
    nothing to do with the metric.
    """
    vals: list[float] = []
    for g in np.unique(groups):
        m = groups == g
        v = auroc(y[m], s[m])
        if v == v:
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------
# Bootstrap machinery
# --------------------------------------------------------------------------


@dataclass
class DiffResult:
    """A paired difference between two metrics, with its uncertainty."""

    name_a: str
    name_b: str
    stat_a: float
    stat_b: float
    difference: float
    ci_lo: float
    ci_hi: float
    #: Fraction of bootstrap replicates where the difference kept its sign.
    sign_consistency: float
    n_replicates: int

    @property
    def distinguishable(self) -> bool:
        """True when the 95% CI excludes zero."""
        return (self.ci_lo > 0.0) or (self.ci_hi < 0.0)

    def verdict(self) -> str:
        """One sentence stating what the data supports, and nothing more."""
        if self.distinguishable:
            direction = "higher" if self.difference > 0 else "lower"
            return (
                f"`{self.name_a}` is {direction} than `{self.name_b}` by "
                f"{abs(self.difference):.3f} "
                f"(95% CI {self.ci_lo:+.3f} to {self.ci_hi:+.3f}, sign held in "
                f"{100 * self.sign_consistency:.0f}% of replicates)."
            )
        return (
            f"`{self.name_a}` and `{self.name_b}` **cannot be distinguished at this "
            f"sample size**: difference {self.difference:+.3f}, 95% CI "
            f"{self.ci_lo:+.3f} to {self.ci_hi:+.3f}, which straddles zero "
            f"(sign held in only {100 * self.sign_consistency:.0f}% of replicates)."
        )


def _group_index(groups: np.ndarray) -> tuple[np.ndarray, dict]:
    uniq = np.unique(groups)
    return uniq, {g: np.flatnonzero(groups == g) for g in uniq}


def paired_bootstrap_difference(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    groups: np.ndarray,
    *,
    name_a: str = "a",
    name_b: str = "b",
    stat: Callable[[np.ndarray, np.ndarray, np.ndarray], float] = mean_within_target_auroc,
    n_boot: int = N_BOOT,
    seed: int = RNG_SEED,
) -> DiffResult:
    """CI on `stat(a) - stat(b)`, resampling targets once per replicate.

    Both metrics are scored on the identical resample, so the target draw
    cancels. This matters: the marginal CI on either metric alone is far wider
    than the CI on their difference, because they rise and fall together as
    easy or hard targets enter the sample.
    """
    valid = ~(np.isnan(score_a) | np.isnan(score_b))
    y, score_a, score_b, groups = y[valid], score_a[valid], score_b[valid], groups[valid]

    point_a = stat(y, score_a, groups)
    point_b = stat(y, score_b, groups)
    rng = np.random.default_rng(seed)
    uniq, idx_by = _group_index(groups)

    diffs: list[float] = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in picked])
        # Resampled groups must be relabelled, or duplicated targets merge into
        # one group and the within-target statistic is computed on a chimera.
        relabelled = np.concatenate(
            [np.full(len(idx_by[g]), f"{g}#{i}") for i, g in enumerate(picked)]
        )
        va = stat(y[idx], score_a[idx], relabelled)
        vb = stat(y[idx], score_b[idx], relabelled)
        if va == va and vb == vb:
            diffs.append(va - vb)

    if not diffs:
        return DiffResult(
            name_a,
            name_b,
            point_a,
            point_b,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            0,
        )

    arr = np.asarray(diffs)
    point = point_a - point_b
    lo, hi = np.percentile(arr, [2.5, 97.5])
    sign = float(np.mean(np.sign(arr) == np.sign(point))) if point != 0 else 0.0
    return DiffResult(
        name_a,
        name_b,
        point_a,
        point_b,
        float(point),
        float(lo),
        float(hi),
        sign,
        len(arr),
    )


def selection_corrected_bootstrap(
    y: np.ndarray,
    scores: dict[str, np.ndarray],
    groups: np.ndarray,
    *,
    stat: Callable[[np.ndarray, np.ndarray, np.ndarray], float] = mean_within_target_auroc,
    n_boot: int = N_BOOT,
    seed: int = RNG_SEED,
) -> dict[str, float]:
    """Interval for "the best of N metrics", re-selecting inside each replicate.

    The naive interval around the winning metric answers "how uncertain is this
    metric's score", but the reported headline is the *maximum* over many
    correlated candidates, which is upward-biased. Re-selecting the winner
    within each resample makes the interval cover the selection step too, and
    the gap between the naive and corrected point estimates is the size of the
    winner's curse.
    """
    rng = np.random.default_rng(seed)
    uniq, idx_by = _group_index(groups)
    names = list(scores)

    observed = {n: stat(y, scores[n], groups) for n in names}
    observed_best = max(observed, key=lambda n: observed[n] if observed[n] == observed[n] else -1)

    best_vals: list[float] = []
    winners: dict[str, int] = {}
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in picked])
        relabelled = np.concatenate(
            [np.full(len(idx_by[g]), f"{g}#{i}") for i, g in enumerate(picked)]
        )
        vals = {}
        for n in names:
            s = scores[n][idx]
            if np.isnan(s).all():
                continue
            v = stat(y[idx], s, relabelled)
            if v == v:
                vals[n] = v
        if not vals:
            continue
        w = max(vals, key=lambda n: vals[n])
        winners[w] = winners.get(w, 0) + 1
        best_vals.append(vals[w])

    arr = np.asarray(best_vals)
    lo, hi = np.percentile(arr, [2.5, 97.5]) if len(arr) else (float("nan"), float("nan"))
    return {
        "observed_best_metric": observed_best,
        "observed_best_value": observed[observed_best],
        "selection_corrected_mean": float(np.mean(arr)) if len(arr) else float("nan"),
        "selection_corrected_lo": float(lo),
        "selection_corrected_hi": float(hi),
        "n_metrics_screened": len(names),
        "winner_stability": winners.get(observed_best, 0) / max(len(best_vals), 1),
        "n_replicates": len(arr),
        "distinct_winners": len(winners),
    }


def within_target_bootstrap(
    y: np.ndarray, s: np.ndarray, *, n_boot: int = N_BOOT, seed: int = RNG_SEED
) -> tuple[float, float, float]:
    """AUROC CI inside a single target, resampling **designs**.

    Once the target is fixed, designs are the unit of replication, so this is
    the one place a design-level bootstrap is correct.
    """
    point = auroc(y, s)
    if point != point:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = auroc(y[idx], s[idx])
        if v == v:
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


# --------------------------------------------------------------------------
# Calibration (NEXT_SESSION section 2.4)
# --------------------------------------------------------------------------


@dataclass
class CalibrationBin:
    lo: float
    hi: float
    n: int
    mean_predicted: float
    observed: float
    obs_lo: float
    obs_hi: float


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used instead of the normal approximation because the extreme bins have few
    designs and an observed rate of 0 or 1, where the normal interval collapses
    to zero width and lies.
    """
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def calibration_bins(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10, min_n: int = 5
) -> tuple[list[CalibrationBin], list[CalibrationBin]]:
    """Reliability bins with Wilson CIs, split into reportable and too-small.

    Session 1 plotted a bin containing a single design as if it were a point
    estimate. Bins with fewer than `min_n` designs are returned separately so
    they can be excluded from the curve and named in the text instead.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    keep: list[CalibrationBin] = []
    dropped: list[CalibrationBin] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        n = int(m.sum())
        if n == 0:
            continue
        k = int(y[m].sum())
        olo, ohi = _wilson(k, n)
        b = CalibrationBin(float(lo), float(hi), n, float(p[m].mean()), k / n, olo, ohi)
        (keep if n >= min_n else dropped).append(b)
    return keep, dropped


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE over all bins, weighted by bin occupancy."""
    keep, dropped = calibration_bins(y, p, n_bins, min_n=0)
    total = len(y)
    return float(sum(b.n / total * abs(b.mean_predicted - b.observed) for b in keep + dropped))


def monotonicity(bins: Sequence[CalibrationBin]) -> tuple[float, float]:
    """Spearman correlation of predicted vs observed across bins.

    ECE can look respectable while the curve is non-monotonic, which is the
    failure that matters: if the observed rate falls as predictions rise, a
    confident prediction is worse than a middling one, and confident
    predictions are exactly the ones acted on.
    """
    from scipy.stats import spearmanr

    if len(bins) < 3:
        return float("nan"), float("nan")
    r, pval = spearmanr([b.mean_predicted for b in bins], [b.observed for b in bins])
    return float(r), float(pval)


def top_decile_rate(y: np.ndarray, s: np.ndarray) -> tuple[float, int]:
    """Observed binder rate among the top 10% of designs by score.

    The number a user actually experiences when they rank and take the top
    slice, which is what this pipeline does.
    """
    n = max(1, int(round(0.1 * len(s))))
    order = np.argsort(-s)[:n]
    return float(y[order].mean()), n


# --------------------------------------------------------------------------
# Conditional signal (NEXT_SESSION section 2.4, disagreement)
# --------------------------------------------------------------------------


def residualise(x: np.ndarray, on: np.ndarray) -> np.ndarray:
    """Residuals of `x` after linear regression on `on`.

    Answers "what is left in this metric once the other one is accounted for".
    If a metric only looks informative because it is a proxy for another, its
    residual carries no signal.
    """
    ok = ~(np.isnan(x) | np.isnan(on))
    out = np.full_like(x, np.nan, dtype=float)
    if ok.sum() < 3:
        return out
    a = np.vstack([on[ok], np.ones(ok.sum())]).T
    coef, *_ = np.linalg.lstsq(a, x[ok], rcond=None)
    out[ok] = x[ok] - (a @ coef)
    return out


def loto_model_scores(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = RNG_SEED
) -> np.ndarray:
    """Out-of-fold probabilities from leave-one-target-out logistic regression.

    Returns NaN for rows whose fold could not be fitted, so callers can mask
    rather than silently receive a default.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    oof = np.full(len(y), np.nan)
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=seed),
        )
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return oof
