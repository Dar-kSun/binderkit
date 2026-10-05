"""Retrospective calibration study (docs/SPEC.md section 8).

The question: **do the in-silico metrics everyone ranks on actually predict
experimental binding?**

Data: the Anthropic de novo binder release (CC BY 4.0), 1,440 designs against
16 targets, co-folded by ten predictors and measured at two independent CROs.
`design_summary.csv` gives one row per design with `ipsae_min_<pred>` and
`sc_dockq_<pred>` for each predictor plus the adjudicated `binder_final` label.

Statistical discipline, in the order it matters:

1. **Grouped everything.** Designs against one target are not independent:
   measured binder rates in this dataset run from 0% (MBP) to 80% (TREM2). Both
   the cross-validation and the bootstrap resample *targets*, never designs. A
   design-level split leaks the target's difficulty into the test fold.
2. **Pooled vs within-target AUROC are reported separately.** Pooled AUROC is
   inflated by between-target structure: easy targets have both higher ipSAE and
   higher binder rates, so a metric gets credit for ranking targets rather than
   designs. Within-target AUROC answers the question a design campaign actually
   asks - "given this target, can the metric rank my designs?" - and it is the
   honest headline.
3. **A deliberately trivial baseline.** `binder_length` separates binders from
   non-binders by half a residue in this data, so it is a genuine floor.
4. **Calibration**, not just ranking: a reliability curve and expected
   calibration error for any score used as a probability.

Run with `binderkit study retrospective`.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

HF_BASE = "https://huggingface.co/datasets/Anthropic/claude-protein-binder-design/resolve/main/data"
SUMMARY_PATH = "tables/design_summary.csv"

HERE = Path(__file__).parent
CACHE = Path("work/hf_cache")
OUT = HERE
FIG_DIR = HERE / "figures"

#: The ten co-fold predictors in the release.
PREDICTORS = (
    "ptxv2",
    "afm3",
    "boltz2",
    "chai1",
    "of3",
    "odde",
    "ef2fast",
    "ef2full",
    "rf3",
    "af3of3",
)

LABEL = "binder_final"
GROUP = "target"
RNG_SEED = 0
N_BOOT = 2000

#: Kyte-Doolittle, duplicated from binderkit.metrics so the study can run
#: standalone against a bare checkout.
_KD = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def fetch_summary(force: bool = False) -> Path:
    """Download and cache `design_summary.csv`. Licence: CC BY 4.0."""
    dest = CACHE / SUMMARY_PATH
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        log.info("cache hit %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF_BASE}/{SUMMARY_PATH}"
    log.info("downloading %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "binderkit-study/0.1"})
    with urllib.request.urlopen(req, timeout=600) as resp:  # noqa: S310
        dest.write_bytes(resp.read())
    return dest


def build_table(path: Path) -> pd.DataFrame:
    """One row per labelled design, with features, label and group."""
    df = pd.read_csv(path, low_memory=False)
    before = len(df)
    df = df[df[LABEL].notna()].copy()
    df[LABEL] = df[LABEL].astype(bool)
    log.info("labelled %d of %d designs (%d unlabelled dropped)", len(df), before, before - len(df))

    # Derived cross-predictor features. Agreement between independent predictors
    # is a different kind of evidence from any single predictor's confidence.
    ipsae_cols = [f"ipsae_min_{p}" for p in PREDICTORS if f"ipsae_min_{p}" in df.columns]
    dockq_cols = [f"sc_dockq_{p}" for p in PREDICTORS if f"sc_dockq_{p}" in df.columns]
    df["ipsae_mean"] = df[ipsae_cols].mean(axis=1)
    df["ipsae_median"] = df[ipsae_cols].median(axis=1)
    df["ipsae_min_across"] = df[ipsae_cols].min(axis=1)
    df["ipsae_max_across"] = df[ipsae_cols].max(axis=1)
    df["ipsae_std_across"] = df[ipsae_cols].std(axis=1)
    #: Fraction of predictors clearing 0.5, i.e. a consensus vote.
    df["ipsae_frac_above_0.5"] = (df[ipsae_cols] > 0.5).mean(axis=1)
    df["dockq_mean"] = df[dockq_cols].mean(axis=1)

    # Trivial baselines.
    df["hydrophobic_fraction"] = df["sequence"].map(
        lambda s: sum(1 for a in str(s) if a in "VILFWYM") / max(len(str(s)), 1)
    )
    df["mean_hydropathy"] = df["sequence"].map(
        lambda s: float(np.mean([_KD.get(a, 0.0) for a in str(s)])) if isinstance(s, str) else 0.0
    )
    return df


def feature_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map feature column -> short family label, for grouping in the report."""
    feats: dict[str, str] = {}
    for p in PREDICTORS:
        if f"ipsae_min_{p}" in df.columns:
            feats[f"ipsae_min_{p}"] = "single-predictor ipSAE"
        if f"sc_dockq_{p}" in df.columns:
            feats[f"sc_dockq_{p}"] = "single-predictor DockQ"
    for c in (
        "ipsae_mean",
        "ipsae_median",
        "ipsae_min_across",
        "ipsae_max_across",
        "ipsae_frac_above_0.5",
        "dockq_mean",
    ):
        feats[c] = "cross-predictor consensus"
    feats["ipsae_std_across"] = "cross-predictor disagreement"
    feats["binder_length"] = "trivial baseline"
    feats["hydrophobic_fraction"] = "trivial baseline"
    feats["mean_hydropathy"] = "trivial baseline"
    return {k: v for k, v in feats.items() if k in df.columns}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC, NaN if only one class is present."""
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _auprc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def grouped_bootstrap_ci(
    y: np.ndarray,
    s: np.ndarray,
    groups: np.ndarray,
    stat=_auroc,  # noqa: ANN001
    n_boot: int = N_BOOT,
    seed: int = RNG_SEED,
) -> tuple[float, float, float]:
    """Point estimate and 95% CI, resampling **targets** with replacement.

    Resampling designs would understate the uncertainty, because designs within
    a target share a difficulty that the metric partly reads off. Resampling
    whole targets is the correct unit of independence here.
    """
    point = stat(y, s)
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in uniq}

    vals: list[float] = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index_by_group[g] for g in picked])
        v = stat(y[idx], s[idx])
        if v == v:  # drop replicates that lost a class
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def within_target_auroc(
    y: np.ndarray, s: np.ndarray, groups: np.ndarray
) -> tuple[float, dict[str, float], dict[str, int]]:
    """Mean AUROC computed **inside** each target, plus the per-target values.

    Targets with only one outcome class contribute no AUROC and are reported as
    excluded rather than silently dropped or counted as 0.5.
    """
    per: dict[str, float] = {}
    n_by: dict[str, int] = {}
    for g in np.unique(groups):
        m = groups == g
        val = _auroc(y[m], s[m])
        n_by[str(g)] = int(m.sum())
        if val == val:
            per[str(g)] = val
    mean = float(np.mean(list(per.values()))) if per else float("nan")
    return mean, per, n_by


def expected_calibration_error(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10
) -> tuple[float, list[dict[str, float]]]:
    """ECE and the reliability curve, equal-width bins on predicted probability."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not m.any():
            continue
        conf = float(p[m].mean())
        obs = float(y[m].mean())
        w = float(m.mean())
        ece += w * abs(conf - obs)
        rows.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "n": int(m.sum()),
                "mean_predicted": conf,
                "observed_rate": obs,
            }
        )
    return float(ece), rows


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


def leave_one_target_out(
    df: pd.DataFrame, features: list[str], seed: int = RNG_SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Leave-one-target-out logistic regression.

    Returns out-of-fold probabilities, the labels, the groups, and the feature
    list used. Every test target is unseen in training, which is the only split
    that answers "would this have helped on a target we had not measured?".
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = df[features].to_numpy(dtype=float)
    y = df[LABEL].to_numpy(dtype=int)
    groups = df[GROUP].to_numpy()

    oof = np.full(len(df), np.nan)
    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
        )
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    keep = ~np.isnan(oof)
    return oof[keep], y[keep], groups[keep], features


def no_target_appears_in_both_folds(df: pd.DataFrame) -> bool:
    """Assertion helper for the grouped-split test (docs/SPEC.md section 13)."""
    from sklearn.model_selection import LeaveOneGroupOut

    groups = df[GROUP].to_numpy()
    y = df[LABEL].to_numpy(dtype=int)
    X = np.zeros((len(df), 1))
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        if set(groups[train_idx]) & set(groups[test_idx]):
            return False
    return True


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def make_figures(
    df: pd.DataFrame,
    single: pd.DataFrame,
    reliability: list[dict[str, float]],
    per_target: dict[str, float],
) -> list[str]:
    """Write the report figures. Returns the filenames written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    # 1. Pooled vs within-target AUROC. Show the best performers AND the metrics
    #    where the two disagree most, so the figure cannot imply a single
    #    direction of bias that the data does not support.
    ranked = single.copy()
    ranked["gap"] = ranked["pooled_auroc"] - ranked["within_target_auroc"]
    best = ranked.sort_values("within_target_auroc", ascending=False).head(9)
    widest = ranked.reindex(ranked["gap"].abs().sort_values(ascending=False).index).head(5)
    top = pd.concat([best, widest]).drop_duplicates(subset="feature")
    top = top.sort_values("within_target_auroc", ascending=False)

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ypos = np.arange(len(top))
    ax.barh(
        ypos - 0.2, top["pooled_auroc"], height=0.38, label="pooled across targets", color="#8fb8de"
    )
    ax.barh(
        ypos + 0.2,
        top["within_target_auroc"],
        height=0.38,
        label="mean within target",
        color="#d97757",
    )
    ax.axvline(0.5, color="black", lw=1, ls="--")
    ax.text(0.505, len(top) - 0.4, "chance", fontsize=8, va="top")
    # Annotate the signed gap so the direction is legible metric by metric.
    for i, (_, r) in enumerate(top.iterrows()):
        ax.annotate(
            f"{r['gap']:+.3f}",
            (max(r["pooled_auroc"], r["within_target_auroc"]) + 0.008, i),
            fontsize=7,
            va="center",
            color="#444444",
        )
    ax.set_yticks(ypos)
    ax.set_yticklabels(top["feature"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("AUROC for predicting measured binding")
    ax.set_xlim(0.3, 1.0)
    ax.set_title(
        "Pooled vs within-target AUROC\n"
        "signed gap annotated: the bias runs both ways and is metric-specific",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "auroc_pooled_vs_within.png", dpi=150)
    plt.close(fig)
    written.append("auroc_pooled_vs_within.png")

    # 2. Reliability curve of the leave-one-target-out model.
    if reliability:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        xs = [r["mean_predicted"] for r in reliability]
        ys = [r["observed_rate"] for r in reliability]
        ns = [r["n"] for r in reliability]
        ax.plot([0, 1], [0, 1], ls="--", color="black", lw=1, label="perfect calibration")
        ax.plot(xs, ys, "o-", color="#d97757", label="leave-one-target-out model")
        for x, yv, n in zip(xs, ys, ns, strict=True):
            ax.annotate(str(n), (x, yv), fontsize=7, xytext=(3, -9), textcoords="offset points")
        ax.set_xlabel("mean predicted probability of binding")
        ax.set_ylabel("observed binder fraction")
        ax.set_title("Calibration (bin counts annotated)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "calibration.png", dpi=150)
        plt.close(fig)
        written.append("calibration.png")

    # 3. Per-target AUROC of the best single metric: where it works and fails.
    if per_target:
        items = sorted(per_target.items(), key=lambda kv: kv[1])
        fig, ax = plt.subplots(figsize=(8, 5))
        names = [k for k, _ in items]
        vals = [v for _, v in items]
        colors = ["#c0392b" if v < 0.5 else "#2e7d52" for v in vals]
        ax.barh(np.arange(len(names)), vals, color=colors)
        ax.axvline(0.5, color="black", lw=1, ls="--")
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("within-target AUROC")
        ax.set_xlim(0, 1)
        ax.set_title("Best single metric, per target (red = worse than chance)")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "per_target_auroc.png", dpi=150)
        plt.close(fig)
        written.append("per_target_auroc.png")

    # 4. Binder rate per target, the reason grouping matters.
    rates = df.groupby(GROUP)[LABEL].agg(["mean", "size"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.arange(len(rates)), 100 * rates["mean"], color="#8fb8de")
    ax.set_yticks(np.arange(len(rates)))
    ax.set_yticklabels(
        [f"{i} (n={int(n)})" for i, n in zip(rates.index, rates["size"], strict=True)], fontsize=8
    )
    ax.set_xlabel("measured binder rate (%)")
    ax.set_title("Binder rate varies 0-80% by target: why splits must be grouped")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "binder_rate_by_target.png", dpi=150)
    plt.close(fig)
    written.append("binder_rate_by_target.png")
    return written


def main(force_download: bool = False) -> int:
    """Run the study. Delegates to the report module (imported lazily to avoid a cycle)."""
    from studies.retrospective.report import main as _main

    return _main(force_download=force_download)
