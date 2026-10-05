"""Generate ``portfolio/numbers.json`` from the canonical study outputs.

``portfolio/`` is a derived view. Its one hard rule (docs/SPEC.md section 18.3) is
that **no number may appear in it that is not already in ``studies/``**. This
script is how that rule is enforced rather than promised: every figure the page
quotes is resolved here from a study output file, and
``tests/test_portfolio.py`` re-resolves each one and fails if they disagree, or
if ``index.html`` displays a value this file does not contain.

Run it with ``python -m portfolio.build`` after re-running either study.

Source syntax
-------------
``path#a.b.c``
    A dotted key path into a JSON file. ``a.b[2].c`` indexes lists.
``path#where=value:column``
    One cell of a CSV: the row whose ``where`` column equals ``value``.
``path#text:substring``
    A fact that exists only as prose in a report. The value is the substring,
    and resolution asserts it appears verbatim in that file. Used only where
    no machine-readable source exists.
``ratio:specA|specB``
    The quotient of two other sources, for figures like enrichment that are
    defined as a ratio and stored nowhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "numbers.json"

CAL = "studies/retrospective/calibration.json"
PER_TARGET = "studies/retrospective/per_target_results.csv"
GEOM = "studies/interface_geometry/summary.json"
GEOM_REPORT = "studies/interface_geometry/REPORT.md"

#: key, human label, source spec, display format.
#:
#: Formats: ``f3`` three decimals; ``s3`` signed three decimals; ``pct1`` a
#: percentage to one decimal; ``int`` a plain count; ``x2`` a multiplier.
SPEC: list[tuple[str, str, str, str]] = [
    # ---- study 1: scope -------------------------------------------------
    ("study1.n_designs", "designs with a measured outcome", f"{CAL}#n_designs", "int"),
    ("study1.n_targets", "targets", f"{CAL}#n_targets", "int"),
    ("study1.n_binders", "designs that bound", f"{CAL}#n_binders", "int"),
    ("study1.base_rate", "binder rate", f"{CAL}#base_rate", "pct1"),
    # ---- study 1: the ranking signal ------------------------------------
    (
        "study1.headline_auroc",
        "co-folding confidence (ipsae_mean), within-target AUROC",
        f"{CAL}#headline_within_target_auroc",
        "f3",
    ),
    (
        "study1.best_single_auroc",
        "best single predictor (ipsae_min_ptxv2)",
        f"{CAL}#paired.ipsae_mean_vs_ipsae_min_ptxv2.stat_b",
        "f3",
    ),
    (
        "study1.loto_model_auroc",
        "model trained on all score columns",
        f"{CAL}#paired.ipsae_mean_vs_LOTO_model.stat_b",
        "f3",
    ),
    (
        "study1.baseline_auroc",
        "trivial baseline (hydrophobic_fraction)",
        f"{CAL}#paired.ipsae_mean_vs_hydrophobic_fraction.stat_b",
        "f3",
    ),
    # ---- study 1: the three paired differences --------------------------
    (
        "study1.vs_baseline.diff",
        "confidence minus trivial baseline",
        f"{CAL}#paired.ipsae_mean_vs_hydrophobic_fraction.difference",
        "s3",
    ),
    (
        "study1.vs_baseline.ci_lo",
        "",
        f"{CAL}#paired.ipsae_mean_vs_hydrophobic_fraction.ci_lo",
        "s3",
    ),
    (
        "study1.vs_baseline.ci_hi",
        "",
        f"{CAL}#paired.ipsae_mean_vs_hydrophobic_fraction.ci_hi",
        "s3",
    ),
    (
        "study1.vs_baseline.sign_held",
        "replicates keeping the sign",
        f"{CAL}#paired.ipsae_mean_vs_hydrophobic_fraction.sign_consistency",
        "pct0",
    ),
    (
        "study1.vs_best_single.diff",
        "averaging ten predictors minus the best single one",
        f"{CAL}#paired.ipsae_mean_vs_ipsae_min_ptxv2.difference",
        "s3",
    ),
    (
        "study1.vs_best_single.ci_lo",
        "",
        f"{CAL}#paired.ipsae_mean_vs_ipsae_min_ptxv2.ci_lo",
        "s3",
    ),
    (
        "study1.vs_best_single.ci_hi",
        "",
        f"{CAL}#paired.ipsae_mean_vs_ipsae_min_ptxv2.ci_hi",
        "s3",
    ),
    (
        "study1.vs_best_single.sign_held",
        "replicates keeping the sign",
        f"{CAL}#paired.ipsae_mean_vs_ipsae_min_ptxv2.sign_consistency",
        "pct0",
    ),
    (
        "study1.vs_loto_model.diff",
        "plain average minus a fitted model",
        f"{CAL}#paired.ipsae_mean_vs_LOTO_model.difference",
        "s3",
    ),
    ("study1.vs_loto_model.ci_lo", "", f"{CAL}#paired.ipsae_mean_vs_LOTO_model.ci_lo", "s3"),
    ("study1.vs_loto_model.ci_hi", "", f"{CAL}#paired.ipsae_mean_vs_LOTO_model.ci_hi", "s3"),
    (
        "study1.vs_loto_model.sign_held",
        "replicates keeping the sign",
        f"{CAL}#paired.ipsae_mean_vs_LOTO_model.sign_consistency",
        "pct0",
    ),
    # ---- study 1: the selection correction ------------------------------
    (
        "study1.selection.ci_lo",
        "selection-corrected interval, low",
        f"{CAL}#selection.selection_corrected_lo",
        "f3",
    ),
    (
        "study1.selection.ci_hi",
        "selection-corrected interval, high",
        f"{CAL}#selection.selection_corrected_hi",
        "f3",
    ),
    (
        "study1.selection.n_screened",
        "metrics screened",
        f"{CAL}#selection.n_metrics_screened",
        "int",
    ),
    (
        "study1.selection.winner_stability",
        "replicates won by the nominal winner",
        f"{CAL}#selection.winner_stability",
        "pct0",
    ),
    (
        "study1.selection.distinct_winners",
        "distinct metrics that won at least once",
        f"{CAL}#selection.distinct_winners",
        "int",
    ),
    # ---- calibration ----------------------------------------------------
    ("calibration.ece", "expected calibration error", f"{CAL}#calibration_ece", "f3"),
    (
        "calibration.bin_0.7_0.8.n",
        "designs predicted 0.7-0.8",
        f"{CAL}#calibration_bins[lo=0.7].n",
        "int",
    ),
    (
        "calibration.bin_0.7_0.8.predicted",
        "mean predicted in that bin",
        f"{CAL}#calibration_bins[lo=0.7].mean_predicted",
        "f3",
    ),
    (
        "calibration.bin_0.7_0.8.observed",
        "observed binder rate in that bin",
        f"{CAL}#calibration_bins[lo=0.7].observed",
        "pct1",
    ),
    ("calibration.bin_0.7_0.8.obs_lo", "", f"{CAL}#calibration_bins[lo=0.7].obs_lo", "pct1"),
    ("calibration.bin_0.7_0.8.obs_hi", "", f"{CAL}#calibration_bins[lo=0.7].obs_hi", "pct1"),
    (
        "calibration.top_decile_rate",
        "binder rate in the top-scoring decile",
        f"{CAL}#top_decile_rate",
        "pct1",
    ),
    ("calibration.top_decile_n", "designs in the top decile", f"{CAL}#top_decile_n", "int"),
    # ---- operating point -------------------------------------------------
    ("operating.threshold", "threshold on ipsae_mean", f"{CAL}#operating_point.threshold", "f3"),
    ("operating.sensitivity", "sensitivity", f"{CAL}#operating_point.sensitivity", "pct1"),
    ("operating.specificity", "specificity", f"{CAL}#operating_point.specificity", "pct1"),
    ("operating.precision", "precision", f"{CAL}#operating_point.precision", "pct1"),
    ("operating.n_selected", "designs selected", f"{CAL}#operating_point.n_selected", "int"),
    (
        "operating.enrichment",
        "enrichment over the base rate",
        f"ratio:{CAL}#operating_point.precision|{CAL}#base_rate",
        "x2",
    ),
    # ---- per target ------------------------------------------------------
    ("per_target.egfr_auroc", "EGFR within-target AUROC", f"{PER_TARGET}#target=EGFR:auroc", "f3"),
    ("per_target.egfr_ci_lo", "", f"{PER_TARGET}#target=EGFR:lo", "f3"),
    ("per_target.egfr_ci_hi", "", f"{PER_TARGET}#target=EGFR:hi", "f3"),
    ("per_target.egfr_n", "EGFR designs", f"{PER_TARGET}#target=EGFR:n", "int"),
    (
        "per_target.egfr_binders",
        "EGFR designs that bound",
        f"{PER_TARGET}#target=EGFR:binders",
        "int",
    ),
    (
        "per_target.egfr_base_rate",
        "EGFR binder rate in the campaign",
        f"ratio:{PER_TARGET}#target=EGFR:binders|{PER_TARGET}#target=EGFR:n",
        "pct1",
    ),
    (
        "per_target.n_ci_includes_chance",
        "targets whose interval includes chance",
        f"{CAL}#n_targets_ci_includes_chance",
        "int",
    ),
    # ---- the label-noise ceiling ----------------------------------------
    (
        "ceiling.agreement",
        "agreement between the two wet-lab assays",
        f"{CAL}#vendor_agreement.agreement",
        "pct1",
    ),
    (
        "ceiling.n_measured_by_both",
        "designs measured by both vendors",
        f"{CAL}#vendor_agreement.n_measured_by_both",
        "int",
    ),
    ("ceiling.both_bind", "called a binder by both", f"{CAL}#vendor_agreement.both_bind", "int"),
    (
        "ceiling.neither_bind",
        "called a non-binder by both",
        f"{CAL}#vendor_agreement.neither_bind",
        "int",
    ),
    (
        "ceiling.adaptyv_only",
        "binder at one vendor only",
        f"{CAL}#vendor_agreement.adaptyv_only_bind",
        "int",
    ),
    (
        "ceiling.twist_only",
        "binder at the other vendor only",
        f"{CAL}#vendor_agreement.twist_only_bind",
        "int",
    ),
    (
        "ceiling.n_discordant",
        "designs the two assays disagree on",
        f"{CAL}#vendor_agreement.n_discordant",
        "int",
    ),
    # ---- study 2 ----------------------------------------------------------
    ("study2.n_designs", "designs with a design model and an outcome", f"{GEOM}#n_designs", "int"),
    ("study2.base_rate", "binder rate", f"{GEOM}#base_rate", "pct1"),
    ("study2.n_metrics", "geometry metrics tested", f"{GEOM}#n_metrics_tested", "int"),
    ("study2.n_helping", "metrics that improved prediction", f"{GEOM}#n_metrics_helping", "int"),
    ("study2.n_hurting", "metrics that measurably hurt", f"{GEOM}#n_metrics_hurting", "int"),
    (
        "study2.reference_auroc",
        "confidence alone, on this subset",
        f"{GEOM}#reference_within_target_auroc",
        "f3",
    ),
    (
        "study2.best_geometry_auroc",
        "best geometry metric alone",
        f"{GEOM}#best_geometry_auroc",
        "f3",
    ),
    (
        "study2.all_geometry.with",
        "confidence plus all 13",
        f"{GEOM}#all_geometry_combined.with_geometry",
        "f3",
    ),
    (
        "study2.all_geometry.without",
        "confidence alone",
        f"{GEOM}#all_geometry_combined.confidence_alone",
        "f3",
    ),
    ("study2.all_geometry.diff", "difference", f"{GEOM}#all_geometry_combined.difference", "s3"),
    ("study2.all_geometry.ci_lo", "", f"{GEOM}#all_geometry_combined.ci_lo", "s3"),
    ("study2.all_geometry.ci_hi", "", f"{GEOM}#all_geometry_combined.ci_hi", "s3"),
    (
        "study2.all_geometry.sign_held",
        "replicates keeping the sign",
        f"{GEOM}#all_geometry_combined.sign_consistency",
        "pct0",
    ),
    # ---- the withdrawn pooled comparison ---------------------------------
    (
        "pooled.contacts.pooled",
        "contacts, pooled across targets",
        f"{GEOM}#group_differences.n_atom_contacts.pooled",
        "s1",
    ),
    (
        "pooled.contacts.within",
        "contacts, within target",
        f"{GEOM}#group_differences.n_atom_contacts.within_target",
        "s1",
    ),
    (
        "pooled.contacts.positive",
        "targets where binders had more",
        f"{GEOM}#group_differences.n_atom_contacts.n_groups_positive",
        "int",
    ),
    (
        "pooled.bsa_total.pooled",
        "buried surface, pooled",
        f"{GEOM}#group_differences.bsa_total.pooled",
        "s1",
    ),
    (
        "pooled.bsa_total.within",
        "buried surface, within target",
        f"{GEOM}#group_differences.bsa_total.within_target",
        "s1",
    ),
    (
        "pooled.bsa_total.positive",
        "targets where binders buried more",
        f"{GEOM}#group_differences.bsa_total.n_groups_positive",
        "int",
    ),
    (
        "pooled.n_groups",
        "targets with both outcomes present",
        f"{GEOM}#group_differences.bsa_total.n_groups",
        "int",
    ),
    # ---- validation of the geometry code ---------------------------------
    (
        "validation.n_compared",
        "designs with a published reference value",
        f"{GEOM}#validation.n_compared",
        "int",
    ),
    (
        "validation.epitope_exact",
        "exact matches on epitope residues",
        f"{GEOM}#validation.epitope_residues.exact_matches",
        "int",
    ),
    (
        "validation.paratope_exact",
        "exact matches on paratope residues",
        f"{GEOM}#validation.paratope_residues.exact_matches",
        "int",
    ),
    (
        "validation.contacts_exact",
        "exact matches on atom contacts",
        f"{GEOM}#validation.atom_contacts.exact_matches",
        "int",
    ),
    (
        "validation.sidechain_n",
        "design models carrying binder side chains",
        f"{GEOM}#sidechain_subset.n",
        "int",
    ),
    (
        "validation.before_bugfix",
        "agreement before the non-protein-chain bug was fixed",
        f"{GEOM_REPORT}#text:agreement from 95.5% to 100%",
        "text",
    ),
    (
        "validation.bug_factor",
        "how far the bug inflated the worst designs",
        f"{GEOM_REPORT}#text:by a factor of eighty",
        "text",
    ),
]


def _select(items: list, predicate: str) -> Any:
    """Pick one list element by a ``key=value`` predicate, or by index.

    Selecting the 0.7-0.8 reliability bin by position breaks silently the
    moment a bin is added or dropped, so bins are selected by their own
    bounds instead.
    """
    if "=" in predicate:
        key, _, want = predicate.partition("=")
        for item in items:
            if f"{item[key]:g}" == want:
                return item
        raise ValueError(f"no element with {key} == {want}")
    return items[int(predicate)]


def _dig(obj: Any, dotted: str) -> Any:
    # Split on dots that are not inside [...] so predicates may contain them.
    parts, depth, buf = [], 0, ""
    for ch in dotted:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "." and depth == 0:
            parts.append(buf)
            buf = ""
        else:
            buf += ch
    parts.append(buf)

    for part in parts:
        if part.endswith("]") and "[" in part:
            name, _, pred = part.partition("[")
            if name:
                obj = obj[name]
            obj = _select(obj, pred.rstrip("]"))
        else:
            obj = obj[part]
    return obj


def resolve(spec: str) -> Any:
    """Read one value out of a ``studies/`` output file. See the module docstring."""
    if spec.startswith("ratio:"):
        a, _, b = spec[len("ratio:") :].partition("|")
        return resolve(a) / resolve(b)

    rel, _, loc = spec.partition("#")
    path = ROOT / rel
    if not path.is_file():
        raise FileNotFoundError(f"{rel} is missing; re-run the study that writes it")

    if loc.startswith("text:"):
        needle = loc[len("text:") :]
        if needle not in path.read_text(encoding="utf-8"):
            raise ValueError(f"{rel} no longer contains {needle!r}")
        return needle

    if rel.endswith(".json"):
        return _dig(json.loads(path.read_text(encoding="utf-8")), loc)

    if rel.endswith(".csv"):
        import csv

        where, _, column = loc.partition(":")
        col, _, want = where.partition("=")
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row[col] == want:
                    raw = row[column]
                    return int(raw) if raw.isdigit() else float(raw)
        raise ValueError(f"{rel} has no row where {col} == {want}")

    raise ValueError(f"cannot resolve {spec}")


def display(value: Any, fmt: str) -> str:
    """Render a value the one way the page is allowed to show it."""
    if fmt == "text":
        return str(value)
    if fmt == "int":
        return f"{int(value)}"
    if fmt == "f3":
        return f"{value:.3f}"
    if fmt == "s3":
        return f"{value:+.3f}"
    if fmt == "s1":
        return f"{value:+.1f}"
    if fmt == "pct1":
        return f"{100 * value:.1f}%"
    if fmt == "pct0":
        return f"{100 * value:.0f}%"
    if fmt == "x2":
        return f"{value:.2f}x"
    raise ValueError(f"unknown format {fmt}")


def build() -> dict:
    """Resolve every entry in :data:`SPEC` into the published payload."""
    numbers = {}
    for key, label, source, fmt in SPEC:
        value = resolve(source)
        numbers[key] = {
            "value": value,
            "display": display(value, fmt),
            "label": label,
            "source": source,
        }
    return {
        "_README": (
            "Generated by portfolio/build.py from studies/. Do not edit by hand. "
            "Every entry's source field names the file and key it came from, and "
            "tests/test_portfolio.py re-resolves all of them."
        ),
        "numbers": numbers,
    }


def main() -> int:
    payload = build()
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} with {len(payload['numbers'])} numbers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
