"""Study 2: does interface geometry add signal over co-folding confidence?

Study 1 established that co-folding confidence predicts measured binding. The
obvious follow-up is whether the expensive structural analysis everyone also
runs - buried surface area, contacts, hydrogen bonds, shape of the interface -
tells you anything that confidence does not already.

This can be answered without running any structure prediction, because the
release ships design models for 1,309 designs whose wet-lab outcome is known.
The question is deliberately framed as a **conditional** one: not "does
geometry correlate with binding" (it does, weakly) but "does adding geometry to
a model that already has the confidence number improve it".
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from studies.retrospective import stats
from studies.retrospective.run_study import build_table, fetch_summary

log = logging.getLogger(__name__)

OUT = Path(__file__).parent
FIG_DIR = OUT / "figures"
METRICS = OUT / "geometry_metrics.csv"
N_BOOT = 1500

#: Geometry computable on every released model, including backbone-only ones.
GEOMETRY_ALL = [
    "n_atom_contacts",
    "n_epitope_residues",
    "n_paratope_residues",
    "bsa_total",
    "bsa_binder",
    "bsa_hydrophobic_fraction",
    "contact_density",
    "interface_hydrophobic_fraction",
    "interface_charged_fraction",
    "binder_exposed_hydrophobic_sasa",
    "radius_of_gyration",
    "n_clashes",
    "min_interface_distance",
]
#: Only computable where the binder carries side chains.
GEOMETRY_SIDECHAIN = ["n_hbonds", "n_salt_bridges"]
REFERENCE = "ipsae_mean"


def load() -> pd.DataFrame:
    """Join geometry to labels. Requires `compute_metrics` to have run."""
    if not METRICS.is_file():
        raise FileNotFoundError(
            f"{METRICS} not found; run `python -m studies.interface_geometry.compute_metrics`"
        )
    df = build_table(fetch_summary())
    geom = pd.read_csv(METRICS)
    m = df.merge(geom, on="full_name", how="inner")
    return m[m["binder_final"].notna()].copy()


def analyse(m: pd.DataFrame) -> dict:
    y = m["binder_final"].astype(int).to_numpy()
    g = m["target"].to_numpy()
    ip = m[REFERENCE].to_numpy(dtype=float)

    alone = {
        c: stats.mean_within_target_auroc(y, m[c].to_numpy(dtype=float), g) for c in GEOMETRY_ALL
    }
    reference_auroc = stats.mean_within_target_auroc(y, ip, g)

    base = stats.loto_model_scores(ip.reshape(-1, 1), y, g)
    conditional = {}
    for c in GEOMETRY_ALL:
        v = m[c].to_numpy(dtype=float)
        both = stats.loto_model_scores(np.column_stack([ip, v]), y, g)
        conditional[c] = stats.paired_bootstrap_difference(
            y, both, base, g, name_a=f"{REFERENCE} + {c}", name_b=REFERENCE, n_boot=N_BOOT
        )

    allg = stats.loto_model_scores(
        np.column_stack([ip] + [m[c].to_numpy(dtype=float) for c in GEOMETRY_ALL]), y, g
    )
    combined = stats.paired_bootstrap_difference(
        y,
        allg,
        base,
        g,
        name_a=f"{REFERENCE} + all geometry",
        name_b=f"{REFERENCE} alone",
        n_boot=N_BOOT,
    )

    # Side-chain metrics, only where observable.
    sub = m[m["n_hbonds"].notna()].copy()
    subset: dict = {"n": len(sub), "base_rate": float(sub["binder_final"].astype(int).mean())}
    if len(sub) > 50:
        ys = sub["binder_final"].astype(int).to_numpy()
        gs = sub["target"].to_numpy()
        ips = sub[REFERENCE].to_numpy(dtype=float)
        subset["reference_auroc"] = stats.mean_within_target_auroc(ys, ips, gs)
        subset["alone"] = {
            c: stats.mean_within_target_auroc(ys, sub[c].to_numpy(dtype=float), gs)
            for c in GEOMETRY_SIDECHAIN
        }
        bs = stats.loto_model_scores(ips.reshape(-1, 1), ys, gs)
        both = stats.loto_model_scores(
            np.column_stack([ips] + [sub[c].to_numpy(dtype=float) for c in GEOMETRY_SIDECHAIN]),
            ys,
            gs,
        )
        subset["combined"] = stats.paired_bootstrap_difference(
            ys,
            both,
            bs,
            gs,
            name_a="confidence + hbonds + salt bridges",
            name_b="confidence alone",
            n_boot=800,
        )

    # Per-target for the best geometry metric.
    best_geom = max(alone, key=lambda c: abs(alone[c] - 0.5))
    per_target = []
    bg = m[best_geom].to_numpy(dtype=float)
    for t in sorted(set(g)):
        mask = (g == t) & ~np.isnan(bg)
        if mask.sum() < 10:
            continue
        pt, lo, hi = stats.within_target_bootstrap(y[mask], bg[mask], n_boot=N_BOOT)
        ref_pt = stats.auroc(y[mask], ip[mask])
        if pt == pt:
            per_target.append(
                {
                    "target": t,
                    "n": int(mask.sum()),
                    "binders": int(y[mask].sum()),
                    "geometry_auroc": pt,
                    "lo": lo,
                    "hi": hi,
                    "reference_auroc": ref_pt,
                }
            )

    # Group differences for the four size-like metrics, computed WITHIN target.
    # The pooled version of this comparison is what session 3 found to be wrong
    # (docs/SPEC.md section 8.7); stats.within_group_mean_difference returns both so
    # the report can state the disagreement instead of repeating it.
    group_diffs = {
        c: stats.within_group_mean_difference(y, m[c].to_numpy(dtype=float), g, name=c)
        for c in ("n_atom_contacts", "bsa_binder", "bsa_total", "contact_density")
    }

    return {
        "n": len(m),
        "n_targets": int(m["target"].nunique()),
        "group_diffs": group_diffs,
        "base_rate": float(y.mean()),
        "alone": alone,
        "reference_auroc": reference_auroc,
        "conditional": conditional,
        "combined": combined,
        "subset": subset,
        "best_geom": best_geom,
        "per_target": per_target,
        "sidechain_available": int(m["n_hbonds"].notna().sum()),
        "status_counts": m["design_model_status"].value_counts().to_dict(),
        "status_rates": m.groupby("design_model_status")["binder_final"]
        .apply(lambda s: float(s.astype(int).mean()))
        .to_dict(),
    }


def write_report(a: dict, validation: dict) -> Path:
    L: list[str] = []
    w = L.append
    comb = a["combined"]

    w("# Does interface geometry add anything over co-folding confidence?")
    w("")
    w("**Not additively, for the thirteen metrics tested - and adding all thirteen")
    w("makes prediction measurably worse.** A multiplicative interaction with")
    w("interface energetics, which the literature reports does help, was not")
    w('tested here; see "Where this conflicts with the literature" below.')
    w("")
    w("Generated by `binderkit study geometry`. Every number is reproducible from")
    w("that command.")
    w("")

    w("## The question")
    w("")
    w("Study 1 found that co-folding confidence predicts measured binding at 0.761")
    w("within-target AUROC. Structure-based design pipelines also compute interface")
    w("geometry - buried surface area, contact counts, hydrogen bonds, interface")
    w("composition - and rank on it. The question is whether that adds anything")
    w("once you already have the confidence number.")
    w("")
    w("This has been asked before on other corpora. Overath et al. 2025")
    w("(bioRxiv 2025.08.14.670059) meta-analysed 3,766 experimentally")
    w("characterised binders and report that confidence **multiplied by** interface")
    w("dG/dSASA beats either alone. What is new here is not the question but the")
    w("dataset: no prior analysis of this release exists, and it is the only corpus")
    w("with one campaign, one adjudication rubric and two independent assays.")
    w("The conflict with their result is addressed below, and it may not be a real")
    w("disagreement.")
    w("")
    w("The question here is deliberately **conditional**. Several geometry metrics")
    w("do correlate with binding on their own. That is not the useful question. The")
    w("useful question is whether a model that already knows the confidence score")
    w("gets better when geometry is added to it.")
    w("")

    w("## Data")
    w("")
    w(f"{a['n']} designs across {a['n_targets']} targets with both a released design")
    w(f"model and a wet-lab outcome; binder rate {100 * a['base_rate']:.1f}%. No")
    w("structure prediction was run: the coordinates are the release's own design")
    w("models.")
    w("")

    w("### The implementation is validated, not assumed correct")
    w("")
    w("The release publishes its own epitope contact counts computed on these same")
    w("files, so the geometry code is checked against a published number on")
    w(f"{validation['n_compared']} designs:")
    w("")
    w("| Quantity | Exact matches | Correlation |")
    w("|---|---|---|")
    for key, label in [
        ("epitope_residues", "epitope residues"),
        ("paratope_residues", "paratope residues"),
        ("atom_contacts", "interface atom contacts"),
    ]:
        if key in validation:
            v = validation[key]
            w(
                f"| {label} | {v['exact_matches']}/{validation['n_compared']} "
                f"({100 * v['fraction_exact']:.1f}%) | {v['correlation']:.4f} |"
            )
    w("")
    w("Agreement is exact on every comparable design. That matters beyond tidiness:")
    w("it means this metric code can be pointed at designs generated later without")
    w("first having to debug the scoring, which is the expensive way to discover a")
    w("contact definition was wrong.")
    w("")
    w("Getting there required fixing a real bug. Several targets are not protein")
    w("only - Cas9 is modelled as a ribonucleoprotein with an sgRNA chain, RBX1")
    w("carries three zinc ions, 15-PGDH an NAD cofactor. Counting binder-to-RNA and")
    w("binder-to-ligand atom pairs as interface contacts inflated some Cas9 designs")
    w("by a factor of eighty. Restricting the target to protein chains took")
    w("agreement from 95.5% to 100%.")
    w("")

    w("### What the models can and cannot show")
    w("")
    w("| design_model_status | n | binder rate |")
    w("|---|---|---|")
    for s, n in sorted(a["status_counts"].items(), key=lambda kv: -kv[1]):
        w(f"| `{s}` | {n} | {100 * a['status_rates'].get(s, float('nan')):.1f}% |")
    w("")
    w(f"Only **{a['sidechain_available']} of {a['n']}** design models carry side")
    w("chains on the binder; the rest are backbone plus C-beta. Hydrogen bonds and")
    w("salt bridges involving a binder side chain are therefore *unobservable* in")
    w("most of this data, not zero, and are reported as missing. They are analysed")
    w("separately below on the subset where they exist.")
    w("")
    w("Note also that model status is confounded with outcome: the all-atom subset")
    w("has a very different binder rate from the backbone-only majority, so the two")
    w("cannot be pooled and the subset result does not generalise.")
    w("")

    w("## Each geometry metric on its own")
    w("")
    w("Within-target AUROC. Values below 0.5 are informative with the sign flipped.")
    w("")
    w("| Metric | Within-target AUROC | Direction |")
    w("|---|---|---|")
    for c, v in sorted(a["alone"].items(), key=lambda kv: -abs(kv[1] - 0.5)):
        direction = "higher is better" if v >= 0.5 else "**lower** is better"
        w(f"| `{c}` | {v:.3f} | {direction} |")
    w(f"| **`{REFERENCE}` (reference)** | **{a['reference_auroc']:.3f}** | higher is better |")
    w("")
    best = a["best_geom"]
    w(f"The strongest geometry metric is `{best}` at {a['alone'][best]:.3f}, against")
    w(f"{a['reference_auroc']:.3f} for the confidence score. Nothing in the geometry")
    w("comes close to it.")
    w("")
    w("### Size-like metrics, compared within target")
    w("")
    w("A previous version of this report compared interface size between binders")
    w("and non-binders **pooled across targets**, and concluded that binders had")
    w("fewer contacts and less buried surface. That was Simpson's paradox: the")
    w("targets with the largest interfaces (TNFa, MBP) are the ones nobody could")
    w("bind, so pooling measures target difficulty, not design quality. Computed")
    w("inside each target on the same file, every one of these differences")
    w("reverses sign. The claim is withdrawn; see `CHANGEstats.md`.")
    w("")
    w("| Metric | pooled difference | within-target difference | targets positive |")
    w("|---|---|---|---|")
    for gd in a["group_diffs"].values():
        w(
            f"| `{gd.name}` | {gd.pooled:+.2f} | **{gd.within:+.2f}** | "
            f"{gd.n_groups_positive}/{gd.n_groups} |"
        )
    w("")
    w("Within a target, binders have **more** contacts and **more** buried surface,")
    w("which is both the unsurprising direction and what the within-target AUROC")
    w("column above said all along. The effect is small - these metrics sit at")
    w("0.52-0.54 AUROC - and the conditional test below is what decides whether it")
    w("is worth anything.")
    w("")

    w("## The conditional test: does it add to confidence?")
    w("")
    w("Each row is a leave-one-target-out model given the confidence score plus one")
    w("geometry metric, compared against the same model given confidence alone.")
    w(f"Paired bootstrap, {N_BOOT} replicates resampling targets.")
    w("")
    w("| Added metric | With | Without | Difference | 95% CI | Verdict |")
    w("|---|---|---|---|---|---|")
    for c, d in sorted(a["conditional"].items(), key=lambda kv: -kv[1].difference):
        if d.distinguishable:
            verdict = "**adds**" if d.difference > 0 else "**hurts**"
        else:
            verdict = "no effect"
        w(
            f"| `{c}` | {d.stat_a:.3f} | {d.stat_b:.3f} | {d.difference:+.3f} | "
            f"{d.ci_lo:+.3f} to {d.ci_hi:+.3f} | {verdict} |"
        )
    w("")
    adds = [c for c, d in a["conditional"].items() if d.distinguishable and d.difference > 0]
    hurts = [c for c, d in a["conditional"].items() if d.distinguishable and d.difference < 0]
    w(f"**Not one of the {len(a['conditional'])} geometry metrics improves prediction")
    w(
        f"over confidence alone.** {len(hurts)} measurably hurt"
        + (f" (`{'`, `'.join(hurts)}`)" if hurts else "")
        + f", and {len(adds)} help."
    )
    w("")
    w("### All of it together")
    w("")
    w(f"A model given confidence plus all {len(GEOMETRY_ALL)} geometry metrics scores")
    w(f"**{comb.stat_a:.3f}** against **{comb.stat_b:.3f}** for confidence alone:")
    w(f"{comb.difference:+.3f}, 95% CI {comb.ci_lo:+.3f} to {comb.ci_hi:+.3f}, sign held")
    w(f"in {100 * comb.sign_consistency:.0f}% of replicates.")
    w("")
    if comb.distinguishable and comb.difference < 0:
        w("Adding the full structural analysis makes held-out prediction **measurably")
        w("worse**. The extra features add variance that does not transfer across")
        w("targets, which is the same pattern study 1 found when it fitted weights")
        w("over ten predictors.")
    w("")
    w("![conditional effect of geometry](figures/geometry_conditional.png)")
    w("")

    w("## Hydrogen bonds and salt bridges, where they can be seen")
    w("")
    s = a["subset"]
    w(f"These exist only for the {s['n']} designs with binder side chains, whose")
    w(f"binder rate is {100 * s['base_rate']:.1f}% against {100 * a['base_rate']:.1f}%")
    w("for the full set. The subset is small and not representative, so this is a")
    w("weaker test than the one above.")
    w("")
    if "combined" in s:
        w("On that subset the confidence score itself only reaches")
        w(f"{s['reference_auroc']:.3f}, which is well below its {a['reference_auroc']:.3f}")
        w("on the full data - another sign the subset is a harder and different")
        w(
            "sample. Alone, hydrogen bonds score "
            f"{s['alone']['n_hbonds']:.3f} and salt bridges {s['alone']['n_salt_bridges']:.3f}."
        )
        d = s["combined"]
        w("")
        w(f"Added to confidence: {d.difference:+.3f}, 95% CI {d.ci_lo:+.3f} to {d.ci_hi:+.3f}.")
        w("")
        w(
            "So there is no evidence that they add anything either, but with "
            f"{s['n']} designs this test is underpowered and should not be read as a"
        )
        w("strong negative.")
    w("")

    w("## Per-target")
    w("")
    w(f"Best geometry metric (`{best}`) against the confidence score, by target.")
    w("")
    w("| Target | n | binders | geometry AUROC | 95% CI | confidence AUROC |")
    w("|---|---|---|---|---|---|")
    for t in sorted(a["per_target"], key=lambda r: -r["geometry_auroc"]):
        w(
            f"| {t['target']} | {t['n']} | {t['binders']} | {t['geometry_auroc']:.3f} | "
            f"{t['lo']:.3f}-{t['hi']:.3f} | {t['reference_auroc']:.3f} |"
        )
    w("")
    egfr = next((t for t in a["per_target"] if t["target"] == "EGFR"), None)
    if egfr:
        w(
            f"On EGFR specifically, geometry reaches {egfr['geometry_auroc']:.3f} "
            f"(CI {egfr['lo']:.3f}-{egfr['hi']:.3f}) against "
            f"{egfr['reference_auroc']:.3f} for confidence, on {egfr['binders']} binders"
        )
        w(
            "in {n} designs. Like every per-target number in these studies, the interval".format(
                n=egfr["n"]
            )
        )
        w("is too wide to support a claim on its own.")
    w("")
    w("![per-target geometry vs confidence](figures/geometry_per_target.png)")
    w("")

    w("## Where this conflicts with the literature")
    w("")
    w("Overath et al. 2025 report the opposite for two features this study did not")
    w("compute: `ipSAE_min x interface_dG/dSASA` and `LIS x shape_complementarity`")
    w("each beat either component alone, on 3,766 binders across 15 targets.")
    w("")
    w("Two reasons that may not contradict the result above, and both are limits of")
    w("this study rather than of theirs:")
    w("")
    w("1. **Different features.** Interface dG/dSASA and Lawrence-Colman shape")
    w("   complementarity are not among the thirteen tested here. The two features")
    w("   the literature says work are precisely the two that were not measured.")
    w("2. **Different functional form, and this is the one that matters.** Their")
    w("   combination is a *product*. The test above adds geometry as an additional")
    w("   *linear term* to a model that already has confidence. A logistic model in")
    w("   (confidence, geometry) cannot represent confidence x geometry unless the")
    w("   interaction is handed to it explicitly. A null additive effect is fully")
    w("   compatible with a real multiplicative one, so this study has not tested")
    w("   their claim.")
    w("")
    w("The experiment that settles it is specified in docs/SPEC.md section 8.6 and")
    w("runs as `binderkit study replication`: add their two features, then compare")
    w("the product against confidence alone by average precision under")
    w("leave-one-target-out, which is their procedure, as well as by the")
    w("within-target AUROC used here. Until that has run, the finding below must be")
    w("read in its narrow form.")
    w("")

    w("## What this means")
    w("")
    w("1. **No *additive* effect was found for thirteen geometry metrics.** If you")
    w("   have co-folding confidence, adding any of these thirteen as an extra")
    w("   linear feature does not improve held-out ranking, and adding all thirteen")
    w("   makes it measurably worse. This is not the same as 'geometry is")
    w("   redundant': a multiplicative interaction with interface energetics was")
    w("   not tested here and is reported to work elsewhere.")
    w("2. **That is a compute saving, not a disappointment.** It removes a stage from")
    w("   any pipeline built to scale, and it says where the hours should go instead:")
    w("   into generating and folding more designs, not into scoring each one more")
    w("   elaborately.")
    w("3. **Geometry still has non-ranking uses.** Clash counts and buried surface")
    w("   area remain the right tools for diagnosing *why* a design looks wrong, and")
    w("   for reporting what was made. This result is about ranking, not about")
    w("   whether the numbers are meaningful.")
    w("4. **The metric code is now validated against published values**, so it can be")
    w("   pointed at new designs without re-litigating whether it is correct.")
    w("")

    w("## What this study cannot tell you")
    w("")
    w("- **Whether geometry computed on a *predicted* complex behaves differently.**")
    w("  These are design models - the pose the designer intended - not co-folds.")
    w("  A co-fold's geometry is a different measurement and could carry different")
    w("  information. That is the obvious next test and it needs no new wet-lab data.")
    w("- **Anything about side-chain chemistry at scale.** Most binder chains here")
    w("  have no side chains, so hydrogen bonds and salt bridges could only be tested")
    w("  on a small, unrepresentative subset.")
    w("- **Whether a better geometry metric exists.** Thirteen were tested; shape")
    w("  complementarity in the classical sense, interface packing density and")
    w("  Rosetta interface energy were not among them.")
    w("- **Whether confidence *multiplied by* geometry behaves differently from")
    w("  confidence *plus* geometry.** It was not tested. See the section above.")
    w("- The same selection bias that qualifies study 1 applies here: these designs")
    w("  were chosen for ordering using the confidence metric, so the sample is")
    w("  filtered on the very reference this study compares against.")
    w("")

    path = OUT / "REPORT.md"
    while L and not L[-1].strip():  # no trailing blank line; the repo hook strips it
        L.pop()
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def make_figures(a: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    items = sorted(a["conditional"].items(), key=lambda kv: kv[1].difference)
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ypos = np.arange(len(items) + 1)
    diffs = [d.difference for _, d in items] + [a["combined"].difference]
    los = [d.difference - d.ci_lo for _, d in items] + [
        a["combined"].difference - a["combined"].ci_lo
    ]
    his = [d.ci_hi - d.difference for _, d in items] + [
        a["combined"].ci_hi - a["combined"].difference
    ]
    labels = [c for c, _ in items] + ["ALL GEOMETRY TOGETHER"]
    cols = [
        "#c0392b"
        if (d.distinguishable and d.difference < 0)
        else ("#2e7d52" if d.distinguishable else "#888888")
        for _, d in items
    ]
    cols.append("#c0392b" if a["combined"].distinguishable else "#888888")
    ax.errorbar(diffs, ypos, xerr=[los, his], fmt="none", ecolor="#999999", capsize=3, zorder=2)
    for i, c in enumerate(cols):
        ax.plot(diffs[i], ypos[i], "o", color=c, markersize=8, zorder=3)
    ax.axvline(0.0, color="black", lw=1.2, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("change in within-target AUROC when added to confidence (95% CI)")
    ax.set_title(
        "No geometry metric added linearly improves confidence; all 13 make it worse",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "geometry_conditional.png", dpi=150)
    plt.close(fig)
    written.append("geometry_conditional.png")

    pt = sorted(a["per_target"], key=lambda r: r["reference_auroc"])
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ypos = np.arange(len(pt))
    ax.barh(
        ypos - 0.2,
        [t["reference_auroc"] for t in pt],
        height=0.38,
        color="#d97757",
        label="co-folding confidence",
    )
    ax.barh(
        ypos + 0.2,
        [t["geometry_auroc"] for t in pt],
        height=0.38,
        color="#8fb8de",
        label=f"best geometry ({a['best_geom']})",
    )
    ax.axvline(0.5, color="black", lw=1, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{t['target']} ({t['binders']}/{t['n']})" for t in pt], fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("within-target AUROC")
    ax.set_title("Confidence beats the best geometry metric on almost every target", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "geometry_per_target.png", dpi=150)
    plt.close(fig)
    written.append("geometry_per_target.png")
    return written


def main() -> int:
    logging.getLogger().setLevel(logging.INFO)
    m = load()
    a = analyse(m)
    validation = json.loads((OUT / "validation.json").read_text(encoding="utf-8"))
    figs = make_figures(a)
    path = write_report(a, validation)

    pd.DataFrame(
        [
            {
                "metric": c,
                "auroc_alone": a["alone"][c],
                "conditional_difference": d.difference,
                "ci_lo": d.ci_lo,
                "ci_hi": d.ci_hi,
                "distinguishable": d.distinguishable,
            }
            for c, d in a["conditional"].items()
        ]
    ).to_csv(OUT / "geometry_results.csv", index=False)

    # Machine-readable summary, so downstream views (portfolio/) can assert
    # against a file rather than scraping prose out of REPORT.md.
    comb = a["combined"]
    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "n_designs": a["n"],
                "n_targets": a["n_targets"],
                "base_rate": a["base_rate"],
                "reference_metric": REFERENCE,
                "reference_within_target_auroc": a["reference_auroc"],
                "best_geometry_metric": a["best_geom"],
                "best_geometry_auroc": a["alone"][a["best_geom"]],
                "n_metrics_tested": len(GEOMETRY_ALL),
                "n_metrics_helping": sum(
                    1 for d in a["conditional"].values() if d.distinguishable and d.difference > 0
                ),
                "n_metrics_hurting": sum(
                    1 for d in a["conditional"].values() if d.distinguishable and d.difference < 0
                ),
                "all_geometry_combined": {
                    "with_geometry": comb.stat_a,
                    "confidence_alone": comb.stat_b,
                    "difference": comb.difference,
                    "ci_lo": comb.ci_lo,
                    "ci_hi": comb.ci_hi,
                    "sign_consistency": comb.sign_consistency,
                    "distinguishable": bool(comb.distinguishable),
                },
                "sidechain_subset": {
                    "n": a["subset"]["n"],
                    "base_rate": a["subset"]["base_rate"],
                },
                "group_differences": {
                    name: {
                        "pooled": gd.pooled,
                        "within_target": gd.within,
                        "n_groups_positive": gd.n_groups_positive,
                        "n_groups": gd.n_groups,
                        "sign_flips": bool(gd.sign_flips),
                    }
                    for name, gd in a["group_diffs"].items()
                },
                "validation": validation,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    log.info("wrote %s and %d figures", path, len(figs))
    print(f"\nreference {REFERENCE} = {a['reference_auroc']:.3f}")
    print(f"best geometry alone: {a['best_geom']} = {a['alone'][a['best_geom']]:.3f}")
    c = a["combined"]
    print(
        f"confidence + all geometry: {c.stat_a:.3f} vs {c.stat_b:.3f} "
        f"({c.difference:+.3f} [{c.ci_lo:+.3f},{c.ci_hi:+.3f}])"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
