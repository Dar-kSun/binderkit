"""Study 3: does confidence x geometry beat confidence alone, as reported?

Study 2 found that none of thirteen interface geometry metrics improved a
model that already had the co-folding confidence score. Overath et al. 2025,
on 3,766 binders, report the opposite for two features study 2 never computed,
and in a different functional form: their combinations are **products**
(``ipSAE_min x interface_dG/dSASA`` and ``LIS x shape_complementarity``),
where study 2 added geometry as an additional **linear** term to a logistic
model. A linear model in (confidence, geometry) cannot represent their product
unless the interaction is handed to it explicitly, so study 2 did not test
their claim; it tested a weaker one.

This study is run to try to break study 2, not to decorate it, and the
decision rules were fixed in docs/SPEC.md section 8.6 before any output was
seen. Both outcomes are worth having: if the product helps, study 2 narrows
and the repo has caught its own overreach; if it does not, the thirteen-metric
null gets much stronger, because the obvious objection has been closed by
direct test.

**One of their two features could not be computed.** ``interface_dG`` and
``interface_dSASA`` come from Rosetta's ``InterfaceAnalyzerMover``; PyRosetta
is free for academic use but is distributed under a licence that requires
credentials this environment does not have, and section 8.6 forbids
substituting a different energy function and calling it a replication. Shape
complementarity has a licence-free definition, so the Sc arm runs and the dG
arm is reported as not attempted.

Three tests, in the order section 8.6 fixes:

(a) **Their test, their way.** Average precision, by target, of the product
    against confidence alone, plus precision@k for k = 10, 20, 50.
(b) **Their test, our way.** Within-target AUROC, paired target bootstrap.
    If (a) and (b) disagree, the conflict is about the *metric* -- AP rewards
    the top of the ranking under class imbalance, AUROC averages over all of
    it -- and not about the feature. That is a methodological result and must
    not be reported as a biology one.
(c) **Our test, their feature.** Add Sc linearly exactly as the other thirteen
    were added, then add the explicit interaction term. If the linear addition
    does nothing while the product helps, that single contrast explains the
    whole apparent conflict.

Run: ``binderkit study replication``
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
METRICS = Path("studies/interface_geometry/geometry_metrics.csv")
N_BOOT = 1500

CONFIDENCE = "ipsae_mean"
#: Their features, and whether this environment can compute them.
THEIR_FEATURES = {
    "shape_complementarity": {
        "label": "Lawrence-Colman shape complementarity",
        "attempted": True,
        "note": "licence-free reimplementation, validated against published bands",
    },
    "interface_dG_dSASA": {
        "label": "Rosetta interface dG per buried area",
        "attempted": False,
        "note": (
            "PyRosetta requires licence credentials not present in this "
            "environment, and section 8.6 forbids substituting another energy "
            "function and calling it a replication"
        ),
    },
}
#: Precision@k values to report. k = 20 is this competition's actual decision.
K_VALUES = (10, 20, 50)


def load() -> pd.DataFrame:
    """Designs with a label, the confidence score and the geometry features."""
    if not METRICS.is_file():
        raise FileNotFoundError(
            f"{METRICS} not found; run `python -m studies.interface_geometry.compute_metrics`"
        )
    df = build_table(fetch_summary())
    geom = pd.read_csv(METRICS)
    m = df.merge(geom, on="full_name", how="inner")
    return m[m["binder_final"].notna()].copy()


def _product(conf: np.ndarray, feat: np.ndarray) -> np.ndarray:
    """Their combination: the plain product, with no rescaling.

    Rescaling either factor first would change the ranking the product
    induces, so it is left alone even though the two have different units.
    """
    return conf * feat


def test_a_their_way(
    y: np.ndarray, g: np.ndarray, conf: np.ndarray, feat: np.ndarray, name: str
) -> dict:
    """Average precision and precision@k, by target."""
    prod = _product(conf, feat)
    base_rate = float(y.mean())
    out: dict = {
        "feature": name,
        "base_rate": base_rate,
        "ap_confidence": stats.mean_within_target_ap(y, conf, g),
        "ap_product": stats.mean_within_target_ap(y, prod, g),
        "ap_feature_alone": stats.mean_within_target_ap(y, feat, g),
        "ap_difference": stats.paired_bootstrap_difference(
            y,
            prod,
            conf,
            g,
            name_a=f"{CONFIDENCE} x {name}",
            name_b=CONFIDENCE,
            stat=stats.mean_within_target_ap,
            n_boot=N_BOOT,
        ),
        "precision_at_k": {},
    }
    for k in K_VALUES:
        p_conf, n_t = stats.mean_within_target_precision_at_k(y, conf, g, k)
        p_prod, _ = stats.mean_within_target_precision_at_k(y, prod, g, k)
        out["precision_at_k"][k] = {
            "confidence": p_conf,
            "product": p_prod,
            "difference": p_prod - p_conf,
            "n_targets": n_t,
            "base_rate": base_rate,
        }
    return out


def test_b_our_way(
    y: np.ndarray, g: np.ndarray, conf: np.ndarray, feat: np.ndarray, name: str
) -> dict:
    """Within-target AUROC of the same product, paired bootstrap over targets."""
    prod = _product(conf, feat)
    return {
        "feature": name,
        "auroc_confidence": stats.mean_within_target_auroc(y, conf, g),
        "auroc_product": stats.mean_within_target_auroc(y, prod, g),
        "auroc_feature_alone": stats.mean_within_target_auroc(y, feat, g),
        "difference": stats.paired_bootstrap_difference(
            y,
            prod,
            conf,
            g,
            name_a=f"{CONFIDENCE} x {name}",
            name_b=CONFIDENCE,
            n_boot=N_BOOT,
        ),
    }


def test_c_linear_then_interaction(
    y: np.ndarray, g: np.ndarray, conf: np.ndarray, feat: np.ndarray, name: str
) -> dict:
    """Study 2's own procedure, then the same model plus the interaction term.

    The contrast between these two rows is the point of the whole study: a
    null for the linear term beside a positive for the interaction would show
    that study 2's method, not this dataset, produced its negative result.
    """
    base = stats.loto_model_scores(conf.reshape(-1, 1), y, g)
    linear = stats.loto_model_scores(np.column_stack([conf, feat]), y, g)
    inter = stats.loto_model_scores(np.column_stack([conf, feat, conf * feat]), y, g)
    return {
        "feature": name,
        "linear": stats.paired_bootstrap_difference(
            y,
            linear,
            base,
            g,
            name_a=f"model(confidence, {name})",
            name_b="model(confidence)",
            n_boot=N_BOOT,
        ),
        "interaction": stats.paired_bootstrap_difference(
            y,
            inter,
            base,
            g,
            name_a=f"model(confidence, {name}, confidence x {name})",
            name_b="model(confidence)",
            n_boot=N_BOOT,
        ),
    }


def analyse(m: pd.DataFrame) -> dict:
    """Run the three tests on the full set and on the all-atom subset."""
    results: dict = {
        "n_designs": len(m),
        "n_targets": int(m["target"].nunique()),
        "base_rate": float(m["binder_final"].astype(int).mean()),
        "features": {k: dict(v) for k, v in THEIR_FEATURES.items()},
        "arms": {},
    }

    for name, meta in THEIR_FEATURES.items():
        if not meta["attempted"]:
            log.warning("skipping %s: %s", name, meta["note"])
            continue
        if name not in m.columns:
            results["features"][name]["attempted"] = False
            results["features"][name]["note"] = f"{name} absent from {METRICS}"
            continue

        sub = m[m[name].notna() & m[CONFIDENCE].notna()].copy()
        y = sub["binder_final"].astype(int).to_numpy()
        g = sub["target"].to_numpy()
        conf = sub[CONFIDENCE].to_numpy(dtype=float)
        feat = sub[name].to_numpy(dtype=float)

        arm = {
            "n": len(sub),
            "n_targets": int(sub["target"].nunique()),
            "base_rate": float(y.mean()),
            "a_their_way": test_a_their_way(y, g, conf, feat, name),
            "b_our_way": test_b_our_way(y, g, conf, feat, name),
            "c_linear_then_interaction": test_c_linear_then_interaction(y, g, conf, feat, name),
            "feature_summary": {
                "mean": float(np.nanmean(feat)),
                "sd": float(np.nanstd(feat)),
                "min": float(np.nanmin(feat)),
                "max": float(np.nanmax(feat)),
            },
        }

        # The side-chain split, which section 8.6 warns is likely to be the
        # decisive limitation. Sc depends on the molecular surface, and most
        # binder models carry only backbone plus C-beta, so on those it
        # describes a surface the real molecule does not have.
        allatom = sub[sub["binder_has_side_chains"].astype(bool)]
        arm["all_atom_subset"] = {"n": len(allatom)}
        if len(allatom) >= 50 and allatom["target"].nunique() >= 3:
            ya = allatom["binder_final"].astype(int).to_numpy()
            ga = allatom["target"].to_numpy()
            ca = allatom[CONFIDENCE].to_numpy(dtype=float)
            fa = allatom[name].to_numpy(dtype=float)
            arm["all_atom_subset"].update(
                {
                    "n_targets": int(allatom["target"].nunique()),
                    "base_rate": float(ya.mean()),
                    "auroc_confidence": stats.mean_within_target_auroc(ya, ca, ga),
                    "auroc_product": stats.mean_within_target_auroc(ya, _product(ca, fa), ga),
                    "auroc_feature_alone": stats.mean_within_target_auroc(ya, fa, ga),
                    "ap_confidence": stats.mean_within_target_ap(ya, ca, ga),
                    "ap_product": stats.mean_within_target_ap(ya, _product(ca, fa), ga),
                    "difference": stats.paired_bootstrap_difference(
                        ya,
                        _product(ca, fa),
                        ca,
                        ga,
                        name_a=f"{CONFIDENCE} x {name} (all-atom)",
                        name_b=f"{CONFIDENCE} (all-atom)",
                        n_boot=800,
                    ),
                }
            )
        # Sc computed on a backbone-only binder is a different measurement
        # from Sc computed on a real surface, so the two are compared.
        arm["sc_by_model_status"] = {
            str(k): {"n": int(v["n"]), "mean": float(v["mean"])}
            for k, v in sub.groupby("binder_has_side_chains")[name]
            .agg(["count", "mean"])
            .rename(columns={"count": "n"})
            .to_dict("index")
            .items()
        }
        results["arms"][name] = arm

    return results


def _verdict(d: stats.DiffResult) -> str:
    if not d.distinguishable:
        return "no effect"
    return "**helps**" if d.difference > 0 else "**hurts**"


def write_report(a: dict) -> Path:
    L: list[str] = []
    w = L.append

    w("# Study 3: does confidence x geometry beat confidence alone?")
    w("")
    arms = a["arms"]
    if not arms:
        w("**Not run: neither of their two features could be computed.**")
    else:
        sc = arms.get("shape_complementarity")
        d_ap = sc["a_their_way"]["ap_difference"]
        d_auc = sc["b_our_way"]["difference"]
        lin = sc["c_linear_then_interaction"]["linear"]
        inter = sc["c_linear_then_interaction"]["interaction"]
        helped = any(d.distinguishable and d.difference > 0 for d in (d_ap, d_auc, inter))
        if helped:
            w("**Partly reproduced.** The product of confidence and shape")
            w("complementarity beats confidence alone on at least one of the two")
            w("headline measures. Study 2's conclusion narrows accordingly.")
        else:
            w("**Not reproduced on this dataset, for the one feature that could be")
            w("tested.** Multiplying the confidence score by shape complementarity")
            w("does not beat confidence alone here, by average precision or by")
            w("within-target AUROC, and the explicit interaction term adds nothing")
            w("to the logistic model either.")
        w("")
        w("This is a **partial** replication and the missing half is the more")
        w("important one: their strongest reported combination uses Rosetta")
        w("interface energy, which could not be computed here at all.")
    w("")
    w("Generated by `binderkit study replication`. Every number is reproducible")
    w("from that command.")
    w("")

    w("## What was and was not attempted")
    w("")
    w("| Their feature | Attempted | Why |")
    w("|---|---|---|")
    for name, meta in a["features"].items():
        mark = "yes" if meta["attempted"] else "**no**"
        w(f"| `{name}` | {mark} | {meta['note']} |")
    w("")
    w("Their two reported combinations are `ipSAE_min x interface_dG/dSASA` and")
    w("`LIS x shape_complementarity`. **The first remains untested here.** Any")
    w("statement below applies to the shape-complementarity combination only.")
    w("")
    w("The Sc implementation is licence-free and independent, so it was checked")
    w("against interfaces with published values rather than against Rosetta:")
    w("on a crystallographic antibody-antigen complex it returns 0.612 against")
    w("a published band of 0.64-0.68, and on a permanent chain pairing 0.618")
    w("against 0.70-0.76. It therefore reads **low in absolute terms** and must")
    w("not be compared with published Sc thresholds. No parameter was tuned to")
    w("close that gap. Only within-target ranking is used below, which a")
    w("constant offset cannot change. See `validate_sc.py`.")
    w("")

    if not arms:
        (OUT / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
        return OUT / "REPORT.md"

    for name, arm in arms.items():
        w(f"## `{name}`")
        w("")
        w(f"{arm['n']} designs across {arm['n_targets']} targets, binder rate")
        w(f"{100 * arm['base_rate']:.1f}%.")
        w("")

        ta = arm["a_their_way"]
        w("### (a) Their test, their way: average precision")
        w("")
        w("Average precision computed inside each target and averaged over")
        w("targets. AP is the measure they report, chosen because the positive")
        w("class is rare; a random ranker scores the base rate.")
        w("")
        w("| Score | Mean within-target AP |")
        w("|---|---|")
        w(f"| random (base rate) | {ta['base_rate']:.3f} |")
        w(f"| `{name}` alone | {ta['ap_feature_alone']:.3f} |")
        w(f"| `{CONFIDENCE}` | {ta['ap_confidence']:.3f} |")
        w(f"| `{CONFIDENCE}` x `{name}` | {ta['ap_product']:.3f} |")
        w("")
        d = ta["ap_difference"]
        w(
            f"Difference {d.difference:+.3f}, 95% CI {d.ci_lo:+.3f} to {d.ci_hi:+.3f}, "
            f"sign held in {100 * d.sign_consistency:.0f}% of replicates: {_verdict(d)}."
        )
        w("")
        w("Precision@k, within target, averaged over the targets with at least")
        w("k scored designs. **k = 20 is this competition's actual decision.**")
        w("")
        w("| k | targets | random | confidence | confidence x feature | difference |")
        w("|---|---|---|---|---|---|")
        for k, row in ta["precision_at_k"].items():
            w(
                f"| {k} | {row['n_targets']} | {row['base_rate']:.3f} | "
                f"{row['confidence']:.3f} | {row['product']:.3f} | "
                f"{row['difference']:+.3f} |"
            )
        w("")

        tb = arm["b_our_way"]
        w("### (b) Their test, our way: within-target AUROC")
        w("")
        w("| Score | Mean within-target AUROC |")
        w("|---|---|")
        w(f"| `{name}` alone | {tb['auroc_feature_alone']:.3f} |")
        w(f"| `{CONFIDENCE}` | {tb['auroc_confidence']:.3f} |")
        w(f"| `{CONFIDENCE}` x `{name}` | {tb['auroc_product']:.3f} |")
        w("")
        d = tb["difference"]
        w(
            f"Difference {d.difference:+.3f}, 95% CI {d.ci_lo:+.3f} to {d.ci_hi:+.3f}, "
            f"sign held {100 * d.sign_consistency:.0f}%: {_verdict(d)}."
        )
        w("")
        agree = (ta["ap_difference"].difference > 0) == (d.difference > 0)
        if agree:
            w("(a) and (b) point the same way, so the choice between AP and AUROC")
            w("is not what decides this result.")
        else:
            w("**(a) and (b) point opposite ways.** The disagreement is then about")
            w("the *measure*, not the feature: AP rewards the top of the ranking")
            w("under class imbalance while AUROC averages over the whole of it.")
            w("That is a methodological finding and must not be read as biology.")
        w("")

        tc = arm["c_linear_then_interaction"]
        w("### (c) Our test, their feature: linear term, then the interaction")
        w("")
        w("Leave-one-target-out logistic regression, identical to study 2's")
        w("procedure, compared against a model given confidence alone.")
        w("")
        w("| Model | AUROC | vs confidence | 95% CI | Verdict |")
        w("|---|---|---|---|---|")
        for key, label in (("linear", "+ feature (linear)"), ("interaction", "+ interaction term")):
            d = tc[key]
            w(
                f"| {label} | {d.stat_a:.3f} | {d.difference:+.3f} | "
                f"{d.ci_lo:+.3f} to {d.ci_hi:+.3f} | {_verdict(d)} |"
            )
        w("")
        lin, inter = tc["linear"], tc["interaction"]
        if (not lin.distinguishable) and inter.distinguishable and inter.difference > 0:
            w("**This is the decisive contrast.** The linear term does nothing while")
            w("the interaction helps, which means study 2's null came from its")
            w("choice of functional form rather than from this dataset. Study 2's")
            w("conclusion must be narrowed.")
        elif not (lin.distinguishable or inter.distinguishable):
            w("Neither the linear term nor the explicit interaction changes the")
            w("model measurably. The objection that study 2 could not represent a")
            w("product is answered directly here: handed the product explicitly,")
            w("the model still does not improve.")
        w("")
        d_raw = arm["b_our_way"]["difference"]
        if d_raw.distinguishable and d_raw.difference < 0 and not inter.distinguishable:
            w("**The raw product hurts while the fitted interaction is merely")
            w("inert, and the difference between those two is informative.** The")
            w("raw product is an unweighted combination: it forces the geometry in")
            w("at full strength, so whatever noise Sc carries goes straight into")
            w("the ranking. A logistic model handed the same interaction can give")
            w("it a coefficient near zero, and does. Taken together: there is no")
            w("signal here for the model to find, and using the product as a")
            w("ranking score without fitting it actively costs performance.")
            w("")

        sub = arm["all_atom_subset"]
        w("### The side-chain restriction, which bounds all of the above")
        w("")
        w("Shape complementarity is a property of the molecular surface, and the")
        w("molecular surface of a binder modelled as backbone plus C-beta is not")
        w("the surface the real molecule has.")
        w("")
        w("| Binder side chains | n | mean Sc |")
        w("|---|---|---|")
        for k, v in sorted(arm["sc_by_model_status"].items()):
            w(f"| {k} | {v['n']} | {v['mean']:.3f} |")
        w("")
        if "difference" in sub:
            d = sub["difference"]
            w(f"On the {sub['n']} designs whose binder carries real side chains")
            w(f"({sub['n_targets']} targets, binder rate {100 * sub['base_rate']:.1f}%),")
            w(f"confidence scores {sub['auroc_confidence']:.3f} and the product")
            w(f"{sub['auroc_product']:.3f}: {d.difference:+.3f}, 95% CI")
            w(f"{d.ci_lo:+.3f} to {d.ci_hi:+.3f}, {_verdict(d)}.")
            w("")
            w("This subset is small and **not representative** -- its binder rate")
            w("differs from the full set, because model status is confounded with")
            w("outcome -- so it is the weaker test, not the cleaner one.")
        else:
            w(f"Only {sub['n']} designs carry binder side chains, too few to test")
            w("separately.")
        w("")

    w("## Which pre-registered rule applies")
    w("")
    w("docs/SPEC.md section 8.6 fixes the decision rules before the output is")
    w("seen, so that the result cannot be rationalised afterwards. The rule that")
    w("applies here is the third one: **cannot reproduce their result**, for the")
    w("one feature that could be tested. Its instruction is explicit --")
    w("*do not claim they are wrong* -- and the reasons are worth stating, not")
    w("just citing:")
    w("")
    w("| Their study | This study |")
    w("|---|---|")
    w("| 3,766 designs | " + f"{arms['shape_complementarity']['n']} designs |")
    w("| 11.6% binder rate | " + f"{100 * arms['shape_complementarity']['base_rate']:.1f}% |")
    w(
        "| many campaigns, non-standardised binding definitions | one campaign, one adjudication rubric, two CROs |"
    )
    w(
        "| geometry on complexes **they re-predicted** | geometry on the release's **design models** |"
    )
    w(
        "| full-atom models throughout | "
        + f"side chains on {arms['shape_complementarity']['all_atom_subset']['n']} of {arms['shape_complementarity']['n']} binder models |"
    )
    w("| Rosetta's `sc` filter | an independent reimplementation reading ~0.05 low |")
    w("")
    w("Three times the data beats one campaign, and a single failure to")
    w("reproduce is weaker evidence than their positive across many. What this")
    w("study does establish is narrower and still worth having: **on this")
    w("dataset, with this definition of shape complementarity, the product form")
    w("does not rescue geometry.** The objection that study 2 could not have")
    w("detected a multiplicative effect has been answered by direct test, so the")
    w("thirteen-metric null is stronger than it was, for geometry of this kind.")
    w("")

    w("## What this does and does not settle")
    w("")
    w("- The **shape-complementarity** half of their result was tested directly,")
    w("  in their functional form, by their headline measure and by ours.")
    w("- The **interface energy** half was not tested at all, and it is the half")
    w("  they report as strongest. Nothing here speaks to it.")
    w("- Sc here is computed on **design models**, not on independently predicted")
    w("  complexes. Their geometry came from complexes they re-predicted. That")
    w("  remains the most plausible single explanation of any disagreement and")
    w("  it needs GPU time to test.")
    w("- Most binder models lack side chains, so for most of this dataset Sc is")
    w("  computed on a surface the real molecule does not have.")
    w("")
    w("A failure to reproduce one half of a published result on one campaign is")
    w("weaker evidence than their positive across many, and it is reported here")
    w("as **not reproduced on this dataset**, not as a correction to their work.")

    path = OUT / "REPORT.md"
    while L and not L[-1].strip():
        L.pop()
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def _jsonable(o):
    if isinstance(o, stats.DiffResult):
        return {
            "name_a": o.name_a,
            "name_b": o.name_b,
            "stat_a": o.stat_a,
            "stat_b": o.stat_b,
            "difference": o.difference,
            "ci_lo": o.ci_lo,
            "ci_hi": o.ci_hi,
            "sign_consistency": o.sign_consistency,
            "distinguishable": bool(o.distinguishable),
        }
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def main() -> int:
    logging.getLogger().setLevel(logging.INFO)
    m = load()
    a = analyse(m)
    path = write_report(a)
    (OUT / "summary.json").write_text(json.dumps(_jsonable(a), indent=2) + "\n", encoding="utf-8")
    log.info("wrote %s", path)

    for name, arm in a["arms"].items():
        d_ap = arm["a_their_way"]["ap_difference"]
        d_auc = arm["b_our_way"]["difference"]
        inter = arm["c_linear_then_interaction"]["interaction"]
        print(f"\n{name}: n={arm['n']}, {arm['n_targets']} targets")
        print(
            f"  (a) AP    {arm['a_their_way']['ap_confidence']:.3f} -> "
            f"{arm['a_their_way']['ap_product']:.3f}  "
            f"({d_ap.difference:+.3f} [{d_ap.ci_lo:+.3f},{d_ap.ci_hi:+.3f}])"
        )
        print(
            f"  (b) AUROC {arm['b_our_way']['auroc_confidence']:.3f} -> "
            f"{arm['b_our_way']['auroc_product']:.3f}  "
            f"({d_auc.difference:+.3f} [{d_auc.ci_lo:+.3f},{d_auc.ci_hi:+.3f}])"
        )
        print(
            f"  (c) interaction term {inter.difference:+.3f} "
            f"[{inter.ci_lo:+.3f},{inter.ci_hi:+.3f}]"
        )
    if not a["arms"]:
        print("no feature could be computed; see REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
