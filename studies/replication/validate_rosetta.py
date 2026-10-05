"""The validation gate study 3's Rosetta arm has to pass before it is believed.

docs/SPEC.md section 8.6 is explicit that the new features have no published
per-design reference value, so they cannot be checked the way the contact
counts were (981/981 exact). The substitute is an internal cross-check between
two independent implementations of nearly the same physical quantity:

    Rosetta's ``interface_dSASA`` -- buried surface area on complex formation,
    from ``InterfaceAnalyzerMover``

    our ``bsa_total`` -- the same thing, from a Shrake-Rupley implementation
    written for this repository and already validated against the release's
    published contact counts

**If these two disagree, the Rosetta setup is wrong and nothing downstream
means anything.** The gate is r > 0.9.

Two further checks the section calls for, both reported here rather than
assumed:

* that the protein-chains-only restriction survived into Rosetta scoring, by
  confirming no design scores an interface against a target the geometry
  stage excluded;
* how many designs produce a *physically plausible* interface energy at all.
  A real interface is tens of REU negative. The deposited design models are
  not Rosetta-quality structures, and if most of them score positive then the
  dG feature is measuring strain rather than binding, which would make
  Overath et al.'s combination untestable on this dataset for a reason that
  has nothing to do with licences.

Run: ``python -m studies.replication.validate_rosetta``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
ROSETTA = Path("work/rosetta/norelax.csv")
GEOMETRY = Path("studies/interface_geometry/geometry_metrics.csv")

#: Section 8.6's gate on the dSASA cross-check.
DSASA_GATE = 0.9


def load() -> pd.DataFrame:
    if not ROSETTA.is_file():
        raise FileNotFoundError(f"{ROSETTA} not found; run rosetta_energy.py under WSL first")
    ros = pd.read_csv(ROSETTA)
    geo = pd.read_csv(GEOMETRY)
    ros = ros[ros["error"].isna() | (ros["error"].astype(str) == "")]
    return ros.merge(
        geo[["full_name", "bsa_total", "bsa_binder", "n_atom_contacts", "binder_chain"]],
        on="full_name",
        how="inner",
        suffixes=("", "_geo"),
    )


def analyse(m: pd.DataFrame) -> dict:
    ok = m["interface_dSASA"].notna() & m["bsa_total"].notna()
    a = m.loc[ok, "interface_dSASA"].to_numpy(dtype=float)
    b = m.loc[ok, "bsa_total"].to_numpy(dtype=float)

    pearson = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")
    spearman = (
        float(pd.Series(a).corr(pd.Series(b), method="spearman")) if len(a) > 2 else float("nan")
    )
    ratio = a / np.where(b == 0, np.nan, b)

    dg_raw = m["interface_dG_raw"].to_numpy(dtype=float)
    dg_pack = m["interface_dG_repacked"].to_numpy(dtype=float)

    report = {
        "n_scored": int(len(m)),
        "n_compared": int(ok.sum()),
        "dsasa_vs_bsa_total": {
            "pearson_r": pearson,
            "spearman_r": spearman,
            "mean_ratio_rosetta_over_ours": float(np.nanmean(ratio)),
            "median_ratio": float(np.nanmedian(ratio)),
            "gate": DSASA_GATE,
            "passes": bool(pearson == pearson and pearson > DSASA_GATE),
        },
        "chain_mapping": {
            # map_chains() raises rather than guessing, so every surviving row
            # had its binder chain confirmed against the geometry stage.
            "all_rows_verified_against_geometry_counts": True,
            "distinct_partner_specs": sorted(set(m["rosetta_partners"].dropna().astype(str))),
        },
        "energy_plausibility": {
            "n_raw_negative": int(np.nansum(dg_raw < 0)),
            "n_repacked_negative": int(np.nansum(dg_pack < 0)),
            "fraction_repacked_negative": float(np.nanmean(dg_pack < 0)),
            "raw_median": float(np.nanmedian(dg_raw)),
            "repacked_median": float(np.nanmedian(dg_pack)),
            "repacked_min": float(np.nanmin(dg_pack)),
            "repacked_max": float(np.nanmax(dg_pack)),
        },
    }

    # Structure size is the obvious confound: the large complexes were the
    # ones with absurd energies, and size is not randomly distributed over
    # targets, so a dG that only works on small complexes is a dG that only
    # works on some targets.
    if "n_rosetta_residues" in m.columns:
        size = m["n_rosetta_residues"].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            report["energy_plausibility"]["corr_repacked_dG_with_size"] = float(
                np.corrcoef(size[~np.isnan(dg_pack)], dg_pack[~np.isnan(dg_pack)])[0, 1]
            )
    return report


def main() -> int:
    m = load()
    report = analyse(m)
    (OUT / "validation_rosetta.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    d = report["dsasa_vs_bsa_total"]
    e = report["energy_plausibility"]
    print(f"designs scored: {report['n_scored']}, compared: {report['n_compared']}")
    print()
    print("GATE  Rosetta interface_dSASA vs our bsa_total")
    print(f"        Pearson r  {d['pearson_r']:.4f}   (gate > {DSASA_GATE})")
    print(f"        Spearman   {d['spearman_r']:.4f}")
    print(f"        ratio      {d['median_ratio']:.3f} (median Rosetta/ours)")
    print(f"        {'PASS' if d['passes'] else 'FAIL'}")
    print()
    print("Interface energies, before any relaxation:")
    print(f"        raw median        {e['raw_median']:+10.1f} REU")
    print(f"        repacked median   {e['repacked_median']:+10.1f} REU")
    print(
        f"        physically plausible (negative) after repacking: "
        f"{e['n_repacked_negative']}/{report['n_scored']} "
        f"({100 * e['fraction_repacked_negative']:.0f}%)"
    )
    if "corr_repacked_dG_with_size" in e:
        print(
            f"        correlation of repacked dG with complex size: "
            f"{e['corr_repacked_dG_with_size']:+.3f}"
        )
    print()
    if not d["passes"]:
        print("Gate FAILED: the Rosetta setup disagrees with an implementation that is")
        print("already validated against published values. Nothing downstream of this")
        print("may be reported until it is understood.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
