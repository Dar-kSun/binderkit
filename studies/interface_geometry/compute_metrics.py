"""Compute interface geometry on the released design complexes (section 3.1).

The Anthropic release ships a design model for 1,309 of the 1,440 designs, so
every metric docs/SPEC.md section 6.1 lists but session 1 left as NaN can be
computed on real coordinates, for designs whose wet-lab outcome is known.

Two things make the result trustworthy rather than merely produced:

* **Validation.** The release publishes its own epitope contact counts computed
  on these same files, so every contact number here is checked against a
  published value rather than assumed correct (section 3.2).
* **Honest absence.** 975 of the design models carry only backbone plus C-beta
  for the binder chain. Hydrogen bonds and salt bridges involving a binder side
  chain are then unobservable, not zero, and are emitted as NaN.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from binderkit.geometry import compute_interface, identify_binder_chain, parse_mmcif

log = logging.getLogger(__name__)

STRUCT_DIR = Path("work/hf_cache/structures")
SUMMARY = Path("work/hf_cache/tables/design_summary.csv")
CONTACTS = Path("work/hf_cache/tables/insilico/epitope_residue_contacts.parquet")
OUT = Path(__file__).parent
RESULTS = OUT / "geometry_metrics.csv"


def _one(args: tuple[str, str]) -> dict:
    """Worker: compute metrics for one design. Never raises."""
    full_name, sequence = args
    path = STRUCT_DIR / f"{full_name}.cif"
    try:
        st = parse_mmcif(path)
        binder, targets = identify_binder_chain(st, sequence)
        m = compute_interface(st, binder, targets, full_name)
        row = asdict(m)
        row["notes"] = "; ".join(m.notes)
        row["full_name"] = full_name
        row["n_atoms"] = len(st)
        row["error"] = ""
        return row
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
        return {"full_name": full_name, "error": f"{type(exc).__name__}: {exc}"}


def compute_all(limit: int | None = None, workers: int = 12) -> pd.DataFrame:
    """Compute geometry for every design that has a structure."""
    ds = pd.read_csv(SUMMARY, low_memory=False)
    ds = ds[ds.full_name.map(lambda n: (STRUCT_DIR / f"{n}.cif").is_file())]
    if limit:
        ds = ds.head(limit)
    jobs = list(zip(ds.full_name, ds.sequence, strict=True))
    log.info("computing interface geometry for %d designs on %d workers", len(jobs), workers)

    rows: list[dict] = []
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for i, row in enumerate(ex.map(_one, jobs, chunksize=8), 1):
            rows.append(row)
            if i % 200 == 0:
                log.info("  %d/%d", i, len(jobs))
    return pd.DataFrame(rows)


def validate_against_release(geom: pd.DataFrame) -> dict:
    """Check our contact counts against the release's published values.

    Only the designs whose published contacts were computed on the design model
    are comparable; for the rest the release used a co-fold instead, so a
    mismatch there would mean nothing.
    """
    ref = pd.read_parquet(CONTACTS)
    ref = ref[ref.source_file_kind == "design_model"]
    m = geom.merge(
        ref[
            [
                "full_name",
                "n_epitope_residues",
                "epitope_paratope_n_residues",
                "epitope_interface_n_atom_contacts",
            ]
        ],
        on="full_name",
        suffixes=("_mine", "_ref"),
        how="inner",
    )
    out: dict = {"n_compared": len(m)}
    pairs = [
        ("n_epitope_residues_mine", "n_epitope_residues_ref", "epitope_residues"),
        ("n_paratope_residues", "epitope_paratope_n_residues", "paratope_residues"),
        ("n_atom_contacts", "epitope_interface_n_atom_contacts", "atom_contacts"),
    ]
    for mine, theirs, label in pairs:
        if mine not in m.columns or theirs not in m.columns:
            continue
        a, b = m[mine].astype(float), m[theirs].astype(float)
        exact = int((a == b).sum())
        out[label] = {
            "exact_matches": exact,
            "fraction_exact": exact / len(m) if len(m) else float("nan"),
            "max_abs_diff": float((a - b).abs().max()),
            "mean_abs_diff": float((a - b).abs().mean()),
            "correlation": float(a.corr(b)),
        }
    return out


def main(limit: int | None = None, workers: int = 12) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    geom = compute_all(limit=limit, workers=workers)
    failed = geom[geom.get("error", "").astype(str) != ""]
    if len(failed):
        log.warning("%d designs failed", len(failed))
        for r in failed.head(5).itertuples():
            log.warning("  %s: %s", r.full_name, r.error)
    geom.to_csv(RESULTS, index=False)
    log.info("wrote %s (%d rows)", RESULTS, len(geom))

    report = validate_against_release(geom)
    print("\nVALIDATION against the release's published contacts")
    print(f"  designs compared: {report['n_compared']}")
    for key in ("epitope_residues", "paratope_residues", "atom_contacts"):
        if key in report:
            v = report[key]
            print(
                f"  {key:20s} exact {v['exact_matches']}/{report['n_compared']} "
                f"({100 * v['fraction_exact']:.1f}%)  max|diff| {v['max_abs_diff']:.0f}  "
                f"r={v['correlation']:.4f}"
            )
    import json

    (OUT / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
