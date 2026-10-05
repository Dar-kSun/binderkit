"""Compute Rosetta interface energetics for study 3's second arm.

docs/SPEC.md section 8.6 names four features. Two of them -- ``interface_dG``
and ``interface_dSASA``, and the ratio ``interface_dG_dSASA`` that Overath et
al. actually multiply the confidence score by -- come from Rosetta's
``InterfaceAnalyzerMover`` and have no licence-free equivalent. Section 8.6 is
explicit that substituting a different energy function would not be a
replication, so this is the real thing or nothing.

**This script does not run on Windows.** PyRosetta publishes Linux and macOS
wheels only, so it runs under WSL against the same files, and is invoked
separately from the rest of the pipeline::

    wsl -d Ubuntu-24.04 -- ~/pyr/bin/python \\
        /mnt/c/Users/aryan/'Protein Design'/studies/replication/rosetta_energy.py

Two restrictions are deliberate and both come from section 8.6's validation
gate:

* **Protein chains only.** The Cas9 sgRNA, the RBX1 zinc ions and the 15-PGDH
  NAD cofactor inflated interface contacts eightyfold before they were
  excluded, and they will distort Rosetta scoring at least as badly. The
  chain assignment is read from ``geometry_metrics.csv``, which already
  applies that restriction, rather than being re-derived here.
* **Side chains required.** A Rosetta energy on a backbone-plus-C-beta model
  is not meaningful: the side chains that make the interactions are simply
  absent. ``interface_dG`` is therefore computed only where the binder
  carries real side chains, and reported as **missing, not zero**, elsewhere.
  That is a small minority of this dataset, and the subset size is carried
  into every downstream number.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

log = logging.getLogger("rosetta_energy")

#: The repository, as seen from inside WSL.
REPO = Path(__file__).resolve().parent.parent.parent
STRUCTURES = REPO / "work" / "hf_cache" / "structures"
GEOMETRY = REPO / "studies" / "interface_geometry" / "geometry_metrics.csv"
OUT = REPO / "studies" / "replication" / "rosetta_metrics.csv"

FIELDS = [
    "full_name",
    "binder_chain",
    "target_chains",
    "binder_has_side_chains",
    "interface_dG",
    "interface_dSASA",
    "interface_dG_dSASA",
    "n_interface_residues",
    "packstat",
    "error",
]


def init_pyrosetta() -> None:
    """Start PyRosetta quietly, with the options the mover needs."""
    import pyrosetta

    pyrosetta.init(
        " ".join(
            [
                "-mute all",
                "-ignore_unrecognized_res true",
                "-ignore_zero_occupancy false",
                "-load_PDB_components false",
                # Design models carry no hydrogens; let Rosetta add them
                # rather than scoring a structure it considers broken.
                "-ignore_waters true",
            ]
        )
    )


def read_targets(limit: int | None, all_atom_only: bool) -> list[dict]:
    """Designs to score, with the chain assignment the geometry stage used."""
    if not GEOMETRY.is_file():
        raise FileNotFoundError(f"{GEOMETRY} not found; run compute_metrics first")
    rows = []
    with GEOMETRY.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("error"):
                continue
            if not row.get("binder_chain") or not row.get("target_chains"):
                continue
            has_sc = str(row.get("binder_has_side_chains", "")).strip().lower() == "true"
            if all_atom_only and not has_sc:
                continue
            if not (STRUCTURES / f"{row['full_name']}.cif").is_file():
                continue
            rows.append(
                {
                    "full_name": row["full_name"],
                    "binder_chain": row["binder_chain"],
                    "target_chains": row["target_chains"],
                    "binder_has_side_chains": has_sc,
                }
            )
    return rows[:limit] if limit else rows


def score_one(spec: dict) -> dict:
    """Interface energetics for one complex, or an error row.

    The interface is given to the mover as ``binder_LH``-style chain groups so
    it separates exactly the two sides the rest of the study uses, rather than
    guessing from chain order.
    """
    import pyrosetta
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    out = dict.fromkeys(FIELDS, "")
    out.update(
        {
            "full_name": spec["full_name"],
            "binder_chain": spec["binder_chain"],
            "target_chains": spec["target_chains"],
            "binder_has_side_chains": spec["binder_has_side_chains"],
        }
    )
    try:
        pose = pyrosetta.pose_from_file(str(STRUCTURES / f"{spec['full_name']}.cif"))
        targets = "".join(spec["target_chains"].split(","))
        interface = f"{spec['binder_chain']}_{targets}"

        mover = InterfaceAnalyzerMover(interface)
        mover.set_pack_separated(True)  # repack the unbound state, as BindCraft does
        mover.set_compute_packstat(True)
        mover.apply(pose)

        data = mover.get_all_data()
        dg = float(mover.get_interface_dG())
        dsasa = float(mover.get_interface_delta_sasa())
        out["interface_dG"] = dg
        out["interface_dSASA"] = dsasa
        out["interface_dG_dSASA"] = dg / dsasa if dsasa else ""
        out["n_interface_residues"] = int(mover.get_num_interface_residues())
        out["packstat"] = float(data.packstat)
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N")
    ap.add_argument(
        "--all",
        action="store_true",
        help=(
            "score every design, including binders with no side chains. "
            "Off by default because a Rosetta energy on a backbone-plus-C-beta "
            "model is not a meaningful number."
        ),
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    specs = read_targets(args.limit, all_atom_only=not args.all)
    log.info("scoring %d designs (all_atom_only=%s)", len(specs), not args.all)
    if not specs:
        log.error("nothing to score")
        return 1

    init_pyrosetta()
    rows = []
    for i, spec in enumerate(specs, 1):
        rows.append(score_one(spec))
        if i % 20 == 0 or i == len(specs):
            ok = sum(1 for r in rows if not r["error"])
            log.info("  %d/%d (%d scored, %d failed)", i, len(specs), ok, len(rows) - ok)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    failed = [r for r in rows if r["error"]]
    log.info("wrote %s (%d rows, %d failed)", OUT, len(rows), len(failed))
    for r in failed[:5]:
        log.warning("  %s: %s", r["full_name"], r["error"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
