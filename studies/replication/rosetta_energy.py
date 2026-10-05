"""Compute Rosetta interface energetics for study 3's second arm.

docs/SPEC.md section 8.6 names four features. Two of them -- ``interface_dG``
and ``interface_dSASA``, and the ratio ``interface_dG_dSASA`` that Overath et
al. actually multiply the confidence score by -- come from Rosetta's
``InterfaceAnalyzerMover`` and have no licence-free equivalent. Section 8.6 is
explicit that substituting a different energy function would not be a
replication, so this is the real thing or nothing.

**This script does not run on Windows.** PyRosetta publishes Linux and macOS
wheels only, so it runs under WSL against the same files::

    wsl -d Ubuntu-24.04 -- ~/pyr/bin/python \\
        "/mnt/c/.../studies/replication/rosetta_energy.py" --limit 25

Four things here are not incidental, and each cost a failed run to find:

* **Paths with a space.** Rosetta's reader reports ``Cannot open file`` on a
  path Python opens happily. The repository lives under "Protein Design", so
  structures are reached through a space-free symlink.
* **Chain letters.** The release's mmCIF files use multi-character chain ids
  (``Axp``, ``Bxp``); Rosetta reduces them to single letters. The mapping is
  resolved per structure and then **checked against the residue counts the
  geometry stage recorded**, because a silently mismatched chain assignment
  would score the wrong interface and still return a plausible number.
* **Protein chains only.** The Cas9 sgRNA, the RBX1 zinc and the 15-PGDH NAD
  inflated interface contacts eightyfold before they were excluded, and they
  distort Rosetta scoring at least as badly. The assignment is read from
  ``geometry_metrics.csv``, which already applies that restriction.
* **These are not Rosetta-quality structures.** A deposited design model
  scores ``dG = +2108`` REU untouched and ``+137`` after repacking, where a
  real interface is tens of REU *negative*. Both numbers are recorded per
  design so the reader can see how far the raw coordinates are from
  anything Rosetta considers physical, and ``--relax`` adds the constrained
  FastRelax that BindCraft runs before scoring. Relaxation moves the
  coordinates being evaluated, which is a real caveat and is reported.

A Rosetta energy on a backbone-plus-C-beta model is meaningless -- the side
chains that make the interactions are absent -- so by default only designs
whose binder carries side chains are scored, and ``interface_dG`` is missing
rather than zero elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

log = logging.getLogger("rosetta_energy")

REPO = Path(__file__).resolve().parent.parent.parent
STRUCTURES = REPO / "work" / "hf_cache" / "structures"
GEOMETRY = REPO / "studies" / "interface_geometry" / "geometry_metrics.csv"
OUT = REPO / "studies" / "replication" / "rosetta_metrics.csv"
_LINK_ROOT = Path("/tmp/binderkit-structures")

#: Accept a chain mapping only if the residue count is within this fraction of
#: what the geometry stage counted. Rosetta drops unrecognised residues, so an
#: exact match is too strict, but a factor-of-two disagreement means the
#: mapping is wrong.
COUNT_TOLERANCE = 0.25

FIELDS = [
    "full_name",
    "binder_chain",
    "target_chains",
    "rosetta_partners",
    "binder_has_side_chains",
    "n_rosetta_residues",
    "interface_dG_raw",
    "interface_dG_repacked",
    "interface_dG",
    "interface_dSASA",
    "interface_dG_dSASA",
    "n_interface_residues",
    "packstat",
    "relaxed",
    "error",
]


def space_free(path: Path) -> Path:
    """The same file by a path with no space in it."""
    if " " not in str(path):
        return path
    if not _LINK_ROOT.is_symlink() or _LINK_ROOT.resolve() != path.parent.resolve():
        if _LINK_ROOT.is_symlink() or _LINK_ROOT.exists():
            _LINK_ROOT.unlink()
        _LINK_ROOT.symlink_to(path.parent, target_is_directory=True)
    return _LINK_ROOT / path.name


def init_pyrosetta() -> None:
    import pyrosetta

    pyrosetta.init(
        "-mute all -ignore_unrecognized_res true -ignore_zero_occupancy false -ignore_waters true"
    )


def read_targets(limit: int | None, all_atom_only: bool) -> list[dict]:
    """Designs to score, with the chain assignment the geometry stage used."""
    if not GEOMETRY.is_file():
        raise FileNotFoundError(f"{GEOMETRY} not found; run compute_metrics first")
    rows = []
    with GEOMETRY.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("error") or not row.get("binder_chain") or not row.get("target_chains"):
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
                    "n_binder_residues": int(float(row.get("n_binder_residues") or 0)),
                    "n_target_residues": int(float(row.get("n_target_residues") or 0)),
                }
            )
    return rows[:limit] if limit else rows


def map_chains(pose, spec: dict) -> str:
    """Translate our chain ids to Rosetta's, and verify the result.

    Raises
    ------
    ValueError
        When the chains cannot be matched, or when the matched binder chain
        holds a residue count incompatible with the geometry stage's. Scoring
        the wrong side of an interface returns a plausible-looking number, so
        this must fail loudly rather than guess.
    """
    info = pose.pdb_info()
    counts: dict[str, int] = {}
    for i in range(1, pose.total_residue() + 1):
        counts[info.chain(i)] = counts.get(info.chain(i), 0) + 1

    binder = spec["binder_chain"][0]
    targets = [t.strip()[0] for t in spec["target_chains"].split(",") if t.strip()]
    missing = [c for c in [binder, *targets] if c not in counts]
    if missing:
        raise ValueError(f"chains {missing} absent from pose (has {sorted(counts)})")

    expected = spec["n_binder_residues"]
    got = counts[binder]
    if expected and abs(got - expected) > COUNT_TOLERANCE * expected:
        raise ValueError(
            f"binder chain {binder} has {got} residues, geometry stage counted "
            f"{expected}: chain mapping is unreliable for this structure"
        )
    return f"{binder}_{''.join(dict.fromkeys(targets))}"


def _analyse(pose, partners: str, pack_input: bool):
    """Run the mover and return (dG, dSASA, n_interface_residues, packstat)."""
    from pyrosetta.rosetta.core.pose import DockingPartners
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    mover = InterfaceAnalyzerMover(DockingPartners.docking_partners_from_string(partners))
    mover.set_pack_separated(True)
    mover.set_pack_input(pack_input)
    mover.set_compute_packstat(True)
    mover.apply(pose)
    return (
        float(mover.get_interface_dG()),
        float(mover.get_interface_delta_sasa()),
        int(mover.get_num_interface_residues()),
        float(mover.get_all_data().packstat),
    )


def _relax(pose) -> None:
    """Constrained FastRelax, as BindCraft runs before interface scoring.

    Coordinate constraints keep the pose near the deposited coordinates. It
    still moves them, which is why the unrelaxed energies are recorded too.
    """
    import pyrosetta
    from pyrosetta.rosetta.core.scoring import ScoreType
    from pyrosetta.rosetta.protocols.relax import FastRelax

    sf = pyrosetta.create_score_function("ref2015")
    sf.set_weight(ScoreType.coordinate_constraint, 1.0)
    fr = FastRelax(sf, 1)
    fr.constrain_relax_to_start_coords(True)
    fr.apply(pose)


def score_one(spec: dict, relax: bool) -> dict:
    """Interface energetics for one complex, or an error row."""
    import pyrosetta

    out = dict.fromkeys(FIELDS, "")
    out.update(
        {
            "full_name": spec["full_name"],
            "binder_chain": spec["binder_chain"],
            "target_chains": spec["target_chains"],
            "binder_has_side_chains": spec["binder_has_side_chains"],
            "relaxed": relax,
        }
    )
    try:
        pose = pyrosetta.pose_from_file(str(space_free(STRUCTURES / f"{spec['full_name']}.cif")))
        out["n_rosetta_residues"] = pose.total_residue()
        partners = map_chains(pose, spec)
        out["rosetta_partners"] = partners

        dg_raw, dsasa, n_res, _ = _analyse(pose.clone(), partners, pack_input=False)
        out["interface_dG_raw"] = dg_raw
        dg_pack, dsasa, n_res, packstat = _analyse(pose.clone(), partners, pack_input=True)
        out["interface_dG_repacked"] = dg_pack

        if relax:
            _relax(pose)
            dg, dsasa, n_res, packstat = _analyse(pose, partners, pack_input=False)
        else:
            dg = dg_pack

        out["interface_dG"] = dg
        out["interface_dSASA"] = dsasa
        out["interface_dG_dSASA"] = dg / dsasa if dsasa else ""
        out["n_interface_residues"] = n_res
        out["packstat"] = packstat
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
        out["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rosetta interface energetics for study 3")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N")
    ap.add_argument("--relax", action="store_true", help="run constrained FastRelax first")
    ap.add_argument("--out", default=None, help="output CSV (default: rosetta_metrics.csv)")
    ap.add_argument(
        "--all",
        action="store_true",
        help=(
            "score every design, including binders with no side chains. Off by "
            "default: a Rosetta energy on a backbone-plus-C-beta model is not a "
            "meaningful number."
        ),
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    specs = read_targets(args.limit, all_atom_only=not args.all)
    log.info(
        "scoring %d designs (all_atom_only=%s, relax=%s)", len(specs), not args.all, args.relax
    )
    if not specs:
        log.error("nothing to score")
        return 1

    init_pyrosetta()
    out_path = Path(args.out) if args.out else OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for i, spec in enumerate(specs, 1):
            row = score_one(spec, relax=args.relax)
            rows.append(row)
            writer.writerow(row)
            fh.flush()  # a long run must be readable while it is still going
            if i % 5 == 0 or i == len(specs):
                ok = sum(1 for r in rows if not r["error"])
                log.info("  %d/%d (%d scored, %d failed)", i, len(specs), ok, len(rows) - ok)

    failed = [r for r in rows if r["error"]]
    log.info("wrote %s (%d rows, %d failed)", out_path, len(rows), len(failed))
    for r in failed[:5]:
        log.warning("  %s: %s", r["full_name"], r["error"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
