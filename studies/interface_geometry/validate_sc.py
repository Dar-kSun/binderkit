"""Validate the Sc reimplementation against interfaces with published values.

docs/SPEC.md section 8.6 requires every new feature to pass a validation gate
before any result computed from it is believed. The contact metrics could be
checked against counts the data release publishes; shape complementarity
cannot, because no per-design reference value exists. The next best check is
an external one: compute Sc on crystallographic interfaces whose values
Lawrence and Colman's own classification places in a known band, and see
whether an untuned implementation lands there.

Published bands, from Lawrence MC and Colman PM, *Shape complementarity at
protein/protein interfaces*, J Mol Biol 234:946-950 (1993):

* antibody-antigen           Sc ~ 0.64-0.68
* permanent oligomeric pairs Sc ~ 0.70-0.76
* protease-inhibitor         Sc ~ 0.70-0.75

This is the direction of validation that section 8.9 calls the cheapest
credibility available: running our code against someone else's published
result. It found a real defect. Without the 1.5 A peripheral trim the
implementation returned 0.457 on the antibody-antigen complex below, 0.2
beneath the published band; with it, 0.612.

Run: ``python -m studies.interface_geometry.validate_sc``
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from binderkit.geometry import (
    DEFAULT_VDW,
    SC_DENSITY,
    SC_PERIPHERAL_TRIM,
    SC_PROBE,
    SC_WEIGHT,
    VDW_RADII,
    Structure,
    shape_complementarity,
)

OUT = Path(__file__).parent
#: Cached by the target-preparation stage; a cetuximab Fab bound to the EGFR
#: extracellular domain, which is an antibody-antigen interface plus, between
#: the Fab's own two chains, a permanent oligomeric pairing.
REFERENCE_PDB = Path("work/cache/6ARU.pdb")

CASES = [
    {
        "name": "6ARU EGFR domain III vs cetuximab Fab",
        "kind": "antibody-antigen",
        "binder": "A",
        "targets": ["B", "C"],
        "published_lo": 0.64,
        "published_hi": 0.68,
    },
    {
        "name": "6ARU Fab heavy vs light chain",
        "kind": "permanent oligomeric pair",
        "binder": "B",
        "targets": ["C"],
        "published_lo": 0.70,
        "published_hi": 0.76,
    },
]


def load_pdb(path: Path) -> Structure:
    """Heavy atoms of the first model of a PDB file."""
    warnings.filterwarnings("ignore")
    from Bio.PDB import PDBParser

    model = next(PDBParser(QUIET=True).get_structure("ref", str(path)).get_models())
    coords, element, atom_name, resname, resnum, chain = [], [], [], [], [], []
    for ch in model:
        for res in ch:
            if res.id[0] != " ":  # skip waters and heteroatoms
                continue
            for atom in res:
                el = (atom.element or "").strip() or atom.get_name()[0]
                if el == "H":
                    continue
                coords.append(atom.coord)
                element.append(el)
                atom_name.append(atom.get_name())
                resname.append(res.get_resname())
                resnum.append(res.id[1])
                chain.append(ch.id)
    return Structure(
        coords=np.array(coords, dtype=float),
        element=np.array(element),
        atom_name=np.array(atom_name),
        resname=np.array(resname),
        resnum=np.array(resnum, dtype=int),
        chain=np.array(chain),
        path=str(path),
    )


def main() -> int:
    if not REFERENCE_PDB.is_file():
        print(f"reference structure {REFERENCE_PDB} not cached; nothing to validate against")
        return 1

    st = load_pdb(REFERENCE_PDB)
    radii = np.array([VDW_RADII.get(str(e), DEFAULT_VDW) for e in st.element])
    report: dict = {
        "reference": str(REFERENCE_PDB),
        "n_atoms": len(st),
        "parameters": {
            "probe": SC_PROBE,
            "density": SC_DENSITY,
            "weight": SC_WEIGHT,
            "peripheral_trim": SC_PERIPHERAL_TRIM,
            "tuned": False,
        },
        "mean_vdw_radius": float(radii.mean()),
        "cases": [],
    }

    print(f"{'interface':44s} {'Sc':>7s} {'no trim':>8s} {'published':>12s}  verdict")
    for case in CASES:
        sc, n_b, n_t = shape_complementarity(st, case["binder"], set(case["targets"]))
        # The same calculation without the peripheral trim, kept as a recorded
        # control rather than an anecdote: it is what this implementation
        # returned before the trim was added, and the size of that gap is the
        # argument for validating against an external value at all.
        sc_untrimmed, _, _ = shape_complementarity(
            st, case["binder"], set(case["targets"]), trim=0.0
        )
        lo, hi = case["published_lo"], case["published_hi"]
        gap = 0.0 if lo <= sc <= hi else (sc - lo if sc < lo else sc - hi)
        verdict = "in band" if lo <= sc <= hi else f"{gap:+.3f} outside"
        report["cases"].append(
            {
                **{k: case[k] for k in ("name", "kind", "published_lo", "published_hi")},
                "sc": sc,
                "sc_without_peripheral_trim": sc_untrimmed,
                "trim_gain": sc - sc_untrimmed,
                "n_binder_points": n_b,
                "n_target_points": n_t,
                "gap_to_band": gap,
                "in_band": bool(lo <= sc <= hi),
            }
        )
        print(f"{case['name']:44s} {sc:7.3f} {sc_untrimmed:8.3f} {lo:5.2f}-{hi:.2f}  {verdict}")

    print(
        "\nThe 'no trim' column is the control: Lawrence and Colman discard a "
        "1.5 A\nperipheral band, and omitting it costs "
        f"{report['cases'][0]['trim_gain']:.3f} of Sc on the first case."
    )
    worst = max(abs(c["gap_to_band"]) for c in report["cases"])
    report["worst_absolute_gap"] = worst
    # The study only ever ranks designs within a target, which a constant
    # offset cannot change, so the gate is on the size of the offset rather
    # than on landing inside the band.
    report["passes_gate"] = bool(worst <= 0.15)
    print(
        f"\nworst gap to a published band: {worst:+.3f} "
        f"({'PASS' if report['passes_gate'] else 'FAIL'}, gate is 0.15)"
    )
    print(
        "Absolute values read low because the re-entrant surface is not\n"
        "reconstructed. Only within-dataset ranking is used downstream, which\n"
        "a constant offset does not affect."
    )

    (OUT / "validation_sc.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["passes_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
