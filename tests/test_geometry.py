"""Tests for interface geometry (docs/SPEC.md section 13).

The headline validation - exact agreement with the release's published contact
counts on 981 designs - runs in the study itself. These tests cover the
properties that validation cannot: that side-chain absence produces NaN rather
than zero, that non-protein chains are excluded, and that the zone-restricted
SASA shortcut is exact.
"""

from __future__ import annotations

import numpy as np
import pytest

from binderkit.geometry import (
    BACKBONE_ATOMS,
    CONTACT_CUTOFF_A,
    Structure,
    compute_interface,
    has_side_chains,
    identify_binder_chain,
    is_protein_chain,
    shrake_rupley,
)


def make_structure(spec: list[tuple[str, int, str, str, str, tuple[float, float, float]]]):
    """Build a Structure from (chain, resnum, resname, atom, element, xyz) tuples."""
    return Structure(
        coords=np.array([s[5] for s in spec], dtype=float),
        element=np.array([s[4] for s in spec]),
        atom_name=np.array([s[3] for s in spec]),
        resname=np.array([s[2] for s in spec]),
        resnum=np.array([s[1] for s in spec], dtype=int),
        chain=np.array([s[0] for s in spec]),
        path="synthetic",
    )


def two_chain_complex(binder_full_sidechains: bool) -> Structure:
    """A tiny binder/target pair placed 4 A apart, so they are in contact."""
    spec = []
    for i in range(4):
        x = float(i * 3.0)
        spec += [
            ("A", i + 1, "LYS", "N", "N", (x, 0.0, 0.0)),
            ("A", i + 1, "LYS", "CA", "C", (x, 1.0, 0.0)),
            ("A", i + 1, "LYS", "C", "C", (x, 2.0, 0.0)),
            ("A", i + 1, "LYS", "O", "O", (x, 2.5, 0.0)),
            ("A", i + 1, "LYS", "CB", "C", (x, 1.5, 1.0)),
        ]
        if binder_full_sidechains:
            spec += [
                ("A", i + 1, "LYS", "CG", "C", (x, 1.8, 2.0)),
                ("A", i + 1, "LYS", "CD", "C", (x, 2.1, 3.0)),
                ("A", i + 1, "LYS", "NZ", "N", (x, 2.4, 3.6)),
            ]
    for i in range(4):
        x = float(i * 3.0)
        spec += [
            ("B", i + 1, "ASP", "N", "N", (x, 0.0, 4.2)),
            ("B", i + 1, "ASP", "CA", "C", (x, 1.0, 4.2)),
            ("B", i + 1, "ASP", "C", "C", (x, 2.0, 4.2)),
            ("B", i + 1, "ASP", "O", "O", (x, 2.5, 4.2)),
            ("B", i + 1, "ASP", "CB", "C", (x, 1.5, 5.0)),
            ("B", i + 1, "ASP", "CG", "C", (x, 2.0, 5.6)),
            ("B", i + 1, "ASP", "OD1", "O", (x, 2.4, 4.6)),
            ("B", i + 1, "ASP", "OD2", "O", (x, 2.6, 6.2)),
        ]
    return make_structure(spec)


# --------------------------------------------------------------------------
# Side chains
# --------------------------------------------------------------------------


def test_backbone_only_chain_is_detected() -> None:
    st = two_chain_complex(binder_full_sidechains=False)
    assert not has_side_chains(st, {"A"}), "backbone + CB must not count as side chains"
    assert has_side_chains(st, {"B"})


def test_full_sidechain_chain_is_detected() -> None:
    st = two_chain_complex(binder_full_sidechains=True)
    assert has_side_chains(st, {"A"})


def test_hbonds_are_nan_not_zero_without_binder_side_chains() -> None:
    """The bug this guards against: 975 of 1,309 released models are backbone
    only, and counting zero hydrogen bonds there reads as a property of the
    design rather than of the model.
    """
    st = two_chain_complex(binder_full_sidechains=False)
    m = compute_interface(st, "A", {"B"}, "backbone_only")
    assert np.isnan(m.n_hbonds), "unobservable must be NaN, not 0"
    assert np.isnan(m.n_salt_bridges)
    assert any("NOT COMPUTABLE" in n for n in m.notes)
    assert m.binder_has_side_chains is False


def test_hbonds_are_counted_when_side_chains_exist() -> None:
    st = two_chain_complex(binder_full_sidechains=True)
    m = compute_interface(st, "A", {"B"}, "all_atom")
    assert not np.isnan(m.n_hbonds)
    assert m.n_hbonds >= 0
    assert m.binder_has_side_chains is True


def test_backbone_atom_set_is_what_we_think() -> None:
    assert {"N", "CA", "C", "O", "CB"} <= BACKBONE_ATOMS


# --------------------------------------------------------------------------
# Chain selection
# --------------------------------------------------------------------------


def test_rna_and_ligand_chains_are_not_targets() -> None:
    """Regression: counting binder-to-RNA and binder-to-ligand pairs as
    interface contacts inflated some Cas9 designs by ~80x and broke agreement
    with the release.
    """
    st = two_chain_complex(binder_full_sidechains=True)
    extra = make_structure(
        [("R", 1, "G", "P", "P", (0.0, 0.0, 6.0)), ("R", 2, "A", "P", "P", (3.0, 0.0, 6.0))]
    )
    merged = Structure(
        coords=np.vstack([st.coords, extra.coords]),
        element=np.concatenate([st.element, extra.element]),
        atom_name=np.concatenate([st.atom_name, extra.atom_name]),
        resname=np.concatenate([st.resname, extra.resname]),
        resnum=np.concatenate([st.resnum, extra.resnum]),
        chain=np.concatenate([st.chain, extra.chain]),
    )
    assert is_protein_chain(merged, "A")
    assert is_protein_chain(merged, "B")
    assert not is_protein_chain(merged, "R"), "an RNA chain is not a protein target"

    binder, targets = identify_binder_chain(merged, "KKKK")
    assert binder == "A"
    assert targets == {"B"}, "the RNA chain must be excluded from the target set"


def test_binder_identified_by_sequence_not_size() -> None:
    st = two_chain_complex(binder_full_sidechains=True)
    binder, targets = identify_binder_chain(st, "KKKK")
    assert binder == "A"
    assert targets == {"B"}
    binder2, targets2 = identify_binder_chain(st, "DDDD")
    assert binder2 == "B"
    assert targets2 == {"A"}


# --------------------------------------------------------------------------
# SASA
# --------------------------------------------------------------------------


def test_isolated_atom_sasa_matches_sphere_area() -> None:
    coords = np.array([[0.0, 0.0, 0.0]])
    area = shrake_rupley(coords, np.array(["C"]))[0]
    expected = 4 * np.pi * (1.70 + 1.4) ** 2
    assert area == pytest.approx(expected, rel=1e-9)


def test_subset_restriction_matches_full_computation() -> None:
    """The zone shortcut must be exact, not approximate: it is what makes BSA
    affordable over a thousand complexes.
    """
    rng = np.random.default_rng(0)
    coords = rng.normal(scale=5.0, size=(60, 3))
    elements = np.array(["C"] * 60)
    full = shrake_rupley(coords, elements)
    subset = np.array([3, 7, 11, 25, 40])
    partial = shrake_rupley(coords, elements, subset=subset)
    assert np.allclose(full[subset], partial[subset])
    outside = np.setdiff1d(np.arange(60), subset)
    assert np.isnan(partial[outside]).all(), "unscored atoms must be NaN, not 0"


def test_buried_atom_has_less_sasa_than_exposed() -> None:
    coords = np.array(
        [[0.0, 0.0, 0.0]]
        + [[2.5 * x, 0.0, 0.0] for x in (-1, 1)]
        + [[0.0, 2.5 * y, 0.0] for y in (-1, 1)]
        + [[0.0, 0.0, 2.5 * z] for z in (-1, 1)]
    )
    areas = shrake_rupley(coords, np.array(["C"] * len(coords)))
    assert areas[0] < areas[1:].min(), "the central atom must be the most buried"


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------


def test_contacts_use_the_release_cutoff() -> None:
    assert CONTACT_CUTOFF_A == 5.0


def test_non_contacting_chains_give_zero_bsa_and_a_note() -> None:
    """Regression: this crashed with "need at least one array to concatenate"."""
    far = make_structure(
        [
            ("A", 1, "ALA", "CA", "C", (0.0, 0.0, 0.0)),
            ("B", 1, "ALA", "CA", "C", (500.0, 0.0, 0.0)),
        ]
    )
    m = compute_interface(far, "A", {"B"}, "far_apart")
    assert m.n_atom_contacts == 0
    assert m.bsa_total == 0.0
    assert any("not in contact" in n for n in m.notes)


def test_interface_metrics_are_populated_for_a_contacting_pair() -> None:
    st = two_chain_complex(binder_full_sidechains=True)
    m = compute_interface(st, "A", {"B"}, "contacting")
    assert m.n_atom_contacts > 0
    assert m.n_epitope_residues > 0
    assert m.n_paratope_residues > 0
    assert m.bsa_total > 0
    assert np.isfinite(m.min_interface_distance)
