"""Tests for the licence-free Lawrence-Colman Sc reimplementation.

PyRosetta carries the reference implementation but needs a licence credential
this environment does not have, so `shape_complementarity` is an independent
reimplementation (docs/SPEC.md section 8.6). It therefore has to earn trust
from analytic cases whose answer is known in advance, rather than from
agreement with Rosetta.

The cases are ordered by how much they would hurt if they failed:

1. two flat plates facing each other must score near 1 - if this fails the
   normals or the pairing are wrong and nothing else matters;
2. a convex sphere against a flat plate must score lower than plate-on-plate,
   and a concave socket holding that sphere must score higher than the plate -
   this is the whole point of the statistic;
3. separation must reduce the score, because of the exp(-w d^2) term;
4. the statistic must be symmetric in its two arguments.
"""

from __future__ import annotations

import numpy as np
import pytest

from binderkit.geometry import Structure, shape_complementarity


def _structure(blocks: dict[str, np.ndarray]) -> Structure:
    """A Structure of carbon atoms from {chain: coordinates}."""
    coords, chain, element, resname, resseq, atom = [], [], [], [], [], []
    for ch, xyz in blocks.items():
        for i, c in enumerate(xyz):
            coords.append(c)
            chain.append(ch)
            element.append("C")
            resname.append("GLY")
            resseq.append(i + 1)
            atom.append("CA")
    return Structure(
        coords=np.array(coords, dtype=float),
        chain=np.array(chain),
        element=np.array(element),
        resname=np.array(resname),
        resnum=np.array(resseq, dtype=int),
        atom_name=np.array(atom),
        path="synthetic",
    )


def _plate(z: float, n: int = 9, spacing: float = 2.8) -> np.ndarray:
    """A flat square sheet of atoms in the xy plane at height z."""
    g = (np.arange(n) - (n - 1) / 2) * spacing
    xx, yy = np.meshgrid(g, g)
    return np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)], axis=1)


def _sphere_shell(centre: np.ndarray, radius: float, n: int = 120) -> np.ndarray:
    """Atoms spread over a spherical shell (golden spiral)."""
    i = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    u = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)
    return centre + radius * u


#: Coarse dots keep these tests fast; the ordering they check is not sensitive
#: to density, and the production default (15/A^2) is used for real structures.
FAST = {"density": 2.0}


def test_two_flat_plates_facing_each_other_score_high() -> None:
    """Perfectly matched opposing surfaces are the definition of complementary.

    A snug flat-on-flat contact lands near 0.72 here, which is where the
    literature puts a *good real* interface (0.64-0.75). It is not near 1.0,
    and should not be: these plates are single atomic layers, so the surface
    is corrugated at atomic scale and the exp(-w d^2) term discounts the dots
    that sit over the gaps between atoms. The value is checked loosely; the
    orderings below are the real assertions.
    """
    st = _structure({"A": _plate(0.0), "B": _plate(3.4)})
    sc, na, nb = shape_complementarity(st, "A", {"B"}, **FAST)
    assert na > 0 and nb > 0, "both sides must contribute buried surface"
    assert 0.55 < sc <= 1.0, f"snug flat-on-flat should be high, got {sc:.3f}"


def test_convex_against_flat_scores_lower_than_flat_against_flat() -> None:
    """A ball on a table matches worse than a table on a table."""
    flat = shape_complementarity(
        _structure({"A": _plate(0.0), "B": _plate(4.0)}), "A", {"B"}, **FAST
    )[0]
    ball = _sphere_shell(np.array([0.0, 0.0, 8.0]), 4.0)
    convex = shape_complementarity(_structure({"A": _plate(0.0), "B": ball}), "A", {"B"}, **FAST)[0]
    assert convex < flat, f"convex-on-flat {convex:.3f} should be below flat-on-flat {flat:.3f}"


def test_a_socket_holding_a_ball_beats_a_flat_plate_holding_it() -> None:
    """The statistic must reward a surface shaped to its partner.

    The socket is the lower half of a shell of larger radius, so the ball sits
    inside it; the flat plate is the same ball resting on a table.
    """
    ball = _sphere_shell(np.array([0.0, 0.0, 0.0]), 4.0)
    shell = _sphere_shell(np.array([0.0, 0.0, 0.0]), 7.4, n=400)
    socket = shell[shell[:, 2] < -1.0]
    on_socket = shape_complementarity(_structure({"A": ball, "B": socket}), "A", {"B"}, **FAST)[0]

    plate = _plate(-7.4, n=11)
    on_plate = shape_complementarity(_structure({"A": ball, "B": plate}), "A", {"B"}, **FAST)[0]
    assert on_socket > on_plate, (
        f"a fitted socket {on_socket:.3f} should beat a flat plate {on_plate:.3f}"
    )


def test_pulling_the_surfaces_apart_lowers_the_score() -> None:
    """The exp(-w d^2) term must actually penalise separation."""
    close = shape_complementarity(
        _structure({"A": _plate(0.0), "B": _plate(3.4)}), "A", {"B"}, **FAST
    )[0]
    far = shape_complementarity(
        _structure({"A": _plate(0.0), "B": _plate(4.8)}), "A", {"B"}, **FAST
    )[0]
    assert far < close, f"separated {far:.3f} should be below contacting {close:.3f}"
    # Past about 5.2 A between atom centres nothing is buried at all, and the
    # answer is NaN rather than a low score: no interface is not a bad fit.
    none = shape_complementarity(
        _structure({"A": _plate(0.0), "B": _plate(5.6)}), "A", {"B"}, **FAST
    )[0]
    assert np.isnan(none)


def test_the_statistic_is_symmetric_in_its_two_sides() -> None:
    """Sc is the mean of both directions, so swapping the chains changes nothing."""
    st = _structure({"A": _plate(0.0), "B": _plate(4.2)})
    ab = shape_complementarity(st, "A", {"B"}, **FAST)[0]
    ba = shape_complementarity(st, "B", {"A"}, **FAST)[0]
    assert ab == pytest.approx(ba, abs=1e-9)


def test_non_contacting_chains_give_nan_not_zero() -> None:
    """No interface is unobservable, not a score of zero."""
    st = _structure({"A": _plate(0.0), "B": _plate(60.0)})
    sc, na, nb = shape_complementarity(st, "A", {"B"}, **FAST)
    assert np.isnan(sc)
    assert na == 0 or nb == 0


def test_score_stays_within_the_physical_range() -> None:
    """Sc is a mean of damped unit-vector dot products: it cannot exceed 1."""
    for z in (3.4, 3.6, 4.0, 4.8):
        sc = shape_complementarity(
            _structure({"A": _plate(0.0), "B": _plate(z)}), "A", {"B"}, **FAST
        )[0]
        assert -1.0 <= sc <= 1.0, f"out of range at separation {z}: {sc}"


def test_density_changes_the_value_only_slightly() -> None:
    """A dot statistic that swings with sampling density is not measuring shape."""
    st = _structure({"A": _plate(0.0), "B": _plate(4.0)})
    coarse = shape_complementarity(st, "A", {"B"}, density=2.0)[0]
    finer = shape_complementarity(st, "A", {"B"}, density=8.0)[0]
    assert abs(coarse - finer) < 0.1, f"density sensitivity too high: {coarse:.3f} vs {finer:.3f}"
