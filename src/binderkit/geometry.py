"""Interface geometry on real coordinates (docs/SPEC.md section 6.1).

Session 1 left every interface metric as NaN because Tier C produced no
structures. The Anthropic release ships design models for 1,309 designs with
known wet-lab outcomes, so these can now be computed for real and, crucially,
**validated**: the release publishes its own epitope contact counts computed on
the same files, so an implementation that disagrees with them is wrong.

Definitions follow the release (`docs/INSILICO.md` section 8) so the two are
comparable:

* a **contact** is a target heavy atom within **5 Å** of a binder heavy atom;
* the **epitope** is the set of target residues with at least one such contact.

SASA is Shrake-Rupley, which is exact enough for buried-surface-area work and
is pure Python plus NumPy.
"""

from __future__ import annotations

import gzip
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

CONTACT_CUTOFF_A = 5.0
CLASH_CUTOFF_A = 2.2  # release definition for a uniform clash recount
HBOND_CUTOFF_A = 3.5  # N/O donor-acceptor heavy-atom distance
SALT_BRIDGE_CUTOFF_A = 4.0

#: Bondi van der Waals radii, Angstrom. Hydrogens are absent from these models.
VDW_RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "SE": 1.90}
DEFAULT_VDW = 1.70
PROBE_RADIUS = 1.4

#: Charged side-chain atoms, for salt bridges.
POSITIVE_ATOMS = {
    ("ARG", "NH1"),
    ("ARG", "NH2"),
    ("ARG", "NE"),
    ("LYS", "NZ"),
    ("HIS", "ND1"),
    ("HIS", "NE2"),
}
NEGATIVE_ATOMS = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}

HYDROPHOBIC_RES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "TYR"}

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


@dataclass
class Structure:
    """Heavy atoms of one model, as flat arrays."""

    coords: np.ndarray  # (N, 3)
    element: np.ndarray  # (N,)
    atom_name: np.ndarray  # (N,)
    resname: np.ndarray  # (N,)
    resnum: np.ndarray  # (N,)
    chain: np.ndarray  # (N,)
    path: str = ""

    def __len__(self) -> int:
        return len(self.coords)

    @property
    def chains(self) -> list[str]:
        return sorted(set(self.chain.tolist()))

    def chain_mask(self, chains: set[str]) -> np.ndarray:
        return np.isin(self.chain, list(chains))

    def sequence_of(self, chain: str) -> str:
        """One-letter sequence of a chain, in residue-number order."""
        m = self.chain == chain
        seen: dict[int, str] = {}
        for rn, rname in zip(self.resnum[m], self.resname[m], strict=True):
            seen.setdefault(int(rn), THREE_TO_ONE.get(str(rname), "X"))
        return "".join(seen[k] for k in sorted(seen))

    def residue_keys(self) -> np.ndarray:
        """Per-atom `chain:resnum` labels."""
        return np.array([f"{c}:{int(r)}" for c, r in zip(self.chain, self.resnum, strict=True)])


def parse_mmcif(path: Path) -> Structure:
    """Parse the `_atom_site` loop of an mmCIF file. Heavy atoms only.

    Deliberately minimal: these files are machine-generated with a stable
    layout, and a full CIF parser would be a dependency for no gain. Reads
    column positions from the loop header rather than assuming an order.
    """
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    cols: list[str] = []
    rows: list[list[str]] = []
    in_loop = False
    with opener(p, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("_atom_site."):
                in_loop = True
                cols.append(s.split(".", 1)[1])
                continue
            if in_loop:
                if not s or s.startswith(("#", "loop_", "data_", "_")):
                    break
                parts = s.split()
                if len(parts) < len(cols):
                    break
                rows.append(parts)
    if not rows:
        raise ValueError(f"no atom_site records in {p}")

    idx = {c: i for i, c in enumerate(cols)}

    def col(name: str, alt: str | None = None) -> int | None:
        if name in idx:
            return idx[name]
        return idx.get(alt) if alt else None

    i_sym = col("type_symbol")
    i_atom = col("label_atom_id", "auth_atom_id")
    i_res = col("label_comp_id", "auth_comp_id")
    i_seq = col("label_seq_id", "auth_seq_id")
    i_ch = col("label_asym_id", "auth_asym_id")
    i_x, i_y, i_z = idx["Cartn_x"], idx["Cartn_y"], idx["Cartn_z"]

    coords, elements, names, resnames, resnums, chains = [], [], [], [], [], []
    for r in rows:
        el = (r[i_sym] if i_sym is not None else r[i_atom][:1]).strip().upper()
        if el == "H" or el == "D":
            continue
        try:
            xyz = (float(r[i_x]), float(r[i_y]), float(r[i_z]))
        except ValueError:
            continue
        seq_raw = r[i_seq] if i_seq is not None else "0"
        try:
            seq = int(seq_raw)
        except ValueError:
            continue
        coords.append(xyz)
        elements.append(el)
        names.append(r[i_atom].strip().strip('"'))
        resnames.append(r[i_res].strip().upper())
        resnums.append(seq)
        chains.append(r[i_ch].strip())

    return Structure(
        coords=np.asarray(coords, dtype=float),
        element=np.asarray(elements),
        atom_name=np.asarray(names),
        resname=np.asarray(resnames),
        resnum=np.asarray(resnums, dtype=int),
        chain=np.asarray(chains),
        path=str(p),
    )


def is_protein_chain(st: Structure, chain: str, min_fraction: float = 0.8) -> bool:
    """Whether a chain is protein rather than nucleic acid or ligand.

    Several targets in the release are not protein-only: Cas9 is modelled as a
    ribonucleoprotein with an sgRNA chain, RBX1 carries three zinc ions and
    15-PGDH an NAD cofactor. Counting binder-to-RNA or binder-to-ligand atom
    pairs as interface contacts inflates the count badly - on some Cas9 designs
    by a factor of eighty - and makes the numbers incomparable with the
    release, which counts protein target chains only.
    """
    m = st.chain == chain
    if m.sum() == 0:
        return False
    names = st.resname[m]
    standard = np.isin(names, list(THREE_TO_ONE))
    return bool(standard.mean() >= min_fraction)


def identify_binder_chain(st: Structure, binder_sequence: str) -> tuple[str, set[str]]:
    """Return (binder chain, protein target chains), matched by sequence.

    Non-protein chains are excluded from the target set; see
    `is_protein_chain` for why that matters.

    Matching by sequence rather than by chain order or size: chain naming in
    these files is not guaranteed, and some targets are oligomeric, so "the
    small one" is not a safe rule either. Falls back to the chain whose length
    is closest to the binder's if no good match is found.
    """
    target_len = len(binder_sequence)
    best_chain, best_score = None, -1.0
    for ch in st.chains:
        seq = st.sequence_of(ch)
        if not seq:
            continue
        n = min(len(seq), target_len)
        if n == 0:
            continue
        matches = sum(1 for a, b in zip(seq[:n], binder_sequence[:n], strict=False) if a == b)
        score = matches / max(len(seq), target_len)
        if score > best_score:
            best_chain, best_score = ch, score
    if best_chain is None:
        raise ValueError("no chains in structure")
    if best_score < 0.5:
        # Fall back on length, and say so: a weak match usually means the model
        # is a threaded backbone whose residue identities differ.
        by_len = min(st.chains, key=lambda c: abs(len(st.sequence_of(c)) - target_len))
        log.debug("weak binder match (%.2f) in %s; falling back to length", best_score, st.path)
        best_chain = by_len
    targets = {c for c in st.chains if c != best_chain and is_protein_chain(st, c)}
    return best_chain, targets


# --------------------------------------------------------------------------
# SASA
# --------------------------------------------------------------------------


def _sphere_points(n: int = 92) -> np.ndarray:
    """Evenly distributed points on a unit sphere (golden spiral)."""
    i = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


#: An atom's SASA can only change on binding if a partner atom comes within two
#: expanded radii. The largest expanded radius here is 1.9 + 1.4 = 3.3, so 6.6 A
#: is the true bound; 10 A is used for margin.
BSA_ZONE_A = 10.0


def shrake_rupley(
    coords: np.ndarray,
    elements: np.ndarray,
    n_points: int = 92,
    subset: np.ndarray | None = None,
) -> np.ndarray:
    """Per-atom solvent-accessible surface area, Angstrom squared.

    Standard Shrake-Rupley: roll a probe over each atom's expanded sphere and
    count the fraction of test points not buried by a neighbour. A KD-tree
    keeps the neighbour search near-linear, with a cutoff equal to the largest
    possible pair of expanded radii so no occluder is missed.

    `subset` restricts which atoms are *scored* while every atom in `coords`
    still occludes. That is what makes buried-surface-area tractable over a
    thousand complexes: an atom far from the partner chain has identical SASA
    bound and unbound, so it contributes exactly zero to BSA and need not be
    evaluated at all. Atoms outside `subset` come back as NaN rather than 0, so
    a caller that forgets to mask gets an obvious failure instead of a silently
    wrong total.
    """
    from scipy.spatial import cKDTree

    n = len(coords)
    if n == 0:
        return np.zeros(0)
    radii = np.array([VDW_RADII.get(str(e), DEFAULT_VDW) for e in elements]) + PROBE_RADIUS
    sphere = _sphere_points(n_points)
    tree = cKDTree(coords)
    max_r = float(radii.max())

    if subset is None:
        scored = range(n)
        areas = np.zeros(n)
    else:
        scored = np.asarray(subset, dtype=int)
        areas = np.full(n, np.nan)

    for i in scored:
        r_i = radii[i]
        neighbours = tree.query_ball_point(coords[i], r_i + max_r)
        neighbours = [j for j in neighbours if j != i]
        test = coords[i] + sphere * r_i
        if neighbours:
            nb = coords[neighbours]
            nr = radii[neighbours]
            d2 = ((test[:, None, :] - nb[None, :, :]) ** 2).sum(axis=2)
            buried = (d2 < (nr**2)[None, :]).any(axis=1)
            accessible = float((~buried).sum()) / n_points
        else:
            accessible = 1.0
        areas[i] = 4.0 * np.pi * r_i**2 * accessible
    return areas


# --------------------------------------------------------------------------
# Interface metrics
# --------------------------------------------------------------------------


#: Atoms present even in a backbone-only model. Anything else is a side chain.
BACKBONE_ATOMS = {"N", "CA", "C", "O", "CB", "OXT"}


def has_side_chains(st: Structure, chains: set[str], min_fraction: float = 0.15) -> bool:
    """Whether `chains` carry real side chains, not just backbone plus C-beta.

    This matters more than it looks. 975 of the 1,309 released design models
    have `design_model_status == "ordered_sequence_backbone"`, and in those the
    **binder** chain contains only N, CA, C, O and CB while the target carries
    full side chains. Hydrogen bonds and salt bridges involving a binder side
    chain are then not absent, they are *unobservable*, and counting them
    returns 0 for a reason that has nothing to do with the design.

    Reporting that 0 would be a silent lie, so callers use this to emit NaN
    instead.
    """
    m = st.chain_mask(chains)
    if m.sum() == 0:
        return False
    names = st.atom_name[m]
    non_backbone = np.array([str(a) not in BACKBONE_ATOMS for a in names])
    return bool(non_backbone.mean() >= min_fraction)


@dataclass
class InterfaceMetrics:
    """Everything computable from one complex, on real coordinates.

    Side-chain-dependent fields are NaN when the relevant chain lacks side
    chains; see `has_side_chains`.
    """

    design_id: str = ""
    binder_chain: str = ""
    target_chains: str = ""
    binder_has_side_chains: bool = False
    target_has_side_chains: bool = False
    n_binder_residues: int = 0
    n_target_residues: int = 0
    # contacts, release-compatible
    n_atom_contacts: int = 0
    n_epitope_residues: int = 0
    n_paratope_residues: int = 0
    min_interface_distance: float = float("nan")
    n_clashes: int = 0
    # buried surface
    bsa_total: float = float("nan")
    bsa_binder: float = float("nan")
    bsa_hydrophobic_fraction: float = float("nan")
    # chemistry - NaN, not 0, when the binder has no side chains to form them
    n_hbonds: float = float("nan")
    n_salt_bridges: float = float("nan")
    # composition and shape
    interface_hydrophobic_fraction: float = float("nan")
    interface_charged_fraction: float = float("nan")
    binder_sasa_alone: float = float("nan")
    binder_exposed_hydrophobic_sasa: float = float("nan")
    radius_of_gyration: float = float("nan")
    contact_density: float = float("nan")
    notes: list[str] = field(default_factory=list)


def compute_interface(
    st: Structure, binder_chain: str, target_chains: set[str], design_id: str = ""
) -> InterfaceMetrics:
    """Compute every interface metric for one complex."""
    from scipy.spatial import cKDTree

    m = InterfaceMetrics(
        design_id=design_id,
        binder_chain=binder_chain,
        target_chains=",".join(sorted(target_chains)),
    )
    bm = st.chain == binder_chain
    tm = st.chain_mask(target_chains)
    if bm.sum() == 0 or tm.sum() == 0:
        m.notes.append("missing binder or target chain")
        return m

    bc, tc = st.coords[bm], st.coords[tm]
    m.n_binder_residues = len(set(st.resnum[bm].tolist()))
    m.n_target_residues = len(set(zip(st.chain[tm].tolist(), st.resnum[tm].tolist(), strict=True)))

    # --- contacts, exactly the release definition -----------------------
    tree_t = cKDTree(tc)
    pairs = tree_t.query_ball_point(bc, CONTACT_CUTOFF_A)
    n_contacts = sum(len(p) for p in pairs)
    m.n_atom_contacts = int(n_contacts)

    binder_keys = st.residue_keys()[bm]
    target_keys = st.residue_keys()[tm]
    epitope, paratope = set(), set()
    min_d = np.inf
    for bi, plist in enumerate(pairs):
        if plist:
            paratope.add(binder_keys[bi])
            for tj in plist:
                epitope.add(target_keys[tj])
            d = np.sqrt(((tc[plist] - bc[bi]) ** 2).sum(axis=1)).min()
            min_d = min(min_d, float(d))
    m.n_epitope_residues = len(epitope)
    m.n_paratope_residues = len(paratope)
    m.min_interface_distance = float(min_d) if np.isfinite(min_d) else float("nan")
    m.n_clashes = int(sum(len(p) for p in tree_t.query_ball_point(bc, CLASH_CUTOFF_A)))

    # --- buried surface area --------------------------------------------
    # Only atoms near the partner chain can change SASA on binding, so BSA is
    # computed on that zone alone. Exact, and roughly fifty times cheaper than
    # scoring every atom of a 5,000-atom complex three times over.
    b_idx_all = np.flatnonzero(bm)
    t_idx_all = np.flatnonzero(tm)

    def _zone(own: np.ndarray, other: np.ndarray) -> np.ndarray:
        """Indices into `own` that lie within the BSA zone of `other`.

        Returns an empty array when the two chains never approach, which
        happens for a handful of designs whose model places the binder away
        from the target entirely.
        """
        if len(own) == 0 or len(other) == 0:
            return np.array([], dtype=int)
        hits = [
            np.asarray(p, dtype=int)
            for p in cKDTree(own).query_ball_point(other, BSA_ZONE_A)
            if len(p)
        ]
        return np.unique(np.concatenate(hits)) if hits else np.array([], dtype=int)

    b_zone_local = _zone(bc, tc)
    t_zone_local = _zone(tc, bc)

    if len(b_zone_local) == 0 or len(t_zone_local) == 0:
        m.bsa_total = 0.0
        m.bsa_binder = 0.0
        m.bsa_hydrophobic_fraction = float("nan")
        m.notes.append("no atoms within the BSA zone: chains are not in contact")
    else:
        sasa_cx = shrake_rupley(
            st.coords,
            st.element,
            subset=np.concatenate([b_idx_all[b_zone_local], t_idx_all[t_zone_local]]),
        )
        sasa_b_alone = shrake_rupley(bc, st.element[bm], subset=b_zone_local)
        sasa_t_alone = shrake_rupley(tc, st.element[tm], subset=t_zone_local)

        buried_b = sasa_b_alone[b_zone_local] - sasa_cx[b_idx_all[b_zone_local]]
        buried_t = sasa_t_alone[t_zone_local] - sasa_cx[t_idx_all[t_zone_local]]
        m.bsa_binder = float(buried_b.sum())
        m.bsa_total = float(buried_b.sum() + buried_t.sum())

        hydro_zone = np.isin(st.resname[b_idx_all[b_zone_local]], list(HYDROPHOBIC_RES))
        denom = buried_b.sum()
        m.bsa_hydrophobic_fraction = (
            float(buried_b[hydro_zone].sum() / denom) if denom > 0 else float("nan")
        )

    # Whole-binder SASA is cheap (the binder is small) and is needed in full.
    sasa_binder_full = shrake_rupley(bc, st.element[bm])
    m.binder_sasa_alone = float(sasa_binder_full.sum())
    hydro_b = np.isin(st.resname[bm], list(HYDROPHOBIC_RES))
    m.binder_exposed_hydrophobic_sasa = float(sasa_binder_full[hydro_b].sum())

    # --- hydrogen bonds and salt bridges ---------------------------------
    # Only meaningful when BOTH partners carry side chains. Where the binder is
    # a backbone-plus-CB model these stay NaN, because a count of zero there
    # would read as "this design forms no hydrogen bonds" when the truth is
    # "the model cannot show them".
    m.binder_has_side_chains = has_side_chains(st, {binder_chain})
    m.target_has_side_chains = has_side_chains(st, target_chains)

    if m.binder_has_side_chains and m.target_has_side_chains:
        polar_b = np.flatnonzero(bm)[np.isin(st.element[bm], ["N", "O"])]
        polar_t = np.flatnonzero(tm)[np.isin(st.element[tm], ["N", "O"])]
        if len(polar_b) and len(polar_t):
            tree_pt = cKDTree(st.coords[polar_t])
            m.n_hbonds = float(
                sum(len(p) for p in tree_pt.query_ball_point(st.coords[polar_b], HBOND_CUTOFF_A))
            )

        def charged(indices: np.ndarray, table: set) -> np.ndarray:
            return np.array(
                [i for i in indices if (str(st.resname[i]), str(st.atom_name[i])) in table],
                dtype=int,
            )

        b_idx, t_idx = np.flatnonzero(bm), np.flatnonzero(tm)
        bpos, bneg = charged(b_idx, POSITIVE_ATOMS), charged(b_idx, NEGATIVE_ATOMS)
        tpos, tneg = charged(t_idx, POSITIVE_ATOMS), charged(t_idx, NEGATIVE_ATOMS)
        n_sb = 0
        for a, b in ((bpos, tneg), (bneg, tpos)):
            if len(a) and len(b):
                n_sb += int(
                    sum(
                        len(p)
                        for p in cKDTree(st.coords[b]).query_ball_point(
                            st.coords[a], SALT_BRIDGE_CUTOFF_A
                        )
                    )
                )
        m.n_salt_bridges = float(n_sb)
    else:
        m.notes.append(
            "hbonds/salt bridges NOT COMPUTABLE: "
            f"binder_side_chains={m.binder_has_side_chains}, "
            f"target_side_chains={m.target_has_side_chains}"
        )

    # --- composition and shape -------------------------------------------
    epi_res = {k.split(":")[0] + ":" + k.split(":")[1] for k in epitope | paratope}
    keys_all = st.residue_keys()
    iface_mask = np.isin(keys_all, list(epi_res))
    if iface_mask.any():
        names = st.resname[iface_mask]
        m.interface_hydrophobic_fraction = float(np.isin(names, list(HYDROPHOBIC_RES)).mean())
        m.interface_charged_fraction = float(
            np.isin(names, ["ARG", "LYS", "ASP", "GLU", "HIS"]).mean()
        )
    centre = bc.mean(axis=0)
    m.radius_of_gyration = float(np.sqrt(((bc - centre) ** 2).sum(axis=1).mean()))
    m.contact_density = (
        m.n_atom_contacts / m.n_binder_residues if m.n_binder_residues else float("nan")
    )
    return m
