"""Target preparation (docs/SPEC.md section 4).

The commonest silent bug in this stage is a numbering mismatch between UniProt
and PDB numbering, so the mapping is built explicitly, carried in the TargetSpec,
and tested (`tests/test_target.py`).
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from binderkit.config import TargetConfig

log = logging.getLogger(__name__)

RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"
DOWNLOAD_TIMEOUT_S = 600  # docs/SPEC.md section 0.5: 10 min cap on a download

#: Three-letter to one-letter, standard residues only. Anything else is skipped
#: when reading SEQRES/ATOM records, which is what we want for a clean chain.
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
class Residue:
    """One observed residue of the target chain."""

    pdb_resnum: int
    icode: str
    resname: str
    one_letter: str


@dataclass
class TargetSpec:
    """Everything downstream stages need to know about the target."""

    name: str
    uniprot: str
    pdb_id: str
    chain: str
    structure_path: str
    sequence_uniprot: str
    sequence_observed: str
    #: PDB residue number -> UniProt residue number. Built by alignment.
    numbering_map: dict[int, int] = field(default_factory=dict)
    epitope_residues_uniprot: list[int] = field(default_factory=list)
    epitope_residues_pdb: list[int] = field(default_factory=list)
    hotspots_pdb: list[int] = field(default_factory=list)
    hotspot_method: str = ""
    epitope_rationale: str = ""
    ortholog_specs: dict[str, str] = field(default_factory=dict)
    conservation: dict[int, float] = field(default_factory=dict)

    def to_json(self, path: Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        # JSON object keys must be strings
        payload["numbering_map"] = {str(k): v for k, v in self.numbering_map.items()}
        payload["conservation"] = {str(k): v for k, v in self.conservation.items()}
        p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return p

    @classmethod
    def from_json(cls, path: Path) -> TargetSpec:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["numbering_map"] = {int(k): v for k, v in raw.get("numbering_map", {}).items()}
        raw["conservation"] = {int(k): v for k, v in raw.get("conservation", {}).items()}
        return cls(**raw)


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------


def _download(url: str, dest: Path) -> Path:
    """Download `url` to `dest`, caching: an existing non-empty file is reused."""
    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size > 0:
        log.info("cache hit %s", dest.name)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "binderkit/0.1"})
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_S) as resp:  # noqa: S310
        data = resp.read()
    dest.write_bytes(data)
    return dest


def fetch_structure(pdb_id: str, cache_dir: Path) -> Path:
    """Fetch a PDB entry from RCSB into `cache_dir`."""
    return _download(
        RCSB_PDB_URL.format(pdb_id=pdb_id.upper()),
        Path(cache_dir) / f"{pdb_id.upper()}.pdb",
    )


def fetch_uniprot_sequence(acc: str, cache_dir: Path) -> str:
    """Fetch a UniProt sequence as a plain string.

    Note: rest.uniprot.org answers HEAD with 403 but GET normally, so a health
    check must use GET. Isoform accessions such as ``P00533-1`` are served
    directly.
    """
    path = _download(
        UNIPROT_FASTA_URL.format(acc=acc),
        Path(cache_dir) / f"{acc}.fasta",
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    return "".join(ln.strip() for ln in lines if ln and not ln.startswith(">"))


# --------------------------------------------------------------------------
# Cleaning and numbering
# --------------------------------------------------------------------------


def parse_chain(pdb_path: Path, chain: str, keep_ligands: bool = False) -> list[Residue]:
    """Read one chain's standard residues from ATOM records, in order.

    Waters and heteroatoms are dropped unless `keep_ligands` is set. Altloc
    duplicates collapse to the first occurrence of each residue.
    """
    residues: list[Residue] = []
    seen: set[tuple[int, str]] = set()
    for line in Path(pdb_path).read_text(encoding="utf-8", errors="replace").splitlines():
        record = line[:6]
        if record not in ("ATOM  ", "HETATM"):
            continue
        if record == "HETATM" and not keep_ligands:
            continue
        if line[21:22] != chain:
            continue
        resname = line[17:20].strip().upper()
        if resname == "HOH":
            continue
        one = THREE_TO_ONE.get(resname)
        if one is None:
            continue
        try:
            resnum = int(line[22:26])
        except ValueError:
            continue
        icode = line[26:27].strip()
        key = (resnum, icode)
        if key in seen:
            continue
        seen.add(key)
        residues.append(Residue(resnum, icode, resname, one))
    residues.sort(key=lambda r: (r.pdb_resnum, r.icode))
    return residues


def build_numbering_map(
    observed: list[Residue],
    uniprot_sequence: str,
) -> tuple[dict[int, int], int]:
    """Map PDB residue numbers to UniProt positions (1-based).

    Strategy: find the offset that makes the observed one-letter sequence agree
    best with the UniProt sequence at the observed numbering. For a construct
    whose author numbered it by UniProt position the offset is 0, which is the
    common case and must come out exactly right.

    Returns the mapping and the number of residues whose identity agrees.
    """
    if not observed:
        return {}, 0

    best_offset, best_score = 0, -1
    lo = min(r.pdb_resnum for r in observed)
    # Try offsets that keep every observed residue inside the UniProt sequence.
    for offset in range(1 - lo, len(uniprot_sequence)):
        score = 0
        for r in observed:
            pos = r.pdb_resnum + offset
            if 1 <= pos <= len(uniprot_sequence) and uniprot_sequence[pos - 1] == r.one_letter:
                score += 1
        if score > best_score:
            best_offset, best_score = offset, score

    mapping = {
        r.pdb_resnum: r.pdb_resnum + best_offset
        for r in observed
        if 1 <= r.pdb_resnum + best_offset <= len(uniprot_sequence)
    }
    return mapping, best_score


# --------------------------------------------------------------------------
# Epitope and hotspots
# --------------------------------------------------------------------------


def epitope_from_range(
    spec_range: tuple[int, int],
    numbering_map: dict[int, int],
) -> tuple[list[int], list[int]]:
    """Residues inside a UniProt-numbered range, as (uniprot, pdb) lists."""
    lo, hi = spec_range
    inverse = {v: k for k, v in numbering_map.items()}
    uni = [u for u in range(lo, hi + 1) if u in inverse]
    pdb = [inverse[u] for u in uni]
    return uni, pdb


def surface_hotspots(
    pdb_path: Path,
    chain: str,
    candidate_pdb_residues: list[int],
    top_n: int = 12,
) -> tuple[list[int], str]:
    """Rank candidate residues as hotspots by a cheap geometric proxy.

    Method, stated plainly because it is reported: for each candidate residue,
    count heavy atoms of the same chain within 10 A of its side-chain centroid.
    Fewer neighbours means more exposed. Among exposed residues, hydrophobics
    score higher, since a hydrophobic patch is what a binder interface wants.
    This is a **proxy for solvent accessibility, not a real SASA calculation**,
    and is reported as such.
    """
    coords: dict[int, list[tuple[float, float, float]]] = {}
    resnames: dict[int, str] = {}
    for line in Path(pdb_path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line[:6] != "ATOM  " or line[21:22] != chain:
            continue
        atom = line[12:16].strip()
        if atom.startswith("H"):
            continue
        try:
            resnum = int(line[22:26])
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        coords.setdefault(resnum, []).append(xyz)
        resnames.setdefault(resnum, line[17:20].strip().upper())

    hydrophobic = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "TYR", "PRO"}
    all_atoms = [xyz for atoms in coords.values() for xyz in atoms]

    scored: list[tuple[float, int]] = []
    for resnum in candidate_pdb_residues:
        atoms = coords.get(resnum)
        if not atoms:
            continue
        cx = sum(a[0] for a in atoms) / len(atoms)
        cy = sum(a[1] for a in atoms) / len(atoms)
        cz = sum(a[2] for a in atoms) / len(atoms)
        neighbours = sum(
            1 for (x, y, z) in all_atoms if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 < 100.0
        )
        exposure = 1.0 / (1.0 + neighbours)
        bonus = 1.5 if resnames.get(resnum) in hydrophobic else 1.0
        scored.append((exposure * bonus, resnum))

    scored.sort(reverse=True)
    method = (
        "neighbour-count exposure proxy (heavy atoms within 10 A of the residue "
        "centroid, same chain), hydrophobic residues up-weighted 1.5x. This is a "
        "proxy, not a SASA calculation."
    )
    return [resnum for _, resnum in scored[:top_n]], method


def prepare(cfg: TargetConfig, cache_dir: Path) -> TargetSpec:
    """Run the whole of section 4 and return a TargetSpec.

    Epitope selection follows the section 4 order of preference: the residues
    the challenge page recommends win if the config names a range for them.
    """
    cache = Path(cache_dir)
    structure = fetch_structure(cfg.pdb_id, cache)
    uniprot_seq = fetch_uniprot_sequence(cfg.uniprot, cache)
    observed = parse_chain(structure, cfg.chain, cfg.keep_ligands)
    numbering_map, agree = build_numbering_map(observed, uniprot_seq)

    log.info(
        "target %s: %d observed residues, numbering map %d entries, %d identities agree",
        cfg.name,
        len(observed),
        len(numbering_map),
        agree,
    )

    epi_uni, epi_pdb = epitope_from_range(cfg.epitope_residue_range, numbering_map)
    hotspots, method = surface_hotspots(structure, cfg.chain, epi_pdb)

    rationale = (
        f"Challenge page recommends {cfg.recommended_epitope!r}. Config maps that to "
        f"UniProt residues {cfg.epitope_residue_range[0]}-{cfg.epitope_residue_range[1]}, "
        f"of which {len(epi_pdb)} are observed in {cfg.pdb_id} chain {cfg.chain}. "
        f"Hotspots ranked within that epitope by: {method}"
    )

    return TargetSpec(
        name=cfg.name,
        uniprot=cfg.uniprot,
        pdb_id=cfg.pdb_id,
        chain=cfg.chain,
        structure_path=str(structure).replace("\\", "/"),
        sequence_uniprot=uniprot_seq,
        sequence_observed="".join(r.one_letter for r in observed),
        numbering_map=numbering_map,
        epitope_residues_uniprot=epi_uni,
        epitope_residues_pdb=epi_pdb,
        hotspots_pdb=hotspots,
        hotspot_method=method,
        epitope_rationale=rationale,
        ortholog_specs=dict(cfg.orthologs),
    )
