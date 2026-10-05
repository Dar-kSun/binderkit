"""The novelty gate (docs/SPEC.md section 6.2).

The challenge requires "de novo designs only" and "adequate sequence- and
structural-diversity from known proteins", without giving numbers, so the
thresholds here are ours and are reported with the actual identities achieved,
not merely as pass/fail.

Scope note: MMseqs2 and Foldseek are not installed, so this module does **not**
search all of UniProt or the PDB. It aligns against an explicit reference set
(the target, its orthologs, and a curated known-binder list) using Biopython's
pairwise aligner, plus full all-pairs within-batch comparison. That is a real
check against the sequences that matter most for a de novo claim - above all the
known binders of this target - but it is weaker than a database search, and
docs/LIMITATIONS.md says so.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from Bio import Align

from binderkit.config import SCHEMAS, NoveltyConfig

log = logging.getLogger(__name__)


def _aligner() -> Align.PairwiseAligner:
    """A BLOSUM62 local aligner, the conventional choice for identity screens."""
    al = Align.PairwiseAligner()
    al.mode = "local"
    al.open_gap_score = -11
    al.extend_gap_score = -1
    al.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
    return al


@dataclass
class Reference:
    """One sequence a design is screened against."""

    name: str
    sequence: str
    is_known_binder: bool = False


def read_fasta(path: Path, is_known_binder: bool = False) -> list[Reference]:
    """Read a FASTA file into References. Tolerates blank lines and CRLF."""
    refs: list[Reference] = []
    name: str | None = None
    chunks: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                refs.append(Reference(name, "".join(chunks), is_known_binder))
            name = line[1:].split()[0] if len(line) > 1 else "unnamed"
            chunks = []
        else:
            chunks.append(line)
    if name is not None:
        refs.append(Reference(name, "".join(chunks), is_known_binder))
    return [r for r in refs if r.sequence]


def pairwise_identity(a: str, b: str, aligner: Align.PairwiseAligner | None = None) -> float:
    """Fraction identity over the aligned region, normalised by the shorter sequence.

    Range 0.0-1.0. Normalising by the shorter sequence is the conservative
    choice for a novelty screen: a short design embedded in a long known binder
    scores high rather than being diluted by the length difference.
    """
    if not a or not b:
        return 0.0
    al = aligner or _aligner()
    try:
        alignment = al.align(a, b)[0]
    except (ValueError, IndexError):
        return 0.0
    idx_a, idx_b = alignment.aligned
    matches = 0
    for (a0, a1), (b0, b1) in zip(idx_a, idx_b, strict=True):
        matches += sum(1 for i, j in zip(range(a0, a1), range(b0, b1), strict=True) if a[i] == b[j])
    return round(matches / max(1, min(len(a), len(b))), 4)


def best_hit(
    sequence: str,
    references: list[Reference],
    aligner: Align.PairwiseAligner | None = None,
) -> tuple[float, str]:
    """Highest identity against `references`, with the hit name."""
    al = aligner or _aligner()
    best, hit = 0.0, ""
    for ref in references:
        ident = pairwise_identity(sequence, ref.sequence, al)
        if ident > best:
            best, hit = ident, ref.name
    return best, hit


def within_batch_clusters(
    designs: pd.DataFrame,
    identity_cutoff: float,
    aligner: Align.PairwiseAligner | None = None,
) -> dict[str, int]:
    """Single-linkage cluster designs by pairwise identity.

    Returns design_id -> cluster index. All-pairs, so O(n^2) alignments; fine
    for the design counts this pipeline produces (tens to low hundreds).
    """
    al = aligner or _aligner()
    ids = list(designs["design_id"])
    seqs = dict(zip(designs["design_id"], designs["sequence"], strict=True))
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if pairwise_identity(seqs[a], seqs[b], al) >= identity_cutoff:
                union(a, b)

    roots: dict[str, int] = {}
    out: dict[str, int] = {}
    for i in ids:
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        out[i] = roots[r]
    return out


def gate(
    designs: pd.DataFrame,
    cfg: NoveltyConfig,
    references: list[Reference],
    structure_tm: dict[str, tuple[float, str]] | None = None,
    db_hits: dict | None = None,
) -> pd.DataFrame:
    """Apply the novelty gate. Returns the `novelty` schema.

    A design is rejected if any of these holds:

    * sequence identity to any reference exceeds `max_seq_identity`;
    * identity to a **known binder** exceeds `known_binder_identity_cutoff`,
      which is stricter - known binders are rejected, not down-weighted
      (section 6.2);
    * structural TM-score to a known fold exceeds `max_tm_score`, when a
      structure comparison is available;
    * the design is the (`max_per_cluster`+1)-th member of its batch cluster.

    `structure_tm` maps design_id -> (best TM, hit). When it is None the
    structural arm of the gate cannot run; `reason` says so explicitly rather
    than silently passing.

    `db_hits` maps design_id -> `binderkit.search.Hit` from a real database
    search. Where present it supersedes the reference-set screen, and it is
    judged on **effective identity** (identity x query coverage) rather than raw
    identity: a 9-residue local match at 77% identity is noise, and rejecting a
    design for it would be a false positive.
    """
    al = _aligner()
    known = [r for r in references if r.is_known_binder]
    clusters = within_batch_clusters(designs, cfg.batch_identity_cutoff, al)

    seen_per_cluster: dict[int, int] = {}
    rows: list[dict[str, object]] = []

    for d in designs.itertuples():
        seq = d.sequence
        best_id, hit = best_hit(seq, references, al)
        known_id, known_hit = best_hit(seq, known, al) if known else (0.0, "")
        tm, tm_hit = (structure_tm or {}).get(d.design_id, (float("nan"), ""))

        cluster = clusters[d.design_id]
        rank_in_cluster = seen_per_cluster.get(cluster, 0)
        seen_per_cluster[cluster] = rank_in_cluster + 1

        reasons: list[str] = []
        passed = True

        db_hit = (db_hits or {}).get(d.design_id)
        if db_hit is not None:
            if db_hit.significant and db_hit.effective_identity > cfg.max_seq_identity:
                passed = False
                reasons.append(
                    f"REJECTED on database search: {db_hit.effective_identity:.1%} effective "
                    f"identity to {db_hit.target} ({db_hit.identity:.1%} over "
                    f"{db_hit.query_coverage:.0%} of the design, e={db_hit.evalue:.1e}) "
                    f"exceeds the {cfg.max_seq_identity:.0%} cutoff"
                )
            elif db_hit.significant:
                reasons.append(
                    f"database search: best significant hit {db_hit.target} at "
                    f"{db_hit.effective_identity:.1%} effective identity - under the cutoff"
                )
            else:
                reasons.append(
                    "database search: no significant PDB hit (best was "
                    f"{db_hit.identity:.1%} identity over only {db_hit.query_coverage:.0%} "
                    f"of the design, e={db_hit.evalue:.1e})"
                )
            if db_hit.effective_identity > best_id:
                best_id, hit = db_hit.effective_identity, db_hit.target

        if known_id > cfg.known_binder_identity_cutoff:
            passed = False
            reasons.append(
                f"REJECTED as resembling a known binder: {known_id:.1%} identity to "
                f"{known_hit} exceeds the {cfg.known_binder_identity_cutoff:.0%} "
                "known-binder cutoff"
            )
        if best_id > cfg.max_seq_identity:
            passed = False
            reasons.append(
                f"sequence identity {best_id:.1%} to {hit} exceeds the "
                f"{cfg.max_seq_identity:.0%} cutoff"
            )
        if tm == tm and tm > cfg.max_tm_score:  # NaN-safe
            passed = False
            reasons.append(
                f"structural TM {tm:.2f} to {tm_hit} exceeds the {cfg.max_tm_score} cutoff"
            )
        if rank_in_cluster >= cfg.max_per_cluster:
            passed = False
            reasons.append(
                f"batch cluster {cluster} already has {cfg.max_per_cluster} representatives"
            )
        if tm != tm:
            reasons.append("structural novelty NOT CHECKED (no Foldseek/TM comparison available)")

        if passed:
            hit_label = hit or "nothing in the reference set"
            reasons.insert(
                0,
                f"passed: best identity {best_id:.1%} to {hit_label}, cluster {cluster}",
            )

        rows.append(
            {
                "design_id": d.design_id,
                "best_seq_identity": best_id,
                "seq_hit": hit,
                "best_tm": tm,
                "struct_hit": tm_hit,
                "batch_cluster": cluster,
                "passed": passed,
                "reason": "; ".join(reasons),
            }
        )

    df = pd.DataFrame(rows, columns=SCHEMAS["novelty"])
    log.info("novelty: %d/%d designs passed the gate", int(df["passed"].sum()), len(df))
    return df
