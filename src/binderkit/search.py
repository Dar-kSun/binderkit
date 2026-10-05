"""Sequence database search for the novelty gate (NEXT_SESSION.md section 4.1).

Session 1 screened designs against a handful of reference sequences because no
search tool was installed. This runs a real search against every protein
sequence in the PDB using MMseqs2.

**The coverage trap.** MMseqs2 reports identity over the aligned region only.
Searching random designs against 1.1M PDB sequences returns hits like "77%
identity" that are 9-residue local alignments at 16% query coverage with
e-values above 1000 - pure noise. A novelty gate that read `fident` alone would
reject novel designs for resembling nothing in particular. Significance here
therefore requires an e-value **and** a coverage threshold, and the gate
reports effective identity (identity x query coverage) alongside raw identity
so the distinction is visible rather than buried.

The Windows build's `search` module shells out to a script that fails, so this
drives `prefilter` / `align` / `convertalis` directly. Same result, no shell.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_BINARY = Path("work/tools/mmseqs/mmseqs/bin/mmseqs.exe")
DEFAULT_DB = Path("work/mmseqs_db/pdb")

#: A hit must clear both to count. An alignment covering a sixth of a design at
#: an e-value of 1000 is not evidence of anything.
SIGNIFICANT_EVALUE = 1e-3
MIN_QUERY_COVERAGE = 0.50
SEARCH_TIMEOUT_S = 1800


class SearchUnavailableError(RuntimeError):
    """MMseqs2 or its database is not present."""


@dataclass
class Hit:
    """Best database hit for one design."""

    design_id: str
    target: str = ""
    identity: float = 0.0
    alignment_length: int = 0
    evalue: float = float("inf")
    query_coverage: float = 0.0
    target_coverage: float = 0.0
    significant: bool = False

    @property
    def effective_identity(self) -> float:
        """Identity scaled by how much of the design the alignment covers.

        This is the number a novelty claim should rest on: a short, high-identity
        local match has a low effective identity and should not reject a design.
        """
        return round(self.identity * self.query_coverage, 4)


def available(binary: Path | None = None, db: Path | None = None) -> bool:
    """Whether a real search can be run."""
    b = Path(binary or DEFAULT_BINARY)
    d = Path(db or DEFAULT_DB)
    return b.is_file() and d.with_suffix(".dbtype").is_file()


def _run(binary: Path, args: list[str]) -> None:
    out = subprocess.run(
        [str(binary), *args],
        capture_output=True,
        text=True,
        timeout=SEARCH_TIMEOUT_S,
        check=False,
    )
    if out.returncode != 0:
        raise SearchUnavailableError(
            f"mmseqs {args[0]} failed ({out.returncode}): {out.stderr[-400:]}"
        )


def search(
    sequences: dict[str, str],
    binary: Path | None = None,
    db: Path | None = None,
    sensitivity: float = 7.5,
    max_seqs: int = 300,
) -> dict[str, Hit]:
    """Search `sequences` against the database. Returns design_id -> best Hit.

    "Best" means the most significant hit that clears both thresholds; if none
    does, the best raw hit is still returned with `significant=False`, so the
    caller can report what was actually found rather than an empty result.
    """
    b = Path(binary or DEFAULT_BINARY)
    d = Path(db or DEFAULT_DB)
    if not available(b, d):
        raise SearchUnavailableError(f"mmseqs binary {b} or database {d} not found")

    tmp = Path(tempfile.mkdtemp(prefix="binderkit_mmseqs_"))
    try:
        fasta = tmp / "query.fasta"
        fasta.write_text("".join(f">{k}\n{v}\n" for k, v in sequences.items()), encoding="utf-8")
        q, pref, aln, tsv = tmp / "q", tmp / "pref", tmp / "aln", tmp / "hits.tsv"
        _run(b, ["createdb", str(fasta), str(q), "--dbtype", "1"])
        _run(
            b,
            [
                "prefilter",
                str(q),
                str(d),
                str(pref),
                "-s",
                str(sensitivity),
                "--max-seqs",
                str(max_seqs),
            ],
        )
        _run(
            b,
            ["align", str(q), str(d), str(pref), str(aln), "-e", "10000", "--alignment-mode", "3"],
        )
        _run(
            b,
            [
                "convertalis",
                str(q),
                str(d),
                str(aln),
                str(tsv),
                "--format-output",
                "query,target,fident,alnlen,evalue,qcov,tcov",
            ],
        )

        if not tsv.is_file() or tsv.stat().st_size == 0:
            return {k: Hit(design_id=k) for k in sequences}

        df = pd.read_csv(
            tsv,
            sep="\t",
            names=["query", "target", "fident", "alnlen", "evalue", "qcov", "tcov"],
        )
        df["significant"] = (df.evalue <= SIGNIFICANT_EVALUE) & (df.qcov >= MIN_QUERY_COVERAGE)

        out: dict[str, Hit] = {k: Hit(design_id=k) for k in sequences}
        for design_id, grp in df.groupby("query"):
            sig = grp[grp.significant]
            pick = (
                sig.sort_values("evalue").iloc[0] if len(sig) else grp.sort_values("evalue").iloc[0]
            )
            out[str(design_id)] = Hit(
                design_id=str(design_id),
                target=str(pick.target),
                identity=float(pick.fident),
                alignment_length=int(pick.alnlen),
                evalue=float(pick.evalue),
                query_coverage=float(pick.qcov),
                target_coverage=float(pick.tcov),
                significant=bool(pick.significant),
            )
        n_sig = sum(1 for h in out.values() if h.significant)
        log.info(
            "mmseqs: %d/%d designs have a significant PDB hit (e<=%.0e and query coverage>=%.0f%%)",
            n_sig,
            len(out),
            SIGNIFICANT_EVALUE,
            100 * MIN_QUERY_COVERAGE,
        )
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
