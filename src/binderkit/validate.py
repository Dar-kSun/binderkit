"""Submission validator (docs/SPEC.md section 7.2).

Runs automatically at the end of packaging and in the test suite. Two jobs:

1. Format: row count, columns, uniqueness, length bounds, allowed
   `molecule_class`, no empty fields, standard amino acids only, ranked order.
2. **Integrity:** scan every text field for anything that reads as an
   instruction to a model. The challenge page states that embedded instructions
   or prompt injection are not permitted, so a hit is a hard failure, never a
   warning (section 12.1).

The scan is deliberately broad and will occasionally flag innocent prose. That
is the correct trade: a false positive costs one rewrite, a false negative risks
disqualification.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from binderkit.config import SubmissionConfig
from binderkit.metrics import STANDARD_AA

log = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when a submission fails validation."""


@dataclass
class ValidationResult:
    """Outcome of validating one submission."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_rows: int = 0

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ValidationError(
                f"submission failed validation with {len(self.errors)} error(s):\n"
                + "\n".join(f"  - {e}" for e in self.errors)
            )


# --------------------------------------------------------------------------
# Instruction / prompt-injection scan
# --------------------------------------------------------------------------

#: Phrases that read as an instruction addressed at a reader or a model.
#: Word-boundary anchored to limit false positives on ordinary prose.
INSTRUCTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bignore\b.{0,40}\b(previous|prior|above|earlier|all)\b", "ignore-previous"),
    (r"\bdisregard\b", "disregard"),
    (r"\byou\s+(must|should|will|need\s+to|have\s+to)\b", "you-must"),
    (r"\bplease\s+(select|choose|rank|pick|prefer|include)\b", "please-select"),
    (r"\b(select|choose|rank|pick|prefer)\s+(this|these|me|us|the\s+following)\b", "select-this"),
    (r"\bas\s+an?\s+(ai|language\s+model|assistant)\b", "role-address"),
    (r"\b(system|assistant|user)\s*:", "chat-role-marker"),
    (r"\bprompt\b.{0,20}\b(injection|override)\b", "prompt-injection"),
    (r"\boverrid(e|ing)\b.{0,30}\b(instruction|rule|prompt|guideline)\b", "override"),
    (r"\bhighest\s+(priority|score|rank)\b", "claim-priority"),
    (r"\bthis\s+design\s+(is|should)\b.{0,30}\b(best|selected|chosen|top)\b", "self-promotion"),
    (r"<\s*/?\s*(system|instruction|prompt|im_start|im_end)\b", "prompt-markup"),
    (r"\{\{.{0,40}\}\}", "template-markup"),
    (r"\[\[.{0,40}\]\]", "template-markup"),
    (r"```", "code-fence"),
    (r"\bdo\s+not\s+(reject|filter|exclude|discard)\b", "do-not-reject"),
    (r"\bimportant\s*[:!]", "attention-grab"),
    (r"\bnote\s+to\s+(the\s+)?(reviewer|selector|model|claude)\b", "note-to-reviewer"),
)

_COMPILED = tuple((re.compile(p, re.IGNORECASE), label) for p, label in INSTRUCTION_PATTERNS)


def scan_text_for_instructions(text: str) -> list[str]:
    """Return labels of every instruction-like pattern found in `text`.

    Empty list means clean. Used on submission fields and on METHODS.md.
    """
    if not isinstance(text, str) or not text:
        return []
    return [label for rx, label in _COMPILED if rx.search(text)]


def scan_dataframe(df: pd.DataFrame) -> list[str]:
    """Scan every text cell in `df`. Returns one error string per hit.

    Deliberately does NOT filter columns by dtype. pandas 3.0 infers
    `StringDtype` rather than `object` for text columns, so a `dtype == object`
    guard silently skips every field that matters and turns this check into a
    no-op that always passes. `scan_text_for_instructions` already ignores
    non-string values, so per-cell dispatch is both simpler and version-proof.
    """
    errors: list[str] = []
    for col in df.columns:
        for idx, value in df[col].items():
            for label in scan_text_for_instructions(value):
                errors.append(
                    f"instruction-like text in column {col!r} row {idx}: "
                    f"pattern {label!r} matched {str(value)[:80]!r}"
                )
    return errors


# --------------------------------------------------------------------------
# Format validation
# --------------------------------------------------------------------------


def validate_frame(df: pd.DataFrame, cfg: SubmissionConfig) -> ValidationResult:
    """Validate an in-memory submission table."""
    errors: list[str] = []
    warnings: list[str] = []

    # Columns, exact names and order.
    expected = list(cfg.columns)
    if list(df.columns) != expected:
        errors.append(f"columns must be exactly {expected} in that order, got {list(df.columns)}")

    # Row count.
    if len(df) == 0:
        errors.append("submission is empty")
    if len(df) > cfg.max_designs:
        errors.append(f"{len(df)} rows exceeds the cap of {cfg.max_designs}")

    if "sequence" in df.columns:
        lo, hi = cfg.length_bounds
        for idx, seq in df["sequence"].items():
            if not isinstance(seq, str) or not seq.strip():
                errors.append(f"row {idx}: empty sequence")
                continue
            # A Fab is submitted as VH:VL, so validate each half.
            parts = seq.split(":") if ":" in seq else [seq]
            for part_i, part in enumerate(parts):
                part = part.strip().upper()
                bad = sorted({c for c in part if c not in STANDARD_AA})
                if bad:
                    errors.append(
                        f"row {idx} chain {part_i}: non-standard amino acid character(s) {bad}"
                    )
                if not (lo <= len(part) <= hi):
                    errors.append(
                        f"row {idx} chain {part_i}: length {len(part)} outside bounds {lo}-{hi}"
                    )
        if cfg.require_unique_sequences:
            dupes = df["sequence"][df["sequence"].duplicated(keep=False)]
            if not dupes.empty:
                errors.append(f"duplicate sequences at rows {sorted(dupes.index)}")

    if "name" in df.columns:
        if df["name"].isna().any() or (df["name"].astype(str).str.strip() == "").any():
            errors.append("one or more empty name fields")
        dn = df["name"][df["name"].duplicated(keep=False)]
        if not dn.empty:
            errors.append(f"duplicate names at rows {sorted(dn.index)}")

    if "molecule_class" in df.columns:
        allowed = set(cfg.allowed_molecule_class)
        bad = sorted(set(df["molecule_class"].astype(str)) - allowed)
        if bad:
            errors.append(f"molecule_class value(s) {bad} not in allowed set {sorted(allowed)}")

    # Any empty cell anywhere is a failure.
    if df.isna().to_numpy().any():
        errors.append("submission contains empty (NaN) cells")

    # Ranked order: best first. The file order IS the ranking, so we can only
    # check that the caller did not ship an index that contradicts it.
    if not df.index.is_monotonic_increasing:
        warnings.append("index is not monotonic; file order is the ranking")

    # Integrity scan - hard failure.
    errors.extend(scan_dataframe(df))

    ok = not errors
    if ok:
        log.info("validate: %d rows passed", len(df))
    else:
        log.error("validate: %d error(s)", len(errors))
    return ValidationResult(ok=ok, errors=errors, warnings=warnings, n_rows=len(df))


def validate_csv(path: Path, cfg: SubmissionConfig | None = None) -> ValidationResult:
    """Validate a submission CSV on disk."""
    cfg = cfg or SubmissionConfig()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # Re-introduce NaN only for genuinely empty strings so the NaN check works.
    df = df.replace("", pd.NA)
    return validate_frame(df, cfg)


def validate_methods_text(path: Path) -> list[str]:
    """Scan METHODS.md for instruction-like text (section 12.1 rule 1).

    The methods document is read by a human reviewer and may be seen by the
    selection workflow, so it is held to the same standard as the CSV.
    """
    if not Path(path).is_file():
        return [f"{path} does not exist"]
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    hits: list[str] = []
    for i, line in enumerate(text.splitlines(), 1):
        for label in scan_text_for_instructions(line):
            hits.append(f"{path.name}:{i}: pattern {label!r} matched {line[:80]!r}")
    return hits
