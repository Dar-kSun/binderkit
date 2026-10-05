"""Run manifest: versions, seeds, hashes, timings, and what was mocked.

Every artefact derived from a mocked stage must carry `mocked: true`
(docs/SPEC.md section 0.5), so the manifest is the single place that records which
stages were real.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from binderkit import __version__


def sequence_hash(sequence: str) -> str:
    """Stable short hash of an amino-acid sequence, used as a fold cache key."""
    return hashlib.sha256(sequence.strip().upper().encode()).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    """Full SHA-256 of a file, streamed so large structures are fine."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    """Current commit hash, or None outside a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def tool_versions() -> dict[str, str]:
    """Versions of the libraries whose numbers end up in reports."""
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "binderkit": __version__,
        "platform": platform.platform(),
    }
    for mod in ("numpy", "pandas", "scipy", "sklearn", "Bio", "matplotlib"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - a missing optional dep is not fatal
            versions[mod] = "absent"
    return versions


@dataclass
class Provenance:
    """Accumulates everything needed to reproduce a run."""

    run_id: str
    config: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    seeds: dict[str, int] = field(default_factory=dict)
    mocked_stages: list[str] = field(default_factory=list)
    skipped_stages: list[str] = field(default_factory=list)
    file_hashes: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def mark_mocked(self, stage: str, reason: str) -> None:
        """Record that `stage` produced mocked values, loudly."""
        if stage not in self.mocked_stages:
            self.mocked_stages.append(stage)
        self.notes.append(f"MOCK: {stage} - {reason}")

    def mark_skipped(self, stage: str, reason: str) -> None:
        if stage not in self.skipped_stages:
            self.skipped_stages.append(stage)
        self.notes.append(f"SKIPPED: {stage} - {reason}")

    @property
    def any_mocked(self) -> bool:
        """True if any artefact of this run is downstream of a mock."""
        return bool(self.mocked_stages)

    @contextmanager
    def time(self, stage: str):
        """Time a stage and record it in seconds."""
        start = time.monotonic()
        try:
            yield
        finally:
            self.timings[stage] = round(time.monotonic() - start, 3)

    def add_file(self, path: Path) -> None:
        p = Path(path)
        if p.is_file():
            self.file_hashes[str(p).replace("\\", "/")] = file_sha256(p)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": git_commit(),
            "tool_versions": tool_versions(),
            "config": self.config,
            "hardware": self.hardware,
            "seeds": self.seeds,
            "timings_seconds": self.timings,
            "mocked_stages": self.mocked_stages,
            "skipped_stages": self.skipped_stages,
            "any_mocked": self.any_mocked,
            "file_sha256": self.file_hashes,
            "notes": self.notes,
        }

    def write(self, path: Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return p
