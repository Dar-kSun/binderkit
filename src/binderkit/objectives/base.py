"""Objective interface (docs/SPEC.md section 6.3).

Each objective returns a value, an uncertainty, and a human-readable reason, so
that a ranking built from objectives can always explain itself in plain English.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ObjectiveScore:
    """One objective's verdict on one design.

    `value` is higher-is-better and should be roughly 0..1 so objectives are
    comparable. `uncertainty` is on the same scale; a large uncertainty is how
    an objective says "weakly validated" in numbers as well as prose.
    """

    value: float
    uncertainty: float
    reason: str

    def __post_init__(self) -> None:
        if self.uncertainty < 0:
            raise ValueError("uncertainty must be non-negative")


class Objective(Protocol):
    """What every objective module must expose."""

    name: str

    def score(self, design: Any, context: dict[str, Any]) -> ObjectiveScore:
        """Score one design in `context` (target spec, metrics row, config)."""
        ...
