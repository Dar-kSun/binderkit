"""Objective registry: name -> Objective (docs/SPEC.md section 6.3).

New objectives are added per challenge by dropping a module in this package and
registering it here. The interface does not change.
"""

from __future__ import annotations

from binderkit.objectives import ortholog, ph_selectivity
from binderkit.objectives.base import Objective, ObjectiveScore

REGISTRY: dict[str, Objective] = {
    ph_selectivity.name: ph_selectivity.OBJECTIVE,
    ortholog.name: ortholog.OBJECTIVE,
}


def get(name: str) -> Objective:
    """Look up an objective by name, with a useful error if it is missing."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown objective {name!r}; registered: {sorted(REGISTRY)}") from None


__all__ = ["REGISTRY", "Objective", "ObjectiveScore", "get"]
