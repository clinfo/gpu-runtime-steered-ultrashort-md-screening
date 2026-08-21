"""Exact named-group parsing and generated ligand-heavy index support."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REQUIRED_GROUPS = ("C-alpha", "LIG", "Protein_LIG", "Water_and_ions")


class IndexValidationError(ValueError):
    """Raised when an index violates exact unique-group requirements."""


@dataclass(frozen=True)
class IndexGroup:
    name: str
    atoms: tuple[int, ...]


def parse_index(path: str | Path) -> tuple[IndexGroup, ...]:
    resolved = Path(path)
    groups: list[IndexGroup] = []
    name: str | None = None
    atoms: list[int] = []
    try:
        lines = resolved.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise IndexValidationError(f"cannot read index file: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            if name is not None:
                groups.append(IndexGroup(name=name, atoms=tuple(atoms)))
            name = stripped[1:-1].strip()
            if not name:
                raise IndexValidationError(
                    f"{resolved}:{line_number}: empty group name"
                )
            atoms = []
            continue
        if name is None:
            raise IndexValidationError(
                f"{resolved}:{line_number}: atom list precedes first group"
            )
        try:
            values = [int(value) for value in stripped.split()]
        except ValueError as exc:
            raise IndexValidationError(
                f"{resolved}:{line_number}: non-integer atom index"
            ) from exc
        if any(value < 1 for value in values):
            raise IndexValidationError(
                f"{resolved}:{line_number}: atom indices must be positive"
            )
        atoms.extend(values)
    if name is not None:
        groups.append(IndexGroup(name=name, atoms=tuple(atoms)))
    if not groups:
        raise IndexValidationError(f"{resolved}: index has no groups")
    return tuple(groups)


def validate_required_groups(
    groups: tuple[IndexGroup, ...],
) -> dict[str, IndexGroup]:
    counts = Counter(group.name for group in groups)
    invalid = [name for name in REQUIRED_GROUPS if counts[name] != 1]
    if invalid:
        detail = ", ".join(f"{name}={counts[name]}" for name in invalid)
        raise IndexValidationError(
            f"required index groups must exist exactly once: {detail}"
        )
    required = {group.name: group for group in groups if group.name in REQUIRED_GROUPS}
    empty = [name for name, group in required.items() if not group.atoms]
    if empty:
        raise IndexValidationError(
            "required index groups must be non-empty: " + ", ".join(empty)
        )
    repeated = [
        name
        for name, group in required.items()
        if len(set(group.atoms)) != len(group.atoms)
    ]
    if repeated:
        raise IndexValidationError(
            "required index groups contain duplicate atom indices: "
            + ", ".join(repeated)
        )
    complex_atoms = set(required["Protein_LIG"].atoms)
    for name in ("C-alpha", "LIG"):
        if not set(required[name].atoms).issubset(complex_atoms):
            raise IndexValidationError(f"{name} must be a subset of Protein_LIG")
    return required
