"""Recursive quoted-include validation for GROMACS topologies."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

INCLUDE_PATTERN = re.compile(r'^\s*#include\s+"([^"]+)"', re.MULTILINE)


class TopologyValidationError(ValueError):
    """Raised when a topology include cannot be resolved safely."""


@dataclass(frozen=True)
class TopologyResolution:
    root: Path
    local_files: tuple[Path, ...]
    external_files: tuple[Path, ...]


def _resolve_include(
    include: str,
    *,
    parent: Path,
    prepared_root: Path,
    external_search_dirs: list[Path],
) -> tuple[Path, bool]:
    external_roots = [directory.resolve() for directory in external_search_dirs]
    candidates = [parent / include, prepared_root / include]
    candidates.extend(directory / include for directory in external_roots)
    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.is_file():
            continue
        if prepared_root in resolved.parents:
            return resolved, True
        if any(search_root in resolved.parents for search_root in external_roots):
            return resolved, False
    raise TopologyValidationError(f"unresolved topology include: {include}")


def resolve_topology(
    topology: str | Path,
    *,
    prepared_root: str | Path,
    external_search_dirs: list[Path] | None = None,
) -> TopologyResolution:
    root = Path(topology).resolve()
    prepared = Path(prepared_root).resolve()
    external_dirs = external_search_dirs or []
    if not root.is_file():
        raise TopologyValidationError(f"topology not found: {root.name}")
    local: set[Path] = set()
    external: set[Path] = set()
    visited: set[Path] = set()

    def visit(path: Path, *, is_local: bool) -> None:
        if path in visited:
            return
        visited.add(path)
        (local if is_local else external).add(path)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise TopologyValidationError(
                f"cannot read topology include {path.name}: {exc}"
            ) from exc
        for include in INCLUDE_PATTERN.findall(text):
            resolved, include_is_local = _resolve_include(
                include,
                parent=path.parent,
                prepared_root=prepared,
                external_search_dirs=external_dirs,
            )
            visit(resolved, is_local=include_is_local)

    visit(root, is_local=True)
    return TopologyResolution(
        root=root,
        local_files=tuple(sorted(local)),
        external_files=tuple(sorted(external)),
    )
