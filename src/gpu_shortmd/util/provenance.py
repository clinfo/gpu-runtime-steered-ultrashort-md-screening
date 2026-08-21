"""Best-effort, non-mutating source revision discovery."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict


class SourceRevision(TypedDict):
    commit_sha: str | None
    checkout_clean: bool | None


def detect_source_revision(
    search_roots: Sequence[Path] | None = None,
) -> SourceRevision:
    """Record an exact Git revision when execution is inside a checkout."""
    candidates = (
        tuple(search_roots)
        if search_roots is not None
        else (Path(__file__).resolve().parent, Path.cwd().resolve())
    )
    visited: set[Path] = set()
    for candidate in candidates:
        current = candidate if candidate.is_dir() else candidate.parent
        for root in (current, *current.parents):
            if root in visited:
                continue
            visited.add(root)
            if not (root / ".git").exists():
                continue
            revision = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
            value = revision.stdout.strip().lower()
            if (
                revision.returncode != 0
                or len(value) != 40
                or any(character not in "0123456789abcdef" for character in value)
            ):
                continue
            status = subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            return {
                "commit_sha": value,
                "checkout_clean": (
                    not status.stdout.strip() if status.returncode == 0 else None
                ),
            }
    return {"commit_sha": None, "checkout_clean": None}
