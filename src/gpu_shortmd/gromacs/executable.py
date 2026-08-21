"""GROMACS executable discovery."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path


class ExecutableNotFoundError(FileNotFoundError):
    """Raised when an explicitly configured executable cannot be found."""


def _validated_executable(path: Path, *, display_name: str) -> Path:
    if not path.is_file():
        raise ExecutableNotFoundError(f"executable not found: {display_name}")
    if not os.access(path, os.X_OK):
        raise ExecutableNotFoundError(f"file is not executable: {display_name}")
    return path.resolve()


def resolve_executable(name: str, *, env: Mapping[str, str]) -> Path:
    candidate = Path(name)
    explicit_path = (
        candidate.is_absolute()
        or os.sep in name
        or (os.altsep is not None and os.altsep in name)
    )
    if explicit_path:
        return _validated_executable(candidate, display_name=name)
    resolved = shutil.which(name, path=env.get("PATH"))
    if resolved is None:
        raise ExecutableNotFoundError(f"executable not found on PATH: {name}")
    return _validated_executable(Path(resolved), display_name=name)
