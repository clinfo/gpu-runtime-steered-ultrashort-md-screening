from __future__ import annotations

from pathlib import Path

import pytest

from gpu_shortmd.gromacs.executable import (
    ExecutableNotFoundError,
    resolve_executable,
)


def test_explicit_non_executable_file_is_rejected(tmp_path: Path) -> None:
    candidate = tmp_path / "gmx"
    candidate.write_text("#!/bin/sh\n", encoding="utf-8")

    with pytest.raises(ExecutableNotFoundError, match="not executable"):
        resolve_executable(str(candidate), env={"PATH": ""})


def test_bare_name_uses_path_even_when_cwd_has_non_executable_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shadow = tmp_path / "gmx"
    shadow.write_text("not executable\n", encoding="utf-8")
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    executable = executable_dir / "gmx"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.chdir(tmp_path)

    assert resolve_executable("gmx", env={"PATH": str(executable_dir)}) == (
        executable.resolve()
    )
