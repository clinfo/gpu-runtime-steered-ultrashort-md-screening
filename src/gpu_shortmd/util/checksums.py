"""Streaming SHA-256 helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path


class ChecksumVerificationError(ValueError):
    """Raised when a checksum manifest is malformed or unsafe."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksum_file(
    paths: Iterable[Path],
    *,
    root: Path,
    output: Path,
) -> None:
    lines = []
    for path in sorted(paths):
        if not path.is_file() or path == output:
            continue
        relative = path.relative_to(root).as_posix()
        lines.append(f"{sha256_file(path)}  {relative}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_checksum_file(*, checksum_file: Path, root: Path) -> list[str]:
    """Return mismatched/missing paths after strictly parsing a SHA-256 file."""
    root = root.resolve()
    failures: list[str] = []
    for line_number, line in enumerate(
        checksum_file.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line:
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ChecksumVerificationError(
                f"invalid checksum line {line_number}"
            ) from exc
        if len(expected) != 64 or any(
            character not in "0123456789abcdef" for character in expected
        ):
            raise ChecksumVerificationError(f"invalid SHA-256 on line {line_number}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ChecksumVerificationError(
                f"unsafe checksum path on line {line_number}"
            )
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected:
            failures.append(relative_path.as_posix())
    return failures
