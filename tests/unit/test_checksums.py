from __future__ import annotations

from pathlib import Path

import pytest

from gpu_shortmd.util.checksums import (
    ChecksumVerificationError,
    verify_checksum_file,
    write_checksum_file,
)


def test_checksum_manifest_verifies_and_detects_change(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("original\n", encoding="utf-8")
    manifest = tmp_path / "checksums.sha256"
    write_checksum_file([artifact], root=tmp_path, output=manifest)

    assert verify_checksum_file(checksum_file=manifest, root=tmp_path) == []
    artifact.write_text("changed\n", encoding="utf-8")
    assert verify_checksum_file(checksum_file=manifest, root=tmp_path) == [
        "artifact.txt"
    ]


def test_checksum_manifest_rejects_parent_traversal(tmp_path: Path) -> None:
    manifest = tmp_path / "checksums.sha256"
    manifest.write_text(f"{'0' * 64}  ../outside.txt\n", encoding="utf-8")

    with pytest.raises(ChecksumVerificationError, match="unsafe checksum path"):
        verify_checksum_file(checksum_file=manifest, root=tmp_path)
