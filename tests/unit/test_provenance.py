from __future__ import annotations

from pathlib import Path

from gpu_shortmd.util.provenance import detect_source_revision


def test_revision_is_explicitly_unavailable_outside_checkout(tmp_path: Path) -> None:
    assert detect_source_revision([tmp_path]) == {
        "commit_sha": None,
        "checkout_clean": None,
    }
