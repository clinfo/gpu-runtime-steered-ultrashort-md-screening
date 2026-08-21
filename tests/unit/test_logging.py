from __future__ import annotations

from pathlib import Path

from gpu_shortmd.util.logging import RunLogger


def test_redaction_does_not_treat_filesystem_root_as_private_substring(
    tmp_path: Path,
) -> None:
    logger = RunLogger(
        run_dir=tmp_path,
        run_id="RUN-TEST",
        redacted_values=["/", "/private/work"],
        redacted_tokens=["node-a"],
    )

    redacted = logger.redact(
        "working=/private/work/stage host=node-a unrelated=node-alpha"
    )
    assert "working=<REDACTED_PATH>/stage" in redacted
    assert "host=<REDACTED>" in redacted
    assert "unrelated=node-alpha" in redacted
