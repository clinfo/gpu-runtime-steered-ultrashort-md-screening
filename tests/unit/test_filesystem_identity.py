from __future__ import annotations

import re

from gpu_shortmd.workflow.filesystem_identity import (
    derive_pose_filesystem_keys,
    pose_filesystem_key,
)


def test_pose_filesystem_keys_are_deterministic_safe_and_collision_resistant() -> None:
    pose_ids = (
        "a/b",
        "a-b",
        "P00533_EGFR|p2|model4",
        "標的：リガンド（候補）",  # noqa: RUF001
        "pose !@#$%^&*()[]{};,.?",
        "long-" + "x" * 500,
    )
    keys = derive_pose_filesystem_keys(pose_ids)

    assert set(keys) == set(pose_ids)
    assert len(set(keys.values())) == len(pose_ids)
    assert keys["a/b"] != keys["a-b"]
    assert keys["P00533_EGFR|p2|model4"].startswith("P00533_EGFR-p2-model4--")
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+", key) for key in keys.values())
    assert all(len(key) <= 106 for key in keys.values())
    assert keys == derive_pose_filesystem_keys(pose_ids)


def test_pose_filesystem_key_does_not_replace_authoritative_raw_id() -> None:
    raw_pose_id = "P00533_EGFR|p2|model4"
    key = pose_filesystem_key(raw_pose_id)
    assert raw_pose_id != key
    assert key.endswith(
        "614367583c14ccf168106fdc31e077fefeee266a92f80910dd81ab1742e12dc4"
    )
