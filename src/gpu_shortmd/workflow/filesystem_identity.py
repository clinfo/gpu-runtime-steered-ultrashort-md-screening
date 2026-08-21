"""Deterministic collision-resistant filesystem identities for raw pose IDs."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

POSE_SLUG_MAX_LENGTH = 40


class PoseFilesystemIdentityError(ValueError):
    """Raised before run creation when pose filesystem keys are unsafe."""


def pose_filesystem_key(pose_id: str) -> str:
    """Return an ASCII slug plus the full SHA-256 of the authoritative ID."""
    if not pose_id or pose_id != pose_id.strip():
        raise PoseFilesystemIdentityError(
            "pose_id for filesystem identity must be nonempty and trimmed"
        )
    if any(ord(character) < 32 for character in pose_id):
        raise PoseFilesystemIdentityError(
            "pose_id for filesystem identity cannot contain control characters"
        )
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", pose_id).strip("-.")
    slug = slug[:POSE_SLUG_MAX_LENGTH].rstrip("-.") or "pose"
    digest = hashlib.sha256(pose_id.encode("utf-8")).hexdigest()
    return f"{slug}--{digest}"


def derive_pose_filesystem_keys(pose_ids: Iterable[str]) -> dict[str, str]:
    """Derive and validate a one-to-one map before any run directory exists."""
    identifiers = tuple(pose_ids)
    if len(set(identifiers)) != len(identifiers):
        raise PoseFilesystemIdentityError("raw pose_id values must be unique")
    resolved = {pose_id: pose_filesystem_key(pose_id) for pose_id in identifiers}
    if len(set(resolved.values())) != len(resolved):
        raise PoseFilesystemIdentityError("derived pose filesystem keys are not unique")
    return resolved
