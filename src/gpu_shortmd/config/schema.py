"""Machine-readable JSON Schema generation."""

from __future__ import annotations

import json
from pathlib import Path

from gpu_shortmd.config.models import AppConfig, PoseManifest


def write_json_schema(path: str | Path) -> None:
    output = Path(path)
    schema = AppConfig.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://example.org/gpu-shortmd/config.schema.json"
    output.write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_pose_manifest_schema(path: str | Path) -> None:
    output = Path(path)
    schema = PoseManifest.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://example.org/gpu-shortmd/pose-manifest.schema.json"
    output.write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
