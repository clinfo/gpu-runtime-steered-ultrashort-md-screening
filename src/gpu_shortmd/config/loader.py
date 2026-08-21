"""Safe YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from gpu_shortmd.config.models import AppConfig


class ConfigLoadError(ValueError):
    """Raised when YAML or strict schema validation fails."""


def load_config(path: str | Path) -> AppConfig:
    resolved = Path(path)
    try:
        value: Any = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigLoadError(f"cannot load configuration: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigLoadError("configuration root must be a mapping")
    try:
        return AppConfig.model_validate(value)
    except ValidationError as exc:
        raise ConfigLoadError(str(exc)) from exc
