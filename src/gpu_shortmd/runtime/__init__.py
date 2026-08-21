"""Transactional single-node runtime for replicas and local GPU workers."""

from gpu_shortmd.runtime.seeds import resolve_seeds
from gpu_shortmd.runtime.state import RuntimeState

__all__ = ["RuntimeState", "resolve_seeds"]
