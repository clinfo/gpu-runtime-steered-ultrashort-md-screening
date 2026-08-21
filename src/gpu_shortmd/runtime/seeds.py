"""Resolve unique GROMACS velocity seeds before task execution."""

from __future__ import annotations

import secrets
from collections.abc import Callable, Sequence

from gpu_shortmd.config.models import GROMACS_SEED_MAX


class SeedResolutionError(ValueError):
    """Raised when replica seeds cannot satisfy the stable contract."""


def _validate(seeds: Sequence[int], replicas: int) -> tuple[int, ...]:
    if len(seeds) != replicas:
        raise SeedResolutionError("seed count must equal replica count")
    if len(set(seeds)) != replicas:
        raise SeedResolutionError("replica seeds must be unique")
    if any(seed < 1 or seed > GROMACS_SEED_MAX for seed in seeds):
        raise SeedResolutionError("seeds must be positive 32-bit integers")
    return tuple(seeds)


def resolve_seeds(
    *,
    replicas: int,
    explicit: Sequence[int] | None,
    base_seed: int | None,
    random_below: Callable[[int], int] = secrets.randbelow,
) -> tuple[int, ...]:
    """Resolve explicit, deterministic base-derived, or random unique seeds."""
    if replicas < 1:
        raise SeedResolutionError("replicas must be positive")
    if explicit is not None and base_seed is not None:
        raise SeedResolutionError("explicit seeds and base_seed are mutually exclusive")
    if explicit is not None:
        return _validate(explicit, replicas)
    if base_seed is not None:
        if base_seed < 1 or base_seed > GROMACS_SEED_MAX:
            raise SeedResolutionError("base_seed must be a positive 32-bit integer")
        if base_seed + replicas - 1 > GROMACS_SEED_MAX:
            raise SeedResolutionError("base_seed range exceeds GROMACS seed limit")
        return _validate(
            tuple(base_seed + offset for offset in range(replicas)),
            replicas,
        )

    resolved: list[int] = []
    seen: set[int] = set()
    while len(resolved) < replicas:
        candidate = random_below(GROMACS_SEED_MAX) + 1
        if candidate not in seen:
            seen.add(candidate)
            resolved.append(candidate)
    return _validate(resolved, replicas)
