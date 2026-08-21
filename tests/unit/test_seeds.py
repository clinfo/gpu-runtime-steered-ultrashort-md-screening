from __future__ import annotations

import pytest

from gpu_shortmd.config.models import GROMACS_SEED_MAX
from gpu_shortmd.runtime.seeds import SeedResolutionError, resolve_seeds


def test_explicit_seeds_are_preserved() -> None:
    assert resolve_seeds(
        replicas=3,
        explicit=[101, 202, 303],
        base_seed=None,
    ) == (101, 202, 303)


def test_base_seed_is_deterministic_and_unique() -> None:
    assert resolve_seeds(
        replicas=5,
        explicit=None,
        base_seed=2026073001,
    ) == (2026073001, 2026073002, 2026073003, 2026073004, 2026073005)


def test_random_seed_collisions_are_retried() -> None:
    values = iter([4, 4, 8, 9])
    assert resolve_seeds(
        replicas=3,
        explicit=None,
        base_seed=None,
        random_below=lambda _: next(values),
    ) == (5, 9, 10)


@pytest.mark.parametrize(
    ("explicit", "base_seed"),
    [
        ([1, 1], None),
        ([1], None),
        (None, GROMACS_SEED_MAX),
        ([1, 2], 5),
    ],
)
def test_invalid_seed_resolution_fails(
    explicit: list[int] | None,
    base_seed: int | None,
) -> None:
    with pytest.raises(SeedResolutionError):
        resolve_seeds(
            replicas=2,
            explicit=explicit,
            base_seed=base_seed,
        )
