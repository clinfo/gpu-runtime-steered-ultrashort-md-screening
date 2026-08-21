"""Strict streaming parser for two-column GROMACS XVG RMSD data."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


class XvgParseError(ValueError):
    """Raised when an XVG file cannot produce a valid numeric RMSD series."""


@dataclass(frozen=True)
class XvgSample:
    time_ps: float
    rmsd: float


@dataclass(frozen=True)
class XvgSeries:
    path: Path
    samples: tuple[XvgSample, ...]

    @property
    def maximum(self) -> XvgSample:
        return max(self.samples, key=lambda sample: sample.rmsd)


def iter_xvg_samples(path: Path) -> Iterator[XvgSample]:
    """Yield finite, strictly time-ordered samples from an XVG file."""
    previous_time: float | None = None
    try:
        handle = path.open(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise XvgParseError(f"cannot read XVG file {path}: {exc}") from exc

    try:
        with handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith(("#", "@")):
                    continue
                columns = stripped.split()
                if len(columns) != 2:
                    raise XvgParseError(
                        f"{path}:{line_number}: expected exactly two numeric columns"
                    )
                try:
                    time_ps, rmsd = (float(value) for value in columns)
                except ValueError as exc:
                    raise XvgParseError(
                        f"{path}:{line_number}: non-numeric XVG data"
                    ) from exc
                if not math.isfinite(time_ps) or not math.isfinite(rmsd):
                    raise XvgParseError(
                        f"{path}:{line_number}: NaN or infinite values are invalid"
                    )
                if rmsd < 0:
                    raise XvgParseError(
                        f"{path}:{line_number}: RMSD must be non-negative"
                    )
                if previous_time is not None and time_ps <= previous_time:
                    raise XvgParseError(
                        f"{path}:{line_number}: time values must be strictly increasing"
                    )
                previous_time = time_ps
                yield XvgSample(time_ps=time_ps, rmsd=rmsd)
    except (OSError, UnicodeError) as exc:
        raise XvgParseError(f"cannot read XVG file {path}: {exc}") from exc


def parse_xvg(path: str | Path) -> XvgSeries:
    """Parse a complete offline XVG file.

    Empty or truncated data is an error. Online retry behavior belongs to the
    runtime monitor and is intentionally not hidden here.
    """
    resolved = Path(path)
    samples = tuple(iter_xvg_samples(resolved))
    if not samples:
        raise XvgParseError(f"{resolved}: no numeric XVG samples")
    return XvgSeries(path=resolved, samples=samples)


def numeric_data_sha256(path: str | Path) -> str:
    """Hash the normalized numeric lines for transformation provenance."""
    numeric_lines: list[str] = []
    resolved = Path(path)
    try:
        lines = resolved.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise XvgParseError(f"cannot read XVG file {resolved}: {exc}") from exc
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith(("#", "@")):
            numeric_lines.append(stripped)
    return hashlib.sha256(("\n".join(numeric_lines) + "\n").encode("utf-8")).hexdigest()
