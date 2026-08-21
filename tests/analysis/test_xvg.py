from __future__ import annotations

from pathlib import Path

import pytest

from gpu_shortmd.analysis.xvg import XvgParseError, parse_xvg


def test_parse_comments_metadata_and_numeric_rows(tmp_path: Path) -> None:
    path = tmp_path / "series.xvg"
    path.write_text(
        '# comment\n@ title "RMSD"\n0 0.10\n10 0.25\n20 0.20\n',
        encoding="utf-8",
    )
    series = parse_xvg(path)
    assert len(series.samples) == 3
    assert series.maximum.time_ps == 10.0
    assert series.maximum.rmsd == 0.25


@pytest.mark.parametrize(
    "contents",
    [
        "",
        "# comments only\n@TYPE xy\n",
        "0 value\n",
        "0 nan\n",
        "0 inf\n",
        "0 -0.1\n",
        "0 0.1 2\n",
        "10 0.1\n5 0.2\n",
        "0 0.1\n0 0.2\n",
    ],
)
def test_invalid_xvg_fails(contents: str, tmp_path: Path) -> None:
    path = tmp_path / "invalid.xvg"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(XvgParseError):
        parse_xvg(path)


def test_non_utf8_xvg_fails_as_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "invalid_encoding.xvg"
    path.write_bytes(b"0.0 0.1\n\xff")

    with pytest.raises(XvgParseError, match="cannot read XVG"):
        parse_xvg(path)
