from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

from gpu_shortmd import __version__

ROOT = Path(__file__).parents[2]
SOFTWARE_CREATORS = [
    "Natsuki Kanazawa",
    "Junta Asano",
    "Mitsugu Araki",
    "Takao Otsuka",
    "Shigeyuki Matsumoto",
]
PREFERRED_CITATION_CREATORS = [
    "Natsuki Kanazawa",
    "Junta Asano",
    "Mitsugu Araki",
    "Shingo Okuno",
    "Takao Otsuka",
    "Yuta Isaka",
    "Hiroaki Iwata",
    "Shuntaro Chiba",
    "Yenni Ng",
    "Yukiko Muramoto",
    "Chiho Onishi",
    "Kiyoshi Takemura",
    "Biao Ma",
    "Takashi Katoh",
    "Kei Terayama",
    "Norihito Arichi",
    "Hiroaki Ohno",
    "Takeshi Noda",
    "Motonari Uesugi",
    "Shigeyuki Matsumoto",
    "Yasushi Okuno",
]
REPOSITORY_URL = "https://github.com/clinfo/gpu-runtime-steered-ultrashort-md-screening"
PREPRINT_URL = "https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15006352/v1"
ICPP_DOI = "10.1145/3832810.3832919"
COPYRIGHT = "Copyright (c) 2026 Kyoto University"


def _citation_names(authors: list[dict[str, str]]) -> list[str]:
    return [f"{author['given-names']} {author['family-names']}" for author in authors]


def test_version_and_license_metadata_are_consistent() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert project["name"] == "gpu-shortmd-screening"
    assert project["version"] == __version__ == str(citation["version"])
    assert project["license"] == citation["license"] == "MIT"
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in license_text
    assert [
        line for line in license_text.splitlines() if line.startswith("Copyright")
    ] == [COPYRIGHT]
    data_license = (ROOT / "docs" / "legal" / "data_license.md").read_text(
        encoding="utf-8"
    )
    assert "Creative Commons Attribution 4.0" in data_license
    assert COPYRIGHT in data_license


def test_creators_repository_urls_and_preferred_citation_are_exact() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))

    assert [author["name"] for author in project["authors"]] == SOFTWARE_CREATORS
    assert _citation_names(citation["authors"]) == SOFTWARE_CREATORS
    assert len(SOFTWARE_CREATORS) == 5
    assert len(PREFERRED_CITATION_CREATORS) == 21
    publication_only_authors = {"Shingo Okuno", "Yuta Isaka"}
    assert publication_only_authors.isdisjoint(SOFTWARE_CREATORS)
    assert publication_only_authors.issubset(PREFERRED_CITATION_CREATORS)
    assert all(
        set(author) == {"given-names", "family-names"} for author in citation["authors"]
    )
    assert citation["title"] == "GPU Runtime-Steered Ultrashort MD Screening"
    assert citation["repository-code"] == REPOSITORY_URL
    assert citation["url"] == REPOSITORY_URL
    assert "doi" not in citation
    assert "identifiers" not in citation

    assert project["urls"] == {
        "Repository": REPOSITORY_URL,
        "Documentation": REPOSITORY_URL + "#readme",
        "Issues": REPOSITORY_URL + "/issues",
        "Preprint": PREPRINT_URL,
    }

    preferred = citation["preferred-citation"]
    assert preferred["type"] == "article"
    assert preferred["title"] == (
        "Runtime-steered ultrashort molecular dynamics enables million-pose "
        "protein\u2013ligand screening"
    )
    assert preferred["journal"] == "ChemRxiv"
    assert preferred["year"] == 2026
    assert preferred["doi"] == "10.26434/chemrxiv.15006352/v1"
    assert preferred["url"] == PREPRINT_URL
    assert _citation_names(preferred["authors"]) == PREFERRED_CITATION_CREATORS
    assert all(
        set(author) == {"given-names", "family-names"}
        for author in preferred["authors"]
    )


def test_related_icpp_citation_is_restrained_and_role_specific() -> None:
    method = (ROOT / "docs" / "method_and_limitations.md").read_text(encoding="utf-8")
    assert "### Related runtime-steering work" in method
    assert (
        "The design of the pruning and work-stealing mechanisms in this software "
        "draws\non the early-termination and parallel-scheduling framework described "
        "by Okuno\n"
        "et al."
    ) in method
    assert "work stealing as a scheduling strategy" in method
    assert "broader evaluation as future work" in method
    assert "in press, 2026" in method
    assert method.count(ICPP_DOI) == 1

    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert ICPP_DOI not in citation
    assert ICPP_DOI not in readme
    assert not (ROOT / "CITATION.bib").exists()


def test_public_installation_uses_version_checked_python3() -> None:
    expected = (
        "python3 --version  # Confirm Python 3.11 or 3.12.\npython3 -m venv .venv"
    )
    for relative in ("README.md", "docs/getting_started.md"):
        content = (ROOT / relative).read_text(encoding="utf-8")
        assert expected in content
        assert "python3." + "11 -m venv" not in content
        assert "python3." + "12 -m venv" not in content


def test_optional_backends_are_not_core_dependencies() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    core = "\n".join(project["dependencies"]).lower()
    for optional in ("vina", "rdkit", "openbabel", "acpype"):
        assert optional not in core
