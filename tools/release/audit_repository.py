#!/usr/bin/env python3
"""Audit a browser-visible checkout or an unpacked public release tree."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import yaml

MAX_FILE_BYTES = 20 * 1024 * 1024
FORBIDDEN_SUFFIXES = {
    ".cpt",
    ".dcd",
    ".edr",
    ".gz",
    ".nc",
    ".tar",
    ".tpr",
    ".trr",
    ".tgz",
    ".xtc",
    ".zip",
}
TEXT_SUFFIXES = {
    "",
    ".cff",
    ".csv",
    ".gro",
    ".ini",
    ".itp",
    ".json",
    ".jsonl",
    ".md",
    ".mdp",
    ".ndx",
    ".py",
    ".pdbqt",
    ".rst",
    ".toml",
    ".top",
    ".txt",
    ".yaml",
    ".yml",
}
RELEASE_METADATA_SUFFIXES = {
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
EXPECTED_REPOSITORY_URL = (
    "https://github.com/clinfo/gpu-runtime-steered-ultrashort-md-screening"
)
EXPECTED_PREPRINT_URL = "https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15006352/v1"
EXPECTED_PREPRINT_DOI = "10.26434/chemrxiv.15006352/v1"
EXPECTED_COPYRIGHT = "Copyright (c) 2026 Kyoto University"
EXPECTED_ICPP_DOI = "10.1145/3832810.3832919"
EXPECTED_PREPRINT_TITLE = (
    "Runtime-steered ultrashort molecular dynamics enables million-pose "
    "protein\u2013ligand screening"
)
EXPECTED_DATASET_DOI = "10.5281/" + "zenodo.21835249"
EXPECTED_DATASET_RECORD = {
    "title": (
        "Six-target docking poses and MD-scores for runtime-steered ultrashort "
        "molecular dynamics"
    ),
    "doi": EXPECTED_DATASET_DOI,
    "url": "https://doi.org/" + EXPECTED_DATASET_DOI,
    "version": 2,
    "publication_date": "2026-08-18",
    "creators": [
        "Natsuki Kanazawa",
        "Junta Asano",
        "Shigeyuki Matsumoto",
    ],
    "license": "CC BY 4.0",
    "resource_type": "Dataset",
}
EXPECTED_DATASET_LINK_FILES = {
    "README.md",
    "docs/getting_started.md",
    "docs/legal/data_license.md",
    "docs/method_and_limitations.md",
    "examples/egfr_p00533_1xkk_fmm_p2_model4/README.md",
    "examples/egfr_p00533_1xkk_fmm_p2_model4/zenodo_mapping.yaml",
}
EXPECTED_SOFTWARE_CREATORS = (
    "Natsuki Kanazawa",
    "Junta Asano",
    "Mitsugu Araki",
    "Takao Otsuka",
    "Shigeyuki Matsumoto",
)
EXPECTED_CITATION_CREATORS = (
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
)
REQUIRED_PUBLIC_FILES = {
    ".github/CODE_OF_CONDUCT.md": "# Code of conduct",
    ".github/CONTRIBUTING.md": "# Contributing",
    ".github/SECURITY.md": "# Security policy",
    "LICENSE": "MIT License",
    "CITATION.cff": "preferred-citation:",
    "README.md": "# GPU Runtime-Steered Ultrashort MD Screening",
    "docs/configuration.md": "# Configuration",
    "docs/getting_started.md": "# Getting started",
    "docs/legal/data_license.md": "Creative Commons Attribution 4.0",
    "docs/legal/third_party_notices.md": "GROMACS",
    "docs/method_and_limitations.md": "# Method and limitations",
    "docs/outputs.md": "# Outputs",
    "docs/prepared_input.md": "# Prepared input",
    "docs/running.md": "# Running",
    "docs/runtime_control.md": "# Runtime control",
    "docs/troubleshooting.md": "# Troubleshooting",
    "docs/validated_environments.md": "# Validated environments",
    "dev/environment.yml": "name: gpu-shortmd",
    "dev/requirements-dev-lock.txt": "mypy==",
    "pyproject.toml": 'name = "gpu-shortmd-screening"',
    "templates/screening/README.md": "# Screening template",
    "templates/screening/poses.yaml": "TARGET2_CMPD0042_pose01",
    "templates/screening/run.yaml": "prepared/TARGET1_CMPD0001_pose01",
    "tools/release/audit_packages.py": "Inspect wheel and sdist members",
    "tools/release/audit_repository.py": "Audit a browser-visible checkout",
}


def _obsolete_attribution_literals() -> tuple[str, ...]:
    organization = "Fujitsu " + "Limited"
    return ("Kyoto University and " + organization, organization)


def _sensitive_literals() -> tuple[str, ...]:
    # Construct fragments so the audit implementation does not match itself.
    return (
        "/" + "Users" + "/",
        "/" + "home" + "/",
        "/" + "archive" + "/",
        "/" + "vol0004" + "/",
        "".join(("s", "i", "m", "c", "r", "n")),
        "".join(("Advance", "Soft")),
    )


def _forbidden_public_paths() -> tuple[str, ...]:
    return (
        "docs/" + "compliance_matrix.md",
        "docs/" + "prepared_input_contract.md",
        "docs/" + "runtime_steering.md",
        "docs/" + "validation.md",
        "examples/egfr_p00533_1xkk_fmm_p2_model4/" + "multi_pose_manifest.example.yaml",
        "src/gpu_shortmd/cli/" + "pipeline.py",
    )


def _forbidden_public_prefixes() -> tuple[str, ...]:
    return (
        "configs/" + "presets/",
        "docs/" + "contract/",
        "docs/" + "decisions/",
        "docs/" + "project/",
        "internal" + "_validation/",
        "notes/",
        "scripts/",
        "src/gpu_shortmd/" + "experimental/",
    )


def _private_development_literals() -> tuple[str, ...]:
    return (
        "internal" + "_validation",
        "C" + "PATCH",
        "NON" + "COMPLIANCE",
        "DEFERRED" + "_EXTERNAL_VALIDATION",
        ".private" + "_validation_inputs",
        ".private" + "-release-audit",
        "deploy" + "-key",
        "create_release_" + "audit_bundle",
    )


def _restricted_release_metadata_patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    activity_keys = tuple(
        "_".join(("activity", field)) for field in ("type", "value", "unit")
    )
    database_names = (
        "PDB" + "bind",
        "Ch" + "EMBL",
        "Pub" + "Chem",
    )
    activity_measures = (
        "K" + "i",
        "K" + "d",
        "IC" + "50",
        "p" + "Ch" + "EMBL",
    )
    label_key = "activity" + "_label"
    bioactivity_key = "bio" + "activity" + "_label"
    active_label = "act" + "ive"
    inactive_label = "in" + "active"
    stale_literals = (
        "10.5281/" + "zenodo." + "20828001",
        "zenodo.org/records/" + "20828001",
        "runtime-steered-ultrashort-md-screening-" + "analysis",
        "clinfo/" + "gpu-shortmd-screening",
        "_".join(("DATA", "LICENSE", "REVIEW", "PENDING")),
    )
    return (
        (
            "activity_key",
            re.compile(r"(?i)\b(?:" + "|".join(map(re.escape, activity_keys)) + r")\b"),
        ),
        (
            "activity_database",
            re.compile(
                r"(?i)\b(?:" + "|".join(map(re.escape, database_names)) + r")\b"
            ),
        ),
        (
            "activity_measure",
            re.compile(
                r"(?i)\b(?:" + "|".join(map(re.escape, activity_measures)) + r")\b"
            ),
        ),
        (
            "activity_label",
            re.compile(
                r"(?i)(?:\b(?:"
                + "|".join(map(re.escape, (label_key, bioactivity_key)))
                + r")\b|\blabel\s*[:=]\s*(?:"
                + "|".join(map(re.escape, (active_label, inactive_label)))
                + r")\b|\b"
                + re.escape(active_label + "/" + inactive_label)
                + r"\b)"
            ),
        ),
        (
            "activity_threshold_label",
            re.compile(r"(?i)\b(?:1|10)[- ]?(?:u|µ|μ)m\b"),
        ),
        (
            "stale_public_metadata",
            re.compile("|".join(map(re.escape, stale_literals)), re.IGNORECASE),
        ),
    )


def _cff_author_names(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    names: list[str] = []
    for author in value:
        if not isinstance(author, dict):
            return ()
        given = author.get("given-names")
        family = author.get("family-names")
        if not isinstance(given, str) or not isinstance(family, str):
            return ()
        names.append(f"{given} {family}")
    return tuple(names)


def _forbidden_code_literals() -> tuple[str, ...]:
    return (
        "".join(("Open", "MM")),
        "".join(("GB", "SA")),
        "".join(("Hyper", "sound")),
        "".join(("ve", "lex")),
        "".join(("shell", "=", "True")),
        "".join(("-maxwarn", " ", "-1")),
    )


def _inside_git_worktree(root: Path) -> bool:
    process = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
        check=False,
        capture_output=True,
        text=True,
    )
    return process.returncode == 0 and process.stdout.strip() == "true"


def candidate_files(root: Path) -> list[Path]:
    if not _inside_git_worktree(root):
        return sorted(
            path for path in root.rglob("*") if path.is_file() or path.is_symlink()
        )
    process = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"git file inventory failed: {process.stderr.strip()}")
    relative_paths = [line for line in process.stdout.splitlines() if line]
    return [root / relative for relative in relative_paths]


def read_text(path: Path) -> str | None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def audit(root: Path) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    files = candidate_files(root)
    restricted_release_metadata = _restricted_release_metadata_patterns()
    relative_files = {path.relative_to(root).as_posix() for path in files}
    if "CITATION.bib" in relative_files:
        findings.append(
            {
                "category": "citation_metadata",
                "path": "CITATION.bib",
                "detail": "a separate BibTeX citation file must not be introduced",
            }
        )
    dataset_link_files = {
        path.relative_to(root).as_posix()
        for path in files
        if (content := read_text(path)) is not None and EXPECTED_DATASET_DOI in content
    }
    if dataset_link_files != EXPECTED_DATASET_LINK_FILES:
        findings.append(
            {
                "category": "reference_dataset_metadata",
                "path": "public tree",
                "detail": "Zenodo Version 2 DOI is missing or appears unexpectedly",
            }
        )
    for relative in sorted(relative_files):
        if relative in _forbidden_public_paths() or any(
            relative.startswith(prefix) for prefix in _forbidden_public_prefixes()
        ):
            findings.append(
                {
                    "category": "private_or_out_of_scope_path",
                    "path": relative,
                    "detail": "path must remain outside the public candidate",
                }
            )
    for required, marker in REQUIRED_PUBLIC_FILES.items():
        if required not in relative_files:
            findings.append(
                {
                    "category": "required_public_file",
                    "path": required,
                    "detail": "required public file is missing",
                }
            )
            continue
        try:
            content = (root / required).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            content = ""
        if marker not in content:
            findings.append(
                {
                    "category": "required_public_file",
                    "path": required,
                    "detail": f"required marker is missing: {marker}",
                }
            )

    attributes = root / ".gitattributes"
    if attributes.is_file():
        export_filter = "-".join(("export", "ignore"))
        try:
            attributes_text = attributes.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            attributes_text = ""
        if export_filter in attributes_text:
            findings.append(
                {
                    "category": "hidden_public_tree_content",
                    "path": ".gitattributes",
                    "detail": "archive filtering must not hide tracked files",
                }
            )

    citation = root / "CITATION.cff"
    if citation.is_file():
        citation_text = ""
        try:
            citation_text = citation.read_text(encoding="utf-8")
            citation_data = yaml.safe_load(citation_text)
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            findings.append(
                {
                    "category": "citation_metadata",
                    "path": "CITATION.cff",
                    "detail": f"CFF parsing failed: {exc}",
                }
            )
        else:
            if not isinstance(citation_data, dict):
                findings.append(
                    {
                        "category": "citation_metadata",
                        "path": "CITATION.cff",
                        "detail": "CFF root must be a mapping",
                    }
                )
            else:
                preferred = citation_data.get("preferred-citation")
                citation_checks = {
                    "cff-version": "1.2.0",
                    "title": "GPU Runtime-Steered Ultrashort MD Screening",
                    "version": "0.1.0",
                    "repository-code": EXPECTED_REPOSITORY_URL,
                }
                for field, expected in citation_checks.items():
                    if str(citation_data.get(field)) != expected:
                        findings.append(
                            {
                                "category": "citation_metadata",
                                "path": "CITATION.cff",
                                "detail": f"unexpected {field}",
                            }
                        )
                if "doi" in citation_data or "identifiers" in citation_data:
                    findings.append(
                        {
                            "category": "citation_metadata",
                            "path": "CITATION.cff",
                            "detail": (
                                "top-level software DOI fields must remain absent"
                            ),
                        }
                    )
                if EXPECTED_ICPP_DOI in citation_text:
                    findings.append(
                        {
                            "category": "citation_metadata",
                            "path": "CITATION.cff",
                            "detail": (
                                "related ICPP work must not be a software citation"
                            ),
                        }
                    )
                if (
                    _cff_author_names(citation_data.get("authors"))
                    != EXPECTED_SOFTWARE_CREATORS
                ):
                    findings.append(
                        {
                            "category": "citation_metadata",
                            "path": "CITATION.cff",
                            "detail": "software creator order is incorrect",
                        }
                    )
                if not isinstance(preferred, dict):
                    findings.append(
                        {
                            "category": "citation_metadata",
                            "path": "CITATION.cff",
                            "detail": "preferred citation is missing",
                        }
                    )
                else:
                    preferred_checks: dict[str, object] = {
                        "type": "article",
                        "title": EXPECTED_PREPRINT_TITLE,
                        "journal": "ChemRxiv",
                        "year": 2026,
                        "doi": EXPECTED_PREPRINT_DOI,
                        "url": EXPECTED_PREPRINT_URL,
                    }
                    for field, expected in preferred_checks.items():
                        if preferred.get(field) != expected:
                            findings.append(
                                {
                                    "category": "citation_metadata",
                                    "path": "CITATION.cff",
                                    "detail": f"unexpected preferred-citation {field}",
                                }
                            )
                    if (
                        _cff_author_names(preferred.get("authors"))
                        != EXPECTED_CITATION_CREATORS
                    ):
                        findings.append(
                            {
                                "category": "citation_metadata",
                                "path": "CITATION.cff",
                                "detail": (
                                    "preferred citation author order is incorrect"
                                ),
                            }
                        )

    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        try:
            project_data = tomllib.loads(pyproject.read_text(encoding="utf-8"))[
                "project"
            ]
        except (KeyError, OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
            findings.append(
                {
                    "category": "project_metadata",
                    "path": "pyproject.toml",
                    "detail": f"project metadata parsing failed: {exc}",
                }
            )
        else:
            author_names = tuple(
                item.get("name")
                for item in project_data.get("authors", [])
                if isinstance(item, dict)
            )
            if author_names != EXPECTED_SOFTWARE_CREATORS:
                findings.append(
                    {
                        "category": "project_metadata",
                        "path": "pyproject.toml",
                        "detail": "project author order is incorrect",
                    }
                )
            if project_data.get("requires-python") != ">=3.11,<3.13":
                findings.append(
                    {
                        "category": "project_metadata",
                        "path": "pyproject.toml",
                        "detail": "supported Python requirement is incorrect",
                    }
                )
            if project_data.get("license") != "MIT":
                findings.append(
                    {
                        "category": "project_metadata",
                        "path": "pyproject.toml",
                        "detail": "software license type is incorrect",
                    }
                )
            expected_urls = {
                "Repository": EXPECTED_REPOSITORY_URL,
                "Documentation": EXPECTED_REPOSITORY_URL + "#readme",
                "Issues": EXPECTED_REPOSITORY_URL + "/issues",
                "Preprint": EXPECTED_PREPRINT_URL,
            }
            if project_data.get("urls") != expected_urls:
                findings.append(
                    {
                        "category": "project_metadata",
                        "path": "pyproject.toml",
                        "detail": "project URLs are incorrect",
                    }
                )

    mapping_path = (
        root / "examples" / "egfr_p00533_1xkk_fmm_p2_model4" / "zenodo_mapping.yaml"
    )
    if mapping_path.is_file():
        try:
            mapping = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            findings.append(
                {
                    "category": "reference_dataset_metadata",
                    "path": mapping_path.relative_to(root).as_posix(),
                    "detail": f"mapping parsing failed: {exc}",
                }
            )
        else:
            if not isinstance(mapping, dict) or mapping.get("record") != (
                EXPECTED_DATASET_RECORD
            ):
                findings.append(
                    {
                        "category": "reference_dataset_metadata",
                        "path": mapping_path.relative_to(root).as_posix(),
                        "detail": "Zenodo Version 2 record metadata is incorrect",
                    }
                )

    license_path = root / "LICENSE"
    if license_path.is_file():
        try:
            license_text = license_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            findings.append(
                {
                    "category": "license_metadata",
                    "path": "LICENSE",
                    "detail": f"license parsing failed: {exc}",
                }
            )
        else:
            copyright_lines = [
                line
                for line in license_text.splitlines()
                if line.startswith("Copyright")
            ]
            if copyright_lines != [EXPECTED_COPYRIGHT]:
                findings.append(
                    {
                        "category": "license_metadata",
                        "path": "LICENSE",
                        "detail": "copyright holder is incorrect",
                    }
                )

    method_path = root / "docs" / "method_and_limitations.md"
    if method_path.is_file():
        try:
            method_text = method_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            method_text = ""
        required_related_work = (
            "### Related runtime-steering work",
            "The design of the pruning and work-stealing mechanisms in this "
            "software draws",
            "work stealing as a scheduling strategy",
            "broader evaluation as future work",
            "in press, 2026",
            EXPECTED_ICPP_DOI,
        )
        for marker in required_related_work:
            if marker not in method_text:
                findings.append(
                    {
                        "category": "related_work_citation",
                        "path": "docs/method_and_limitations.md",
                        "detail": (
                            "required restrained related-work marker is missing: "
                            f"{marker}"
                        ),
                    }
                )
        forbidden_claims = (
            "job " + "stealing",
            "validated " + "by",
            "implemented " + "from",
            "reproduces",
        )
        for claim in forbidden_claims:
            if claim in method_text:
                findings.append(
                    {
                        "category": "related_work_citation",
                        "path": "docs/method_and_limitations.md",
                        "detail": (
                            f"overstated or incorrect related-work wording: {claim}"
                        ),
                    }
                )

    expected_installation = (
        "python3 --version  # Confirm Python 3.11 or 3.12.\npython3 -m venv .venv"
    )
    for relative in ("README.md", "docs/getting_started.md"):
        installation_path = root / relative
        try:
            installation_text = installation_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            installation_text = ""
        if expected_installation not in installation_text:
            findings.append(
                {
                    "category": "installation_documentation",
                    "path": relative,
                    "detail": (
                        "version-checked python3 virtual environment command is missing"
                    ),
                }
            )
        for obsolete_command in (
            "python3." + "11 -m venv",
            "python3." + "12 -m venv",
        ):
            if obsolete_command in installation_text:
                findings.append(
                    {
                        "category": "installation_documentation",
                        "path": relative,
                        "detail": (
                            f"version-specific installation command: {obsolete_command}"
                        ),
                    }
                )

    for path in files:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            findings.append(
                {
                    "category": "symlink",
                    "path": relative,
                    "detail": "public candidate must not contain symlinks",
                }
            )
            continue
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            findings.append(
                {"category": "large_file", "path": relative, "detail": str(size)}
            )
        suffix = path.suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES:
            findings.append(
                {"category": "forbidden_extension", "path": relative, "detail": suffix}
            )
        content = read_text(path)
        if content is None:
            continue
        for literal in _obsolete_attribution_literals():
            if literal in content:
                findings.append(
                    {
                        "category": "obsolete_attribution",
                        "path": relative,
                        "detail": "obsolete joint software attribution",
                    }
                )
        for literal in _private_development_literals():
            if literal in content:
                findings.append(
                    {
                        "category": "private_development_content",
                        "path": relative,
                        "detail": literal,
                    }
                )
        if suffix in RELEASE_METADATA_SUFFIXES:
            for label, pattern in restricted_release_metadata:
                match = pattern.search(content)
                if match is not None:
                    findings.append(
                        {
                            "category": "restricted_release_metadata",
                            "path": relative,
                            "detail": f"{label}: {match.group(0)}",
                        }
                    )
        for literal in _sensitive_literals():
            if literal in content:
                findings.append(
                    {
                        "category": "private_path_or_identity",
                        "path": relative,
                        "detail": literal,
                    }
                )
        if re.search(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", content):
            findings.append(
                {
                    "category": "email_address",
                    "path": relative,
                    "detail": "email-like value",
                }
            )
        if "-----BEGIN " + "PRIVATE" + " KEY-----" in content or re.search(
            r"\b(?:AKIA[0-9A-Z]{16}|gh[opsu]_[A-Za-z0-9]{30,})\b",
            content,
        ):
            findings.append(
                {
                    "category": "secret",
                    "path": relative,
                    "detail": "credential-like value",
                }
            )
        if path.suffix == ".py":
            for literal in _forbidden_code_literals():
                if literal in content:
                    findings.append(
                        {
                            "category": "forbidden_feature_or_execution",
                            "path": relative,
                            "detail": literal,
                        }
                    )
            if re.search(r"\bshell\s*=\s*True\b", content):
                findings.append(
                    {
                        "category": "forbidden_feature_or_execution",
                        "path": relative,
                        "detail": "shell execution enabled",
                    }
                )

    return {
        "status": "PASS" if not findings else "FAIL",
        "files_checked": len(files),
        "max_file_bytes": MAX_FILE_BYTES,
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.root.resolve())
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
