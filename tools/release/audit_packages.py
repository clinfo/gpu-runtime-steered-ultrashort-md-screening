#!/usr/bin/env python3
"""Inspect wheel and sdist members without extracting them."""

from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

MAX_MEMBER_BYTES = 20 * 1024 * 1024
FORBIDDEN_SUFFIXES = {
    ".cpt",
    ".dcd",
    ".edr",
    ".nc",
    ".tpr",
    ".trr",
    ".xtc",
    ".zip",
}
FORBIDDEN_PACKAGE_MEMBERS = {"gpu_shortmd/cli/pipeline.py"}
FORBIDDEN_PACKAGE_PREFIXES = {"gpu_shortmd/experimental/"}
REQUIRED_SDIST_SUFFIXES = {
    "/README.md",
    "/LICENSE",
    "/configs/config.schema.json",
    "/configs/pose-manifest.schema.json",
    "/docs/configuration.md",
    "/docs/getting_started.md",
    "/docs/legal/data_license.md",
    "/docs/legal/third_party_notices.md",
    "/docs/method_and_limitations.md",
    "/docs/outputs.md",
    "/docs/prepared_input.md",
    "/docs/running.md",
    "/docs/runtime_control.md",
    "/docs/troubleshooting.md",
    "/docs/validated_environments.md",
    "/examples/egfr_p00533_1xkk_fmm_p2_model4/reference/"
    "p2_model4_rmsd_replica_01_nm.xvg",
    "/src/gpu_shortmd/runtime/state.py",
    "/templates/screening/README.md",
    "/templates/screening/poses.yaml",
    "/templates/screening/run.yaml",
}


def _forbidden_sdist_paths() -> tuple[str, ...]:
    return (
        "docs/" + "compliance_matrix.md",
        "docs/" + "prepared_input_contract.md",
        "docs/" + "runtime_steering.md",
        "docs/" + "validation.md",
    )


def _forbidden_sdist_prefixes() -> tuple[str, ...]:
    return (
        "configs/" + "presets/",
        "dev/",
        "docs/" + "contract/",
        "docs/" + "decisions/",
        "docs/" + "project/",
        "internal" + "_validation/",
        "notes/",
        "scripts/",
        "tools/" + "release/",
    )


def _member_findings(name: str, size: int) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        findings.append(
            {
                "category": "unsafe_member_path",
                "path": name,
                "detail": "absolute or parent-traversing archive member",
            }
        )
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        findings.append(
            {
                "category": "forbidden_member",
                "path": name,
                "detail": path.suffix.lower(),
            }
        )
    if size > MAX_MEMBER_BYTES:
        findings.append(
            {
                "category": "oversized_member",
                "path": name,
                "detail": str(size),
            }
        )
    return findings


def audit_wheel(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
    names = [member.filename for member in members if not member.is_dir()]
    findings = [
        finding
        for member in members
        for finding in _member_findings(member.filename, member.file_size)
    ]
    required_suffixes = {
        "gpu_shortmd/cli/app.py",
        "gpu_shortmd/runtime/state.py",
        ".dist-info/METADATA",
        ".dist-info/RECORD",
        ".dist-info/entry_points.txt",
    }
    for required in required_suffixes:
        if not any(name.endswith(required) for name in names):
            findings.append(
                {
                    "category": "missing_wheel_member",
                    "path": required,
                    "detail": "required member is missing",
                }
            )
    for name in names:
        if name in FORBIDDEN_PACKAGE_MEMBERS or any(
            name.startswith(prefix) for prefix in FORBIDDEN_PACKAGE_PREFIXES
        ):
            findings.append(
                {
                    "category": "out_of_scope_wheel_member",
                    "path": name,
                    "detail": "prepared-system package boundary violated",
                }
            )
    return names, findings


def audit_sdist(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with tarfile.open(path, mode="r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
    names = [member.name for member in members]
    findings = [
        finding
        for member in members
        for finding in _member_findings(member.name, member.size)
    ]
    for required in REQUIRED_SDIST_SUFFIXES:
        if not any(name.endswith(required) for name in names):
            findings.append(
                {
                    "category": "missing_sdist_member",
                    "path": required,
                    "detail": "required public member is missing",
                }
            )
    for name in names:
        relative = name.split("/", maxsplit=1)[-1]
        if relative in _forbidden_sdist_paths() or any(
            relative.startswith(prefix) for prefix in _forbidden_sdist_prefixes()
        ):
            findings.append(
                {
                    "category": "private_sdist_member",
                    "path": name,
                    "detail": "member must remain outside public packages",
                }
            )
        if relative == "src/gpu_shortmd/cli/pipeline.py" or relative.startswith(
            "src/gpu_shortmd/experimental/"
        ):
            findings.append(
                {
                    "category": "out_of_scope_sdist_member",
                    "path": name,
                    "detail": "prepared-system package boundary violated",
                }
            )
    return names, findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    args = parser.parse_args()
    wheel_names, wheel_findings = audit_wheel(args.wheel)
    sdist_names, sdist_findings = audit_sdist(args.sdist)
    findings = wheel_findings + sdist_findings
    report: dict[str, Any] = {
        "status": "PASS" if not findings else "FAIL",
        "wheel": args.wheel.name,
        "wheel_members": len(wheel_names),
        "sdist": args.sdist.name,
        "sdist_members": len(sdist_names),
        "max_member_bytes": MAX_MEMBER_BYTES,
        "findings": findings,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
