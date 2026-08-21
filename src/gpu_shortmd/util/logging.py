"""Append-only structured run logging with path redaction."""

from __future__ import annotations

import json
import re
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpu_shortmd.util.files import write_json

EVENT_CATEGORIES = {
    "CONFIG",
    "INPUT",
    "IO",
    "SCHEMA",
    "ORCHESTRATION",
    "COMPUTE",
    "EXECUTION",
    "VALIDATION",
    "TEST",
    "BUILD",
    "OUTPUT",
    "ENV",
    "DEPENDENCY",
    "TIMEOUT",
    "MEMORY",
    "SECURITY",
    "RMSD",
    "PRUNING",
    "SCHEDULER",
    "STATE",
    "UNKNOWN",
}
EVENT_STATUSES = {
    "STARTED",
    "SUCCEEDED",
    "FAILED",
    "SKIPPED",
    "PARTIAL",
    "OBSERVED",
}
EVENT_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}
ISSUE_SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class RunLogger:
    def __init__(
        self,
        *,
        run_dir: Path,
        run_id: str,
        redacted_values: list[str],
        redacted_tokens: list[str] | None = None,
        resume: bool = False,
    ) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.logs_dir = run_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.logs_dir / f"{run_id}_events.jsonl"
        self.issues_path = self.logs_dir / f"{run_id}_issues.jsonl"
        self.stdout_path = self.logs_dir / f"{run_id}_stdout.txt"
        self.stderr_path = self.logs_dir / f"{run_id}_stderr.txt"
        self.redacted_values = sorted(
            {
                value
                for value in redacted_values
                if value and value not in {"/", "\\", ".", ".."}
            },
            key=len,
            reverse=True,
        )
        self.redacted_tokens = sorted(
            {value for value in (redacted_tokens or []) if value},
            key=len,
            reverse=True,
        )
        self._write_lock = threading.Lock()
        self._context = threading.local()
        for path in (
            self.events_path,
            self.issues_path,
            self.stdout_path,
            self.stderr_path,
        ):
            if resume and not path.is_file():
                raise FileNotFoundError(f"required resume log is missing: {path.name}")
            path.touch(exist_ok=resume)

    def redact(self, value: str) -> str:
        redacted = value
        for private in self.redacted_values:
            redacted = redacted.replace(private, "<REDACTED_PATH>")
        for private in self.redacted_tokens:
            redacted = re.sub(
                rf"(?<![\w.-]){re.escape(private)}(?![\w.-])",
                "<REDACTED>",
                redacted,
            )
        redacted = re.sub(
            r"(?i)\b(token|password|secret|api[_-]?key)=\S+",
            r"\1=<REDACTED>",
            redacted,
        )
        return redacted

    def _redact_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.redact(value)
        if isinstance(value, list):
            return [self._redact_value(item) for item in value]
        if isinstance(value, tuple):
            return [self._redact_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self._redact_value(item) for key, item in value.items()}
        return value

    def set_pose_context(self, pose_id: str) -> None:
        if not pose_id:
            raise ValueError("pose context cannot be empty")
        self._context.pose_id = pose_id

    def event(
        self,
        *,
        step_id: str,
        level: str,
        category: str,
        status: str,
        code: str,
        message: str,
        pose_id: str | None = None,
        params: dict[str, Any] | None = None,
        input_refs: list[str] | None = None,
        artifact_refs: list[str] | None = None,
        metrics: dict[str, Any] | None = None,
        duration_ms: int = 0,
        exception_type: str | None = None,
        exception_excerpt: str | None = None,
        suggested_action: str | None = None,
    ) -> None:
        if level not in EVENT_LEVELS:
            raise ValueError(f"invalid event level: {level}")
        if category not in EVENT_CATEGORIES:
            raise ValueError(f"invalid event category: {category}")
        if status not in EVENT_STATUSES:
            raise ValueError(f"invalid event status: {status}")
        resolved_pose_id = pose_id or getattr(
            self._context,
            "pose_id",
            "__RUN__",
        )
        payload = {
            "ts": utc_now(),
            "run_id": self.run_id,
            "pose_id": resolved_pose_id,
            "step_id": step_id,
            "event_id": str(uuid.uuid4()),
            "level": level,
            "category": category,
            "status": status,
            "code": code,
            "message": self.redact(message),
            "params": self._redact_value(params or {}),
            "input_refs": self._redact_value(input_refs or []),
            "artifact_refs": self._redact_value(artifact_refs or []),
            "metrics": self._redact_value(metrics or {}),
            "duration_ms": duration_ms,
            "exception_type": exception_type,
            "exception_excerpt": (
                self.redact(exception_excerpt) if exception_excerpt else None
            ),
            "suggested_action": (
                self.redact(suggested_action) if suggested_action else None
            ),
        }
        with (
            self._write_lock,
            self.events_path.open(
                "a",
                encoding="utf-8",
            ) as handle,
        ):
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def issue(
        self,
        *,
        step_id: str,
        severity: str,
        code: str,
        message: str,
        evidence: list[str],
        suggested_action: str,
    ) -> None:
        if severity not in ISSUE_SEVERITIES:
            raise ValueError(f"invalid issue severity: {severity}")
        payload = {
            "ts": utc_now(),
            "run_id": self.run_id,
            "issue_id": str(uuid.uuid4()),
            "step_id": step_id,
            "severity": severity,
            "code": code,
            "message": self.redact(message),
            "evidence": [self.redact(item) for item in evidence],
            "suggested_action": self.redact(suggested_action),
        }
        with (
            self._write_lock,
            self.issues_path.open(
                "a",
                encoding="utf-8",
            ) as handle,
        ):
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def command_output(self, *, step_id: str, stdout: str, stderr: str) -> None:
        header = f"\n[{utc_now()}] {step_id}\n"
        with self._write_lock:
            with self.stdout_path.open("a", encoding="utf-8") as handle:
                handle.write(header + self.redact(stdout))
            with self.stderr_path.open("a", encoding="utf-8") as handle:
                handle.write(header + self.redact(stderr))

    def write_manifest(self, value: dict[str, Any]) -> None:
        write_json(self.logs_dir / f"{self.run_id}_manifest.json", value)

    def write_artifacts(self, value: dict[str, Any]) -> None:
        write_json(self.logs_dir / f"{self.run_id}_artifacts.json", value)

    def write_summary(self, value: dict[str, Any]) -> None:
        required = {
            "overall_status",
            "top_issues",
            "likely_root_causes",
            "first_checks",
            "failed_steps",
            "partial_outputs",
            "next_actions",
        }
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"log summary missing keys: {', '.join(missing)}")
        write_json(self.logs_dir / f"{self.run_id}_summary.json", value)
