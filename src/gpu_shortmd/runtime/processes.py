"""Ownership-checked process-group signaling for stop and pruning."""

from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass
from enum import StrEnum

import psutil  # type: ignore[import-untyped]


class ProcessOwnershipError(RuntimeError):
    """Raised instead of signaling a process whose ownership is uncertain."""


class ProcessNotFoundError(ProcessOwnershipError):
    """Raised when the recorded operating-system process no longer exists."""


class ProcessIdentityStatus(StrEnum):
    MATCHING_LIVE = "MATCHING_LIVE"
    EXITED = "EXITED"
    PID_REUSED = "PID_REUSED"
    UNVERIFIABLE = "UNVERIFIABLE"


@dataclass(frozen=True)
class ProcessIdentityResult:
    pid: int
    status: ProcessIdentityStatus
    reason: str


@dataclass(frozen=True)
class TerminationResult:
    pid: int
    terminated: bool
    forced: bool
    reason: str


def process_start_token(pid: int) -> str:
    """Return an OS-observed start token used to detect stale PID reuse."""
    if pid <= 1:
        raise ProcessOwnershipError("refusing an invalid process ID")
    proc_stat = f"/proc/{pid}/stat"
    try:
        with open(proc_stat, encoding="utf-8") as handle:
            fields = handle.read().split()
        if len(fields) > 21:
            return f"proc:{fields[21]}"
    except OSError:
        pass
    try:
        created = psutil.Process(pid).create_time()
    except psutil.NoSuchProcess as exc:
        raise ProcessNotFoundError("recorded process no longer exists") from exc
    except (psutil.Error, OSError) as exc:
        raise ProcessOwnershipError("cannot verify process start time") from exc
    return f"psutil:{created:.6f}"


def inspect_process_identity(
    *,
    pid: int,
    expected_start_token: str,
) -> ProcessIdentityResult:
    """Classify a recorded process without signaling it or guessing staleness."""
    if pid <= 1 or not expected_start_token:
        return ProcessIdentityResult(
            pid=pid,
            status=ProcessIdentityStatus.UNVERIFIABLE,
            reason="recorded process identity is incomplete or invalid",
        )
    try:
        actual_start_token = process_start_token(pid)
    except ProcessNotFoundError as exc:
        return ProcessIdentityResult(
            pid=pid,
            status=ProcessIdentityStatus.EXITED,
            reason=str(exc),
        )
    except ProcessOwnershipError as exc:
        return ProcessIdentityResult(
            pid=pid,
            status=ProcessIdentityStatus.UNVERIFIABLE,
            reason=str(exc),
        )
    if actual_start_token != expected_start_token:
        return ProcessIdentityResult(
            pid=pid,
            status=ProcessIdentityStatus.PID_REUSED,
            reason="process start token mismatch; recorded owner has exited",
        )
    try:
        process = psutil.Process(pid)
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return ProcessIdentityResult(
                pid=pid,
                status=ProcessIdentityStatus.EXITED,
                reason="recorded process is no longer running",
            )
        confirmed_start_token = process_start_token(pid)
    except (ProcessNotFoundError, psutil.NoSuchProcess):
        return ProcessIdentityResult(
            pid=pid,
            status=ProcessIdentityStatus.EXITED,
            reason="recorded process exited during identity verification",
        )
    except (ProcessOwnershipError, psutil.Error, OSError) as exc:
        return ProcessIdentityResult(
            pid=pid,
            status=ProcessIdentityStatus.UNVERIFIABLE,
            reason=f"cannot confirm recorded process ownership: {exc}",
        )
    if confirmed_start_token != expected_start_token:
        return ProcessIdentityResult(
            pid=pid,
            status=ProcessIdentityStatus.PID_REUSED,
            reason="PID was reused during identity verification",
        )
    return ProcessIdentityResult(
        pid=pid,
        status=ProcessIdentityStatus.MATCHING_LIVE,
        reason="matching recorded process is still live",
    )


def terminate_owned_process_group(
    *,
    pid: int,
    expected_start_token: str,
    grace_seconds: float,
) -> TerminationResult:
    """Signal only a validated session leader recorded by this run."""
    if grace_seconds < 0:
        raise ValueError("grace_seconds cannot be negative")
    try:
        actual_start_token = process_start_token(pid)
    except ProcessNotFoundError as exc:
        return TerminationResult(
            pid=pid,
            terminated=True,
            forced=False,
            reason=str(exc),
        )
    except ProcessOwnershipError as exc:
        return TerminationResult(
            pid=pid,
            terminated=False,
            forced=False,
            reason=str(exc),
        )
    if actual_start_token != expected_start_token:
        return TerminationResult(
            pid=pid,
            terminated=False,
            forced=False,
            reason="process start token mismatch; PID may have been reused",
        )
    try:
        process_group = os.getpgid(pid)
    except ProcessLookupError:
        return TerminationResult(
            pid=pid,
            terminated=True,
            forced=False,
            reason="process already exited",
        )
    if process_group != pid:
        return TerminationResult(
            pid=pid,
            terminated=False,
            forced=False,
            reason="recorded process is not its process-group leader",
        )

    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return TerminationResult(
            pid=pid,
            terminated=True,
            forced=False,
            reason="process exited before SIGTERM",
        )
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return TerminationResult(
                pid=pid,
                terminated=True,
                forced=False,
                reason="process exited after SIGTERM",
            )
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        return TerminationResult(
            pid=pid,
            terminated=True,
            forced=False,
            reason="process exited during grace period",
        )
    return TerminationResult(
        pid=pid,
        terminated=True,
        forced=True,
        reason="process group required SIGKILL after grace period",
    )
