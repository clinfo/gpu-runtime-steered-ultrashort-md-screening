"""Subprocess execution using argument arrays and isolated process groups."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_ms: int
    interrupted: bool = False


class CommandTimeoutError(RuntimeError):
    """Raised when an external command exceeds its explicit timeout."""


def _terminate_process_group(
    process: subprocess.Popen[str],
    *,
    grace_seconds: float,
) -> None:
    """Terminate a still-live session leader and wait before ownership clears."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_command(
    args: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stdin_text: str | None = None,
    timeout_seconds: float | None = None,
) -> CommandResult:
    if not args or any(not isinstance(item, str) for item in args):
        raise ValueError("external command must be a non-empty string argument list")
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(args),
            cwd=cwd,
            env=dict(env),
            input=stdin_text,
            capture_output=True,
            text=True,
            check=False,
            start_new_session=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise CommandTimeoutError(
            f"external command timed out after {timeout_seconds} seconds"
        ) from exc
    return CommandResult(
        args=tuple(args),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_ms=round((time.monotonic() - started) * 1000),
    )


def run_cancellable_command(
    args: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stop_requested: Callable[[], bool],
    on_start: Callable[[int], None],
    on_exit: Callable[[int], None],
    stdin_text: str | None = None,
    timeout_seconds: float | None = None,
    poll_interval_seconds: float = 0.2,
    termination_grace_seconds: float = 5.0,
) -> CommandResult:
    """Run an owned process group and durably report its confirmed exit."""
    if not args or any(not isinstance(item, str) for item in args):
        raise ValueError("external command must be a non-empty string argument list")
    if poll_interval_seconds <= 0 or termination_grace_seconds < 0:
        raise ValueError("invalid subprocess polling or grace interval")
    started = time.monotonic()
    process = subprocess.Popen(
        list(args),
        cwd=cwd,
        env=dict(env),
        stdin=subprocess.PIPE if stdin_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    interrupted = False
    first_communicate = True
    registered = False
    try:
        on_start(process.pid)
        registered = True
        while True:
            elapsed = time.monotonic() - started
            if timeout_seconds is not None and elapsed >= timeout_seconds:
                _terminate_process_group(
                    process,
                    grace_seconds=termination_grace_seconds,
                )
                raise CommandTimeoutError(
                    f"external command timed out after {timeout_seconds} seconds"
                )
            if stop_requested():
                interrupted = True
                _terminate_process_group(
                    process,
                    grace_seconds=termination_grace_seconds,
                )
            try:
                stdout, stderr = process.communicate(
                    input=stdin_text if first_communicate else None,
                    timeout=poll_interval_seconds,
                )
                break
            except subprocess.TimeoutExpired:
                first_communicate = False
                continue
        if stop_requested():
            interrupted = True
            _terminate_process_group(
                process,
                grace_seconds=termination_grace_seconds,
            )
    except BaseException:
        _terminate_process_group(
            process,
            grace_seconds=termination_grace_seconds,
        )
        raise
    finally:
        if registered:
            if process.poll() is None:
                _terminate_process_group(
                    process,
                    grace_seconds=termination_grace_seconds,
                )
            if process.returncode is None:
                raise RuntimeError("owned process exit could not be confirmed")
            on_exit(process.returncode)
    return CommandResult(
        args=tuple(args),
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
        duration_ms=round((time.monotonic() - started) * 1000),
        interrupted=interrupted,
    )
