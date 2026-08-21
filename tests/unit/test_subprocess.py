from __future__ import annotations

import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

from gpu_shortmd.util.subprocess import run_cancellable_command


def test_external_stop_during_communicate_returns_interrupted(tmp_path: Path) -> None:
    stop_intent = threading.Event()
    process_started = threading.Event()
    process_pid: list[int] = []
    recorded_returncodes: list[int] = []

    def register(pid: int) -> None:
        process_pid.append(pid)
        process_started.set()

    def external_stop() -> None:
        assert process_started.wait(timeout=2)
        time.sleep(0.05)
        stop_intent.set()
        os.killpg(process_pid[0], signal.SIGTERM)

    stop_thread = threading.Thread(target=external_stop)
    stop_thread.start()
    result = run_cancellable_command(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        env=os.environ,
        stop_requested=stop_intent.is_set,
        on_start=register,
        on_exit=recorded_returncodes.append,
        poll_interval_seconds=1.0,
        termination_grace_seconds=1.0,
    )
    stop_thread.join(timeout=2)

    assert not stop_thread.is_alive()
    assert result.interrupted is True
    assert result.returncode != 0
    assert recorded_returncodes == [result.returncode]


def test_keyboard_interrupt_terminates_process_before_exit_is_recorded(
    tmp_path: Path,
) -> None:
    process_pid: list[int] = []
    recorded_returncodes: list[int] = []

    def interrupt() -> bool:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_cancellable_command(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            env=os.environ,
            stop_requested=interrupt,
            on_start=process_pid.append,
            on_exit=recorded_returncodes.append,
            termination_grace_seconds=1.0,
        )

    assert len(recorded_returncodes) == 1
    assert recorded_returncodes[0] != 0
    with pytest.raises(ProcessLookupError):
        os.kill(process_pid[0], 0)
