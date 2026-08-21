from __future__ import annotations

import signal

import pytest

from gpu_shortmd.runtime import processes


def test_process_token_mismatch_never_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signaled: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(processes, "process_start_token", lambda _: "actual")
    monkeypatch.setattr(
        processes.os,
        "killpg",
        lambda pid, sig: signaled.append((pid, sig)),
    )
    result = processes.terminate_owned_process_group(
        pid=100,
        expected_start_token="stale",
        grace_seconds=0,
    )
    assert result.terminated is False
    assert "mismatch" in result.reason
    assert signaled == []


def test_non_session_leader_is_never_signaled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(processes, "process_start_token", lambda _: "token")
    monkeypatch.setattr(processes.os, "getpgid", lambda _: 99)
    result = processes.terminate_owned_process_group(
        pid=100,
        expected_start_token="token",
        grace_seconds=0,
    )
    assert result.terminated is False
    assert "not its process-group leader" in result.reason
