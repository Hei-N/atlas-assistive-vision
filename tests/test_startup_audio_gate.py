"""Unit tests for src/audio/startup_audio_gate.py -- the INITIALIZING /
STARTUP_SPEAKING / READY state machine that gives the cinematic startup
announcement exclusive ownership of the audio system. No real audio is
ever produced: the same autouse subprocess-mocking fixture pattern as
tests/test_startup_announcer.py is used, driving a real StartupAnnouncer
through its actual lifecycle so gate state is derived from genuine
announcer behavior, not a hand-rolled fake.
"""

from unittest.mock import MagicMock

import pytest

from src.audio.startup_announcer import StartupAnnouncer
from src.audio.startup_audio_gate import (
    INITIALIZING,
    READY,
    STARTUP_SPEAKING,
    StartupAudioGate,
)


def _voices_stdout(names) -> str:
    return "\n".join(f"{name}               en_US    # Hello, I'm {name}." for name in names)


@pytest.fixture(autouse=True)
def mock_subprocess(monkeypatch):
    run_result = MagicMock()
    run_result.stdout = _voices_stdout(("Samantha", "Karen"))
    mock_run = MagicMock(return_value=run_result)

    popen_instance = MagicMock()
    popen_instance.poll.return_value = None  # "still running" by default
    mock_popen = MagicMock(return_value=popen_instance)

    monkeypatch.setattr("src.audio.startup_announcer.subprocess.run", mock_run)
    monkeypatch.setattr("src.audio.startup_announcer.subprocess.Popen", mock_popen)

    return {"run": mock_run, "popen": mock_popen, "popen_instance": popen_instance}


def make_announcer(enabled: bool = True, delay_seconds: float = 0.0) -> StartupAnnouncer:
    return StartupAnnouncer(
        enabled=enabled,
        message="Atlas online. Detection system activated.",
        play_once=True,
        delay_seconds=delay_seconds,
        require_first_valid_frame=True,
        configured_voice=None,
        fallback_voice=None,
        rate_wpm=165,
        sound_enabled=False,
        sound_path=None,
        sound_volume=0.7,
    )


def advance_ready(announcer: StartupAnnouncer, monkeypatch) -> None:
    announcer.mark_camera_ready()
    announcer.notify_frame_read()
    future = announcer._ready_since + announcer._delay_seconds + 0.01
    monkeypatch.setattr("src.audio.startup_announcer.time.time", lambda: future)


# --- state: INITIALIZING --------------------------------------------------


def test_initializing_before_camera_ready() -> None:
    announcer = make_announcer()
    gate = StartupAudioGate(announcer)

    assert gate.state == INITIALIZING
    assert gate.warnings_blocked is True


def test_initializing_after_ready_but_before_delay_elapses(monkeypatch) -> None:
    announcer = make_announcer(delay_seconds=5.0)
    announcer.mark_camera_ready()
    announcer.notify_frame_read()
    gate = StartupAudioGate(announcer)

    announcer.tick()  # ready_since set, but delay hasn't elapsed yet

    assert gate.state == INITIALIZING
    assert gate.warnings_blocked is True


# --- state: STARTUP_SPEAKING -----------------------------------------------


def test_startup_speaking_while_process_running(monkeypatch, mock_subprocess) -> None:
    announcer = make_announcer()
    gate = StartupAudioGate(announcer)
    advance_ready(announcer, monkeypatch)

    announcer.tick()  # triggers _speak() -- popen_instance.poll() returns None

    assert gate.state == STARTUP_SPEAKING
    assert gate.warnings_blocked is True


# --- state: READY ------------------------------------------------------


def test_ready_after_process_finishes(monkeypatch, mock_subprocess) -> None:
    announcer = make_announcer()
    gate = StartupAudioGate(announcer)
    advance_ready(announcer, monkeypatch)
    announcer.tick()
    assert gate.state == STARTUP_SPEAKING

    mock_subprocess["popen_instance"].poll.return_value = 0  # process finished

    assert gate.state == READY
    assert gate.warnings_blocked is False


def test_ready_immediately_when_disabled() -> None:
    announcer = make_announcer(enabled=False)
    gate = StartupAudioGate(announcer)

    assert gate.state == READY
    assert gate.warnings_blocked is False


def test_ready_when_say_unavailable_and_announced_without_process(
    monkeypatch, mock_subprocess
) -> None:
    # _check_say_available() fails -- tick() marks announced without
    # ever creating a process. Must not get stuck blocked forever.
    monkeypatch.setattr(
        "src.audio.startup_announcer.StartupAnnouncer._check_say_available",
        lambda self: False,
    )
    announcer = make_announcer()
    gate = StartupAudioGate(announcer)
    advance_ready(announcer, monkeypatch)

    announcer.tick()

    assert announcer.has_announced is True
    assert gate.state == READY
    assert gate.warnings_blocked is False
    mock_subprocess["popen"].assert_not_called()


# --- transition correctness -------------------------------------------


def test_full_transition_sequence(monkeypatch, mock_subprocess) -> None:
    announcer = make_announcer()
    gate = StartupAudioGate(announcer)

    assert gate.state == INITIALIZING

    advance_ready(announcer, monkeypatch)
    announcer.tick()
    assert gate.state == STARTUP_SPEAKING

    mock_subprocess["popen_instance"].poll.return_value = 0
    assert gate.state == READY


# --- read-only: never triggers speech itself -------------------------


def test_querying_gate_state_never_triggers_speech(mock_subprocess) -> None:
    announcer = make_announcer()
    gate = StartupAudioGate(announcer)

    for _ in range(10):
        _ = gate.state
        _ = gate.warnings_blocked

    mock_subprocess["popen"].assert_not_called()
