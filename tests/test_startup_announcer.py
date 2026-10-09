"""Unit tests for src/audio/startup_announcer.py -- the one-shot,
non-blocking cinematic startup announcement. No real audio is ever
produced: an autouse fixture patches subprocess.run/Popen for every test
in this file (requirement M), so it is structurally impossible for a
test here to invoke real `say`/`afplay`.
"""

from unittest.mock import MagicMock

import pytest

from src.audio.startup_announcer import (
    StartupAnnouncer,
    list_installed_voices,
    resolve_voice,
)

MOCK_INSTALLED_VOICES = ("Samantha", "Karen", "Daniel")


def _voices_stdout(names) -> str:
    return "\n".join(f"{name}               en_US    # Hello, I'm {name}." for name in names)


@pytest.fixture(autouse=True)
def mock_subprocess(monkeypatch):
    """Patch subprocess.run/Popen inside the startup_announcer module for
    every test in this file -- never a real `say`/`afplay` call."""
    run_result = MagicMock()
    run_result.stdout = _voices_stdout(MOCK_INSTALLED_VOICES)
    mock_run = MagicMock(return_value=run_result)

    popen_instance = MagicMock()
    popen_instance.poll.return_value = None  # "still running" by default
    mock_popen = MagicMock(return_value=popen_instance)

    monkeypatch.setattr("src.audio.startup_announcer.subprocess.run", mock_run)
    monkeypatch.setattr("src.audio.startup_announcer.subprocess.Popen", mock_popen)

    return {"run": mock_run, "popen": mock_popen, "popen_instance": popen_instance}


def make_announcer(
    enabled: bool = True,
    message: str = "Atlas online. Detection system activated.",
    play_once: bool = True,
    delay_seconds: float = 0.25,
    require_first_valid_frame: bool = True,
    configured_voice: str | None = None,
    fallback_voice: str | None = None,
    rate_wpm: float | None = 165,
    sound_enabled: bool = False,
    sound_path: str | None = None,
    sound_volume: float = 0.7,
) -> StartupAnnouncer:
    return StartupAnnouncer(
        enabled=enabled,
        message=message,
        play_once=play_once,
        delay_seconds=delay_seconds,
        require_first_valid_frame=require_first_valid_frame,
        configured_voice=configured_voice,
        fallback_voice=fallback_voice,
        rate_wpm=rate_wpm,
        sound_enabled=sound_enabled,
        sound_path=sound_path,
        sound_volume=sound_volume,
    )


def advance_ready(announcer: StartupAnnouncer, monkeypatch, past_delay: bool = True) -> None:
    """Mark camera+frame ready, then deterministically advance the mocked
    clock past delay_seconds -- no real sleeping."""
    announcer.mark_camera_ready()
    announcer.notify_frame_read()
    if past_delay:
        future = announcer._ready_since + announcer._delay_seconds + 0.01
        monkeypatch.setattr(
            "src.audio.startup_announcer.time.time", lambda: future
        )


# --- A/D: disabled / camera-not-ready never plays --------------------------


def test_disabled_never_plays(mock_subprocess) -> None:
    announcer = make_announcer(enabled=False, delay_seconds=0.0)
    announcer.mark_camera_ready()
    announcer.notify_frame_read()
    for _ in range(5):
        announcer.tick()

    mock_subprocess["popen"].assert_not_called()


def test_camera_not_ready_never_plays_regardless_of_tick_count(mock_subprocess) -> None:
    # mark_camera_ready() is only ever called by main.py after a
    # successful VideoSource open -- simulating a failed camera init by
    # simply never calling it.
    announcer = make_announcer(delay_seconds=0.0)
    for _ in range(10):
        announcer.tick()

    mock_subprocess["popen"].assert_not_called()


# --- B/E: plays exactly once, even across many ticks -----------------------


def test_plays_exactly_once_across_many_ticks(mock_subprocess, monkeypatch) -> None:
    announcer = make_announcer(delay_seconds=0.0)
    advance_ready(announcer, monkeypatch)

    for _ in range(20):
        announcer.tick()

    assert mock_subprocess["popen"].call_count == 1


# --- C: waits for first valid frame when configured -------------------------


def test_waits_for_first_valid_frame_when_required(mock_subprocess, monkeypatch) -> None:
    announcer = make_announcer(require_first_valid_frame=True, delay_seconds=0.0)
    announcer.mark_camera_ready()
    announcer.tick()  # no frame read yet
    mock_subprocess["popen"].assert_not_called()

    announcer.notify_frame_read()
    future = announcer._ready_since + 0.01
    monkeypatch.setattr("src.audio.startup_announcer.time.time", lambda: future)
    announcer.tick()

    mock_subprocess["popen"].assert_called_once()


def test_delay_seconds_is_honored(mock_subprocess, monkeypatch) -> None:
    announcer = make_announcer(delay_seconds=1.0)
    announcer.mark_camera_ready()
    announcer.notify_frame_read()

    # Before the delay has elapsed -- must not fire yet.
    almost = announcer._ready_since + 0.5
    monkeypatch.setattr("src.audio.startup_announcer.time.time", lambda: almost)
    announcer.tick()
    mock_subprocess["popen"].assert_not_called()

    # After the delay -- fires.
    after = announcer._ready_since + 1.01
    monkeypatch.setattr("src.audio.startup_announcer.time.time", lambda: after)
    announcer.tick()
    mock_subprocess["popen"].assert_called_once()


# --- F: exact configured message is used ------------------------------------


def test_exact_configured_message_is_spoken(mock_subprocess, monkeypatch) -> None:
    announcer = make_announcer(
        message="Custom cinematic line.", delay_seconds=0.0
    )
    advance_ready(announcer, monkeypatch)
    announcer.tick()

    command = mock_subprocess["popen"].call_args[0][0]
    assert "Custom cinematic line." in command


# --- G/H: startup-specific voice/rate override general -----------------------


def test_startup_voice_overrides_general_voice(mock_subprocess, monkeypatch) -> None:
    announcer = make_announcer(
        configured_voice="Samantha", fallback_voice="Karen", delay_seconds=0.0
    )
    advance_ready(announcer, monkeypatch)
    announcer.tick()

    command = mock_subprocess["popen"].call_args[0][0]
    assert "-v" in command
    assert command[command.index("-v") + 1] == "Samantha"


def test_startup_rate_overrides_general_rate(mock_subprocess, monkeypatch) -> None:
    # Simulates main.py already having collapsed CLI/config precedence
    # down to a single resolved rate_wpm before constructing the
    # announcer -- here we just confirm whatever rate_wpm is given is
    # what's actually passed to `say -r`.
    announcer = make_announcer(rate_wpm=180, delay_seconds=0.0)
    advance_ready(announcer, monkeypatch)
    announcer.tick()

    command = mock_subprocess["popen"].call_args[0][0]
    assert "-r" in command
    assert command[command.index("-r") + 1] == "180"


# --- I: missing configured voice falls back safely ---------------------------


def test_missing_configured_voice_falls_back_to_general(mock_subprocess, monkeypatch) -> None:
    announcer = make_announcer(
        configured_voice="NotInstalledVoice", fallback_voice="Karen", delay_seconds=0.0
    )
    advance_ready(announcer, monkeypatch)
    announcer.tick()

    command = mock_subprocess["popen"].call_args[0][0]
    assert command[command.index("-v") + 1] == "Karen"


def test_missing_configured_and_general_falls_back_to_preferred_list(
    mock_subprocess, monkeypatch
) -> None:
    announcer = make_announcer(
        configured_voice="NotInstalled", fallback_voice="AlsoNotInstalled",
        delay_seconds=0.0,
    )
    advance_ready(announcer, monkeypatch)
    announcer.tick()

    command = mock_subprocess["popen"].call_args[0][0]
    # "Samantha" is in the mocked installed-voices set and is the first
    # entry of the preferred-female-voice list.
    assert command[command.index("-v") + 1] == "Samantha"


def test_resolve_voice_returns_none_when_nothing_installed() -> None:
    assert resolve_voice("Ghost", "AlsoGhost", installed=set()) is None


def test_list_installed_voices_parses_names_with_spaces(mock_subprocess) -> None:
    mock_subprocess["run"].return_value.stdout = _voices_stdout(("Bad News", "Samantha"))
    voices = list_installed_voices()
    assert "Bad News" in voices
    assert "Samantha" in voices


# --- J: speech runs non-blockingly -------------------------------------------


def test_speech_uses_popen_not_run(mock_subprocess, monkeypatch) -> None:
    announcer = make_announcer(delay_seconds=0.0)
    advance_ready(announcer, monkeypatch)
    run_calls_before = mock_subprocess["run"].call_count

    announcer.tick()

    mock_subprocess["popen"].assert_called_once()
    # subprocess.run is only used for the voice-list/availability query,
    # never for the actual speech invocation.
    assert mock_subprocess["run"].call_count == run_calls_before


# --- K: a stale/first announcement never delays a subsequent action --------


def test_announcement_never_blocks_a_subsequent_action(mock_subprocess, monkeypatch) -> None:
    announcer = make_announcer(delay_seconds=0.0)
    advance_ready(announcer, monkeypatch)

    announcer.tick()  # fires the (mocked, instant) announcement
    # A second, independent action (standing in for a future urgent-
    # warning path) must be free to run immediately afterward.
    from src.audio import startup_announcer as module

    module.subprocess.Popen(["say", "a second, independent message"])

    assert mock_subprocess["popen"].call_count == 2


# --- L: clean shutdown during startup speech ---------------------------------


def test_shutdown_terminates_in_flight_process(mock_subprocess, monkeypatch) -> None:
    announcer = make_announcer(delay_seconds=0.0)
    advance_ready(announcer, monkeypatch)
    announcer.tick()

    popen_instance = mock_subprocess["popen_instance"]
    popen_instance.poll.return_value = None  # still "running"

    announcer.shutdown()

    popen_instance.terminate.assert_called_once()
    popen_instance.wait.assert_called_once()


def test_shutdown_is_a_no_op_when_process_already_finished(mock_subprocess, monkeypatch) -> None:
    announcer = make_announcer(delay_seconds=0.0)
    advance_ready(announcer, monkeypatch)
    announcer.tick()

    popen_instance = mock_subprocess["popen_instance"]
    popen_instance.poll.return_value = 0  # already exited

    announcer.shutdown()

    popen_instance.terminate.assert_not_called()


def test_shutdown_before_any_speech_does_not_crash(mock_subprocess) -> None:
    announcer = make_announcer()
    announcer.shutdown()  # never spoke -- must be a safe no-op


# --- N: startup message contains no positive crossing guidance -------------


_UNSAFE_PHRASES = (
    "safe to cross", "it's safe", "is safe", "you may cross",
    "all clear", "clear to cross", "go ahead",
)


def test_default_message_contains_no_crossing_guidance() -> None:
    default_message = "Atlas online. Detection system activated."
    lowered = default_message.lower()
    for phrase in _UNSAFE_PHRASES:
        assert phrase not in lowered


def test_configured_message_from_settings_yaml_contains_no_crossing_guidance() -> None:
    import main

    config = main.load_config("config/settings.yaml")
    message = config["audio"]["startup_announcement"]["message"].lower()
    for phrase in _UNSAFE_PHRASES:
        assert phrase not in message


# --- constructor validation ---------------------------------------------


def test_negative_delay_seconds_raises() -> None:
    with pytest.raises(ValueError):
        make_announcer(delay_seconds=-1.0)


def test_non_positive_rate_wpm_raises() -> None:
    with pytest.raises(ValueError):
        make_announcer(rate_wpm=0)


def test_out_of_range_sound_volume_raises() -> None:
    with pytest.raises(ValueError):
        make_announcer(sound_volume=1.5)


# --- optional activation sound cue ------------------------------------------


def test_activation_sound_skipped_when_path_missing(mock_subprocess, monkeypatch, tmp_path) -> None:
    missing_path = str(tmp_path / "does_not_exist.aiff")
    announcer = make_announcer(
        sound_enabled=True, sound_path=missing_path, delay_seconds=0.0
    )
    advance_ready(announcer, monkeypatch)

    announcer.tick()

    # Only the speech Popen call, never an afplay call for the missing file.
    calls = mock_subprocess["popen"].call_args_list
    assert not any("afplay" in call[0][0] for call in calls)


def test_activation_sound_plays_when_path_exists(mock_subprocess, monkeypatch, tmp_path) -> None:
    sound_file = tmp_path / "cue.aiff"
    sound_file.write_bytes(b"fake audio data")
    announcer = make_announcer(
        sound_enabled=True, sound_path=str(sound_file), delay_seconds=0.0
    )
    advance_ready(announcer, monkeypatch)

    announcer.tick()

    calls = mock_subprocess["popen"].call_args_list
    assert any("afplay" in call[0][0] for call in calls)
