"""Unit tests for src/audio/audio_worker.py -- the final stage of the
audio-warning pipeline, speaking queued AudioEvents via macOS `say`
(optionally preceded by a sequenced local alert cue via `afplay`). No
real audio is ever produced: an autouse fixture patches subprocess.run/
Popen for every test in this file, mirroring tests/
test_startup_announcer.py's established pattern.
"""

from unittest.mock import MagicMock

import pytest

from src.audio.audio_worker import AudioWorker
from src.audio.event_policy import EventPolicy
from src.audio.speech_queue import SpeechQueue
from src.models import AudioEvent, DeliveryProfile


def _voices_stdout(names) -> str:
    return "\n".join(f"{name}               en_US    # Hello, I'm {name}." for name in names)


@pytest.fixture(autouse=True)
def mock_subprocess(monkeypatch):
    """Patch subprocess.run/Popen inside both audio_worker.py and the
    startup_announcer.py helpers it imports (list_installed_voices/
    resolve_voice) -- never a real `say`/`afplay` call from any test
    here. subprocess.Popen is shared by both modules (same underlying
    `subprocess` module object) -- patched exactly once, with a
    side_effect returning a fresh, independently controllable instance
    per call, so a cue-then-speech test can distinguish the two."""
    run_result = MagicMock()
    run_result.stdout = _voices_stdout(("Samantha", "Karen"))
    mock_run = MagicMock(return_value=run_result)

    instances: list = []

    def make_popen(*args, **kwargs):
        instance = MagicMock()
        instance.poll.return_value = None  # "still running" by default
        instances.append(instance)
        return instance

    mock_popen = MagicMock(side_effect=make_popen)

    monkeypatch.setattr("src.audio.startup_announcer.subprocess.run", mock_run)
    monkeypatch.setattr("src.audio.audio_worker.subprocess.Popen", mock_popen)

    return {"run": mock_run, "popen": mock_popen, "instances": instances}


def make_event(
    key: str = "LEVEL_1_MOVING_FAR:1:left",
    priority: str = "INFORMATIONAL",
    message: str = "Vehicle moving from the left.",
    track_id: int | None = 1,
    created_at: float = 0.0,
    delivery_profile: str = "level_1",
) -> AudioEvent:
    return AudioEvent(
        key=key, event_type="LEVEL_1_MOVING_FAR", priority=priority,
        message=message, track_id=track_id, created_at=created_at,
        delivery_profile=delivery_profile,
    )


def make_profile(
    rate_wpm: float = 170,
    cue_enabled: bool = False,
    cue_path: str | None = None,
    cue_gain: float = 0.0,
) -> DeliveryProfile:
    return DeliveryProfile(
        rate_wpm=rate_wpm, cue_enabled=cue_enabled, cue_path=cue_path, cue_gain=cue_gain
    )


def make_worker(
    enabled: bool = True,
    rate_wpm: float | None = 170,
    delivery_profiles: dict[str, DeliveryProfile] | None = None,
) -> AudioWorker:
    queue = SpeechQueue(max_size=5, max_age_seconds=3.0)
    policy = EventPolicy(
        cooldown_same_event_seconds=4.0,
        cooldown_informational_seconds=8.0,
    )
    if delivery_profiles is None:
        delivery_profiles = {"level_1": make_profile(rate_wpm=170)}
    return AudioWorker(
        enabled=enabled,
        queue=queue,
        event_policy=policy,
        configured_voice=None,
        fallback_voice=None,
        rate_wpm=rate_wpm,
        delivery_profiles=delivery_profiles,
    )


# --- end-to-end enqueue -> speak flow ---------------------------------


def test_enqueued_event_is_spoken_on_tick(mock_subprocess) -> None:
    worker = make_worker()
    worker.enqueue_events([make_event()], now=0.0)

    worker.tick(now=0.0)

    mock_subprocess["popen"].assert_called_once()
    command = mock_subprocess["popen"].call_args[0][0]
    assert "Vehicle moving from the left." in command


def test_tick_with_empty_queue_does_not_speak(mock_subprocess) -> None:
    worker = make_worker()
    worker.tick(now=0.0)
    mock_subprocess["popen"].assert_not_called()


# --- sequential, non-overlapping playback (no audio flood) -----------------


def test_does_not_start_new_speech_while_one_is_in_flight(mock_subprocess) -> None:
    worker = make_worker()
    worker.enqueue_events([make_event(key="a", track_id=1)], now=0.0)
    worker.enqueue_events(
        [make_event(key="b", track_id=2, message="Person moving ahead.")], now=0.0
    )

    worker.tick(now=0.0)  # starts speaking "a"
    worker.tick(now=0.1)  # must NOT start "b" yet -- instances[0] still "running"

    assert mock_subprocess["popen"].call_count == 1


def test_starts_next_event_once_previous_finishes(mock_subprocess) -> None:
    worker = make_worker()
    worker.enqueue_events([make_event(key="a", track_id=1)], now=0.0)
    worker.enqueue_events(
        [make_event(key="b", track_id=2, message="Person moving ahead.")], now=0.0
    )

    worker.tick(now=0.0)  # starts "a"
    mock_subprocess["instances"][0].poll.return_value = 0  # "a" finished
    worker.tick(now=0.5)  # should now start "b"

    assert mock_subprocess["popen"].call_count == 2


def test_higher_priority_queued_event_speaks_first(mock_subprocess) -> None:
    worker = make_worker()
    informational = make_event(key="LEVEL_1_MOVING_FAR:1:left", priority="INFORMATIONAL")
    warning = make_event(
        key="LEVEL_3_APPROACHING:2:right", priority="WARNING", track_id=2,
        message="Possible vehicle approach from the right. Please wait.",
        delivery_profile="level_1",
    )
    worker.enqueue_events([informational, warning], now=0.0)

    worker.tick(now=0.0)

    command = mock_subprocess["popen"].call_args[0][0]
    assert "Possible vehicle approach from the right. Please wait." in command


# --- cooldown integration (via EventPolicy) --------------------------------


def test_marks_spoken_at_actual_speech_time_enforcing_cooldown(mock_subprocess) -> None:
    worker = make_worker()
    event = make_event(key="LEVEL_1_MOVING_FAR:1:left")

    worker.enqueue_events([event], now=0.0)
    worker.tick(now=0.0)  # spoken now
    mock_subprocess["instances"][0].poll.return_value = 0  # finished

    # Same event re-built one frame later -- must be suppressed by
    # cooldown (informational = 8s), not re-queued/re-spoken.
    worker.enqueue_events([event], now=0.05)
    worker.tick(now=0.1)

    assert mock_subprocess["popen"].call_count == 1


# --- escalation: fresh event for a track supersedes a stale pending one ---


def test_enqueue_evicts_stale_pending_entry_for_the_same_track(mock_subprocess) -> None:
    worker = make_worker(
        delivery_profiles={"level_1": make_profile(), "level_5": make_profile(rate_wpm=200)}
    )
    low = make_event(key="LEVEL_1_MOVING_FAR:1:left", track_id=1, message="Vehicle moving from the left.")
    high = make_event(
        key="LEVEL_5_HIGH_DANGER:1:left", track_id=1, priority="WARNING",
        message="Warning! Vehicle approaching from the left. Please wait.",
        delivery_profile="level_5",
    )
    worker.enqueue_events([low], now=0.0)
    worker.enqueue_events([high], now=0.1)  # escalation -- different key

    worker.tick(now=0.2)

    command = mock_subprocess["popen"].call_args[0][0]
    assert "Warning!" in " ".join(command)
    # The stale Level-1 entry must never be spoken afterward.
    mock_subprocess["instances"][0].poll.return_value = 0
    worker.tick(now=0.3)
    assert mock_subprocess["popen"].call_count == 1


# --- disabled worker -----------------------------------------------------


def test_disabled_worker_never_speaks(mock_subprocess) -> None:
    worker = make_worker(enabled=False)
    worker.enqueue_events([make_event()], now=0.0)
    worker.tick(now=0.0)

    mock_subprocess["popen"].assert_not_called()


def test_disabled_worker_does_not_query_installed_voices(mock_subprocess) -> None:
    make_worker(enabled=False)
    mock_subprocess["run"].assert_not_called()


# --- shutdown --------------------------------------------------------------


def test_shutdown_terminates_in_flight_process(mock_subprocess) -> None:
    worker = make_worker()
    worker.enqueue_events([make_event()], now=0.0)
    worker.tick(now=0.0)

    worker.shutdown()

    mock_subprocess["instances"][0].terminate.assert_called_once()
    mock_subprocess["instances"][0].wait.assert_called_once()


def test_shutdown_before_any_speech_does_not_crash(mock_subprocess) -> None:
    worker = make_worker()
    worker.shutdown()  # never spoke -- must be a safe no-op


def test_shutdown_is_a_no_op_when_process_already_finished(mock_subprocess) -> None:
    worker = make_worker()
    worker.enqueue_events([make_event()], now=0.0)
    worker.tick(now=0.0)
    mock_subprocess["instances"][0].poll.return_value = 0  # already exited

    worker.shutdown()

    mock_subprocess["instances"][0].terminate.assert_not_called()


# --- rate/voice wiring (per hazard-level delivery profile) -----------------


def test_rate_wpm_resolved_from_events_delivery_profile(mock_subprocess) -> None:
    worker = make_worker(delivery_profiles={"level_3": make_profile(rate_wpm=188)})
    event = make_event(delivery_profile="level_3")
    worker.enqueue_events([event], now=0.0)

    worker.tick(now=0.0)

    command = mock_subprocess["popen"].call_args[0][0]
    assert "-r" in command
    assert command[command.index("-r") + 1] == "188"


def test_unrecognized_delivery_profile_falls_back_to_general_rate(mock_subprocess) -> None:
    worker = make_worker(rate_wpm=180, delivery_profiles={})
    event = make_event(delivery_profile="level_99")
    worker.enqueue_events([event], now=0.0)

    worker.tick(now=0.0)

    command = mock_subprocess["popen"].call_args[0][0]
    assert command[command.index("-r") + 1] == "180"


def test_non_positive_rate_wpm_raises() -> None:
    with pytest.raises(ValueError):
        make_worker(rate_wpm=0)


# --- alert cue sequencing ---------------------------------------------


def test_cue_plays_before_speech_and_never_overlaps(mock_subprocess, tmp_path) -> None:
    cue_file = tmp_path / "cue.wav"
    cue_file.write_bytes(b"fake")
    worker = make_worker(
        delivery_profiles={
            "level_5": make_profile(rate_wpm=200, cue_enabled=True, cue_path=str(cue_file), cue_gain=1.0)
        }
    )
    event = make_event(delivery_profile="level_5", message="Warning! Vehicle approaching from the left. Please wait.")
    worker.enqueue_events([event], now=0.0)

    worker.tick(now=0.0)  # starts the cue
    assert mock_subprocess["popen"].call_count == 1
    cue_command = mock_subprocess["popen"].call_args[0][0]
    assert cue_command[0] == "afplay"
    assert str(cue_file) in cue_command
    assert "1.0" in cue_command

    worker.tick(now=0.05)  # cue still "playing" -- must not start speech yet
    assert mock_subprocess["popen"].call_count == 1

    mock_subprocess["instances"][0].poll.return_value = 0  # cue finished
    worker.tick(now=0.1)  # now speech starts

    assert mock_subprocess["popen"].call_count == 2
    speech_command = mock_subprocess["popen"].call_args[0][0]
    assert speech_command[0] == "say"
    assert "Warning! Vehicle approaching from the left. Please wait." in speech_command


def test_missing_cue_file_does_not_crash_and_speech_plays_normally(mock_subprocess) -> None:
    worker = make_worker(
        delivery_profiles={
            "level_5": make_profile(rate_wpm=200, cue_enabled=True, cue_path="/no/such/file.wav", cue_gain=1.0)
        }
    )
    event = make_event(delivery_profile="level_5", message="Warning! Vehicle approaching from the left. Please wait.")
    worker.enqueue_events([event], now=0.0)

    worker.tick(now=0.0)

    assert mock_subprocess["popen"].call_count == 1  # cue skipped -- speech fired directly
    command = mock_subprocess["popen"].call_args[0][0]
    assert command[0] == "say"


def test_cue_disabled_speaks_immediately(mock_subprocess) -> None:
    worker = make_worker(delivery_profiles={"level_1": make_profile(cue_enabled=False)})
    worker.enqueue_events([make_event()], now=0.0)

    worker.tick(now=0.0)

    assert mock_subprocess["popen"].call_count == 1
    assert mock_subprocess["popen"].call_args[0][0][0] == "say"


# --- safety invariants: no shell=True, never touches global volume --------


def test_never_uses_shell_true(mock_subprocess) -> None:
    worker = make_worker()
    worker.enqueue_events([make_event()], now=0.0)
    worker.tick(now=0.0)

    for call in mock_subprocess["popen"].call_args_list:
        assert "shell" not in call.kwargs or call.kwargs["shell"] is False


def test_never_invokes_osascript_or_system_volume(mock_subprocess, tmp_path) -> None:
    cue_file = tmp_path / "cue.wav"
    cue_file.write_bytes(b"fake")
    worker = make_worker(
        delivery_profiles={
            "level_5": make_profile(cue_enabled=True, cue_path=str(cue_file), cue_gain=1.0)
        }
    )
    worker.enqueue_events([make_event(delivery_profile="level_5")], now=0.0)
    worker.tick(now=0.0)
    mock_subprocess["instances"][0].poll.return_value = 0
    worker.tick(now=0.1)

    for call in mock_subprocess["popen"].call_args_list:
        command = call[0][0]
        assert command[0] != "osascript"
        assert "volume" not in " ".join(command).lower()


# --- no crash on missing/empty data ----------------------------------------


def test_enqueue_empty_events_list_does_not_crash(mock_subprocess) -> None:
    worker = make_worker()
    worker.enqueue_events([], now=0.0)
    worker.tick(now=0.0)
    mock_subprocess["popen"].assert_not_called()
