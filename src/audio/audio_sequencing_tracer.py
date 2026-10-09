"""Runtime observability for startup/warning audio sequencing.

AudioSequencingTracer records what src/audio/startup_audio_gate.py and
main.py's process_audio_events() already decided, one AudioSequencingTrace
(see src/models.py) per frame -- it makes no admission/speaking decisions
itself. This exists so the "startup and warning speech never overlap"
guarantee can be verified from actual runtime state and timestamps
(logged, inspectable after a real run) rather than only judged by
listening to a recording.

Pure logic, no OpenCV/network calls -- fully unit testable with a real
StartupAnnouncer/StartupAudioGate/SpeechQueue/AudioWorker driven through
mocked subprocess (never real `say`).
"""

from __future__ import annotations

import logging

from src.audio.audio_worker import AudioWorker
from src.audio.speech_queue import SpeechQueue
from src.audio.startup_announcer import StartupAnnouncer
from src.audio.startup_audio_gate import StartupAudioGate
from src.models import AudioSequencingTrace

logger = logging.getLogger("atlas")

SUPPRESSED_STARTUP_EXCLUSIVE = "SUPPRESSED_STARTUP_EXCLUSIVE"
ADMITTED = "ADMITTED"


class AudioSequencingTracer:
    """Stateful recorder of startup/warning audio-sequencing decisions.

    Holds only the small amount of state needed to answer "when did this
    first become true" across frames (start/finish/first-candidate/
    first-speech timestamps, and the sticky pre-READY-candidate safety
    check) -- everything else in AudioSequencingTrace is recomputed
    fresh from the live components' own read-only status every call.
    """

    def __init__(self) -> None:
        self._startup_speech_start_time: float | None = None
        self._startup_speech_finish_time: float | None = None
        self._first_warning_candidate_time: float | None = None
        self._first_warning_speech_start_time: float | None = None
        self._candidate_created_during_startup = False
        self._previous_state: str | None = None

    def record_frame(
        self,
        startup_announcer: StartupAnnouncer,
        startup_audio_gate: StartupAudioGate,
        speech_queue: SpeechQueue,
        audio_worker: AudioWorker,
        now: float,
        candidates_built: bool,
    ) -> AudioSequencingTrace:
        """Call once per frame, after process_audio_events() has already
        made this frame's admission decision -- records what happened,
        never influences it.

        Args:
            candidates_built: Whether AudioEventBuilder.build_events()
                was actually invoked this frame (the caller knows this
                directly; recomputing it here would require duplicating
                the gate check this tracer is supposed to only observe).
        """
        state = startup_audio_gate.state
        blocked = startup_audio_gate.warnings_blocked

        if startup_announcer.is_speaking and self._startup_speech_start_time is None:
            self._startup_speech_start_time = now
        if (
            self._startup_speech_start_time is not None
            and self._startup_speech_finish_time is None
            and startup_announcer.has_announced
            and not startup_announcer.is_speaking
        ):
            self._startup_speech_finish_time = now

        if blocked and candidates_built:
            self._candidate_created_during_startup = True
        if not blocked and candidates_built and self._first_warning_candidate_time is None:
            self._first_warning_candidate_time = now
        if (
            not blocked
            and audio_worker.is_speaking
            and self._first_warning_speech_start_time is None
        ):
            self._first_warning_speech_start_time = now

        if state != self._previous_state:
            logger.info(
                "Audio sequencing: %s -> %s (pid=%s)",
                self._previous_state, state, startup_announcer.process_pid,
            )
            self._previous_state = state

        active_speech_process_count = int(startup_announcer.is_speaking) + int(
            audio_worker.is_speaking
        )

        return AudioSequencingTrace(
            startup_state=state,
            startup_process_pid=startup_announcer.process_pid,
            startup_speech_start_time=self._startup_speech_start_time,
            startup_speech_finish_time=self._startup_speech_finish_time,
            warning_pipeline_enabled=not blocked,
            candidate_created_during_startup=self._candidate_created_during_startup,
            startup_gate_decision=SUPPRESSED_STARTUP_EXCLUSIVE if blocked else ADMITTED,
            startup_gate_reason=f"startup_state={state}",
            queue_size_during_startup=len(speech_queue) if blocked else 0,
            first_warning_candidate_time=self._first_warning_candidate_time,
            first_warning_speech_start_time=self._first_warning_speech_start_time,
            active_speech_process_count=active_speech_process_count,
        )
