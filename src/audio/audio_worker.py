"""Speaks queued AudioEvents via macOS `say` -- the final stage of
Atlas's audio-warning pipeline: AudioEventBuilder -> EventPolicy ->
SpeechQueue -> AudioWorker -> macOS `say`.

Mirrors src/audio/startup_announcer.py's proven non-blocking pattern
(subprocess.Popen, never shell=True; terminate() -> wait(timeout) ->
kill() on shutdown) rather than inventing a new one, and reuses its
already-standalone, side-effect-free list_installed_voices()/
resolve_voice() helpers directly. startup_announcer.py itself is not
imported for its StartupAnnouncer class and is not modified.

Per-utterance volume: NOT supported and not implemented. macOS `say` has
no official, stable per-utterance volume or pitch flag (only `-v` for
voice and `-r` for rate) -- an undocumented, voice-dependent embedded
text command exists but is inconsistent across modern voices and isn't
built on here. Urgency is carried entirely by wording (hazard-level
message prefixes, decided upstream by AudioHazardResolver), speaking
rate (this module's per-level `DeliveryProfile.rate_wpm`), and an
optional short local alert cue played via `afplay -v <gain>` (the one
per-utterance volume control this codebase actually has proof of --
already used by StartupAnnouncer for its own activation sound) --
never a global system-volume change, never AppleScript, never
`shell=True`.

Delivery profile selection: this module never decides hazard urgency
itself (no perception logic here) -- it only looks up the pre-decided
`AudioEvent.delivery_profile` string in a config-provided
`dict[str, DeliveryProfile]`, falling back to this worker's general
`rate_wpm` for an unrecognized/missing profile.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from src.audio.event_policy import EventPolicy
from src.audio.speech_queue import SpeechQueue
from src.audio.startup_announcer import list_installed_voices, resolve_voice
from src.models import AudioEvent, DeliveryProfile

logger = logging.getLogger("atlas")

_PROCESS_TERMINATE_TIMEOUT_SECONDS = 1.0

_IDLE = "IDLE"
_CUE_PLAYING = "CUE_PLAYING"
_SPEAKING = "SPEAKING"


class AudioWorker:
    """Pops the highest-priority non-stale event from `queue` and speaks
    it, one utterance at a time, non-blocking -- optionally preceded by
    a short local alert cue (sequenced, never overlapping the speech).

    Args:
        enabled: Master on/off switch (config audio.warnings.enabled AND
            not --no-audio). When False, enqueue_events()/tick() are
            no-ops -- events are never even filtered/queued, matching
            StartupAnnouncer's own enabled-flag convention.
        queue: The bounded, deduplicated, priority-ordered SpeechQueue
            events are enqueued into.
        event_policy: The EventPolicy cooldown gate -- events are
            filtered through it in enqueue_events(), and marked spoken
            in it (at actual speech time, not admission time) here.
        configured_voice: Resolved warnings-specific voice name (after
            CLI/config precedence has already been applied by the
            caller), or None. One voice is used across every hazard
            level -- only pacing/cue/wording carry urgency.
        fallback_voice: Resolved general voice name, used only if
            configured_voice isn't installed.
        rate_wpm: Words per minute for `say -r`, used as a fallback when
            an event's delivery_profile isn't found in delivery_profiles.
            None means let `say` use its own default rate.
        delivery_profiles: hazard-level-name -> DeliveryProfile
            (rate_wpm/cue), looked up by each event's own
            `delivery_profile` field at speak time.
    """

    def __init__(
        self,
        enabled: bool,
        queue: SpeechQueue,
        event_policy: EventPolicy,
        configured_voice: str | None,
        fallback_voice: str | None,
        rate_wpm: float | None,
        delivery_profiles: dict[str, DeliveryProfile],
    ) -> None:
        if rate_wpm is not None and rate_wpm <= 0:
            raise ValueError(f"rate_wpm must be > 0, got {rate_wpm}")

        self._enabled = enabled
        self._queue = queue
        self._event_policy = event_policy
        self._rate_wpm = rate_wpm
        self._delivery_profiles = delivery_profiles

        self._resolved_voice = None
        if enabled:
            installed = list_installed_voices()
            self._resolved_voice = resolve_voice(configured_voice, fallback_voice, installed)

        self._process: subprocess.Popen | None = None
        self._phase = _IDLE
        self._pending_event: AudioEvent | None = None
        self._pending_rate: float | None = None

    @property
    def is_speaking(self) -> bool:
        """Whether the audio system is currently busy -- either playing
        an alert cue or speaking -- i.e. whether a subprocess launched by
        this worker is currently running. False before any activity,
        after it finishes, or while disabled. Read-only status, does not
        affect behavior -- mirrors StartupAnnouncer.is_speaking exactly,
        for cross-checking that the two never report True at the same
        time."""
        return self._process is not None and self._process.poll() is None

    def enqueue_events(self, events: list[AudioEvent], now: float) -> None:
        """Run candidate events through the cooldown gate, then queue
        the survivors. Before enqueueing an event with a known track_id,
        evicts any OTHER pending entry for that same track (see
        SpeechQueue.evict_track) -- a fresh assessment of an object
        (escalation OR de-escalation) always fully supersedes whatever
        was previously pending for it, since an escalation's different
        hazard-level key would otherwise just coexist rather than
        replace the stale one. Never blocks; never speaks directly. A
        no-op when disabled -- events are never even filtered/queued."""
        if not self._enabled:
            return
        for event in self._event_policy.filter(events, now):
            if event.track_id is not None:
                self._queue.evict_track(event.track_id)
            self._queue.enqueue(event)

    def tick(self, now: float) -> None:
        """Call once per loop iteration. Non-blocking. If a cue or an
        utterance is already in flight, returns immediately -- exactly
        one subprocess at a time, always sequential, never overlapping.
        A no-op when disabled."""
        if not self._enabled:
            return
        if self._process is not None and self._process.poll() is None:
            return

        if self._phase == _CUE_PLAYING:
            event = self._pending_event
            rate = self._pending_rate
            self._pending_event = None
            self._pending_rate = None
            self._process = None
            self._event_policy.mark_spoken(event.key, now)
            self._speak(event, rate)
            self._phase = _SPEAKING
            return

        self._process = None
        self._phase = _IDLE
        event = self._queue.pop_next(now)
        if event is None:
            return

        profile = self._delivery_profiles.get(event.delivery_profile)
        rate = profile.rate_wpm if profile is not None else self._rate_wpm

        if profile is not None and profile.cue_enabled and profile.cue_path and Path(profile.cue_path).exists():
            self._pending_event = event
            self._pending_rate = rate
            self._play_cue(profile.cue_path, profile.cue_gain)
            self._phase = _CUE_PLAYING
        else:
            self._event_policy.mark_spoken(event.key, now)
            self._speak(event, rate)
            self._phase = _SPEAKING

    def _play_cue(self, cue_path: str, cue_gain: float) -> None:
        try:
            self._process = subprocess.Popen(["afplay", "-v", str(cue_gain), cue_path])
        except OSError as exc:
            logger.error("Alert cue failed to start: %s", exc)
            self._process = None  # tick() will proceed straight to speech next call

    def _speak(self, event: AudioEvent, rate_wpm: float | None) -> None:
        command = ["say"]
        if self._resolved_voice:
            command += ["-v", self._resolved_voice]
        if rate_wpm is not None:
            command += ["-r", str(rate_wpm)]
        command.append(event.message)

        try:
            self._process = subprocess.Popen(command)
        except OSError as exc:
            logger.error("Audio warning failed to start: %s", exc)

    def shutdown(self) -> None:
        """Terminate any in-flight cue or speech process. Safe to call
        even if nothing was ever spoken, or the process already
        finished."""
        if self._process is None:
            return
        if self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=_PROCESS_TERMINATE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self._process.kill()
