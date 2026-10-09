"""Explicit startup-audio state machine gating the warning pipeline until
the one-time startup announcement (src/audio/startup_announcer.py) has
completely finished speaking -- so the two audio sources can never
overlap or mix.

    INITIALIZING -> STARTUP_SPEAKING -> READY

INITIALIZING: startup announcement not yet triggered (still waiting on
    camera/frame readiness or its configured delay). Warning audio
    blocked.
STARTUP_SPEAKING: startup announcement triggered and its `say` process
    is still running. Warning audio blocked -- startup speech has
    exclusive ownership of the audio system.
READY: startup announcement has fully finished speaking, was skipped
    (e.g. `say` unavailable), or was never enabled in the first place
    (nothing to sequence around). Warning audio enabled.

Perception (detection/tracking/motion estimation/filtering/trajectory/
corridor analysis) is NEVER gated by this -- only warning-event
generation and speech are. See main.py's process_audio_events(): the
`core` motion result it receives is always computed unconditionally by
the caller, before this gate is even consulted.

Read-only with respect to StartupAnnouncer -- never calls any of its
speaking/timing methods, only its status properties, and never blocks
(status is polled once per frame from the main loop).
"""

from __future__ import annotations

from src.audio.startup_announcer import StartupAnnouncer

INITIALIZING = "INITIALIZING"
STARTUP_SPEAKING = "STARTUP_SPEAKING"
READY = "READY"


class StartupAudioGate:
    """Derives INITIALIZING/STARTUP_SPEAKING/READY from a
    StartupAnnouncer's read-only status. Stateless beyond the
    StartupAnnouncer reference itself -- state is recomputed fresh on
    every access, never cached.
    """

    def __init__(self, startup_announcer: StartupAnnouncer) -> None:
        self._startup_announcer = startup_announcer
        self._ready_seen = False

    @property
    def state(self) -> str:
        if not self._startup_announcer.is_enabled:
            return READY
        if not self._startup_announcer.has_announced:
            return INITIALIZING
        if self._startup_announcer.is_speaking:
            return STARTUP_SPEAKING
        return READY

    @property
    def warnings_blocked(self) -> bool:
        """True whenever state is anything other than READY -- the
        single question main.py's audio pipeline needs answered."""
        return self.state != READY

    def is_first_ready_cycle(self) -> bool:
        """True exactly once -- the first call where state == READY.
        False before that (still blocked) and False on every call after
        (already consumed) -- a one-shot flag, mirroring
        StartupAnnouncer's own _announced convention. Lets the caller
        limit the very first post-READY cycle to a single admitted
        event, so READY doesn't itself sound like a burst of speech."""
        if self.state != READY:
            return False
        if self._ready_seen:
            return False
        self._ready_seen = True
        return True
