"""Cinematic startup announcement for Atlas -- a one-shot, non-blocking
spoken system-status announcement ("Atlas online. Detection system
activated.") played via macOS's built-in `say` command.

This is a STYLISTIC system-status announcement only. It must never imply
that the road is safe or that perception is infallible -- it only
confirms the software itself has started.

No general multi-message priority/alert-queue framework exists in this
codebase, and none is built here -- this module is deliberately a single-
purpose, self-contained announcer, engineered so a future alert system
could sit alongside it without conflict (never blocks, never repeats,
never gets stuck "queued"), not a preview of that system.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from pathlib import Path

logger = logging.getLogger("atlas")

# Tried in order; each is only ever used if actually installed (queried
# live via `say -v ?`, never assumed) -- satisfies "do not hard-code a
# voice that may not exist on every Mac" while still honoring "female-
# presenting voice when available." "Samantha" (US English) is the
# highest-quality, most universally-installed macOS female voice.
_PREFERRED_FEMALE_VOICES = (
    "Samantha", "Ava", "Allison", "Susan", "Victoria",
    "Karen", "Moira", "Tessa", "Serena",
)

_VOICE_LINE_PATTERN = re.compile(r"^(.+?)\s{2,}[a-zA-Z]{2}_[a-zA-Z0-9]{2,3}\s")
_SAY_QUERY_TIMEOUT_SECONDS = 5.0
_PROCESS_TERMINATE_TIMEOUT_SECONDS = 1.0


def list_installed_voices() -> set[str]:
    """The set of voice names installed for macOS `say`, queried live via
    `say -v ?` (list-form subprocess.run, never shell=True).

    Returns an empty set -- never raises -- if `say` is missing, the
    query fails, or it times out, so callers always have a safe value to
    fall back from.
    """
    try:
        result = subprocess.run(
            ["say", "-v", "?"],
            capture_output=True,
            text=True,
            timeout=_SAY_QUERY_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()

    voices: set[str] = set()
    for line in result.stdout.splitlines():
        match = _VOICE_LINE_PATTERN.match(line)
        if match:
            voices.add(match.group(1))
    return voices


def resolve_voice(
    configured: str | None, fallback: str | None, installed: set[str]
) -> str | None:
    """Pick the voice to pass to `say -v`.

    Order: `configured` -> `fallback` -> first installed preferred-female
    voice -> None (caller omits -v entirely, letting `say` use the
    machine's own default voice). Only ever returns a name confirmed
    present in `installed`.
    """
    for candidate in (configured, fallback):
        if candidate and candidate in installed:
            return candidate
    for candidate in _PREFERRED_FEMALE_VOICES:
        if candidate in installed:
            return candidate
    return None


class StartupAnnouncer:
    """One-shot, non-blocking spoken startup announcement.

    Lifecycle, driven by main.py's run():
        mark_camera_ready()  -- call once, right after the video source
            opens successfully (by construction, every perception
            component already succeeded before this point is reachable).
        notify_frame_read()  -- call every frame, after a frame is
            successfully read.
        tick()  -- call once per loop iteration. A no-op until every
            readiness condition is met (camera ready, audio backend
            available, and -- if require_first_valid_frame -- a frame has
            been read) and delay_seconds has elapsed since then; fires
            the announcement exactly once via a non-blocking
            subprocess.Popen call, then never fires again for the rest
            of this process's life (play_once, and even on failure --
            never retried every frame).
        shutdown()  -- terminate any in-flight speech process. Call from
            run()'s finally block so quitting mid-speech never leaves an
            orphaned `say`/`afplay` process.

    Args:
        enabled: Master on/off switch (config enabled AND not --no-audio).
        message: The exact text to speak.
        play_once: Whether this announcement should ever repeat. Only
            True is currently supported/tested -- kept as an explicit
            parameter (matching this codebase's config-driven-behavior
            convention) rather than hard-coded, for forward compatibility.
        delay_seconds: Seconds to wait after readiness before speaking
            (a small cinematic beat, not a network/IO wait). Must be >= 0.
        require_first_valid_frame: If True, readiness additionally
            requires notify_frame_read() to have been called at least once.
        configured_voice: The resolved startup-specific voice name (after
            CLI/config precedence has already been applied by the
            caller), or None.
        fallback_voice: The resolved general voice name, used only if
            configured_voice isn't installed.
        rate_wpm: Words per minute passed to `say -r`. None means let
            `say` use its own default rate. If provided, must be > 0.
        sound_enabled: Whether to play a local activation sound cue
            immediately before speaking. Default-off.
        sound_path: Path to a user-provided local audio file. Never a
            bundled/downloaded sound. Ignored if sound_enabled is False.
        sound_volume: afplay volume, 0.0-1.0.
    """

    def __init__(
        self,
        enabled: bool,
        message: str,
        play_once: bool,
        delay_seconds: float,
        require_first_valid_frame: bool,
        configured_voice: str | None,
        fallback_voice: str | None,
        rate_wpm: float | None,
        sound_enabled: bool,
        sound_path: str | None,
        sound_volume: float,
    ) -> None:
        if delay_seconds < 0:
            raise ValueError(f"delay_seconds must be >= 0, got {delay_seconds}")
        if rate_wpm is not None and rate_wpm <= 0:
            raise ValueError(f"rate_wpm must be > 0, got {rate_wpm}")
        if not (0.0 <= sound_volume <= 1.0):
            raise ValueError(f"sound_volume must be within [0, 1], got {sound_volume}")

        self._enabled = enabled
        self._message = message
        self._play_once = play_once
        self._delay_seconds = delay_seconds
        self._require_first_valid_frame = require_first_valid_frame
        self._rate_wpm = rate_wpm
        self._sound_enabled = sound_enabled
        self._sound_path = sound_path
        self._sound_volume = sound_volume

        self._say_available = self._check_say_available()
        self._resolved_voice: str | None = None
        if self._enabled and self._say_available:
            installed = list_installed_voices()
            self._resolved_voice = resolve_voice(configured_voice, fallback_voice, installed)
            if configured_voice and self._resolved_voice != configured_voice:
                logger.warning(
                    "Configured startup voice %r is not installed -- "
                    "falling back to %r.",
                    configured_voice,
                    self._resolved_voice or "the system default voice",
                )

        self._camera_ready = False
        self._frame_ready = not require_first_valid_frame
        self._ready_since: float | None = None
        self._announced = False
        self._logged_backend_error = False
        self._process: subprocess.Popen | None = None

    @property
    def is_enabled(self) -> bool:
        """Whether this announcer is enabled at all (config AND not
        --no-audio) -- read-only status, does not affect behavior."""
        return self._enabled

    @property
    def has_announced(self) -> bool:
        """Whether tick() has already decided to speak (or skip because
        `say` is unavailable) -- True forever after, since this is
        play_once. Read-only status, does not affect behavior."""
        return self._announced

    @property
    def is_speaking(self) -> bool:
        """Whether the startup speech subprocess is currently running.
        False before it starts, after it finishes, or if it was skipped
        entirely (e.g. `say` unavailable) -- never blocks, just polls
        the existing process handle. Read-only status, does not affect
        behavior."""
        return self._process is not None and self._process.poll() is None

    @property
    def process_pid(self) -> int | None:
        """PID of the startup speech subprocess, or None before it
        starts (or if it was never created, e.g. `say` unavailable).
        Read-only status, does not affect behavior."""
        return self._process.pid if self._process is not None else None

    @staticmethod
    def _check_say_available() -> bool:
        try:
            subprocess.run(
                ["say", "-v", "?"],
                capture_output=True,
                timeout=_SAY_QUERY_TIMEOUT_SECONDS,
            )
            return True
        except (OSError, subprocess.TimeoutExpired):
            return False

    def mark_camera_ready(self) -> None:
        """Call once, right after the video source opens successfully."""
        self._camera_ready = True
        self._update_ready_since()

    def notify_frame_read(self) -> None:
        """Call every frame, after a frame is successfully read. Only the
        first call (while still pending) has any effect."""
        self._frame_ready = True
        self._update_ready_since()

    def _update_ready_since(self) -> None:
        if self._ready_since is None and self._camera_ready and self._frame_ready:
            self._ready_since = time.time()

    def tick(self) -> None:
        """Call once per loop iteration. Non-blocking; a cheap no-op
        before eligibility and forever after the one-time firing."""
        if self._announced or not self._enabled:
            return

        if not self._say_available:
            if not self._logged_backend_error:
                logger.error(
                    "Audio backend ('say') unavailable -- startup "
                    "announcement skipped."
                )
                self._logged_backend_error = True
            self._announced = True
            return

        if self._ready_since is None:
            return
        if time.time() - self._ready_since < self._delay_seconds:
            return

        self._speak()
        # Marked announced even if _speak() failed internally -- never
        # retried every frame (failure is logged once, inside _speak()).
        self._announced = True

    def _speak(self) -> None:
        if self._sound_enabled:
            self._play_activation_sound()

        command = ["say"]
        if self._resolved_voice:
            command += ["-v", self._resolved_voice]
        if self._rate_wpm is not None:
            command += ["-r", str(self._rate_wpm)]
        command.append(self._message)

        try:
            self._process = subprocess.Popen(command)
        except OSError as exc:
            logger.error("Startup announcement failed to start: %s", exc)

    def _play_activation_sound(self) -> None:
        if not self._sound_path:
            return
        if not Path(self._sound_path).exists():
            logger.warning(
                "Startup activation sound not found at %r -- skipping.",
                self._sound_path,
            )
            return
        try:
            subprocess.Popen(["afplay", "-v", str(self._sound_volume), self._sound_path])
        except OSError as exc:
            logger.warning("Could not play startup activation sound: %s", exc)

    def shutdown(self) -> None:
        """Terminate any in-flight speech process. Safe to call even if
        nothing was ever spoken, or the process already finished."""
        if self._process is None:
            return
        if self._process.poll() is not None:
            return  # already finished
        self._process.terminate()
        try:
            self._process.wait(timeout=_PROCESS_TERMINATE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self._process.kill()
