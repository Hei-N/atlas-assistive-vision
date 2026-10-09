"""Video input handling for Atlas Phase 1.

Wraps OpenCV's VideoCapture to provide a small, testable interface that
distinguishes webcam indices from video file paths and surfaces clear
errors instead of failing silently.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("atlas")


class VideoSourceError(Exception):
    """Raised when a video source cannot be opened or read from."""


class VideoSource:
    """Manages a camera or video file input stream.

    The source string is interpreted as follows:
        * A string containing only digits (e.g. "0", "1") is treated as a
          webcam device index.
        * Any other string is treated as a path to a video file, which
          must exist on disk.
    """

    def __init__(self, source: str) -> None:
        """Initialize and open the video source.

        Args:
            source: Either a numeric webcam index (as a string) or a path
                to a video file.

        Raises:
            VideoSourceError: If the source is a non-existent file path,
                or if the underlying capture device/file cannot be opened.
        """
        self._raw_source = source
        self._capture_target = self._resolve_source(source)

        if isinstance(self._capture_target, int) and sys.platform == "darwin":
            # On macOS, OpenCV's auto-selected backend can probe multiple
            # camera APIs and hang (rather than fail fast) when camera
            # permission hasn't been granted to the calling application.
            # Requesting AVFoundation directly opens faster and surfaces a
            # clear isOpened()-is-False failure instead of hanging.
            #
            # Even with AVFoundation requested explicitly, the *first*
            # camera open in a fresh terminal session can still take several
            # seconds while macOS's camera daemon spins up -- logging this
            # upfront so it doesn't look like a hang. Later opens in the
            # same session are fast.
            logger.info(
                "Opening camera %s (the first camera open in a new "
                "terminal session can take several seconds -- please "
                "wait rather than interrupting).",
                self._capture_target,
            )
            self._capture = cv2.VideoCapture(
                self._capture_target, cv2.CAP_AVFOUNDATION
            )
        else:
            self._capture = cv2.VideoCapture(self._capture_target)

        if not self._capture.isOpened():
            self._capture.release()
            raise VideoSourceError(
                f"Unable to open video source: {source!r}. "
                "Check that the webcam is connected and not in use by "
                "another application, or that the video file path is correct."
            )

    @staticmethod
    def _resolve_source(source: str) -> int | str:
        """Resolve a raw source string into a webcam index or file path.

        Args:
            source: Raw source string, e.g. "0" or "data/input/video.mp4".

        Returns:
            An int webcam index if the source is purely numeric, otherwise
            the original string path.

        Raises:
            VideoSourceError: If the source looks like a file path but the
                file does not exist.
        """
        if source.isdigit():
            return int(source)

        path = Path(source)
        if not path.exists():
            raise VideoSourceError(f"Video file does not exist: {source!r}")
        if not path.is_file():
            raise VideoSourceError(f"Video source is not a file: {source!r}")
        return str(path)

    def is_opened(self) -> bool:
        """Return True if the underlying capture device/file is open."""
        return self._capture.isOpened()

    def is_live_source(self) -> bool:
        """Return True if this source is a live camera device (a webcam
        index) rather than a video file. Used to distinguish a genuine
        live-feed failure (worth confirming/announcing as CAMERA_FEED_
        LOST) from a video file simply reaching its normal end -- the
        latter is expected, not a failure, and is never announced as
        one (see src/system/system_health_monitor.py)."""
        return isinstance(self._capture_target, int)

    def read_frame(self) -> np.ndarray:
        """Read a single frame from the video source.

        Returns:
            The frame as a numpy array (BGR, as returned by OpenCV).

        Raises:
            VideoSourceError: If a frame could not be read (e.g. end of
                video file, or webcam disconnected).
        """
        success, frame = self._capture.read()
        if not success or frame is None:
            raise VideoSourceError(
                f"Failed to read frame from source: {self._raw_source!r}"
            )
        return frame

    def get_frame_size(self) -> tuple[int, int]:
        """Return the (width, height) of frames from this source, in pixels."""
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def get_fps(self) -> float:
        """Return the source's reported frames-per-second.

        Returns 0.0 (rather than raising) if the source doesn't report a
        usable value -- common for some webcams/synthetic sources -- so
        callers can detect this and fall back to a safe default.
        """
        return float(self._capture.get(cv2.CAP_PROP_FPS))

    def release(self) -> None:
        """Release the underlying capture device/file."""
        self._capture.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
