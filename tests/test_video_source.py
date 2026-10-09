"""Unit tests for VideoSource.

cv2.VideoCapture is mocked throughout so these tests require no physical
webcam, no real video file playback, and no network access. A real
(temporary, empty) file is used only to test path-existence logic.
"""

import sys
from unittest.mock import MagicMock, patch

import cv2
import pytest

from src.video_source import VideoSource, VideoSourceError


def _expected_webcam_call_args(index: int) -> tuple:
    """VideoSource passes an explicit AVFoundation backend on macOS only
    (to avoid a silent open-hang when camera permission is missing); other
    platforms get the plain index so OpenCV picks its default backend."""
    if sys.platform == "darwin":
        return (index, cv2.CAP_AVFOUNDATION)
    return (index,)


@patch("src.video_source.cv2.VideoCapture")
def test_webcam_index_zero_interpreted_as_int(mock_capture_cls: MagicMock) -> None:
    """A source string of '0' must be passed to VideoCapture as int 0."""
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = True
    mock_capture_cls.return_value = mock_instance

    VideoSource("0")

    mock_capture_cls.assert_called_once_with(*_expected_webcam_call_args(0))


@patch("src.video_source.cv2.VideoCapture")
def test_webcam_index_other_digit(mock_capture_cls: MagicMock) -> None:
    """A source string of '1' must be passed to VideoCapture as int 1."""
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = True
    mock_capture_cls.return_value = mock_instance

    VideoSource("1")

    mock_capture_cls.assert_called_once_with(*_expected_webcam_call_args(1))


@patch("src.video_source.cv2.VideoCapture")
def test_video_path_string_interpreted_as_path(
    mock_capture_cls: MagicMock, tmp_path
) -> None:
    """A non-numeric source string pointing to an existing file is passed
    through as a string path, not converted to an int."""
    video_file = tmp_path / "test_video.mp4"
    video_file.write_bytes(b"fake video bytes")

    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = True
    mock_capture_cls.return_value = mock_instance

    VideoSource(str(video_file))

    mock_capture_cls.assert_called_once_with(str(video_file))


def test_nonexistent_video_path_raises() -> None:
    """A path to a file that does not exist must raise VideoSourceError
    before ever touching cv2.VideoCapture."""
    with pytest.raises(VideoSourceError):
        VideoSource("data/input/does_not_exist.mp4")


@patch("src.video_source.cv2.VideoCapture")
def test_webcam_cannot_open_raises(mock_capture_cls: MagicMock) -> None:
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = False
    mock_capture_cls.return_value = mock_instance

    with pytest.raises(VideoSourceError):
        VideoSource("0")


@patch("src.video_source.cv2.VideoCapture")
def test_read_frame_failure_raises(mock_capture_cls: MagicMock) -> None:
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = True
    mock_instance.read.return_value = (False, None)
    mock_capture_cls.return_value = mock_instance

    source = VideoSource("0")
    with pytest.raises(VideoSourceError):
        source.read_frame()


@patch("src.video_source.cv2.VideoCapture")
def test_read_frame_success_returns_frame(mock_capture_cls: MagicMock) -> None:
    fake_frame = object()
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = True
    mock_instance.read.return_value = (True, fake_frame)
    mock_capture_cls.return_value = mock_instance

    source = VideoSource("0")
    assert source.read_frame() is fake_frame


@patch("src.video_source.cv2.VideoCapture")
def test_release_calls_underlying_release(mock_capture_cls: MagicMock) -> None:
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = True
    mock_capture_cls.return_value = mock_instance

    source = VideoSource("0")
    source.release()

    mock_instance.release.assert_called_once()


@patch("src.video_source.cv2.VideoCapture")
def test_is_live_source_true_for_webcam_index(mock_capture_cls: MagicMock) -> None:
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = True
    mock_capture_cls.return_value = mock_instance

    source = VideoSource("0")

    assert source.is_live_source() is True


@patch("src.video_source.cv2.VideoCapture")
def test_is_live_source_false_for_video_file(
    mock_capture_cls: MagicMock, tmp_path
) -> None:
    video_file = tmp_path / "test_video.mp4"
    video_file.write_bytes(b"fake video bytes")

    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = True
    mock_capture_cls.return_value = mock_instance

    source = VideoSource(str(video_file))

    assert source.is_live_source() is False
