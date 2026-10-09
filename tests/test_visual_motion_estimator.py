"""Unit tests for VisualMotionEstimator. No camera, model, or network
required -- all inputs are synthetic, seeded (deterministic) frames.
"""

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from src.models import BoundingBox
from src.motion.motion_estimator import MotionEstimator
from src.motion.visual_motion_estimator import VisualMotionEstimator

FRAME_SIZE = 300


def make_textured_frame(size: int = FRAME_SIZE, seed: int = 42) -> np.ndarray:
    """A deterministic, corner-rich synthetic grayscale frame."""
    rng = np.random.default_rng(seed)
    frame = np.zeros((size, size), dtype=np.uint8)
    for _ in range(150):
        x, y = rng.integers(0, size - 10, 2)
        value = int(rng.integers(50, 255))
        frame[y : y + 8, x : x + 8] = value
    return frame


def shift_frame(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))


def rotate_frame(frame: np.ndarray, angle_degrees: float) -> np.ndarray:
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))


def make_estimator(**overrides) -> VisualMotionEstimator:
    params = dict(
        max_features=400,
        quality_level=0.01,
        min_distance_px=8,
        block_size=7,
        lk_window_size=21,
        lk_max_level=3,
        ransac_threshold_px=3.0,
        min_tracked_features=20,
        min_inliers=12,
        exclude_foreground=True,
        exclusion_padding_px=10,
        smoothing_enabled=True,
        smoothing_alpha=0.25,
        max_consecutive_invalid_frames=5,
        low_confidence_threshold=0.6,
        max_translation_px=250.0,
        max_rotation_degrees=30.0,
    )
    params.update(overrides)
    return VisualMotionEstimator(**params)


@pytest.fixture
def estimator() -> VisualMotionEstimator:
    return make_estimator()


def test_is_a_motion_estimator(estimator: VisualMotionEstimator) -> None:
    assert isinstance(estimator, MotionEstimator)


def test_identical_frames_report_near_zero_motion(
    estimator: VisualMotionEstimator,
) -> None:
    frame = make_textured_frame()
    result = estimator.estimate(frame, frame)

    assert result.valid
    assert abs(result.dx) < 1.0
    assert abs(result.dy) < 1.0
    assert abs(result.rotation_degrees) < 1.0


def test_horizontal_translation_recovered_with_correct_sign(
    estimator: VisualMotionEstimator,
) -> None:
    frame = make_textured_frame()
    shifted = shift_frame(frame, dx=10, dy=0)

    result = estimator.estimate(frame, shifted)

    assert result.valid
    # Content shifted +10px right -> dx should be positive (right), per
    # the documented convention, with no sign flip.
    assert result.dx == pytest.approx(10, abs=3)
    assert abs(result.dy) < 2


def test_horizontal_translation_left_recovered_with_correct_sign(
    estimator: VisualMotionEstimator,
) -> None:
    frame = make_textured_frame()
    shifted = shift_frame(frame, dx=-10, dy=0)

    result = estimator.estimate(frame, shifted)

    assert result.valid
    assert result.dx == pytest.approx(-10, abs=3)


def test_vertical_translation_recovered_with_correct_sign(
    estimator: VisualMotionEstimator,
) -> None:
    frame = make_textured_frame()
    shifted = shift_frame(frame, dx=0, dy=8)

    result = estimator.estimate(frame, shifted)

    assert result.valid
    assert abs(result.dx) < 2
    # Content shifted +8px down -> dy should be positive (down).
    assert result.dy == pytest.approx(8, abs=3)


def test_small_rotation_recovered_with_documented_sign(
    estimator: VisualMotionEstimator,
) -> None:
    frame = make_textured_frame()
    rotated = rotate_frame(frame, angle_degrees=5.0)

    result = estimator.estimate(frame, rotated)

    assert result.valid
    # Empirically verified (see visual_motion_estimator.py's module
    # docstring): content rotated +5deg via cv2.getRotationMatrix2D
    # (OpenCV's own counter-clockwise-positive convention) is recovered by
    # this estimator as approximately -5deg -- the opposite sign.
    assert result.rotation_degrees == pytest.approx(-5.0, abs=2.0)


def test_blank_frames_are_invalid_without_crashing(
    estimator: VisualMotionEstimator,
) -> None:
    blank = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
    result = estimator.estimate(blank, blank)

    assert not result.valid
    assert result.reason_invalid is not None
    assert "feature" in result.reason_invalid
    assert result.dx == 0.0
    assert result.dy == 0.0


def make_low_contrast_background(
    size: int = FRAME_SIZE, seed: int = 1, count: int = 150, box: int = 8
) -> np.ndarray:
    """A dim, low-contrast textured background -- weak corners, so a
    higher-contrast foreground patch dominates feature selection unless
    it's masked out."""
    rng = np.random.default_rng(seed)
    frame = np.full((size, size), 120, dtype=np.uint8)
    for _ in range(count):
        x, y = rng.integers(0, size - box, 2)
        value = int(rng.integers(100, 145))
        frame[y : y + box, x : x + box] = value
    return frame


def make_high_contrast_patch(
    size: int, seed: int = 2, count: int = 150, box: int = 6
) -> np.ndarray:
    """A stark black/white textured patch -- strong corners, simulating a
    prominent foreground object that would otherwise dominate feature
    detection over a dim background."""
    rng = np.random.default_rng(seed)
    frame = np.zeros((size, size), dtype=np.uint8)
    for _ in range(count):
        x, y = rng.integers(0, size - box, 2)
        value = 255 if rng.integers(0, 2) == 0 else 0
        frame[y : y + box, x : x + box] = value
    return frame


def test_foreground_exclusion_protects_background_estimate() -> None:
    background = make_low_contrast_background()
    foreground_size = 180
    foreground_texture = make_high_contrast_patch(foreground_size)

    x0, y0 = 60, 60
    x1, y1 = x0 + foreground_size, y0 + foreground_size

    frame_a = background.copy()
    frame_a[y0:y1, x0:x1] = foreground_texture

    true_background_dx = 10
    background_shifted = shift_frame(background, dx=true_background_dx, dy=0)
    foreground_shifted = shift_frame(foreground_texture, dx=-40, dy=0)

    frame_b = background_shifted.copy()
    frame_b[y0:y1, x0:x1] = foreground_shifted

    foreground_box = BoundingBox(x1=x0, y1=y0, x2=x1, y2=y1)

    # A small max_features budget makes the (much stronger) foreground
    # corners dominate selection when not masked out -- demonstrating
    # exactly the failure mode foreground exclusion exists to prevent.
    result_excluded = make_estimator(max_features=60).estimate(
        frame_a, frame_b, excluded_regions=[foreground_box]
    )
    result_included = make_estimator(max_features=60).estimate(
        frame_a, frame_b, excluded_regions=None
    )

    assert result_excluded.valid
    assert result_excluded.dx == pytest.approx(true_background_dx, abs=2)

    # Without exclusion, the dominant high-contrast foreground corners
    # should either make the estimate outright invalid (RANSAC can't find
    # a confident consensus) or clearly worse than the excluded estimate
    # -- either way demonstrates exclusion protects the global estimate.
    excluded_error = abs(result_excluded.dx - true_background_dx)
    if result_included.valid:
        included_error = abs(result_included.dx - true_background_dx)
        assert included_error > excluded_error
    else:
        assert result_included.reason_invalid is not None


def test_outlier_robustness_recovers_dominant_background_motion(
    estimator: VisualMotionEstimator,
) -> None:
    frame = make_textured_frame(seed=3)
    true_dx = 12
    shifted = shift_frame(frame, dx=true_dx, dy=0)

    # A handful of small patches moved independently/inconsistently,
    # simulating a few bad correspondences amid consistent background
    # motion.
    outlier_shifted = shifted.copy()
    rng = np.random.default_rng(7)
    for _ in range(4):
        x, y = rng.integers(20, FRAME_SIZE - 40, 2)
        patch = make_textured_frame(size=20, seed=int(rng.integers(0, 1000)))
        outlier_shifted[y : y + 20, x : x + 20] = patch

    result = estimator.estimate(frame, outlier_shifted)

    assert result.valid
    assert result.dx == pytest.approx(true_dx, abs=4)


def test_reset_clears_smoothing_state_without_leaking_history() -> None:
    smoothing_estimator = make_estimator(smoothing_alpha=0.5)
    frame = make_textured_frame(seed=5)

    shifted_a = shift_frame(frame, dx=10, dy=0)
    smoothing_estimator.estimate(frame, shifted_a)

    shifted_b = shift_frame(frame, dx=20, dy=0)
    second_result = smoothing_estimator.estimate(frame, shifted_b)
    # Smoothed second estimate should have been pulled toward the first
    # estimate's value, away from its own raw reading.
    assert second_result.dx != pytest.approx(second_result.raw_dx, abs=0.01)

    smoothing_estimator.reset()

    shifted_c = shift_frame(frame, dx=30, dy=0)
    third_result = smoothing_estimator.estimate(frame, shifted_c)
    # First estimate since reset() -- no leaked history, smoothed == raw.
    assert third_result.dx == pytest.approx(third_result.raw_dx, abs=0.01)


def test_mismatched_frame_shapes_are_invalid_without_crashing(
    estimator: VisualMotionEstimator,
) -> None:
    small = make_textured_frame(size=100)
    large = make_textured_frame(size=200)

    result = estimator.estimate(small, large)

    assert not result.valid
    assert "dimensions" in result.reason_invalid


def test_sequential_different_frame_sizes_handled_safely(
    estimator: VisualMotionEstimator,
) -> None:
    small = make_textured_frame(size=100, seed=10)
    small_shifted = shift_frame(small, dx=5, dy=0)
    estimator.estimate(small, small_shifted)

    large = make_textured_frame(size=200, seed=11)
    large_shifted = shift_frame(large, dx=5, dy=0)
    # Must not crash; a fresh, differently-sized frame pair is a valid
    # scenario on its own (only *mismatched* prev/current shapes within a
    # single call are rejected).
    result = estimator.estimate(large, large_shifted)
    assert isinstance(result.valid, bool)


def test_max_features_must_be_positive() -> None:
    with pytest.raises(ValueError):
        make_estimator(max_features=0)


def test_quality_level_must_be_in_range() -> None:
    with pytest.raises(ValueError):
        make_estimator(quality_level=1.5)


def test_min_inliers_cannot_exceed_min_tracked_features() -> None:
    with pytest.raises(ValueError):
        make_estimator(min_tracked_features=10, min_inliers=20)


def test_smoothing_alpha_must_be_in_range() -> None:
    with pytest.raises(ValueError):
        make_estimator(smoothing_alpha=0.0)
    with pytest.raises(ValueError):
        make_estimator(smoothing_alpha=1.5)


def test_max_consecutive_invalid_frames_must_be_positive() -> None:
    with pytest.raises(ValueError):
        make_estimator(max_consecutive_invalid_frames=0)


def test_low_confidence_threshold_must_be_in_range() -> None:
    with pytest.raises(ValueError):
        make_estimator(low_confidence_threshold=1.5)


def test_max_translation_px_must_be_positive() -> None:
    with pytest.raises(ValueError):
        make_estimator(max_translation_px=0)


def test_max_rotation_degrees_must_be_positive() -> None:
    with pytest.raises(ValueError):
        make_estimator(max_rotation_degrees=0)


# --- new diagnostics: feature_count / inlier_ratio / status --------------


def test_feature_count_is_at_least_tracked_feature_count(
    estimator: VisualMotionEstimator,
) -> None:
    frame = make_textured_frame()
    shifted = shift_frame(frame, dx=10, dy=0)
    result = estimator.estimate(frame, shifted)

    assert result.valid
    assert result.feature_count >= result.tracked_feature_count


def test_inlier_ratio_matches_inlier_and_tracked_counts(
    estimator: VisualMotionEstimator,
) -> None:
    frame = make_textured_frame()
    shifted = shift_frame(frame, dx=10, dy=0)
    result = estimator.estimate(frame, shifted)

    assert result.valid
    assert result.inlier_ratio == pytest.approx(
        result.inlier_count / result.tracked_feature_count
    )


def test_status_is_valid_for_a_good_estimate(estimator: VisualMotionEstimator) -> None:
    frame = make_textured_frame()
    shifted = shift_frame(frame, dx=10, dy=0)
    result = estimator.estimate(frame, shifted)

    assert result.status == "VALID"


def test_status_is_low_confidence_when_below_threshold() -> None:
    # This frame/shift pair naturally tracks ~142 features with a perfect
    # (1.0) inlier ratio, so confidence saturates at 1.0 by default.
    # Raising min_tracked_features (while staying well under the actual
    # count, so the estimate remains valid) drags feature_sufficiency --
    # and therefore confidence -- below an otherwise-reasonable
    # threshold, isolating the LOW_CONFIDENCE labeling logic without
    # needing a genuinely noisy/ambiguous frame pair.
    partial_confidence_estimator = make_estimator(
        min_tracked_features=100, low_confidence_threshold=0.95
    )
    frame = make_textured_frame()
    shifted = shift_frame(frame, dx=10, dy=0)

    result = partial_confidence_estimator.estimate(frame, shifted)

    assert result.valid
    assert result.confidence < 0.95
    assert result.status == "LOW_CONFIDENCE"


def test_status_is_unavailable_for_blank_frames(
    estimator: VisualMotionEstimator,
) -> None:
    blank = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
    result = estimator.estimate(blank, blank)

    assert not result.valid
    assert result.status == "UNAVAILABLE"


# --- new rejection paths: non-finite / implausibly large transform -------


def test_non_finite_transform_is_rejected_without_crashing(
    estimator: VisualMotionEstimator,
) -> None:
    frame = make_textured_frame()
    shifted = shift_frame(frame, dx=10, dy=0)

    def fake_estimate_affine(good_prev, good_next, method=None, ransacReprojThreshold=None):
        # Real corner data essentially never produces a NaN transform;
        # mocking is the clean, deterministic way to exercise this
        # rejection path.
        n = len(good_prev)
        matrix = np.array([[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        inlier_mask = np.ones((n, 1), dtype=np.uint8)
        return matrix, inlier_mask

    with patch("cv2.estimateAffinePartial2D", side_effect=fake_estimate_affine):
        result = estimator.estimate(frame, shifted)

    assert not result.valid
    assert result.status == "UNAVAILABLE"
    assert "finite" in result.reason_invalid


def test_implausibly_large_translation_is_rejected() -> None:
    # A normal, well-tracked shift that simply exceeds an artificially
    # tight max_translation_px -- isolates the new magnitude check
    # without needing a genuinely degenerate frame pair.
    tight_estimator = make_estimator(max_translation_px=5.0)
    frame = make_textured_frame()
    shifted = shift_frame(frame, dx=10, dy=0)

    result = tight_estimator.estimate(frame, shifted)

    assert not result.valid
    assert "implausibly large" in result.reason_invalid


def test_implausibly_large_rotation_is_rejected() -> None:
    tight_estimator = make_estimator(max_rotation_degrees=1.0)
    frame = make_textured_frame()
    rotated = rotate_frame(frame, angle_degrees=5.0)

    result = tight_estimator.estimate(frame, rotated)

    assert not result.valid
    assert "implausibly large" in result.reason_invalid
