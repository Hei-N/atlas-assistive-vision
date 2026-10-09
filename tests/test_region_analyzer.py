"""Unit tests for RegionAnalyzer. No camera, model, or network required."""

import pytest

from src.models import BoundingBox
from src.region_analyzer import RegionAnalyzer


FRAME_WIDTH = 900
FRAME_HEIGHT = 600


@pytest.fixture
def analyzer() -> RegionAnalyzer:
    return RegionAnalyzer(
        left_boundary=0.33,
        right_boundary=0.66,
        attention_zone_fractions=(0.25, 0.75, 0.20, 0.80),
    )


def test_classify_point_left(analyzer: RegionAnalyzer) -> None:
    # x = 100 -> fraction ~0.11, well below 0.33
    assert analyzer.classify_point(100, FRAME_WIDTH) == "left"


def test_classify_point_center(analyzer: RegionAnalyzer) -> None:
    # x = 450 -> fraction 0.5, between 0.33 and 0.66
    assert analyzer.classify_point(450, FRAME_WIDTH) == "center"


def test_classify_point_right(analyzer: RegionAnalyzer) -> None:
    # x = 800 -> fraction ~0.89, above 0.66
    assert analyzer.classify_point(800, FRAME_WIDTH) == "right"


def test_classify_point_invalid_frame_width(analyzer: RegionAnalyzer) -> None:
    with pytest.raises(ValueError):
        analyzer.classify_point(100, 0)


def test_compute_attention_zone_coordinates(analyzer: RegionAnalyzer) -> None:
    zone = analyzer.compute_attention_zone(FRAME_WIDTH, FRAME_HEIGHT)
    assert zone.x_min == int(0.25 * FRAME_WIDTH)
    assert zone.x_max == int(0.75 * FRAME_WIDTH)
    assert zone.y_min == int(0.20 * FRAME_HEIGHT)
    assert zone.y_max == int(0.80 * FRAME_HEIGHT)


def test_compute_attention_zone_invalid_dimensions(analyzer: RegionAnalyzer) -> None:
    with pytest.raises(ValueError):
        analyzer.compute_attention_zone(0, FRAME_HEIGHT)
    with pytest.raises(ValueError):
        analyzer.compute_attention_zone(FRAME_WIDTH, -1)


def test_point_in_zone_true(analyzer: RegionAnalyzer) -> None:
    zone = analyzer.compute_attention_zone(FRAME_WIDTH, FRAME_HEIGHT)
    center_x = (zone.x_min + zone.x_max) // 2
    center_y = (zone.y_min + zone.y_max) // 2
    assert RegionAnalyzer.point_in_zone(center_x, center_y, zone) is True


def test_point_in_zone_false(analyzer: RegionAnalyzer) -> None:
    zone = analyzer.compute_attention_zone(FRAME_WIDTH, FRAME_HEIGHT)
    assert RegionAnalyzer.point_in_zone(0, 0, zone) is False


def test_box_overlaps_zone_true(analyzer: RegionAnalyzer) -> None:
    zone = analyzer.compute_attention_zone(FRAME_WIDTH, FRAME_HEIGHT)
    # Box fully inside the zone.
    box = BoundingBox(
        x1=zone.x_min + 5,
        y1=zone.y_min + 5,
        x2=zone.x_min + 20,
        y2=zone.y_min + 20,
    )
    assert RegionAnalyzer.box_overlaps_zone(box, zone) is True


def test_box_overlaps_zone_false(analyzer: RegionAnalyzer) -> None:
    zone = analyzer.compute_attention_zone(FRAME_WIDTH, FRAME_HEIGHT)
    # Box entirely to the left of and above the zone -> no overlap.
    box = BoundingBox(x1=0, y1=0, x2=5, y2=5)
    assert RegionAnalyzer.box_overlaps_zone(box, zone) is False


def test_get_region_boundaries_px(analyzer: RegionAnalyzer) -> None:
    left_px, right_px = analyzer.get_region_boundaries_px(FRAME_WIDTH)
    assert left_px == int(0.33 * FRAME_WIDTH)
    assert right_px == int(0.66 * FRAME_WIDTH)


def test_invalid_boundaries_raise() -> None:
    with pytest.raises(ValueError):
        RegionAnalyzer(
            left_boundary=0.7,
            right_boundary=0.3,
            attention_zone_fractions=(0.25, 0.75, 0.2, 0.8),
        )
