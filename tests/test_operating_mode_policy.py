"""Unit tests for src/audio/operating_mode_policy.py -- the two-stage
Minimal/Balanced/Detailed filter (class/person eligibility) and
finalizer (Level-5 message form, danger-episode repeat policy, Detailed
summary caps). Pure logic, no camera/model/network/subprocess involved
-- synthetic AudioHazardResult/AudioEvent data throughout.
"""

import pytest

from src.audio.operating_mode_policy import (
    ALLOWED_DANGER_REPEAT,
    ALLOWED_DETAILED_CONTEXT,
    ALLOWED_NEW_DANGER_EPISODE,
    ALLOWED_RELEVANT_PERSON,
    ALLOWED_TRAFFIC_EVENT,
    BALANCED,
    DETAILED,
    MINIMAL,
    PERSON_BROAD,
    PERSON_INELIGIBLE,
    PERSON_RELEVANT_ONLY,
    SUPPRESSED_DETAILED_REPEAT,
    SUPPRESSED_IRRELEVANT_PERSON_IN_BALANCED,
    SUPPRESSED_PERSON_IN_MINIMAL,
    SUPPRESSED_REPEAT_LIMIT,
    SUPPRESSED_SUMMARY_GROUP_LIMIT,
    SUPPRESSED_UNCHANGED_SUMMARY,
    OperatingModePolicy,
    parse_operating_mode,
)
from src.audio.scene_summarizer import SCENE_SUMMARY
from src.models import AudioEvent, AudioHazardResult, OperatingModeProfile


def make_profile(
    mode: str,
    person_eligibility: str,
    repeat_enabled: bool = True,
    delay: float = 3.0,
    max_cycles: int = 2,
    structured: bool = False,
    max_groups: int = 2,
    max_words: int = 15,
) -> OperatingModeProfile:
    return OperatingModeProfile(
        mode=mode, announce_traffic=True, person_eligibility=person_eligibility,
        highest_danger_repeat_enabled=repeat_enabled,
        highest_danger_repeat_delay_seconds=delay, highest_danger_max_cycles=max_cycles,
        structured_scene_summaries=structured, max_summary_object_groups=max_groups,
        max_summary_words=max_words,
    )


def make_policy(mode: str = MINIMAL, **detailed_kwargs) -> OperatingModePolicy:
    profiles = {
        MINIMAL: make_profile(MINIMAL, PERSON_INELIGIBLE),
        BALANCED: make_profile(BALANCED, PERSON_RELEVANT_ONLY),
        DETAILED: make_profile(
            DETAILED, PERSON_BROAD, repeat_enabled=False, max_cycles=1,
            structured=True, **detailed_kwargs,
        ),
    }
    return OperatingModePolicy(mode=mode, mode_source="CLI", profiles=profiles)


def make_hazard(
    track_id: int = 1, level: str = "LEVEL_1_MOVING_FAR", class_name: str = "car",
    region: str = "left", priority: str = "INFORMATIONAL",
) -> AudioHazardResult:
    return AudioHazardResult(
        track_id=track_id, level=level, priority=priority, class_name=class_name,
        region=region, proximity_zone="NEAR", approach_state="APPROACHING",
        intersects_corridor=False, uncertain=False, reason_codes=(),
        recommended_message="placeholder", delivery_profile="level_1",
    )


def make_level5_event(
    region: str = "left", track_id: int = 1, created_at: float = 0.0, name: str = "Vehicle"
) -> AudioEvent:
    direction = {"left": "from the left", "right": "from the right", "center": "ahead"}[region]
    return AudioEvent(
        key=f"LEVEL_5_HIGH_DANGER:{track_id}:{region}", event_type="LEVEL_5_HIGH_DANGER",
        priority="WARNING", message=f"Warning! {name} approaching {direction}. Please wait.",
        track_id=track_id, created_at=created_at, delivery_profile="level_5",
    )


def make_summary_event(region: str = "left", message: str = "Vehicle and bicycle moving from the left.") -> AudioEvent:
    return AudioEvent(
        key=f"{SCENE_SUMMARY}:LEVEL_1_MOVING_FAR:{region}", event_type=SCENE_SUMMARY,
        priority="INFORMATIONAL", message=message, track_id=None, created_at=0.0,
        delivery_profile="level_1",
    )


# --- A: mode parsing/validation -----------------------------------------


def test_parse_operating_mode_accepts_lowercase() -> None:
    assert parse_operating_mode("minimal") == MINIMAL
    assert parse_operating_mode("balanced") == BALANCED
    assert parse_operating_mode("detailed") == DETAILED


def test_parse_operating_mode_is_case_insensitive() -> None:
    assert parse_operating_mode("MiNiMaL") == MINIMAL


def test_parse_operating_mode_rejects_invalid_value() -> None:
    with pytest.raises(ValueError):
        parse_operating_mode("extreme")


def test_policy_exposes_mode_through_one_central_interface() -> None:
    policy = make_policy(mode=BALANCED)
    assert policy.mode == BALANCED
    assert policy.mode_source == "CLI"


def test_constructor_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError):
        OperatingModePolicy(mode="EXTREME", mode_source="CLI", profiles={
            MINIMAL: make_profile(MINIMAL, PERSON_INELIGIBLE),
            BALANCED: make_profile(BALANCED, PERSON_RELEVANT_ONLY),
            DETAILED: make_profile(DETAILED, PERSON_BROAD),
        })


def test_constructor_rejects_missing_profile() -> None:
    with pytest.raises(ValueError):
        OperatingModePolicy(mode=MINIMAL, mode_source="CLI", profiles={
            MINIMAL: make_profile(MINIMAL, PERSON_INELIGIBLE),
        })


# --- C/D: class/person eligibility per mode -------------------------------


def test_minimal_allows_all_traffic_classes() -> None:
    policy = make_policy(MINIMAL)
    hazards = {i: make_hazard(track_id=i, class_name=c) for i, c in enumerate(
        ("car", "bus", "truck", "motorcycle", "bicycle")
    )}
    allowed, traces = policy.filter_events(hazards, now=0.0)
    assert set(allowed) == set(hazards)
    assert all(t.mode_filter_reason == ALLOWED_TRAFFIC_EVENT for t in traces.values())


def test_minimal_suppresses_ordinary_person() -> None:
    policy = make_policy(MINIMAL)
    hazards = {1: make_hazard(class_name="person", level="LEVEL_1_MOVING_FAR")}
    allowed, traces = policy.filter_events(hazards, now=0.0)
    assert allowed == {}
    assert traces[1].mode_filter_reason == SUPPRESSED_PERSON_IN_MINIMAL


def test_minimal_suppresses_close_person_too() -> None:
    # Minimal has no relevance exception in this codebase (confirmed:
    # no validated direct-human-collision emergency rule exists) --
    # ALL person levels are suppressed, not just ordinary movement.
    policy = make_policy(MINIMAL)
    hazards = {1: make_hazard(class_name="person", level="LEVEL_5_HIGH_DANGER", priority="WARNING")}
    allowed, _ = policy.filter_events(hazards, now=0.0)
    assert allowed == {}


def test_balanced_suppresses_distant_irrelevant_person() -> None:
    policy = make_policy(BALANCED)
    hazards = {1: make_hazard(class_name="person", level="LEVEL_1_MOVING_FAR")}
    allowed, traces = policy.filter_events(hazards, now=0.0)
    assert allowed == {}
    assert traces[1].mode_filter_reason == SUPPRESSED_IRRELEVANT_PERSON_IN_BALANCED


@pytest.mark.parametrize(
    "level", ["LEVEL_2_MOVING_NEARBY", "LEVEL_3_APPROACHING", "LEVEL_4_PATH_CONFLICT", "LEVEL_5_HIGH_DANGER"]
)
def test_balanced_allows_relevant_person(level: str) -> None:
    policy = make_policy(BALANCED)
    hazards = {1: make_hazard(class_name="person", level=level, priority="WARNING")}
    allowed, traces = policy.filter_events(hazards, now=0.0)
    assert 1 in allowed
    assert traces[1].mode_filter_reason == ALLOWED_RELEVANT_PERSON


def test_balanced_allows_all_minimal_traffic_events() -> None:
    policy = make_policy(BALANCED)
    hazards = {i: make_hazard(track_id=i, class_name=c) for i, c in enumerate(
        ("car", "bus", "truck", "motorcycle", "bicycle")
    )}
    allowed, _ = policy.filter_events(hazards, now=0.0)
    assert set(allowed) == set(hazards)


def test_detailed_allows_all_person_levels() -> None:
    policy = make_policy(DETAILED)
    hazards = {1: make_hazard(class_name="person", level="LEVEL_1_MOVING_FAR")}
    allowed, traces = policy.filter_events(hazards, now=0.0)
    assert 1 in allowed
    assert traces[1].mode_filter_reason == ALLOWED_DETAILED_CONTEXT


# --- C20/D32/E41: Level-5 message form per mode ---------------------------


def test_minimal_level5_uses_doubled_wording_left() -> None:
    policy = make_policy(MINIMAL)
    final, _ = policy.finalize_events([make_level5_event(region="left")], now=0.0)
    assert final[0].message == "Warning! Vehicle on the left! Vehicle on the left!"


def test_minimal_level5_uses_doubled_wording_right() -> None:
    policy = make_policy(MINIMAL)
    final, _ = policy.finalize_events([make_level5_event(region="right")], now=0.0)
    assert final[0].message == "Warning! Vehicle on the right! Vehicle on the right!"


def test_minimal_level5_directly_ahead_for_center() -> None:
    policy = make_policy(MINIMAL)
    final, _ = policy.finalize_events([make_level5_event(region="center")], now=0.0)
    assert final[0].message == "Warning! Vehicle directly ahead! Vehicle directly ahead!"


def test_balanced_level5_uses_same_doubled_wording_as_minimal() -> None:
    policy = make_policy(BALANCED)
    final, _ = policy.finalize_events([make_level5_event(region="left")], now=0.0)
    assert final[0].message == "Warning! Vehicle on the left! Vehicle on the left!"


def test_detailed_level5_uses_single_concise_phrase() -> None:
    policy = make_policy(DETAILED)
    final, _ = policy.finalize_events([make_level5_event(region="left")], now=0.0)
    assert final[0].message == "Warning! Vehicle on the left!"
    assert final[0].message.count("Vehicle") == 1


# --- C21-25/D33-34: danger repeat cycle limits (Minimal/Balanced) --------


def _continuous_calls(policy, region, start, end, step=0.1, name="Vehicle", track_id=1):
    """Simulates realistic every-frame calling -- last_seen_time is
    refreshed continuously, matching main.py's real per-frame loop."""
    spoken = []
    t = start
    while t <= end + 1e-9:
        event = make_level5_event(region=region, name=name, track_id=track_id, created_at=t)
        final, traces = policy.finalize_events([event], now=t)
        if final:
            spoken.append((round(t, 2), final[0].message, traces[0]))
        t = round(t + step, 2)
    return spoken


def test_initial_danger_admitted_immediately() -> None:
    policy = make_policy(MINIMAL)
    spoken = _continuous_calls(policy, "left", 0.0, 0.0)
    assert len(spoken) == 1
    assert spoken[0][2].mode_filter_reason == ALLOWED_NEW_DANGER_EPISODE


def test_same_active_danger_repeats_once_after_three_seconds() -> None:
    policy = make_policy(MINIMAL)
    spoken = _continuous_calls(policy, "left", 0.0, 4.0)
    times = [s[0] for s in spoken]
    assert times[0] == 0.0
    assert 3.0 in times
    assert len(spoken) == 2


def test_same_danger_cannot_repeat_a_third_time() -> None:
    policy = make_policy(MINIMAL)
    spoken = _continuous_calls(policy, "left", 0.0, 8.0)
    assert len(spoken) == 2  # never a 3rd cycle despite 8s of continuous danger


def test_repeat_does_not_create_overlap_only_one_event_per_call() -> None:
    policy = make_policy(MINIMAL)
    final, _ = policy.finalize_events([make_level5_event(region="left")], now=0.0)
    assert len(final) == 1


def test_stale_repeat_is_dropped_before_delay_elapses() -> None:
    policy = make_policy(MINIMAL)
    policy.finalize_events([make_level5_event(region="left")], now=0.0)
    final, traces = policy.finalize_events([make_level5_event(region="left")], now=1.0)
    assert final == []
    assert traces[0].mode_filter_reason == SUPPRESSED_REPEAT_LIMIT


# --- E42-45: Detailed danger does not repeat automatically ----------------


def test_detailed_danger_does_not_repeat_after_three_seconds() -> None:
    policy = make_policy(DETAILED)
    spoken = _continuous_calls(policy, "left", 0.0, 4.0)
    assert len(spoken) == 1
    assert spoken[0][0] == 0.0


def test_detailed_second_admission_reason_is_suppressed_detailed_repeat() -> None:
    policy = make_policy(DETAILED)
    policy.finalize_events([make_level5_event(region="left")], now=0.0)
    _, traces = policy.finalize_events([make_level5_event(region="left")], now=0.1)
    assert traces[0].mode_filter_reason == SUPPRESSED_DETAILED_REPEAT


def test_new_hazard_in_different_region_may_speak_during_detailed_hold() -> None:
    policy = make_policy(DETAILED)
    policy.finalize_events([make_level5_event(region="left")], now=0.0)
    final, _ = policy.finalize_events([make_level5_event(region="right")], now=0.1)
    assert len(final) == 1


def test_new_episode_after_hazard_fully_clears() -> None:
    policy = make_policy(MINIMAL)
    _continuous_calls(policy, "left", 0.0, 4.0)  # 2 cycles used up
    # Hazard clears for > grace period (3.0s of no calls for this region)
    final, traces = policy.finalize_events([make_level5_event(region="left")], now=10.0)
    assert len(final) == 1
    assert traces[0].mode_filter_reason == ALLOWED_NEW_DANGER_EPISODE
    assert traces[0].hazard_episode_cycle_count == 1


# --- track-ID churn does not reset the episode (region-based identity) ---


def test_different_track_id_same_region_continues_the_same_episode() -> None:
    policy = make_policy(MINIMAL)
    policy.finalize_events([make_level5_event(region="left", track_id=1)], now=0.0)
    # Occlusion-style track-ID change -- same region, different track_id,
    # shortly after -- must be treated as the SAME episode (no reset).
    final, traces = policy.finalize_events(
        [make_level5_event(region="left", track_id=2)], now=0.2
    )
    assert final == []  # still within the repeat delay -- correctly suppressed, not restarted
    assert traces[0].hazard_episode_cycle_count == 1


# --- E38-40: Detailed summary group/word caps -----------------------------


def test_detailed_enforces_max_two_summary_groups() -> None:
    policy = make_policy(DETAILED)
    events = [
        make_summary_event(region="left", message="Vehicle moving from the left."),
        make_summary_event(region="right", message="Bicycle moving from the right."),
        make_summary_event(region="center", message="Person moving ahead."),
    ]
    final, traces = policy.finalize_events(events, now=0.0)
    assert len(final) == 2


def test_detailed_group_limit_suppression_reason() -> None:
    policy = make_policy(DETAILED)
    events = [
        make_summary_event(region="left", message="Vehicle moving from the left."),
        make_summary_event(region="right", message="Bicycle moving from the right."),
        make_summary_event(region="center", message="Person moving ahead."),
    ]
    _, traces = policy.finalize_events(events, now=0.0)
    assert traces[-1].mode_filter_reason == SUPPRESSED_SUMMARY_GROUP_LIMIT


def test_detailed_word_limit_falls_back_to_short_message() -> None:
    policy = make_policy(DETAILED, max_words=3)
    event = make_summary_event(message="Vehicle, bicycle, and person moving from the left.")
    final, traces = policy.finalize_events([event], now=0.0)
    assert len(final[0].message.split()) <= 3
    assert final[0].message != event.message


def test_minimal_and_balanced_never_apply_summary_caps() -> None:
    # structured_scene_summaries is False for Minimal/Balanced -- caps
    # and unchanged-suppression are Detailed-only.
    for mode in (MINIMAL, BALANCED):
        policy = make_policy(mode)
        events = [
            make_summary_event(region="left"),
            make_summary_event(region="right"),
            make_summary_event(region="center"),
        ]
        final, _ = policy.finalize_events(events, now=0.0)
        assert len(final) == 3


# --- E46: unchanged semantic summary does not repeat -----------------------


def test_detailed_unchanged_summary_suppressed() -> None:
    policy = make_policy(DETAILED)
    event = make_summary_event(region="left", message="Vehicle moving from the left.")
    policy.finalize_events([event], now=0.0)
    final, traces = policy.finalize_events([event], now=0.1)
    assert final == []
    assert traces[0].mode_filter_reason == SUPPRESSED_UNCHANGED_SUMMARY


def test_detailed_changed_summary_is_spoken() -> None:
    policy = make_policy(DETAILED)
    first = make_summary_event(region="left", message="Vehicle moving from the left.")
    second = make_summary_event(region="left", message="2 vehicles moving from the left.")
    policy.finalize_events([first], now=0.0)
    final, _ = policy.finalize_events([second], now=0.1)
    assert len(final) == 1


# --- E48: summaries never delay a Level-5 warning --------------------------


def test_warning_and_summary_both_admitted_in_same_frame() -> None:
    policy = make_policy(DETAILED)
    events = [make_level5_event(region="left"), make_summary_event(region="right")]
    final, _ = policy.finalize_events(events, now=0.0)
    assert len(final) == 2
    assert any(e.message.startswith("Warning!") for e in final)


# --- levels 1-4 pass through finalize_events unchanged ---------------------


def test_levels_1_to_4_pass_through_unchanged() -> None:
    policy = make_policy(MINIMAL)
    event = AudioEvent(
        key="LEVEL_3_APPROACHING:1:left", event_type="LEVEL_3_APPROACHING", priority="WARNING",
        message="Vehicle approaching from the left.", track_id=1, created_at=0.0,
        delivery_profile="level_3",
    )
    final, _ = policy.finalize_events([event], now=0.0)
    assert final == [event]


# --- F: set_mode / future app contract -------------------------------------


def test_set_mode_validates_centrally() -> None:
    policy = make_policy(MINIMAL)
    with pytest.raises(ValueError):
        policy.set_mode("nonsense")


def test_set_mode_switches_active_mode() -> None:
    policy = make_policy(MINIMAL)
    policy.set_mode(BALANCED, source="APP")
    assert policy.mode == BALANCED
    assert policy.mode_source == "APP"


def test_set_mode_clears_pending_episode_state() -> None:
    policy = make_policy(MINIMAL)
    policy.finalize_events([make_level5_event(region="left")], now=0.0)  # cycle=1
    policy.set_mode(MINIMAL, source="APP")  # same mode, but explicit reset

    # A fresh call right after switching must start a NEW episode
    # (cycle=1 again), not continue the old one's state.
    final, traces = policy.finalize_events([make_level5_event(region="left")], now=0.1)
    assert len(final) == 1
    assert traces[0].hazard_episode_cycle_count == 1


def test_set_mode_clears_summary_dedup_state() -> None:
    policy = make_policy(DETAILED)
    event = make_summary_event(region="left", message="Vehicle moving from the left.")
    policy.finalize_events([event], now=0.0)
    policy.set_mode(DETAILED, source="APP")

    final, _ = policy.finalize_events([event], now=0.1)
    assert len(final) == 1  # not suppressed as "unchanged" after reset


# =====================================================================
# Episode-identity correction: (normalized_class, region, hazard_family)
# -- a vehicle and a bicycle in the same region must never share one
# episode/repeat budget. See docs/OPERATING_MODES.md for the full
# rationale (track_id alone is insufficient; mode is redundant given
# set_mode() already clears all episode state).
# =====================================================================


# --- 1: different class, same region -- not the same episode --------


def test_vehicle_then_bicycle_same_region_are_independent_episodes() -> None:
    policy = make_policy(MINIMAL)
    vehicle_spoken = _continuous_calls(policy, "left", 0.0, 3.4, name="Vehicle", track_id=1)
    bicycle_spoken = _continuous_calls(policy, "left", 3.5, 6.9, name="Bicycle", track_id=2)

    # The vehicle used its full 2-cycle budget (0.0s, 3.0s)...
    assert [round(t, 1) for t, _, _ in vehicle_spoken] == [0.0, 3.0]
    # ...and the bicycle, a genuinely different object, gets its OWN
    # full 2-cycle budget -- never suppressed, never inheriting the
    # vehicle's exhausted state.
    assert [round(t, 1) for t, _, _ in bicycle_spoken] == [3.5, 6.5]
    assert bicycle_spoken[0][2].mode_filter_reason == ALLOWED_NEW_DANGER_EPISODE
    assert bicycle_spoken[0][2].hazard_episode_cycle_count == 1


# --- 2: same class, different region -- new episode -------------------


def test_vehicle_left_then_vehicle_right_are_independent_episodes() -> None:
    policy = make_policy(MINIMAL)
    left, _ = policy.finalize_events([make_level5_event(region="left", name="Vehicle")], now=0.0)
    right, traces = policy.finalize_events(
        [make_level5_event(region="right", name="Vehicle")], now=0.1
    )
    assert len(left) == 1
    assert len(right) == 1
    assert traces[0].mode_filter_reason == ALLOWED_NEW_DANGER_EPISODE
    assert traces[0].hazard_episode_cycle_count == 1


# --- 3: vehicle vs. person danger, same region, Balanced --------------


def test_vehicle_danger_and_person_danger_same_region_are_distinct_in_balanced() -> None:
    policy = make_policy(BALANCED)
    vehicle, _ = policy.finalize_events(
        [make_level5_event(region="left", name="Vehicle", track_id=1)], now=0.0
    )
    person, traces = policy.finalize_events(
        [make_level5_event(region="left", name="Person", track_id=2)], now=0.1
    )
    assert len(vehicle) == 1
    assert len(person) == 1
    assert traces[0].mode_filter_reason == ALLOWED_NEW_DANGER_EPISODE


# --- 4: same track, same class, same direction, same hazard -> same episode


def test_same_track_same_class_same_region_is_the_same_episode() -> None:
    policy = make_policy(MINIMAL)
    policy.finalize_events([make_level5_event(region="left", track_id=1)], now=0.0)
    _, traces = policy.finalize_events(
        [make_level5_event(region="left", track_id=1)], now=0.1
    )
    assert traces[0].hazard_episode_cycle_count == 1  # still the first cycle, not a new episode


# --- 5: same semantic hazard, new track ID within grace -> still suppressed


def test_new_track_id_same_class_region_within_grace_is_same_episode() -> None:
    policy = make_policy(MINIMAL)
    policy.finalize_events([make_level5_event(region="left", track_id=1)], now=0.0)
    # Occlusion-style track-ID change, same class/region, well within the
    # 3.0s grace window -- must be treated as the SAME episode (still
    # suppressed, not restarted), exactly like before this correction.
    final, traces = policy.finalize_events(
        [make_level5_event(region="left", track_id=2)], now=0.2
    )
    assert final == []
    assert traces[0].hazard_episode_cycle_count == 1


# --- 6: new track ID with a DIFFERENT class within grace -> new episode


def test_new_track_id_different_class_within_grace_is_a_new_episode() -> None:
    policy = make_policy(MINIMAL)
    policy.finalize_events(
        [make_level5_event(region="left", track_id=1, name="Vehicle")], now=0.0
    )
    final, traces = policy.finalize_events(
        [make_level5_event(region="left", track_id=2, name="Bicycle")], now=0.2
    )
    assert len(final) == 1  # NOT suppressed -- this is the exact bug being corrected
    assert traces[0].mode_filter_reason == ALLOWED_NEW_DANGER_EPISODE


# --- 7: new track ID with a DIFFERENT direction within grace -> new episode


def test_new_track_id_different_region_within_grace_is_a_new_episode() -> None:
    policy = make_policy(MINIMAL)
    policy.finalize_events(
        [make_level5_event(region="left", track_id=1, name="Vehicle")], now=0.0
    )
    final, traces = policy.finalize_events(
        [make_level5_event(region="right", track_id=2, name="Vehicle")], now=0.2
    )
    assert len(final) == 1
    assert traces[0].mode_filter_reason == ALLOWED_NEW_DANGER_EPISODE


# --- 9: hazard clears beyond reset interval, same class/region returns ---


def test_same_class_region_after_full_clear_is_a_new_episode() -> None:
    policy = make_policy(MINIMAL)
    _continuous_calls(policy, "left", 0.0, 3.4, name="Vehicle", track_id=1)  # 2 cycles used
    # Hazard clears for > grace (3.0s of no sighting for this key).
    final, traces = policy.finalize_events(
        [make_level5_event(region="left", track_id=1, name="Vehicle")], now=10.0
    )
    assert len(final) == 1
    assert traces[0].mode_filter_reason == ALLOWED_NEW_DANGER_EPISODE
    assert traces[0].hazard_episode_cycle_count == 1


# --- 10/11: multiple objects in the same region ---------------------------


def test_two_same_class_hazards_same_region_share_one_deterministic_episode() -> None:
    # Two DIFFERENT vehicles, same region, both "seen" within the same
    # grace window -- a disclosed, known limitation (no visual re-ID):
    # they deterministically share one episode/cycle budget, same as any
    # single-object continuation would. Never a crash, never ambiguous.
    policy = make_policy(MINIMAL)
    policy.finalize_events(
        [make_level5_event(region="left", track_id=1, name="Vehicle")], now=0.0
    )
    final, traces = policy.finalize_events(
        [make_level5_event(region="left", track_id=2, name="Vehicle")], now=0.2
    )
    assert final == []  # deterministically treated as the same ongoing episode
    assert traces[0].hazard_episode_cycle_count == 1


def test_two_different_class_hazards_same_region_neither_silently_lost() -> None:
    # Confirms the core fix directly: distinct classes in the same
    # region are never suppressed "merely because they share a region."
    policy = make_policy(MINIMAL)
    vehicle, _ = policy.finalize_events(
        [make_level5_event(region="left", track_id=1, name="Vehicle")], now=0.0
    )
    bicycle, traces = policy.finalize_events(
        [make_level5_event(region="left", track_id=2, name="Bicycle")], now=0.1
    )
    assert len(vehicle) == 1
    assert len(bicycle) == 1
    assert traces[0].hazard_episode_key != None  # noqa: E711 (explicit presence check)
    assert "bicycle" in traces[0].hazard_episode_key


# --- 12/13/14: mode cycle policy unaffected by the identity correction ---


def test_minimal_still_allows_exactly_two_cycles() -> None:
    policy = make_policy(MINIMAL)
    spoken = _continuous_calls(policy, "left", 0.0, 8.0)
    assert len(spoken) == 2


def test_balanced_still_allows_exactly_two_cycles() -> None:
    policy = make_policy(BALANCED)
    spoken = _continuous_calls(policy, "left", 0.0, 8.0)
    assert len(spoken) == 2


def test_detailed_still_allows_exactly_one_cycle() -> None:
    policy = make_policy(DETAILED)
    spoken = _continuous_calls(policy, "left", 0.0, 8.0)
    assert len(spoken) == 1


# --- 16: no duplicate backlog / overlap introduced ------------------------


def test_finalize_events_never_returns_more_than_one_event_per_input_event() -> None:
    policy = make_policy(MINIMAL)
    events = [
        make_level5_event(region="left", track_id=1, name="Vehicle"),
        make_level5_event(region="right", track_id=2, name="Bicycle"),
    ]
    final, _ = policy.finalize_events(events, now=0.0)
    assert len(final) == 2  # one output per distinct input, never duplicated
