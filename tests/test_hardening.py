from app.hardening import (
    ANALYSIS_ENGINE_VERSION,
    ANALYSIS_SCHEMA_VERSION,
    harden_analysis_result,
)


def _base_result(actions, duration=100.0):
    return {
        "available": True,
        "engine": "test-engine",
        "videoDuration": duration,
        "mainMaleTrackId": "p",
        "protagonistSelection": {},
        "sampleInterval": 1.0,
        "sampledFrames": 10,
        "videoPrompt": "test",
        "semanticVideoMap": [],
        "actions": actions,
        "warnings": [],
    }


def test_hardening_versions_output_and_preserves_valid_actions():
    result = harden_analysis_result(
        _base_result([
            {
                "actionId": "a",
                "label": "A",
                "startTime": 10,
                "endTime": 20,
                "sourceVerified": True,
                "confidence": 1.0,
                "subjectTrackId": "p",
            }
        ])
    )
    assert result["schemaVersion"] == ANALYSIS_SCHEMA_VERSION
    assert result["engineVersion"] == ANALYSIS_ENGINE_VERSION
    assert result["analysisComplete"] is True
    assert [item["actionId"] for item in result["actions"]] == ["a"]
    assert result["integrity"]["droppedActionCount"] == 0


def test_hardening_drops_invalid_times_without_inventing_replacements():
    result = harden_analysis_result(
        _base_result([
            {
                "actionId": "bad",
                "label": "Bad",
                "startTime": 30,
                "endTime": 20,
                "sourceVerified": True,
                "confidence": 1.0,
                "subjectTrackId": "p",
            },
            {
                "actionId": "good",
                "label": "Good",
                "startTime": 40,
                "endTime": 50,
                "sourceVerified": True,
                "confidence": 1.0,
                "subjectTrackId": "p",
            },
        ])
    )
    assert [item["actionId"] for item in result["actions"]] == ["good"]
    assert result["integrity"]["droppedActionCount"] == 1
    assert result["integrity"]["dropped"][0]["actionId"] == "bad"
    assert len(result["actions"]) == 1


def test_segment_hardening_rejects_actions_outside_requested_window():
    result = harden_analysis_result(
        _base_result([
            {
                "actionId": "early",
                "label": "Early",
                "startTime": 5,
                "endTime": 9,
                "sourceVerified": True,
                "confidence": 1.0,
                "subjectTrackId": "p",
            },
            {
                "actionId": "inside",
                "label": "Inside",
                "startTime": 20,
                "endTime": 30,
                "sourceVerified": True,
                "confidence": 1.0,
                "subjectTrackId": "p",
            },
        ])
    , requested_start=15, requested_end=35)
    assert [item["actionId"] for item in result["actions"]] == ["inside"]
    reasons = result["integrity"]["dropped"][0]["reasons"]
    assert "BEFORE_REQUESTED_SEGMENT" in reasons


def test_confidence_is_clamped_but_source_action_is_not_rewritten():
    result = harden_analysis_result(
        _base_result([
            {
                "actionId": "a",
                "label": "A",
                "startTime": 1,
                "endTime": 2,
                "sourceVerified": True,
                "confidence": 2.5,
                "subjectTrackId": "p",
            }
        ], duration=5)
    )
    assert result["actions"][0]["confidence"] == 1.0
    assert result["actions"][0]["label"] == "A"
