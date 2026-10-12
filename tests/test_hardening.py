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


def test_analyze_endpoints_use_local_pose_engine(monkeypatch, tmp_path):
    """Both video endpoints invoke the local pose engine without an external API key."""
    from fastapi.testclient import TestClient
    import app.main as analysis_server

    calls = []

    def fake_save_upload(upload, _max_bytes):
        path = tmp_path / "source.mp4"
        path.write_bytes(upload.file.read())
        return str(path)

    def fake_pose_analysis(path, start_time=None, end_time=None):
        assert path.endswith("source.mp4")
        calls.append((start_time, end_time))
        return _base_result([{
            "actionId": "pose-1",
            "label": "Görünür hareket",
            "startTime": 10,
            "endTime": 12,
            "sourceVerified": True,
            "confidence": 0.8,
            "subjectTrackId": "p",
        }], duration=30)

    monkeypatch.setattr(analysis_server, "save_upload_to_temp", fake_save_upload)
    monkeypatch.setattr(analysis_server, "analyze_video", fake_pose_analysis)

    client = TestClient(analysis_server.app)
    full = client.post("/analyze", files={"video": ("clip.mp4", b"test", "video/mp4")})
    segment = client.post(
        "/analyze-segment",
        data={"startTime": "10", "endTime": "20"},
        files={"video": ("clip.mp4", b"test", "video/mp4")},
    )

    assert full.status_code == 200, full.text
    assert segment.status_code == 200, segment.text
    assert [action["actionId"] for action in full.json()["actions"]] == ["pose-1"]
    assert [action["actionId"] for action in segment.json()["actions"]] == ["pose-1"]
    assert calls == [(None, None), (10.0, 20.0)]
