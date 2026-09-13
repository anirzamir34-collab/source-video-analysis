from __future__ import annotations

import math
from typing import Any


ANALYSIS_SCHEMA_VERSION = 4
ANALYSIS_ENGINE_VERSION = "source-video-analysis-hardening-v1"


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def harden_analysis_result(result: dict[str, Any], *, requested_start: float | None = None, requested_end: float | None = None) -> dict[str, Any]:
    source = dict(result or {})
    duration = max(0.0, float(source.get("videoDuration") or 0.0))
    actions = list(source.get("actions") or [])
    accepted: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for index, item in enumerate(actions):
        action = dict(item or {})
        action_id = str(action.get("actionId") or f"action-{index + 1}")
        start = action.get("startTime")
        end = action.get("endTime")
        reasons: list[str] = []

        if not _finite(start) or not _finite(end):
            reasons.append("NON_FINITE_TIME")
        else:
            start_value = float(start)
            end_value = float(end)
            if start_value < 0:
                reasons.append("NEGATIVE_START")
            if end_value <= start_value:
                reasons.append("NON_POSITIVE_DURATION")
            if duration and end_value > duration + 0.5:
                reasons.append("OUTSIDE_VIDEO")
            if requested_start is not None and start_value < float(requested_start) - 0.5:
                reasons.append("BEFORE_REQUESTED_SEGMENT")
            if requested_end is not None and end_value > float(requested_end) + 0.5:
                reasons.append("AFTER_REQUESTED_SEGMENT")

        confidence = action.get("confidence")
        if confidence is not None and _finite(confidence):
            action["confidence"] = max(0.0, min(1.0, float(confidence)))

        if reasons:
            dropped.append({"actionId": action_id, "reasons": reasons})
            continue

        action["actionId"] = action_id
        action["sourceVerified"] = bool(action.get("sourceVerified", True))
        accepted.append(action)

    accepted.sort(key=lambda action: float(action.get("startTime") or 0.0))

    span_start = float(accepted[0]["startTime"]) if accepted else 0.0
    span_end = max((float(action["endTime"]) for action in accepted), default=0.0)
    requested_span_start = float(requested_start) if requested_start is not None else 0.0
    requested_span_end = float(requested_end) if requested_end is not None else duration
    requested_span = max(0.0, requested_span_end - requested_span_start)
    observed_span = max(0.0, span_end - max(span_start, requested_span_start))
    coverage = min(1.0, observed_span / requested_span) if requested_span > 0 else 0.0

    warnings = list(source.get("warnings") or [])
    if dropped:
        warnings.append(f"Integrity hardening dropped {len(dropped)} invalid action(s); no replacement actions were invented.")

    source.update(
        {
            "schemaVersion": ANALYSIS_SCHEMA_VERSION,
            "engineVersion": ANALYSIS_ENGINE_VERSION,
            "analysisComplete": True,
            "analysisCoverage": round(coverage, 3),
            "actions": accepted,
            "warnings": warnings,
            "integrity": {
                "valid": True,
                "reviewed": True,
                "inputActionCount": len(actions),
                "acceptedActionCount": len(accepted),
                "droppedActionCount": len(dropped),
                "dropped": dropped,
                "requestedStartTime": requested_start,
                "requestedEndTime": requested_end,
            },
        }
    )
    return source
