import json
import os
import time

from twelvelabs import TwelveLabs
from twelvelabs.types import AsyncResponseFormat, VideoContext_AssetId


SEGMENT_DEFINITIONS = [
    {
        "id": "protagonist_actions",
        "description": (
            "Choose one primary recurring protagonist at the beginning of the video "
            "based on narrative continuity and screen presence, then keep tracking "
            "that exact same person for the entire video. Do not switch to another "
            "person when camera focus changes. Create a new segment whenever the "
            "protagonist's directly visible action, movement, interaction, or location "
            "meaningfully changes. If the protagonist is off-screen, do not substitute "
            "another person. Describe only events directly visible in the source video. "
            "Do not infer intentions or invent actions, objects, dialogue, outcomes, "
            "or off-screen events."
        ),
        "fields": [
            {
                "name": "person",
                "type": "string",
                "description": "Visible description of the tracked protagonist",
            },
            {
                "name": "activity",
                "type": "string",
                "description": "The protagonist's directly visible action",
            },
            {
                "name": "interaction",
                "type": "string",
                "description": "Visible interaction with a person or object, or none",
            },
            {
                "name": "screen_position",
                "type": "string",
                "enum": ["center", "left", "right", "background", "off_screen"],
            },
            {
                "name": "speaking",
                "type": "boolean",
                "description": "Whether the protagonist is visibly speaking",
            },
        ],
    }
]


def _wait_for_asset(client, asset_id, timeout=600):
    started = time.time()

    while time.time() - started < timeout:
        asset = client.assets.retrieve(asset_id)

        if asset.status == "ready":
            return asset

        if asset.status == "failed":
            raise RuntimeError("TWELVELABS_ASSET_PROCESSING_FAILED")

        time.sleep(3)

    raise TimeoutError("TWELVELABS_ASSET_TIMEOUT")


def _wait_for_task(client, task_id, timeout=900):
    started = time.time()

    while time.time() - started < timeout:
        task = client.analyze_async.tasks.retrieve(task_id)

        if task.status == "ready":
            return task

        if task.status == "failed":
            raise RuntimeError("TWELVELABS_ANALYSIS_FAILED")

        time.sleep(3)

    raise TimeoutError("TWELVELABS_ANALYSIS_TIMEOUT")


def analyze_video_twelvelabs(path, start_time=None, end_time=None):
    api_key = os.environ.get("TWELVELABS_API_KEY")

    if not api_key:
        raise RuntimeError("TWELVELABS_API_KEY_NOT_CONFIGURED")

    client = TwelveLabs(api_key=api_key)

    print("[twelvelabs] uploading asset", flush=True)

    with open(path, "rb") as video_file:
        asset = client.assets.create(
            method="direct",
            file=video_file,
        )

    asset = _wait_for_asset(client, asset.id)

    print(f"[twelvelabs] asset ready: {asset.id}", flush=True)

    task_args = {}

    if start_time is not None:
        task_args["start_time"] = float(start_time)

    if end_time is not None:
        task_args["end_time"] = float(end_time)

    task = client.analyze_async.tasks.create(
        video=VideoContext_AssetId(asset_id=asset.id),
        model_name="pegasus1.5",
        analysis_mode="time_based_metadata",
        response_format=AsyncResponseFormat(
            type="segment_definitions",
            segment_definitions=SEGMENT_DEFINITIONS,
        ),
        min_segment_duration=2,
        max_segment_duration=20,
        **task_args,
    )

    print(f"[twelvelabs] task created: {task.task_id}", flush=True)

    task = _wait_for_task(client, task.task_id)

    if not task.result or not task.result.data:
        raise RuntimeError("TWELVELABS_EMPTY_RESULT")

    parsed = json.loads(task.result.data)
    segments = parsed.get("protagonist_actions", [])

    if not segments:
        raise RuntimeError("TWELVELABS_NO_PROTAGONIST_SEGMENTS")

    semantic_map = []
    actions = []

    for index, segment in enumerate(segments):
        start = float(segment.get("start_time", 0))
        end = float(segment.get("end_time", start))
        metadata = segment.get("metadata") or {}

        activity = metadata.get("activity") or "Visible protagonist action"
        person = metadata.get("person")
        interaction = metadata.get("interaction")
        position = metadata.get("screen_position")
        speaking = metadata.get("speaking")

        semantic_map.append(
            {
                "startTime": start,
                "endTime": end,
                "person": person,
                "activity": activity,
                "interaction": interaction,
                "screenPosition": position,
                "speaking": speaking,
            }
        )

        actions.append(
            {
                "actionId": f"tl-{index + 1:03d}",
                "label": activity,
                "startTime": start,
                "endTime": end,
                "beforeState": None,
                "afterState": {
                    "person": person,
                    "interaction": interaction,
                    "screenPosition": position,
                    "speaking": speaking,
                },
                "sourceVerified": True,
                "confidence": 1.0,
                "subjectTrackId": "twelvelabs-primary",
                "evidence": {
                    "provider": "TwelveLabs",
                    "model": "pegasus1.5",
                    "segmentIndex": index,
                },
            }
        )

    duration = max(float(x.get("end_time", 0)) for x in segments)

    first_metadata = segments[0].get("metadata") or {}
    protagonist = first_metadata.get("person") or "Primary recurring protagonist"

    video_prompt = "\n".join(
        f"{float(x.get('start_time', 0)):.1f}-{float(x.get('end_time', 0)):.1f}s: "
        f"{(x.get('metadata') or {}).get('activity', '')}"
        for x in segments
    )

    return {
        "available": True,
        "engine": "TwelveLabs Pegasus 1.5",
        "videoDuration": duration,
        "mainMaleTrackId": "twelvelabs-primary",
        "protagonistSelection": {
            "method": "primary recurring protagonist with identity lock",
            "person": protagonist,
            "provider": "TwelveLabs",
        },
        "sampleInterval": 0.0,
        "sampledFrames": len(segments),
        "videoPrompt": video_prompt,
        "semanticVideoMap": semantic_map,
        "actions": actions,
        "warnings": [],
    }
