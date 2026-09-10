import json
import os
import time

from twelvelabs import TwelveLabs
from twelvelabs.types import AsyncResponseFormat, VideoContext_AssetId


SEGMENT_DEFINITIONS = [
    {
        "id": "protagonist_actions",
        "description": (
            "Choose one recurring male protagonist at the beginning of the video "
            "based on narrative continuity and screen presence, then keep tracking "
            "that exact same person for the entire video. Do not switch to another "
            "person when camera focus changes. Create a new segment whenever the "
            "protagonist's directly visible action, movement, interaction, or location "
            "meaningfully changes. Also create a separate segment whenever the male protagonist's visible tempo, direction, posture, contact point, body orientation or interaction variation changes, even when the base action remains the same. Do not merge visibly different variations into one segment. Keep segments short and precise, preferably between 2 and 15 seconds. If the protagonist is off-screen, do not substitute "
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
                "description": "Describe only the tracked male protagonist's directly visible movement or body position in one concise sentence. This applies to every video genre. Do not summarize the scene and do not describe another person here.",
            },
            {
                "name": "male_choice",
                "type": "string",
                "description": "Create a short Turkish interactive-game choice containing only the tracked male protagonist's directly visible action. Use imperative form, preferably 3 to 10 words. Apply the same rule to normal, action, romantic and adult videos. Mention another adult only when necessary to identify the man's visible movement. Never summarize the scene, switch protagonist or invent an action.",
            },
            {
                "name": "movement_type",
                "type": "string",
                "description": "Return a short reusable category for only the male protagonist's directly visible action, such as sitting, standing, walking, running, turning, holding, kissing, pushing or another literal action visible in the source. Do not describe the scene.",
            },
            {
                "name": "movement_variant",
                "type": "string",
                "description": "State what makes this male action visibly different from nearby male actions: speed change, direction change, posture change, contact change or continuation. Use a short Turkish phrase. Never invent a variation that is not directly visible.",
            },
            {
                "name": "movement_tempo",
                "type": "string",
                "description": "Classify the directly visible speed of the male action.",
                "enum": ["still", "slow", "moderate", "fast", "changing", "unclear"],
            },
            {
                "name": "interaction",
                "type": "string",
                "description": "Describe the directly visible physical interaction in concrete terms: who touches whom, which visible body parts make contact, relative body orientation, and whether proximity is none, close, embracing or intimate. Describe clearly visible adult intimacy literally without euphemisms. If contact is hidden, uncertain or off-screen, state that it is not visible.",
            },
            {
                "name": "male_body_detail",
                "type": "string",
                "description": "Describe the male protagonist's entire visible body literally and precisely: head and gaze, neck, shoulders, chest, back, abdomen, waist, pelvis, hips, buttocks, arms, hands, thighs, knees, lower legs and feet. State the position and movement of every relevant visible region. Do not infer anything hidden by clothing, another body, framing or camera angle.",
            },
            {
                "name": "other_adult_body_detail",
                "type": "string",
                "description": "Describe the other clearly adult person's entire visible body with the same anatomical precision, including body orientation and position relative to the male protagonist. Use neutral anatomical terms. If a region is obscured or outside the frame, explicitly state that it is not visible.",
            },
            {
                "name": "physical_contact_detail",
                "type": "string",
                "description": "Map every directly visible point of physical contact: identify whose body part touches which body part of the other adult, the direction of contact, body orientation and movement. Name lips, mouth, chest, breasts, waist, pelvis, hips, buttocks, thighs, legs or other regions only when the contact is directly visible and unambiguous. Never infer contact through occlusion.",
            },
            {
                "name": "adult_intimacy_detail",
                "type": "string",
                "description": "If clearly adult physical or sexual intimacy is directly visible, describe the visible action literally, anatomically and without euphemisms. Distinguish kissing, embracing, lap-sitting, body pressing, touching and any other directly visible intimate action. If no such action is visible, return none. Never infer intention, arousal, hidden anatomy or an off-screen act.",
            },
            {
                "name": "visibility_limits",
                "type": "string",
                "description": "List important body regions or contact details that cannot be verified because they are covered, obscured, outside the frame or visually ambiguous.",
            },
            {
                "name": "screen_position",
                "type": "string",
                "description": "Where the protagonist appears on screen",
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
        max_segment_duration=15,
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
        male_choice = metadata.get("male_choice")
        movement_type = metadata.get("movement_type") or activity
        movement_variant = metadata.get("movement_variant") or ""
        movement_tempo = metadata.get("movement_tempo") or "unclear"
        choice_key = "|".join(
            str(value).strip().lower()
            for value in (movement_type, movement_variant, movement_tempo)
            if value and str(value).strip()
        )
        male_body_detail = metadata.get("male_body_detail")
        other_adult_body_detail = metadata.get("other_adult_body_detail")
        physical_contact_detail = metadata.get("physical_contact_detail")
        adult_intimacy_detail = metadata.get("adult_intimacy_detail")
        visibility_limits = metadata.get("visibility_limits")
        position = metadata.get("screen_position")
        speaking = metadata.get("speaking")

        choice_label = str(male_choice or activity).strip()

        semantic_map.append(
            {
                "startTime": start,
                "endTime": end,
                "person": person,
                "activity": activity,
                "interaction": interaction,
                "maleChoice": male_choice,
                "choiceKey": choice_key,
                "movementType": movement_type,
                "movementVariant": movement_variant,
                "movementTempo": movement_tempo,
                "maleBodyDetail": male_body_detail,
                "otherAdultBodyDetail": other_adult_body_detail,
                "physicalContactDetail": physical_contact_detail,
                "adultIntimacyDetail": adult_intimacy_detail,
                "visibilityLimits": visibility_limits,
                "screenPosition": position,
                "speaking": speaking,
            }
        )

        actions.append(
            {
                "actionId": f"tl-{index + 1:03d}",
                "label": choice_label,
                "choiceKey": choice_key,
                "startTime": start,
                "endTime": end,
                "beforeState": None,
                "afterState": {
                    "person": person,
                    "interaction": interaction,
                    "maleChoice": male_choice,
                    "choiceKey": choice_key,
                    "movementType": movement_type,
                    "movementVariant": movement_variant,
                    "movementTempo": movement_tempo,
                    "maleBodyDetail": male_body_detail,
                    "otherAdultBodyDetail": other_adult_body_detail,
                    "physicalContactDetail": physical_contact_detail,
                    "adultIntimacyDetail": adult_intimacy_detail,
                    "visibilityLimits": visibility_limits,
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
