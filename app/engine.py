from __future__ import annotations

import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import cv2
import mediapipe as mp
import numpy as np

from .config import MAX_FINE_SAMPLES, MAX_POSES, MIN_POSE_CONFIDENCE, MODEL_PATH


CORE_IDS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]
LEFT_IDS = {"shoulder": 11, "elbow": 13, "wrist": 15, "hip": 23, "knee": 25, "ankle": 27, "foot": 31}
RIGHT_IDS = {"shoulder": 12, "elbow": 14, "wrist": 16, "hip": 24, "knee": 26, "ankle": 28, "foot": 32}


@dataclass
class PoseObs:
    time: float
    track_id: str
    lm: np.ndarray  # [33, 5] x,y,z,visibility,presence
    world: np.ndarray | None
    center: tuple[float, float]
    area: float
    visibility: float
    motion_energy: float = 0.0
    nearest_npc_distance: float | None = None
    nearest_npc_overlap: float | None = None
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class Track:
    track_id: str
    observations: list[PoseObs] = field(default_factory=list)

    @property
    def last(self) -> PoseObs:
        return self.observations[-1]


def _dist(a: np.ndarray | tuple[float, float], b: np.ndarray | tuple[float, float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    den = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if den < 1e-8:
        return 180.0
    cosv = float(np.clip(np.dot(ba, bc) / den, -1.0, 1.0))
    return float(math.degrees(math.acos(cosv)))


def _bbox(lm: np.ndarray) -> tuple[float, float, float, float]:
    valid = lm[:, 3] >= 0.25
    pts = lm[valid, :2] if valid.any() else lm[:, :2]
    x1, y1 = np.min(pts, axis=0)
    x2, y2 = np.max(pts, axis=0)
    return float(x1), float(y1), float(x2), float(y2)


def _bbox_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    amin = min(max(1e-8, (ax2 - ax1) * (ay2 - ay1)), max(1e-8, (bx2 - bx1) * (by2 - by1)))
    return float(inter / amin)


def _pose_arrays(result: Any) -> list[tuple[np.ndarray, np.ndarray | None]]:
    out: list[tuple[np.ndarray, np.ndarray | None]] = []
    pose_landmarks = getattr(result, "pose_landmarks", None) or []
    world_landmarks = getattr(result, "pose_world_landmarks", None) or []
    for i, pose in enumerate(pose_landmarks):
        rows = []
        for p in pose:
            rows.append([
                float(p.x), float(p.y), float(p.z),
                float(getattr(p, "visibility", 1.0) or 0.0),
                float(getattr(p, "presence", 1.0) or 0.0),
            ])
        lm = np.asarray(rows, dtype=np.float32)
        world = None
        if i < len(world_landmarks):
            world = np.asarray([[float(p.x), float(p.y), float(p.z)] for p in world_landmarks[i]], dtype=np.float32)
        out.append((lm, world))
    return out


def _sample_intervals(duration: float) -> tuple[float, float]:
    if duration <= 60:
        return 0.75, 0.25
    if duration <= 180:
        return 1.0, 0.60
    if duration <= 600:
        return 1.5, 0.50
    if duration <= 1800:
        return 2.5, 0.75
    coarse = max(3.0, duration / 720.0)
    fine = max(1.0, duration / MAX_FINE_SAMPLES)
    return coarse, fine


def _read_frame_at(cap: cv2.VideoCapture, t: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
    ok, frame = cap.read()
    return frame if ok else None


def _coarse_motion_windows(path: str, duration: float) -> tuple[list[float], list[tuple[float, float]], dict[float, float]]:
    coarse, fine = _sample_intervals(duration)
    times = list(np.arange(0.0, max(duration, 0.001), coarse, dtype=float))
    if duration > 0 and (not times or duration - times[-1] > coarse * 0.4):
        times.append(max(0.0, duration - 0.001))

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("VIDEO_OPEN_FAILED")

    energies: dict[float, float] = {}
    prev_gray: np.ndarray | None = None
    raw: list[float] = []
    for t in times:
        frame = _read_frame_at(cap, t)
        if frame is None:
            energies[round(t, 3)] = 0.0
            continue
        small = cv2.resize(frame, (192, 108), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if prev_gray is None:
            energy = 0.0
        else:
            diff = cv2.absdiff(gray, prev_gray)
            energy = float(np.mean(diff) / 255.0)
        prev_gray = gray
        energies[round(t, 3)] = energy
        raw.append(energy)
    cap.release()

    nonzero = np.asarray([x for x in raw if x > 0], dtype=np.float32)
    if nonzero.size:
        med = float(np.median(nonzero))
        mad = float(np.median(np.abs(nonzero - med)))
        threshold = max(0.018, med + 0.75 * mad)
    else:
        threshold = 0.018

    windows: list[tuple[float, float]] = []
    for i in range(1, len(times)):
        e = energies.get(round(times[i], 3), 0.0)
        if e >= threshold:
            a = max(0.0, times[i - 1] - coarse * 0.35)
            b = min(duration, times[i] + coarse * 0.35)
            if windows and a <= windows[-1][1] + coarse * 0.5:
                windows[-1] = (windows[-1][0], max(windows[-1][1], b))
            else:
                windows.append((a, b))

    # Always refine a short opening window so initial posture changes are not missed.
    if duration > 0:
        opening = (0.0, min(duration, max(4.0, coarse * 3)))
        windows.insert(0, opening)

    # Merge overlaps.
    merged: list[tuple[float, float]] = []
    for a, b in sorted(windows):
        if merged and a <= merged[-1][1] + 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))

    refine_times: set[float] = set(times)
    desired: list[float] = []
    for a, b in merged:
        desired.extend(np.arange(a, b + 1e-6, fine, dtype=float).tolist())

    if len(desired) > MAX_FINE_SAMPLES:
        stride = len(desired) / MAX_FINE_SAMPLES
        desired = [desired[min(len(desired) - 1, int(i * stride))] for i in range(MAX_FINE_SAMPLES)]

    refine_times.update(round(float(t), 3) for t in desired if 0 <= t < max(duration, 0.001))
    return sorted(refine_times), merged, energies


def _new_pose_obs(time_s: float, track_id: str, lm: np.ndarray, world: np.ndarray | None, motion_energy: float) -> PoseObs:
    x1, y1, x2, y2 = _bbox(lm)
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    area = max(1e-6, (x2 - x1) * (y2 - y1))
    vis = float(np.mean(lm[CORE_IDS, 3]))
    return PoseObs(time=time_s, track_id=track_id, lm=lm, world=world, center=center, area=area, visibility=vis, motion_energy=motion_energy)


def _association_cost(prev: PoseObs, lm: np.ndarray, time_s: float) -> float:
    x1, y1, x2, y2 = _bbox(lm)
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    area = max(1e-6, (x2 - x1) * (y2 - y1))
    center_cost = _dist(prev.center, center)
    area_cost = abs(math.log(max(1e-6, area / prev.area))) * 0.08

    # Relative shoulder/hip geometry helps reduce identity swaps in close interactions.
    ids = [11, 12, 23, 24, 0]
    p = prev.lm[ids, :2]
    q = lm[ids, :2]
    p = p - np.mean(p, axis=0)
    q = q - np.mean(q, axis=0)
    ps = max(1e-4, float(np.linalg.norm(prev.lm[11, :2] - prev.lm[12, :2])))
    qs = max(1e-4, float(np.linalg.norm(lm[11, :2] - lm[12, :2])))
    shape_cost = float(np.mean(np.linalg.norm(p / ps - q / qs, axis=1))) * 0.06

    gap = max(0.0, time_s - prev.time)
    return center_cost + area_cost + shape_cost + min(0.12, gap * 0.02)


def _associate_tracks(frames: list[tuple[float, list[tuple[np.ndarray, np.ndarray | None]], float]]) -> dict[str, Track]:
    tracks: dict[str, Track] = {}
    next_id = 1
    for time_s, detections, motion_energy in frames:
        available = set(tracks.keys())
        candidates: list[tuple[float, str, int]] = []
        for di, (lm, _world) in enumerate(detections):
            for tid in available:
                prev = tracks[tid].last
                if time_s - prev.time <= 5.0:
                    candidates.append((_association_cost(prev, lm, time_s), tid, di))
        candidates.sort(key=lambda x: x[0])
        assigned_t: set[str] = set()
        assigned_d: set[int] = set()
        for cost, tid, di in candidates:
            if tid in assigned_t or di in assigned_d:
                continue
            if cost > 0.38:
                continue
            lm, world = detections[di]
            obs = _new_pose_obs(time_s, tid, lm, world, motion_energy)
            tracks[tid].observations.append(obs)
            assigned_t.add(tid)
            assigned_d.add(di)

        for di, (lm, world) in enumerate(detections):
            if di in assigned_d:
                continue
            tid = f"TRACK_{next_id:02d}"
            next_id += 1
            tracks[tid] = Track(track_id=tid, observations=[_new_pose_obs(time_s, tid, lm, world, motion_energy)])
    return tracks


def _track_score(track: Track, duration: float) -> float:
    if not track.observations:
        return -1.0
    obs = track.observations
    covered = max(0.0, obs[-1].time - obs[0].time)
    persistence = min(1.0, covered / max(duration, 1.0))
    count_factor = min(1.0, len(obs) / max(12.0, duration / 1.5))
    area = float(np.median([o.area for o in obs]))
    centrality = float(np.mean([1.0 - min(1.0, _dist(o.center, (0.5, 0.5)) / 0.72) for o in obs]))
    visibility = float(np.mean([o.visibility for o in obs]))
    return 0.34 * persistence + 0.22 * count_factor + 0.20 * min(1.0, area / 0.20) + 0.12 * centrality + 0.12 * visibility


def _body_scale(lm: np.ndarray) -> float:
    shoulder = _mid(lm[11, :2], lm[12, :2])
    hip = _mid(lm[23, :2], lm[24, :2])
    return max(0.04, _dist(shoulder, hip))


def _state_for(obs: PoseObs) -> dict[str, Any]:
    lm = obs.lm
    scale = _body_scale(lm)
    shoulder = _mid(lm[11, :2], lm[12, :2])
    hip = _mid(lm[23, :2], lm[24, :2])
    knee = _mid(lm[25, :2], lm[26, :2])
    ankle = _mid(lm[27, :2], lm[28, :2])
    nose = lm[0, :2]

    body_vec = hip - shoulder
    from_vertical = abs(math.degrees(math.atan2(float(body_vec[0]), float(body_vec[1]) + 1e-8)))
    left_knee = _angle(lm[23], lm[25], lm[27])
    right_knee = _angle(lm[24], lm[26], lm[28])
    knee_angle = (left_knee + right_knee) / 2.0

    if from_vertical > 62:
        posture = "lying"
    elif knee_angle < 112:
        posture = "crouched"
    elif knee_angle < 150 and abs(float(knee[1] - hip[1])) < scale * 1.25:
        posture = "seated"
    else:
        posture = "standing"

    shoulder_w = max(0.03, _dist(lm[11, :2], lm[12, :2]))
    head_offset = float((nose[0] - shoulder[0]) / shoulder_w)
    if head_offset < -0.24:
        head_dir = "frame_left"
    elif head_offset > 0.24:
        head_dir = "frame_right"
    else:
        head_dir = "center"

    torso_dx = float((shoulder[0] - hip[0]) / scale)
    if torso_dx < -0.22:
        lean = "frame_left"
    elif torso_dx > 0.22:
        lean = "frame_right"
    else:
        lean = "center"

    chest = shoulder * 0.58 + hip * 0.42
    def side_state(ids: dict[str, int]) -> dict[str, Any]:
        wrist = lm[ids["wrist"], :2]
        sh = lm[ids["shoulder"], :2]
        el = lm[ids["elbow"], :2]
        return {
            "arm_up": bool(wrist[1] < sh[1] - scale * 0.12),
            "arm_extended": bool(_angle(sh, el, wrist) > 150 and _dist(wrist, sh) > scale * 0.95),
            "hand_face": bool(_dist(wrist, nose) < scale * 0.58),
            "hand_torso": bool(_dist(wrist, chest) < scale * 0.52),
        }

    return {
        "posture": posture,
        "head": head_dir,
        "lean": lean,
        "left": side_state(LEFT_IDS),
        "right": side_state(RIGHT_IDS),
        "center": [round(float(hip[0]), 4), round(float(hip[1]), 4)],
        "bodyScale": round(scale, 4),
        "nearestNpcDistance": None if obs.nearest_npc_distance is None else round(obs.nearest_npc_distance, 4),
        "npcOverlap": None if obs.nearest_npc_overlap is None else round(obs.nearest_npc_overlap, 4),
    }


def _enrich_npc_proximity(tracks: dict[str, Track], protagonist_id: str) -> None:
    protagonist = tracks[protagonist_id]
    by_time: dict[float, list[PoseObs]] = {}
    for tid, track in tracks.items():
        if tid == protagonist_id:
            continue
        for o in track.observations:
            by_time.setdefault(round(o.time, 3), []).append(o)

    for o in protagonist.observations:
        others = by_time.get(round(o.time, 3), [])
        if not others:
            continue
        scale = _body_scale(o.lm)
        nearest = min(others, key=lambda x: _dist(o.center, x.center))
        o.nearest_npc_distance = _dist(o.center, nearest.center) / max(scale, 1e-4)
        o.nearest_npc_overlap = _bbox_overlap(_bbox(o.lm), _bbox(nearest.lm))


def _mean_landmark_motion(a: PoseObs, b: PoseObs, ids: Iterable[int]) -> float:
    ids = list(ids)
    scale = max(0.04, (_body_scale(a.lm) + _body_scale(b.lm)) / 2.0)
    d = np.linalg.norm(b.lm[ids, :2] - a.lm[ids, :2], axis=1)
    return float(np.mean(d) / scale)


def _event(label: str, a: PoseObs, b: PoseObs, confidence: float, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label,
        "start": float(a.time),
        "end": float(b.time),
        "confidence": float(max(0.01, min(0.99, confidence))),
        "before": a.state,
        "after": b.state,
        "evidence": evidence,
    }


def _events_between(a: PoseObs, b: PoseObs) -> list[dict[str, Any]]:
    dt = max(0.05, b.time - a.time)
    conf = min(a.visibility, b.visibility)
    out: list[dict[str, Any]] = []
    sa, sb = a.state, b.state

    posture_map = {
        ("standing", "seated"): "Otur",
        ("seated", "standing"): "Ayağa kalk",
        ("standing", "crouched"): "Çömel",
        ("seated", "crouched"): "Çömel",
        ("crouched", "standing"): "Doğrul",
        ("crouched", "seated"): "Oturur pozisyona geç",
        ("standing", "lying"): "Yatay pozisyona geç",
        ("seated", "lying"): "Uzan",
        ("lying", "standing"): "Yatay pozisyondan ayağa kalk",
        ("lying", "seated"): "Yatay pozisyondan otur",
    }
    if sa["posture"] != sb["posture"]:
        label = posture_map.get((sa["posture"], sb["posture"]), "Beden pozisyonunu değiştir")
        out.append(_event(label, a, b, conf, {"type": "posture_transition", "from": sa["posture"], "to": sb["posture"]}))

    for side, tr in (("left", "Sol"), ("right", "Sağ")):
        aa, bb = sa[side], sb[side]
        transitions = [
            ("arm_up", f"{tr} kolunu kaldır", f"{tr} kolunu indir"),
            ("hand_face", f"{tr} elini yüzüne götür", f"{tr} elini yüzünden uzaklaştır"),
            ("hand_torso", f"{tr} elini gövdesine götür", f"{tr} elini gövdesinden uzaklaştır"),
            ("arm_extended", f"{tr} kolunu uzat", f"{tr} kolunu geri çek"),
        ]
        for key, on_label, off_label in transitions:
            if aa[key] != bb[key]:
                out.append(_event(on_label if bb[key] else off_label, a, b, conf * 0.96, {"type": key, "side": side, "to": bool(bb[key])}))

    if sa["head"] != sb["head"] and sb["head"] != "center":
        label = "Başını kadrajın soluna çevir" if sb["head"] == "frame_left" else "Başını kadrajın sağına çevir"
        out.append(_event(label, a, b, conf * 0.84, {"type": "head_orientation", "to": sb["head"]}))

    if sa["lean"] != sb["lean"] and sb["lean"] != "center":
        label = "Gövdesini kadrajın soluna eğ" if sb["lean"] == "frame_left" else "Gövdesini kadrajın sağına eğ"
        out.append(_event(label, a, b, conf * 0.82, {"type": "torso_lean", "to": sb["lean"]}))

    scale = max(0.04, (float(sa["bodyScale"]) + float(sb["bodyScale"])) / 2.0)
    dx = (float(sb["center"][0]) - float(sa["center"][0])) / scale
    center_speed = abs(dx) / dt
    feet_motion = _mean_landmark_motion(a, b, [27, 28, 29, 30, 31, 32]) / dt
    body_motion = _mean_landmark_motion(a, b, CORE_IDS) / dt
    left_hand_motion = _mean_landmark_motion(a, b, [13, 15, 17, 19, 21]) / dt
    right_hand_motion = _mean_landmark_motion(a, b, [14, 16, 18, 20, 22]) / dt
    head_motion = _mean_landmark_motion(a, b, [0, 7, 8, 9, 10]) / dt

    if center_speed > 0.18 and feet_motion > 0.11:
        if dx < -0.08:
            label = "Kadrajda sola ilerle"
        elif dx > 0.08:
            label = "Kadrajda sağa ilerle"
        else:
            label = "Yürü"
        out.append(_event(label, a, b, conf * min(1.0, 0.65 + center_speed), {"type": "locomotion", "dxBodyScale": round(dx, 3), "feetMotion": round(feet_motion, 3)}))

    specific_left = any(e["evidence"].get("side") == "left" for e in out)
    specific_right = any(e["evidence"].get("side") == "right" for e in out)
    if left_hand_motion > 0.48 and not specific_left:
        out.append(_event("Sol el/kol hareketi yap", a, b, conf * 0.72, {"type": "limb_motion", "side": "left", "motion": round(left_hand_motion, 3)}))
    if right_hand_motion > 0.48 and not specific_right:
        out.append(_event("Sağ el/kol hareketi yap", a, b, conf * 0.72, {"type": "limb_motion", "side": "right", "motion": round(right_hand_motion, 3)}))
    if head_motion > 0.40 and not any(e["evidence"].get("type") == "head_orientation" for e in out):
        out.append(_event("Başını hareket ettir", a, b, conf * 0.68, {"type": "head_motion", "motion": round(head_motion, 3)}))

    da, db = sa.get("nearestNpcDistance"), sb.get("nearestNpcDistance")
    oa, ob = sa.get("npcOverlap"), sb.get("npcOverlap")
    if da is not None and db is not None:
        delta = float(db) - float(da)
        if delta < -0.45:
            out.append(_event("Diğer karaktere yaklaş", a, b, conf * 0.78, {"type": "npc_distance", "delta": round(delta, 3)}))
        elif delta > 0.55:
            out.append(_event("Diğer karakterden uzaklaş", a, b, conf * 0.76, {"type": "npc_distance", "delta": round(delta, 3)}))
    if oa is not None and ob is not None and float(oa) < 0.18 <= float(ob):
        out.append(_event("Diğer karakterle fiziksel temasa gir", a, b, conf * 0.76, {"type": "npc_overlap", "overlap": round(float(ob), 3)}))

    if body_motion > 0.55 and not out:
        out.append(_event("Beden pozisyonunu değiştir", a, b, conf * 0.62, {"type": "whole_body_motion", "motion": round(body_motion, 3)}))

    return out


def _merge_events(events: list[dict[str, Any]], track_id: str, duration: float) -> list[dict[str, Any]]:
    if not events:
        return []
    events.sort(key=lambda x: (x["start"], x["end"], x["label"]))
    merged: list[dict[str, Any]] = []
    for e in events:
        e["start"] = max(0.0, min(duration, e["start"]))
        e["end"] = max(e["start"] + 0.05, min(duration, e["end"] + 0.08))
        if merged and e["label"] == merged[-1]["label"] and e["start"] <= merged[-1]["end"] + 0.55:
            merged[-1]["end"] = max(merged[-1]["end"], e["end"])
            merged[-1]["confidence"] = max(merged[-1]["confidence"], e["confidence"])
            merged[-1]["after"] = e["after"]
            merged[-1]["evidence"] = e["evidence"]
        else:
            merged.append(dict(e))

    # Remove extremely short low-confidence jitter and near-duplicate actions.
    cleaned: list[dict[str, Any]] = []
    for e in merged:
        if e["confidence"] < 0.48:
            continue
        if cleaned and e["label"] == cleaned[-1]["label"] and e["start"] - cleaned[-1]["end"] < 0.35:
            cleaned[-1]["end"] = max(cleaned[-1]["end"], e["end"])
            continue
        cleaned.append(e)

    max_actions = min(250, max(24, int(max(duration, 1.0) / 3.5)))
    if len(cleaned) > max_actions:
        # Preserve chronology and strongest events instead of hard-capping the first N.
        ranked = sorted(range(len(cleaned)), key=lambda i: cleaned[i]["confidence"], reverse=True)[:max_actions]
        cleaned = [cleaned[i] for i in sorted(ranked)]

    out: list[dict[str, Any]] = []
    for i, e in enumerate(cleaned, 1):
        out.append({
            "actionId": f"ACTION_{i:04d}",
            "label": e["label"],
            "startTime": round(float(e["start"]), 3),
            "endTime": round(float(e["end"]), 3),
            "beforeState": e["before"],
            "afterState": e["after"],
            "sourceVerified": True,
            "confidence": round(float(e["confidence"]), 3),
            "subjectTrackId": track_id,
            "evidence": e["evidence"],
        })
    return out


def _video_prompt(duration: float, track_id: str, obs: list[PoseObs], actions: list[dict[str, Any]]) -> str:
    if not obs:
        return "Ana karakter için doğrulanmış poz verisi bulunamadı."
    lines = [
        f"Video süresi {duration:.2f} sn. Ana odak {track_id}. Bu açıklama yalnızca ölçülen beden/poz hareketlerinden oluşturuldu; sahne nesneleri veya diyalog uydurulmadı.",
        f"Başlangıç durumu: {obs[0].state.get('posture', 'unknown')}.",
    ]
    for a in actions[:80]:
        lines.append(f"{a['startTime']:.2f}-{a['endTime']:.2f} sn: {a['label']}.")
    if len(actions) > 80:
        lines.append(f"... devamında {len(actions) - 80} doğrulanmış hareket daha var.")
    return "\n".join(lines)


def analyze_video(path: str, start_time: float | None = None, end_time: float | None = None) -> dict[str, Any]:
    started = time.time()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("VIDEO_OPEN_FAILED")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    duration = frame_count / fps if fps > 0 and frame_count > 0 else float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
    cap.release()
    if duration <= 0:
        raise ValueError("VIDEO_DURATION_UNKNOWN")

    seg_start = max(0.0, float(start_time or 0.0))
    seg_end = min(duration, float(end_time)) if end_time is not None else duration
    if seg_end <= seg_start:
        raise ValueError("INVALID_SEGMENT_RANGE")

    effective_duration = seg_end - seg_start
    sample_times_rel, motion_windows_rel, coarse_energy_rel = _coarse_motion_windows(path, duration)
    sample_times = [t for t in sample_times_rel if seg_start <= t <= seg_end]
    if not sample_times:
        sample_times = [seg_start, max(seg_start, seg_end - 0.001)]

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"POSE_MODEL_MISSING:{MODEL_PATH}")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=MAX_POSES,
        min_pose_detection_confidence=MIN_POSE_CONFIDENCE,
        min_pose_presence_confidence=MIN_POSE_CONFIDENCE,
        min_tracking_confidence=0.40,
        output_segmentation_masks=False,
    )

    cap = cv2.VideoCapture(path)
    frames: list[tuple[float, list[tuple[np.ndarray, np.ndarray | None]], float]] = []
    with PoseLandmarker.create_from_options(options) as landmarker:
        for idx, t in enumerate(sample_times):
            frame = _read_frame_at(cap, t)
            if frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            result = landmarker.detect_for_video(mp_image, int(round(t * 1000.0)))
            detections = _pose_arrays(result)
            energy = coarse_energy_rel.get(round(t, 3), 0.0)
            frames.append((float(t), detections, float(energy)))
            if idx and idx % 75 == 0:
                print(f"[analysis] pose samples {idx}/{len(sample_times)}", flush=True)
    cap.release()

    tracks = _associate_tracks(frames)
    tracks = {tid: tr for tid, tr in tracks.items() if len(tr.observations) >= 3}
    if not tracks:
        raise LookupError("NO_POSE_DETECTED")

    protagonist_id = max(tracks, key=lambda tid: _track_score(tracks[tid], effective_duration))
    protagonist = tracks[protagonist_id]
    _enrich_npc_proximity(tracks, protagonist_id)
    for o in protagonist.observations:
        o.state = _state_for(o)

    events: list[dict[str, Any]] = []
    for a, b in zip(protagonist.observations, protagonist.observations[1:]):
        if b.time <= a.time or b.time - a.time > 4.5:
            continue
        events.extend(_events_between(a, b))

    actions = _merge_events(events, protagonist_id, duration)

    semantic_map: list[dict[str, Any]] = []
    # Keep enough temporal detail for debugging without returning thousands of rows.
    step = max(1, int(math.ceil(len(protagonist.observations) / 320)))
    for o in protagonist.observations[::step]:
        semantic_map.append({
            "time": round(o.time, 3),
            "trackId": o.track_id,
            "posture": o.state.get("posture"),
            "head": o.state.get("head"),
            "lean": o.state.get("lean"),
            "center": o.state.get("center"),
            "visibility": round(o.visibility, 3),
            "nearestNpcDistance": o.state.get("nearestNpcDistance"),
            "npcOverlap": o.state.get("npcOverlap"),
        })

    selected_score = _track_score(protagonist, effective_duration)
    warnings = [
        "mainMaleTrackId alanı uygulama uyumluluğu için kullanılır; motor cinsiyet tahmini yapmaz. Ana karakter, zamansal süreklilik/boyut/merkezilik puanıyla seçilir.",
        "Bu sürüm sahne nesnesi veya diyalog tanımaz; yalnızca gözlenen beden pozu, hareket ve kişiler arası yakınlık/örtüşme verilerini kullanır.",
    ]
    if len(tracks) > 1:
        warnings.append(f"Videoda {len(tracks)} kalıcı pose track bulundu; ana odak otomatik olarak {protagonist_id} seçildi.")

    coarse, fine = _sample_intervals(duration)
    return {
        "available": True,
        "engine": "mediapipe-pose-landmarker-lite+temporal-geometry-v2",
        "videoDuration": round(duration, 3),
        "mainMaleTrackId": protagonist_id,
        "protagonistSelection": {
            "trackId": protagonist_id,
            "method": "persistence+visible_area+centrality+pose_visibility",
            "score": round(selected_score, 3),
            "genderInferenceUsed": False,
            "trackCount": len(tracks),
        },
        "sampleInterval": round(fine, 3),
        "sampledFrames": len(frames),
        "motionWindows": [[round(a, 3), round(b, 3)] for a, b in motion_windows_rel if b >= seg_start and a <= seg_end],
        "processingSeconds": round(time.time() - started, 2),
        "videoPrompt": _video_prompt(duration, protagonist_id, protagonist.observations, actions),
        "semanticVideoMap": semantic_map,
        "actions": actions,
        "warnings": warnings,
    }


def save_upload_to_temp(upload_file: Any, max_bytes: int) -> str:
    suffix = os.path.splitext(getattr(upload_file, "filename", "video.mp4") or "video.mp4")[1] or ".mp4"
    total = 0
    tmp = tempfile.NamedTemporaryFile(prefix="svi_", suffix=suffix, delete=False)
    try:
        while True:
            chunk = upload_file.file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("VIDEO_TOO_LARGE")
            tmp.write(chunk)
        tmp.flush()
        return tmp.name
    except Exception:
        tmp.close()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise
    finally:
        tmp.close()
