from __future__ import annotations

from .twelvelabs_engine import analyze_video_twelvelabs

import os
import traceback
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .config import ENVIRONMENT, MAX_POSES, MAX_UPLOAD_MB, MODEL_PATH, SERVICE_NAME, SERVICE_VERSION
from .engine import save_upload_to_temp
from .schemas import AnalysisResponse, CapabilityResponse, HealthResponse


app = FastAPI(
    title="Source Video Interactive - External Analysis Service",
    version=SERVICE_VERSION,
    description=(
        "Real source-video motion analysis service. Uses MediaPipe Pose Landmarker Lite, "
        "multi-pose temporal association, posture/limb geometry and verified timestamps. "
        "It does not synthesize missing actions."
    ),
)


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": SERVICE_NAME,
        "status": "ok",
        "version": SERVICE_VERSION,
        "docs": "/docs",
        "health": "/health",
        "capabilities": "/capabilities",
        "analyze": "/analyze",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        environment=ENVIRONMENT,
    )


@app.get("/capabilities", response_model=CapabilityResponse)
def capabilities():
    model_ready = os.path.exists(MODEL_PATH)
    return CapabilityResponse(
        external_analysis_configured=model_ready,
        tracking=model_ready,
        whole_body_pose=model_ready,
        temporal_action_localization=model_ready,
        engine="MediaPipe Pose Landmarker Lite + multi-pose temporal geometry",
        max_poses=MAX_POSES,
        note=(
            "Real pose/tracking/action timing is enabled. The engine uses 33 pose landmarks and "
            "does not invent scene objects or dialogue. Protagonist selection is based on persistence/size/centrality, not gender inference."
            if model_ready else
            f"Pose model is missing at {MODEL_PATH}. No fake inference will be returned."
        ),
    )


def _raise_analysis_error(exc: Exception) -> None:
    reason = str(exc)
    if isinstance(exc, LookupError) or reason == "NO_POSE_DETECTED":
        status = 422
        message = "Videoda yeterli sürekliliğe sahip insan pozu algılanamadı. Sahte hareket üretilmedi."
    elif reason == "VIDEO_TOO_LARGE":
        status = 413
        message = f"Video {MAX_UPLOAD_MB} MB yükleme sınırını aşıyor."
    elif reason in {"VIDEO_OPEN_FAILED", "VIDEO_DURATION_UNKNOWN", "INVALID_SEGMENT_RANGE"}:
        status = 422
        message = "Video açılamadı veya zaman aralığı geçersiz."
    elif reason.startswith("POSE_MODEL_MISSING"):
        status = 503
        message = "Pose modeli sunucuda bulunamadı; inference çalıştırılmadı."
    else:
        status = 500
        message = "Gerçek analiz pipeline'ı çalışırken hata oluştu."
    print(f"[analysis-error] {type(exc).__name__}: {exc}", flush=True)
    traceback.print_exc()
    raise HTTPException(status_code=status, detail={"available": False, "reason": reason, "message": message})


@app.post("/analyze", response_model=AnalysisResponse)
def analyze(video: UploadFile = File(...)):
    path = None
    try:
        print(f"[analyze] received name={video.filename} type={video.content_type}", flush=True)
        path = save_upload_to_temp(video, MAX_UPLOAD_MB * 1024 * 1024)
        result = analyze_video(path)
        print(f"[analyze] complete actions={len(result['actions'])} frames={result['sampledFrames']} seconds={result['processingSeconds']}", flush=True)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        _raise_analysis_error(exc)
    finally:
        try:
            video.file.close()
        except Exception:
            pass
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


@app.post("/analyze-segment", response_model=AnalysisResponse)
def analyze_segment(
    video: UploadFile = File(...),
    startTime: float | None = Form(default=None),
    endTime: float | None = Form(default=None),
):
    path = None
    try:
        path = save_upload_to_temp(video, MAX_UPLOAD_MB * 1024 * 1024)
        result = analyze_video_twelvelabs(path, start_time=startTime, end_time=endTime)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        _raise_analysis_error(exc)
    finally:
        try:
            video.file.close()
        except Exception:
            pass
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass
