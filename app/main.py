from fastapi import FastAPI, HTTPException

from .config import SERVICE_NAME, SERVICE_VERSION, ENVIRONMENT
from .schemas import HealthResponse, CapabilityResponse

app = FastAPI(
    title="Source Video Interactive - External Analysis Service",
    version=SERVICE_VERSION,
    description=(
        "External analysis backend for the Source Video Interactive project. "
        "This first deployment exposes connectivity and capability checks only. "
        "No fake ML inference is returned."
    ),
)


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": SERVICE_NAME,
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "capabilities": "/capabilities",
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
    return CapabilityResponse(
        external_analysis_configured=False,
        tracking=False,
        whole_body_pose=False,
        temporal_action_localization=False,
        note="Connectivity is working. Real tracker / pose / action models are not installed yet.",
    )


@app.post("/analyze")
def analyze_not_configured():
    raise HTTPException(
        status_code=501,
        detail={
            "available": False,
            "reason": "MODEL_PIPELINE_NOT_CONFIGURED",
            "message": (
                "The external service is reachable, but real video inference has not been installed yet. "
                "No synthetic action data is returned."
            ),
        },
    )


@app.post("/analyze-segment")
def analyze_segment_not_configured():
    raise HTTPException(
        status_code=501,
        detail={
            "available": False,
            "reason": "MODEL_PIPELINE_NOT_CONFIGURED",
            "message": "Segment analysis will be enabled only after the real inference pipeline is installed.",
        },
    )
