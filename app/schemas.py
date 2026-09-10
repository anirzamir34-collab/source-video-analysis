from typing import Any
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str


class CapabilityResponse(BaseModel):
    external_analysis_configured: bool
    tracking: bool
    whole_body_pose: bool
    temporal_action_localization: bool
    engine: str
    max_poses: int
    note: str


class ActionItem(BaseModel):
    actionId: str
    label: str
    startTime: float
    endTime: float
    beforeState: dict[str, Any] | None = None
    afterState: dict[str, Any] | None = None
    sourceVerified: bool
    confidence: float
    subjectTrackId: str
    evidence: dict[str, Any] | None = None


class AnalysisResponse(BaseModel):
    available: bool = True
    engine: str
    videoDuration: float
    mainMaleTrackId: str | None
    protagonistSelection: dict[str, Any]
    sampleInterval: float
    sampledFrames: int
    videoPrompt: str
    semanticVideoMap: list[dict[str, Any]]
    actions: list[ActionItem]
    warnings: list[str] = []
