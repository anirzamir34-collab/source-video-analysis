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
    choiceKey: str | None = None
    startTime: float
    endTime: float
    beforeState: dict[str, Any] | None = None
    afterState: dict[str, Any] | None = None
    sourceVerified: bool
    confidence: float
    subjectTrackId: str
    evidence: dict[str, Any] | None = None

    # Optional gameplay metadata. The pose engine may omit these fields; keeping
    # them optional preserves compatibility while allowing richer upstream
    # analyzers to pass source-verified scene structure through this service.
    actionLevel: str | None = None
    actionType: str | None = None
    adultScene: bool = False
    adultSceneId: str | None = None
    adultSceneStartTime: float | None = None
    adultSceneEndTime: float | None = None
    postSceneTime: float | None = None
    positionId: str | None = None
    positionOccurrenceId: str | None = None
    activityType: str | None = None
    positionLabel: str | None = None
    positionStartTime: float | None = None
    positionEndTime: float | None = None
    movementType: str | None = None
    loopStartTime: float | None = None
    loopEndTime: float | None = None
    maleProgressRate: float | None = None
    femaleProgressRate: float | None = None
    outcomeType: str | None = None
    outcomeLabel: str | None = None
    outcomeStartTime: float | None = None
    outcomeEndTime: float | None = None
    outcomeUnlockProgress: float | None = None


class AnalysisResponse(BaseModel):
    available: bool = True
    schemaVersion: int = 4
    engineVersion: str = "source-video-analysis-hardening-v1"
    analysisComplete: bool = True
    analysisCoverage: float = 0.0
    integrity: dict[str, Any] = {}
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
