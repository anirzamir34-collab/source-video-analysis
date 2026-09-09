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
    note: str
