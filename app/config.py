import os

SERVICE_NAME = os.getenv("SERVICE_NAME", "video-analysis")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "0.2.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

MODEL_PATH = os.getenv("POSE_MODEL_PATH", "/app/models/pose_landmarker_lite.task")
MAX_POSES = max(1, min(6, int(os.getenv("MAX_POSES", "4"))))
MAX_FINE_SAMPLES = max(120, int(os.getenv("MAX_FINE_SAMPLES", "900")))
MAX_UPLOAD_MB = max(10, int(os.getenv("MAX_UPLOAD_MB", "250")))
MIN_POSE_CONFIDENCE = float(os.getenv("MIN_POSE_CONFIDENCE", "0.45"))
