import os

SERVICE_NAME = os.getenv("SERVICE_NAME", "video-analysis")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "0.1.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
