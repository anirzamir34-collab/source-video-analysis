FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000 \
    POSE_MODEL_PATH=/app/models/pose_landmarker_lite.task

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl libgl1 libegl1 libgles2 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/models && \
    curl -fL --retry 4 --retry-delay 3 \
      "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" \
      -o /app/models/pose_landmarker_lite.task && \
    test -s /app/models/pose_landmarker_lite.task

COPY app ./app

EXPOSE 10000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
