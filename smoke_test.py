from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

response = client.get("/health")
assert response.status_code == 200, response.text
payload = response.json()
assert payload["status"] == "ok"
assert payload["service"] == "video-analysis"

capabilities = client.get("/capabilities")
assert capabilities.status_code == 200
assert capabilities.json()["external_analysis_configured"] is False

analyze = client.post("/analyze")
assert analyze.status_code == 501

print("Smoke test passed.")
