from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

r = client.get('/health')
assert r.status_code == 200, r.text
assert r.json()['status'] == 'ok'

r = client.get('/capabilities')
assert r.status_code == 200, r.text
assert 'external_analysis_configured' in r.json()
print('smoke OK')
