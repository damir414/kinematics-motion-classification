import sys
import joblib
from fastapi.testclient import TestClient


class MockModel:
    def predict(self, X):
        return [1]


def test_infer_integration(monkeypatch):
    # Подготовка данных
    sys.modules.pop('infer', None)
    monkeypatch.setattr(joblib, 'load', lambda path: MockModel())
    import infer
    client = TestClient(infer.app)
    resp = client.post('/infer', json={
        'wrist': 0,
        'acceleration_x': 0,
        'acceleration_y': 0,
        'acceleration_z': 0,
        'gyro_x': 0,
        'gyro_y': 0,
        'gyro_z': 0,
    })
    
    # Тест
    assert resp.status_code == 200
    assert resp.json() == {'activity': 1}