from fastapi.testclient import TestClient

import app as app_module

client = TestClient(app_module.app)
VALID = {"feature_1": 1.5, "feature_2": 2.0, "feature_3": 3.1, "feature_4": 0.5}


def test_predict_ok():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["prediction"] in (0, 1)


def test_missing_field_rejected():
    r = client.post("/predict", json={"feature_1": 1.0})
    assert r.status_code == 422


def test_bad_type_rejected():
    r = client.post("/predict", json={**VALID, "feature_1": "abc"})
    assert r.status_code == 422


def test_model_missing_returns_500(monkeypatch):
    monkeypatch.setattr(app_module, "model", None)
    r = client.post("/predict", json=VALID)
    assert r.status_code == 500
    assert "Model file not found" in r.json()["detail"]
