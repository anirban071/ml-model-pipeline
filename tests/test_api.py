import pytest
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


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_health_503_when_model_missing(monkeypatch):
    monkeypatch.setattr(app_module, "model", None)
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["detail"] == "Model not loaded"


@pytest.mark.parametrize("bad_value", ["NaN", "inf", "-inf", 1e308, -1e308])
def test_non_finite_or_huge_value_rejected(bad_value):
    r = client.post("/predict", json={**VALID, "feature_1": bad_value})
    assert r.status_code == 422


def test_extra_field_rejected():
    r = client.post("/predict", json={**VALID, "extra": 9})
    assert r.status_code == 422


def test_prediction_error_returns_json_500(monkeypatch):
    class BrokenModel:
        def predict(self, X):
            raise ValueError("boom")

    monkeypatch.setattr(app_module, "model", BrokenModel())
    r = client.post("/predict", json=VALID)
    assert r.status_code == 500
    assert r.json() == {"detail": "Prediction failed"}
