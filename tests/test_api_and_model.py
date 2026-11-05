import importlib

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from drowsy_eeg.model import cross_validate, make_model
from drowsy_eeg.signal import FEATURES, feature_matrix

SFREQ = 100.0
T = np.arange(int(30 * SFREQ)) / SFREQ


def synthetic_epochs(n, drowsy, rng):
    """Alpha-dominant epochs for awake, theta-dominant for drowsy, plus noise."""
    a, t = (3.0, 10.0) if drowsy else (10.0, 3.0)
    return np.stack([
        a * np.sin(2 * np.pi * rng.uniform(9, 11) * T)
        + t * np.sin(2 * np.pi * rng.uniform(5, 7) * T)
        + rng.normal(0, 2, T.size)
        for _ in range(n)
    ])


@pytest.fixture(scope="module")
def dataset():
    rng = np.random.default_rng(1)
    X, y, groups = [], [], []
    for subject in range(6):
        for drowsy in (0, 1):
            X.append(feature_matrix(synthetic_epochs(8, drowsy, rng), SFREQ))
            y += [drowsy] * 8
            groups += [subject] * 8
    return np.vstack(X), np.array(y), np.array(groups)


def test_cross_validate_separates_synthetic_classes(dataset):
    X, y, groups = dataset
    m = cross_validate(X, y, groups, n_splits=3)
    assert m["n_subjects"] == 6 and m["n_epochs"] == 96
    assert m["model"]["balanced_accuracy"] > 0.95
    assert set(m["baseline_theta_alpha_only"]) == set(m["model"])


@pytest.fixture()
def client(dataset, tmp_path, monkeypatch):
    X, y, _ = dataset
    path = tmp_path / "model.joblib"
    joblib.dump({"model": make_model().fit(X, y), "features": FEATURES}, path)
    monkeypatch.setenv("DROWSY_MODEL", str(path))
    import drowsy_eeg.api as api

    api = importlib.reload(api)
    return TestClient(api.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_flags_theta_epoch_as_drowsy(client):
    rng = np.random.default_rng(2)
    drowsy = synthetic_epochs(1, 1, rng)[0]
    awake = synthetic_epochs(1, 0, rng)[0]
    r1 = client.post("/predict", json={"sfreq": SFREQ, "samples": drowsy.tolist()}).json()
    r2 = client.post("/predict", json={"sfreq": SFREQ, "samples": awake.tolist()}).json()
    assert r1["label"] == "drowsy" and r2["label"] == "awake"
    assert set(r1["features"]) == set(FEATURES)


def test_predict_rejects_wrong_length(client):
    r = client.post("/predict", json={"sfreq": SFREQ, "samples": [0.0] * 100})
    assert r.status_code == 422
    assert "expected 3000 samples" in r.json()["detail"]


def test_predict_rejects_flat_signal(client):
    r = client.post("/predict", json={"sfreq": SFREQ, "samples": [0.0] * 3000})
    assert r.status_code == 422
