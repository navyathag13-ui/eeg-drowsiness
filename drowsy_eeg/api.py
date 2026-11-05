"""HTTP API: send one 30-second EEG epoch, get its features and drowsiness probability.

    uvicorn drowsy_eeg.api:app
"""

from __future__ import annotations

import os
from functools import lru_cache

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .io import EPOCH_SEC
from .signal import FEATURES, epoch_features

MODEL_PATH = os.environ.get("DROWSY_MODEL", "models/model.joblib")

app = FastAPI(title="EEG drowsiness API", version="0.1.0")


class Epoch(BaseModel):
    sfreq: float = Field(gt=60, description="sampling rate in Hz (must exceed 60 so 30 Hz is below Nyquist)")
    samples: list[float] = Field(description="one 30-second epoch of EEG, in microvolts")


@lru_cache
def load_model():
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}


@app.post("/predict")
def predict(epoch: Epoch) -> dict:
    expected = int(round(EPOCH_SEC * epoch.sfreq))
    if len(epoch.samples) != expected:
        raise HTTPException(422, f"expected {expected} samples (30 s at {epoch.sfreq} Hz), got {len(epoch.samples)}")
    x = np.asarray(epoch.samples, dtype=float)
    if not np.isfinite(x).all():
        raise HTTPException(422, "samples contain NaN or infinite values")
    try:
        feats = epoch_features(x, epoch.sfreq)
    except ValueError as err:
        raise HTTPException(422, str(err)) from err
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(503, "model not trained yet; run `python -m drowsy_eeg.pipeline train`")
    bundle = load_model()
    row = np.array([[feats[name] for name in bundle["features"]]])
    prob = float(bundle["model"].predict_proba(row)[0, 1])
    return {
        "drowsy_probability": round(prob, 4),
        "label": "drowsy" if prob >= 0.5 else "awake",
        "features": {name: round(feats[name], 5) for name in FEATURES},
    }
