"""Filtering and FFT band-power features for one EEG epoch."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
}
FEATURES = [
    "rel_delta",
    "rel_theta",
    "rel_alpha",
    "rel_beta",
    "theta_alpha_ratio",
    "slow_fast_ratio",
    "log_total_power",
]


def bandpass(x: np.ndarray, sfreq: float, low: float = 0.5, high: float = 30.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter (removes drift and high-frequency noise)."""
    if not 0 < low < high < sfreq / 2:
        raise ValueError(f"need 0 < low < high < Nyquist ({sfreq / 2} Hz)")
    sos = butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def band_powers(x: np.ndarray, sfreq: float) -> dict[str, float]:
    """Absolute power in each band from a Welch (averaged FFT) spectrum."""
    freqs, psd = welch(x, fs=sfreq, nperseg=int(4 * sfreq))
    step = freqs[1] - freqs[0]
    return {
        name: float(psd[(freqs >= lo) & (freqs < hi)].sum() * step)
        for name, (lo, hi) in BANDS.items()
    }


def epoch_features(x: np.ndarray, sfreq: float) -> dict[str, float]:
    """Filter one epoch and return the features the model uses.

    Relative powers make the features less sensitive to electrode gain; the
    ratios capture the shift from alpha (relaxed waking) toward theta (drowsy).
    """
    p = band_powers(bandpass(x, sfreq), sfreq)
    total = sum(p.values())
    if total <= 0:
        raise ValueError("epoch has no signal power")
    return {
        "rel_delta": p["delta"] / total,
        "rel_theta": p["theta"] / total,
        "rel_alpha": p["alpha"] / total,
        "rel_beta": p["beta"] / total,
        "theta_alpha_ratio": p["theta"] / p["alpha"],
        "slow_fast_ratio": (p["delta"] + p["theta"]) / (p["alpha"] + p["beta"]),
        "log_total_power": float(np.log10(total)),
    }


def feature_matrix(epochs: np.ndarray, sfreq: float) -> np.ndarray:
    """Features for many epochs, in FEATURES column order."""
    rows = [epoch_features(e, sfreq) for e in epochs]
    return np.array([[r[name] for name in FEATURES] for r in rows])
