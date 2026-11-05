import numpy as np
import pytest

from drowsy_eeg.signal import FEATURES, band_powers, bandpass, epoch_features, feature_matrix

SFREQ = 100.0
T = np.arange(int(30 * SFREQ)) / SFREQ


def sine(freq, amp=10.0):
    return amp * np.sin(2 * np.pi * freq * T)


def test_bandpass_keeps_in_band_and_removes_out_of_band():
    kept = bandpass(sine(10), SFREQ)
    removed = bandpass(sine(45), SFREQ)
    drift = bandpass(np.full_like(T, 50.0), SFREQ)  # DC offset
    assert np.std(kept[500:-500]) == pytest.approx(np.std(sine(10)), rel=0.05)
    assert np.std(removed[500:-500]) < 0.05 * np.std(sine(45))
    assert np.abs(drift[500:-500]).max() < 1.0


def test_bandpass_rejects_cutoff_above_nyquist():
    with pytest.raises(ValueError):
        bandpass(sine(10), sfreq=50.0, high=30.0)


@pytest.mark.parametrize("freq,band", [(2, "delta"), (6, "theta"), (10, "alpha"), (20, "beta")])
def test_band_powers_put_a_pure_tone_in_its_band(freq, band):
    p = band_powers(sine(freq), SFREQ)
    assert max(p, key=p.get) == band
    assert p[band] > 0.95 * sum(p.values())


def test_band_power_matches_sine_power():
    # A sine of amplitude A has power A^2 / 2.
    p = band_powers(sine(10, amp=4.0), SFREQ)
    assert p["alpha"] == pytest.approx(8.0, rel=0.05)


def test_theta_alpha_ratio_rises_when_rhythm_slows():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, T.size)
    relaxed = epoch_features(sine(10) + sine(6, 3) + noise, SFREQ)
    drowsy = epoch_features(sine(10, 3) + sine(6) + noise, SFREQ)
    assert drowsy["theta_alpha_ratio"] > 5 * relaxed["theta_alpha_ratio"]
    rel = [relaxed[k] for k in ("rel_delta", "rel_theta", "rel_alpha", "rel_beta")]
    assert sum(rel) == pytest.approx(1.0)


def test_flat_epoch_is_rejected():
    with pytest.raises(ValueError):
        epoch_features(np.zeros_like(T), SFREQ)


def test_feature_matrix_shape_and_order():
    X = feature_matrix(np.stack([sine(10), sine(6)]), SFREQ)
    assert X.shape == (2, len(FEATURES))
    assert X[1, FEATURES.index("rel_theta")] > X[0, FEATURES.index("rel_theta")]
