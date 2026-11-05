import mne
import numpy as np
import pytest

from drowsy_eeg.io import EPOCH_SEC, crop_to_night, cut_epochs, parse_name

SFREQ = 100.0


def test_parse_name():
    assert parse_name("data/SC4012E0-PSG.edf") == (1, 2)
    assert parse_name("SC4191E0-PSG.edf") == (19, 1)


def test_cut_epochs_labels_and_skips_other_stages():
    signal = np.arange(int(300 * SFREQ), dtype=float)
    ann = mne.Annotations(
        onset=[0, 60, 120, 180],
        duration=[60, 60, 60, 75],  # last one is 2.5 epochs long
        description=["Sleep stage W", "Sleep stage 1", "Sleep stage 2", "Sleep stage W"],
    )
    epochs, labels = cut_epochs(signal, SFREQ, ann)
    assert list(labels) == ["awake", "awake", "drowsy", "drowsy", "awake", "awake"]
    assert epochs.shape == (6, int(EPOCH_SEC * SFREQ))
    # the stage-1 epoch at 60 s starts at sample 6000
    assert epochs[2, 0] == 60 * SFREQ


def test_cut_epochs_drops_epoch_past_end_of_signal():
    signal = np.zeros(int(45 * SFREQ))
    ann = mne.Annotations(onset=[0], duration=[60], description=["Sleep stage W"])
    epochs, labels = cut_epochs(signal, SFREQ, ann)
    assert len(labels) == 1


def test_crop_to_night_trims_long_daytime_wake():
    hour = 3600.0
    ann = mne.Annotations(
        onset=[0, 10 * hour, 18 * hour],
        duration=[10 * hour, 8 * hour, 6 * hour],
        description=["Sleep stage W", "Sleep stage 2", "Sleep stage W"],
    )
    cropped = crop_to_night(ann)
    wake = sum(a["duration"] for a in cropped if a["description"] == "Sleep stage W")
    assert wake == pytest.approx(3600.0)  # 30 min before + 30 min after


def test_crop_to_night_needs_sleep():
    ann = mne.Annotations(onset=[0], duration=[600], description=["Sleep stage W"])
    with pytest.raises(ValueError):
        crop_to_night(ann)
