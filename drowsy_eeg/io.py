"""Load Sleep-EDF recordings and cut them into labeled 30-second epochs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np

EPOCH_SEC = 30.0
CHANNEL = "EEG Fpz-Cz"

# Sleep-EDF hypnogram labels -> the two classes this project uses.
# Stage 1 (N1) is the light, drowsy transition between waking and sleep.
LABELS = {"Sleep stage W": "awake", "Sleep stage 1": "drowsy"}
SLEEP_STAGES = {"Sleep stage 1", "Sleep stage 2", "Sleep stage 3", "Sleep stage 4", "Sleep stage R"}

# Recordings run for about a day, mostly awake. Keep only the wake time
# close to the night so the classes are not dominated by daytime wake.
WAKE_MARGIN_SEC = 30 * 60


@dataclass
class Recording:
    """One night of single-channel EEG plus its labeled epochs."""

    subject: int
    night: int
    sfreq: float
    epochs: np.ndarray  # shape (n_epochs, n_samples), microvolts
    labels: np.ndarray  # "awake" or "drowsy" per epoch
    metadata: dict = field(default_factory=dict)


def parse_name(psg_path: str | Path) -> tuple[int, int]:
    """Return (subject, night) from a Sleep-EDF file name such as SC4012E0-PSG.edf."""
    name = Path(psg_path).name
    return int(name[3:5]), int(name[5])


def crop_to_night(annotations: mne.Annotations) -> mne.Annotations:
    """Drop wake time more than WAKE_MARGIN_SEC before the first or after the last sleep stage."""
    sleep = [a for a in annotations if a["description"] in SLEEP_STAGES]
    if not sleep:
        raise ValueError("hypnogram has no sleep stages")
    start = max(sleep[0]["onset"] - WAKE_MARGIN_SEC, 0.0)
    end = sleep[-1]["onset"] + sleep[-1]["duration"] + WAKE_MARGIN_SEC
    return annotations.copy().crop(start, end, use_orig_time=False)


def cut_epochs(
    signal: np.ndarray, sfreq: float, annotations: mne.Annotations
) -> tuple[np.ndarray, np.ndarray]:
    """Split every labeled stretch into whole 30-second epochs.

    Stages outside LABELS (deeper sleep, REM, movement, unscored) are skipped,
    as is any epoch that would run past the end of the signal.
    """
    size = int(round(EPOCH_SEC * sfreq))
    epochs, labels = [], []
    for ann in annotations:
        label = LABELS.get(ann["description"])
        if label is None:
            continue
        for k in range(int(ann["duration"] // EPOCH_SEC)):
            start = int(round((ann["onset"] + k * EPOCH_SEC) * sfreq))
            if start + size > signal.size:
                break
            epochs.append(signal[start : start + size])
            labels.append(label)
    if not epochs:
        return np.empty((0, size)), np.empty(0, dtype=object)
    return np.stack(epochs), np.array(labels, dtype=object)


def load_recording(psg_path: str | Path, hyp_path: str | Path, channel: str = CHANNEL) -> Recording:
    """Read one PSG file and its hypnogram into a Recording."""
    raw = mne.io.read_raw_edf(psg_path, include=[channel], preload=True, verbose="error")
    annotations = mne.read_annotations(hyp_path)
    annotations = crop_to_night(annotations)
    signal = raw.get_data()[0] * 1e6  # volts -> microvolts
    sfreq = float(raw.info["sfreq"])
    epochs, labels = cut_epochs(signal, sfreq, annotations)
    subject, night = parse_name(psg_path)
    metadata = {
        "subject": subject,
        "night": night,
        "file": Path(psg_path).name,
        "channel": channel,
        "sfreq": sfreq,
        "hours_recorded": round(signal.size / sfreq / 3600, 2),
        "epochs_awake": int((labels == "awake").sum()),
        "epochs_drowsy": int((labels == "drowsy").sum()),
    }
    return Recording(subject, night, sfreq, epochs, labels, metadata)
