"""Command-line steps: download, build the feature table, train and evaluate.

    python -m drowsy_eeg.pipeline download --subjects 20
    python -m drowsy_eeg.pipeline features
    python -m drowsy_eeg.pipeline train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .io import load_recording
from .model import cross_validate, make_model
from .signal import FEATURES, feature_matrix

DATA = Path("data")
FEATURES_CSV = Path("results/features.csv")
RECORDINGS_JSON = Path("results/recordings.json")
METRICS_JSON = Path("results/metrics.json")
MODEL_PATH = Path("models/model.joblib")


def find_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Match each *-PSG.edf with its *-Hypnogram.edf (same first 7 characters)."""
    hyps = {p.name[:7]: p for p in root.rglob("*-Hypnogram.edf")}
    pairs = []
    for psg in sorted(root.rglob("*-PSG.edf")):
        hyp = hyps.get(psg.name[:7])
        if hyp is not None:
            pairs.append((psg, hyp))
    return pairs


def download(n_subjects: int) -> None:
    from mne.datasets.sleep_physionet.age import fetch_data

    fetch_data(subjects=list(range(n_subjects)), recording=[1], path=str(DATA), on_missing="warn")


def build_features() -> pd.DataFrame:
    frames, meta = [], []
    for psg, hyp in find_pairs(DATA):
        rec = load_recording(psg, hyp)
        df = pd.DataFrame(feature_matrix(rec.epochs, rec.sfreq), columns=FEATURES)
        df.insert(0, "label", rec.labels)
        df.insert(0, "night", rec.night)
        df.insert(0, "subject", rec.subject)
        frames.append(df)
        meta.append(rec.metadata)
        print(f"{psg.name}: {rec.metadata['epochs_awake']} awake, {rec.metadata['epochs_drowsy']} drowsy")
    if not frames:
        raise SystemExit(f"no PSG/Hypnogram pairs found under {DATA}/")
    table = pd.concat(frames, ignore_index=True)
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(FEATURES_CSV, index=False)
    RECORDINGS_JSON.write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(table)} epochs from {len(meta)} recordings to {FEATURES_CSV}")
    return table


def train() -> dict:
    table = pd.read_csv(FEATURES_CSV)
    X = table[FEATURES].to_numpy()
    y = (table["label"] == "drowsy").to_numpy().astype(int)
    metrics = cross_validate(X, y, table["subject"].to_numpy())
    METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": make_model().fit(X, y), "features": FEATURES}, MODEL_PATH)
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    dl = sub.add_parser("download", help="fetch Sleep-EDF recordings from PhysioNet")
    dl.add_argument("--subjects", type=int, default=20)
    sub.add_parser("features", help="build results/features.csv")
    sub.add_parser("train", help="cross-validate and save models/model.joblib")
    args = parser.parse_args()
    if args.step == "download":
        download(args.subjects)
    elif args.step == "features":
        build_features()
    else:
        train()


if __name__ == "__main__":
    main()
