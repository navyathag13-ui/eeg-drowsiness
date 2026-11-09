# EEG Drowsiness Detection

My B.E. mini-project detected driver drowsiness with an Arduino Uno and Nano by noticing when the eyes stayed closed too long. This project asks whether brain signals can show the same change. It detects the drowsy transition between waking and sleep (sleep stage 1) from a single EEG channel, using public recordings from PhysioNet.

**Why it matters:** drowsy drivers often do not notice they are drifting off. A single forehead-to-crown channel (Fpz-Cz) is the kind of signal a headband could record, so the real question is whether one channel is enough. **Answer:** it carries real signal (ROC AUC 0.84 on people the model never saw), but the single ratio usually cited is close to chance, and this is sleep-lab data, not driving data.

It is a small Python package with four parts:

1. **Load**: read each EEG recording and its sleep-stage labels, keep the night (plus 30 minutes of waking on each side) and cut it into 30-second epochs labeled `awake` or `drowsy`.
2. **Filter and extract features**: a 0.5-30 Hz zero-phase Butterworth band-pass filter, then a Welch (averaged FFT) spectrum per epoch. The features are relative delta, theta, alpha and beta power, the theta/alpha ratio, the slow/fast ratio and log total power.
3. **Model**: scaled logistic regression with balanced class weights, evaluated so that **no person appears in both training and testing**.
4. **API**: a FastAPI service. You send one 30-second epoch and get back the features and a drowsiness probability.

## Results

The data is 20 nights from 20 people (Sleep-EDF Expanded, Sleep Cassette, subjects 0-19, night 1, channel Fpz-Cz at 100 Hz). That gives 4,689 epochs, of which 1,240 are drowsy. Evaluation is 5-fold cross-validation grouped by person.

| Features | Balanced accuracy | ROC AUC | Drowsy recall | Awake recall |
|---|---|---|---|---|
| Theta/alpha ratio alone (baseline) | 58.5% | 0.60 | 68.5% | 48.5% |
| All 7 band-power features | **77.0%** | **0.84** | 78.5% | 75.5% |

The full metrics are in `results/metrics.json`, and the per-recording epoch counts are in `results/recordings.json`.

**What this means:** the single ratio most often linked to drowsiness barely beats chance on people the model has never seen. Combining all the bands lifts balanced accuracy by 18.5 points. Stage 1 is the hardest sleep stage to score, even for human experts, so 77% on unseen people is a realistic result rather than a solved problem.

## User guide

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Get the data and train

```bash
python -m drowsy_eeg.pipeline download --subjects 20   # about 1 GB from PhysioNet into data/
python -m drowsy_eeg.pipeline features                  # writes results/features.csv and results/recordings.json
python -m drowsy_eeg.pipeline train                     # writes results/metrics.json and models/model.joblib
```

### Run the API

```bash
uvicorn drowsy_eeg.api:app
```

- `GET /health` reports whether a trained model is available.
- `POST /predict` takes one 30-second epoch in microvolts:

```json
{"sfreq": 100, "samples": [12.1, 10.4, "... 3000 values in total"]}
```

It returns:

```json
{"drowsy_probability": 0.73, "label": "drowsy", "features": {"rel_theta": 0.31, "...": "..."}}
```

The API rejects epochs that are the wrong length, contain NaN values or are completely flat, and explains why in the error message.

### Use it from Python

```python
from drowsy_eeg.io import load_recording
from drowsy_eeg.signal import epoch_features

rec = load_recording("data/.../SC4001E0-PSG.edf", "data/.../SC4001EC-Hypnogram.edf")
print(rec.metadata)                       # subject, night, hours recorded, epochs per class
print(epoch_features(rec.epochs[0], rec.sfreq))
```

### Tests

```bash
pytest -q
```

There are 20 tests, and none of them need the downloaded data. They use synthetic signals with known answers:
- The filter keeps a 10 Hz tone, removes a 45 Hz tone and removes DC drift.
- A pure tone puts more than 95% of its power in the correct band.
- The power of a sine matches A²/2.
- Epoch cutting and night cropping follow the hypnogram exactly.
- The API labels theta-heavy epochs as drowsy and rejects bad input.

GitHub Actions runs the tests on every push.

## Limitations

- Sleep stage 1 in bed is not the same as drowsiness while driving. This shows the EEG side of the signal, not a driving study.
- The model uses one frontal channel and seven features. Adding more channels, eye-movement (EOG) signals or neighboring epochs would likely help.
- This is a learning project. It is not a medical device.

## Data credit

Kemp B, Zwinderman AH, Tuk B, Kamphuisen HAC, Oberyé JJL. *Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG.* IEEE-BME 47(9):1185-1194 (2000). The data is on PhysioNet: Goldberger et al., *Circulation* 101(23):e215-e220 (2000), and the download uses MNE-Python's Sleep-EDF fetcher.
