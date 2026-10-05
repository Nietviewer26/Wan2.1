"""
Baby cry model - step 2 of the cry-helper app.

  python train_model.py cry_dataset.npz

Trains a small classifier on the spectrograms from cry_preprocess.py, reports
honest accuracy (cross-validated by recording, so no clip is tested on its own
windows), then writes the model into app/index.html so the web app can run it
on the phone with no server.

The donateacry corpus is recorded at 8 kHz, so it holds nothing above 4 kHz.
Only the mel bands below 4 kHz are used, and the app band-limits the phone
microphone the same way before analysing it.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import librosa
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cry_preprocess import SR, N_MELS, N_FFT, FMAX, HOP, WINDOW_SEC, STEP_SEC, DB_FLOOR, SILENCE_RMS, LABELS

TRAIN_FMAX = 4000      # highest frequency present in the 8 kHz training recordings
C = 0.1                # regularisation strength, picked by cross-validation
APP_HTML = Path(__file__).parent / "app" / "index.html"

# Mel bands whose upper edge is at or below 4 kHz
N_BANDS = int(np.sum(librosa.mel_frequencies(n_mels=N_MELS + 2, fmin=0, fmax=FMAX)[2:] <= TRAIN_FMAX))


def window_features(spec):
    """(N_MELS, frames) spectrogram -> per-band mean, spread and frame-to-frame change."""
    s = spec[:N_BANDS]
    return np.concatenate([s.mean(-1), s.std(-1), np.abs(np.diff(s, axis=-1)).mean(-1)])


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def evaluate(F, y, groups, labels):
    """Cross-validate by recording and score one prediction per recording."""
    file_label = {g: y[groups == g][0] for g in np.unique(groups)}
    yt, yp = [], []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(F, y, groups):
        probs = make_model().fit(F[tr], y[tr]).predict_proba(F[te])
        for g in np.unique(groups[te]):
            yt.append(file_label[g])
            yp.append(probs[groups[te] == g].mean(0).argmax())
    yt, yp = np.array(yt), np.array(yp)
    bal = balanced_accuracy_score(yt, yp)
    cm = confusion_matrix(yt, yp, labels=range(len(labels)))
    recall = {labels[i]: round(float(cm[i, i] / max(cm[i].sum(), 1)), 2) for i in range(len(labels))}
    print(f"Balanced accuracy per recording: {bal:.2f} (random guessing: {1 / len(labels):.2f})")
    print("Confusion matrix, rows = true, cols = predicted:", labels)
    print(cm)
    return {"balanced_accuracy": round(float(bal), 2), "chance": round(1 / len(labels), 2),
            "recall": recall, "recordings": int(len(yt))}


def sparse_mel_filters():
    """librosa's mel filterbank as [first_bin, weights...] per band, for the browser."""
    fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmax=FMAX)
    rows = []
    for row in fb:
        nz = np.nonzero(row)[0]
        rows.append([int(nz[0]), [round(float(v), 8) for v in row[nz[0]:nz[-1] + 1]]])
    return rows


def main(npz="cry_dataset.npz"):
    d = np.load(npz)
    X, y, groups = d["X"][..., 0], d["y"], d["groups"]
    labels = [str(l) for l in d["labels"]]
    F = np.stack([window_features(s) for s in X])
    print(f"{len(np.unique(groups))} recordings, {len(F)} windows, {F.shape[1]} features")

    metrics = evaluate(F, y, groups, labels)
    model = make_model().fit(F, y)
    scaler, clf = model.named_steps.values()

    export = {
        "settings": {"sr": SR, "nFft": N_FFT, "hop": HOP, "nMels": N_MELS, "nBands": N_BANDS,
                     "windowSec": WINDOW_SEC, "stepSec": STEP_SEC, "dbFloor": DB_FLOOR,
                     "silenceRms": SILENCE_RMS, "trainFmax": TRAIN_FMAX},
        "labels": labels,
        "messages": [LABELS[l] for l in labels],
        "melFilters": sparse_mel_filters(),
        "mean": [round(float(v), 6) for v in scaler.mean_],
        "scale": [round(float(v), 6) for v in scaler.scale_],
        "coef": [[round(float(v), 6) for v in row] for row in clf.coef_],
        "intercept": [round(float(v), 6) for v in clf.intercept_],
        "metrics": metrics,
    }
    html = APP_HTML.read_text()
    block = "/*MODEL_START*/" + json.dumps(export, separators=(",", ":")) + "/*MODEL_END*/"
    html, n = re.subn(r"/\*MODEL_START\*/.*?/\*MODEL_END\*/", lambda _: block, html, flags=re.S)
    if n != 1:
        sys.exit(f"Could not find the model placeholder in {APP_HTML}")
    APP_HTML.write_text(html)
    print(f"Wrote model into {APP_HTML}")


if __name__ == "__main__":
    main(*sys.argv[1:])
