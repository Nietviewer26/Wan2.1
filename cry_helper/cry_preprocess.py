"""
Baby cry preprocessing - step 1 of the cry-helper app.

Two modes:
  python cry_preprocess.py some_cry.wav      -> shows/saves the spectrogram for one file
  python cry_preprocess.py path/to/dataset   -> builds cry_dataset.npz for training

Dataset folders are named by label, e.g. dataset/hungry/*.wav, dataset/tired/*.wav
(this matches the free donateacry-corpus layout).
"""
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Settings - keep these identical in the phone app later, or predictions break
# ---------------------------------------------------------------------------
SR = 16000            # 16 kHz: standard for cry/speech models, and what YAMNet expects
N_MELS = 64           # 64 bands is enough for cries and keeps the model phone-sized
N_FFT = 1024          # ~64 ms analysis window
HOP = 256             # ~16 ms step -> 188 time frames per 3 s window
FMAX = SR // 2        # highest frequency available at 16 kHz
WINDOW_SEC = 3.0      # length of each model input
STEP_SEC = 1.5        # 50% overlap when slicing longer clips (more training examples)
DB_FLOOR = -80.0      # fixed dB floor so every clip is scaled the same way
SILENCE_RMS = 0.005   # windows quieter than this are skipped (no cry in them)

AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}

# Folder name -> what the app shows the parent
LABELS = {
    "hungry": "I'm hungry",
    "tired": "I'm tired",
    "burping": "I need burping",
    "belly_pain": "My tummy hurts",
    "discomfort": "Something's uncomfortable",
}
LABEL_ORDER = list(LABELS)  # index 0..4 used as the training target


def load_audio(path):
    """Load as mono 16 kHz and trim silence at the start/end."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    return y


def slice_windows(y):
    """Cut audio into fixed 3 s windows with overlap. Short clips are padded."""
    win = int(SR * WINDOW_SEC)
    step = int(SR * STEP_SEC)
    if len(y) < win:
        return [np.pad(y, (0, win - len(y)))]
    return [y[s:s + win] for s in range(0, len(y) - win + 1, step)]


def is_silent(window):
    return np.sqrt(np.mean(window ** 2)) < SILENCE_RMS


def window_to_mel(window):
    """3 s waveform -> normalised mel-spectrogram, shape (N_MELS, frames, 1)."""
    mel = librosa.feature.melspectrogram(
        y=window, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmax=FMAX
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)          # loudest point = 0 dB
    mel_db = np.clip(mel_db, DB_FLOOR, 0.0)
    norm = (mel_db - DB_FLOOR) / -DB_FLOOR                  # fixed scale to 0..1
    return norm[..., np.newaxis].astype(np.float32)         # channels-last for TensorFlow/TFLite


def file_to_examples(path):
    """One audio file -> list of model-ready spectrograms (silent windows dropped)."""
    y = load_audio(path)
    return [window_to_mel(w) for w in slice_windows(y) if not is_silent(w)]


def build_dataset(root, out_file="cry_dataset.npz"):
    """Walk a labelled folder tree and save X (spectrograms) and y (label indexes)."""
    X, y, groups, skipped = [], [], [], []
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() not in AUDIO_EXTS:
            continue
        label = path.parent.name.lower()
        if label not in LABELS:
            continue
        try:
            examples = file_to_examples(path)
        except Exception as err:                            # corrupt/unsupported file
            skipped.append(f"{path.name}: {err}")
            continue
        for ex in examples:
            X.append(ex)
            y.append(LABEL_ORDER.index(label))
            groups.append(str(path))                        # source file, to keep train/test split honest

    if not X:
        sys.exit(f"No usable audio found under {root}. Folders must be named: {', '.join(LABELS)}")

    X, y = np.stack(X), np.array(y)
    np.savez_compressed(out_file, X=X, y=y, groups=np.array(groups), labels=np.array(LABEL_ORDER))

    print(f"Saved {out_file}: X {X.shape}, y {y.shape}")
    print("Examples per label (watch for imbalance):")
    for i, n in sorted(Counter(y.tolist()).items()):
        print(f"  {LABEL_ORDER[i]:<12} {n:>5}   -> app shows: {LABELS[LABEL_ORDER[i]]}")
    if skipped:
        print(f"Skipped {len(skipped)} unreadable files, e.g. {skipped[0]}")


def plot_file(path, save_to="spectrogram.png"):
    """Show the first non-silent window of one file, and save it as an image."""
    examples = file_to_examples(path)
    if not examples:
        sys.exit("Clip is silent or too quiet - nothing to plot.")
    spec = examples[0][..., 0]
    print(f"Model input shape: {examples[0].shape}  |  windows in file: {len(examples)}")
    print(f"Value range: {spec.min():.2f} to {spec.max():.2f}")

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spec * -DB_FLOOR + DB_FLOOR, sr=SR, hop_length=HOP,
                             x_axis="time", y_axis="mel", fmax=FMAX)
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Cry mel-spectrogram: {Path(path).name}")
    plt.tight_layout()
    plt.savefig(save_to, dpi=120)
    print(f"Saved image: {save_to}")
    if matplotlib.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "baby_cry_hungry.wav"
    if Path(target).is_dir():
        build_dataset(target)
    else:
        plot_file(target)
