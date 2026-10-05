# Cry Helper

Record a baby's cry and get a best guess at why they are crying, with things to try.
Everything runs on the phone: no audio leaves the device.

| File | What it does |
| --- | --- |
| `cry_preprocess.py` | Step 1: audio -> 3 s mel-spectrograms, builds `cry_dataset.npz` |
| `train_model.py` | Step 2: trains the classifier, reports accuracy, writes the model into `app/index.html` |
| `app/index.html` | The app: records 10 s from the mic (or opens an audio file), runs the model, learns this baby's cries |

## Rebuild the model

```bash
pip install -r requirements.txt scikit-learn
git clone https://github.com/gveres/donateacry-corpus
python cry_preprocess.py donateacry-corpus/donateacry_corpus_cleaned_and_updated_data
python train_model.py cry_dataset.npz
```

## Run the app

The microphone only works on `https://` or `localhost`:

```bash
cd app && python -m http.server 8000   # then open http://localhost:8000
```

To use it on a phone, host `app/index.html` anywhere with HTTPS (for example GitHub Pages).

## Accuracy

Trained on the Donate-a-Cry corpus (457 clips, 382 of them "hungry"). Cross-validated by
recording, it scores 0.32 balanced accuracy across the five reasons, where random guessing
scores 0.20. Treat its answer as a hint. Teaching it your own baby's cries in the app
(the "What was it really?" buttons) is the realistic way to improve it.

This is not a medical device. The app lists warning signs that need a doctor.
