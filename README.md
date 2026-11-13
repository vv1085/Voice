# VOICE: Adaptive P300 Speller

VOICE is a brain-computer interface speller that runs in the browser. It decodes which letter a person was looking at on a 6×6 grid from their EEG responses (the P300 brain wave). It replays **recorded, held-out** trials from the public [BNCI2014-008](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_008.html) dataset (8 participants, 8 channels) and stops each trial as soon as the decoder is confident.

Built with Python, NumPy and [Gradio](https://www.gradio.app/), and deployable on [Render](https://render.com/) in one step.

## Features

- **Adaptive stopping**: evidence builds up repetition by repetition. A trial stops once the decoder is confident and stable, or after 10 repetitions.
- **Accept / reject**: rejecting a letter offers the next most likely ones (up to 3).
- **Message editor** with space, backspace, clear, undo and redo. When the decoder is idle you can also click a letter to type it.
- **Honest evaluation**: each participant's decoder is trained on the other 7 participants plus that participant's first 10 trials. The 25 trials it spells were never used in training.
- **Session report**: download a JSON file with every trial, decision and edit.

## Project layout

```
app.py                      Gradio web app (entry point)
voice/
  config.py                 grid, flash codes, preprocessing, stopping rule
  decoder.py                linear P300 model, evidence accumulation, adaptive stopping
  artifacts.py              reads and writes the model and trial files
  session.py                speller session: decode, accept/reject, editing, reports
scripts/
  build_artifacts.py        downloads the dataset, trains the decoders, writes artifacts/
  render_build.sh           Render build step
render.yaml                 Render Blueprint
requirements.txt            app dependencies
requirements-build.txt      + dataset/training dependencies
```

## How the model files are made

`scripts/build_artifacts.py` produces `artifacts/`, which is about 23 MB:

1. Loads BNCI2014-008 with MOABB's P300 paradigm: 0.5–10 Hz band-pass, epochs from −0.2 to 0.8 s, 8 channels.
2. Rebuilds each flash's row or column code (3–14) from the `Flash stim` channel. It checks them against the Target/NonTarget annotations: 35 trials × 120 flashes per participant.
3. Converts to microvolts and downsamples 256 → 64 Hz, giving 8 × 65 = 520 features per flash.
4. Trains a shrinkage LDA per participant with streaming 520 × 520 sums. The maths matches scikit-learn's `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")`, and this is checked on every build. Peak memory stays under 400 MB.
5. Scores each decoder on its held-out trials and saves the weights plus the held-out trials.

The web app needs only NumPy to run these models.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-build.txt
python scripts/build_artifacts.py                        # downloads the dataset on first run
python app.py                                            # http://localhost:7860
```

No internet access to the dataset? To try the app offline, use clearly labelled synthetic data:

```bash
pip install -r requirements-build.txt
python scripts/build_artifacts.py --source synthetic
python app.py
```

## Deploy on Render

1. Push this repository to GitHub.
2. In Render, choose **New → Blueprint** and select the repository. Render reads `render.yaml`.
3. Click **Apply**. The build installs the app, downloads BNCI2014-008, trains the 8 decoders and starts `python app.py` on Render's port.

The first build takes several minutes because of the download and training. Later builds rebuild the files the same way.

Settings you can change in Render (Environment tab):

| Variable | Default | Purpose |
| --- | --- | --- |
| `VOICE_BUILD_SOURCE` | `bnci` | `synthetic` builds offline test data (shown as SYNTHETIC in the app) |
| `VOICE_MNE_DATA` | `/tmp/mne_data` | where the dataset is downloaded during the build |
| `VOICE_ARTIFACT_DIR` | `artifacts` | where the app looks for model files |
| `VOICE_STEP_DELAY_S` | `0.35` | pause between repetitions in the animation |

**If the dataset download fails on Render** (the dataset host can be slow), build the files on your own computer or in Colab with `python scripts/build_artifacts.py`. Then commit `artifacts/` by removing it from `.gitignore`, and the build will reuse it. Check the dataset's licence first if your repository is public.

**Free plan notes:** the service sleeps after about 15 minutes of inactivity and takes a short time to wake up. Its disk is temporary, so download session reports to keep them.

## Use your own EEG

`voice/decoder.py` has `TrialDecoder.repetition_probabilities(epochs, flash_codes)`. Give it 12 epochs shaped `(12, 8, 65)` in microvolts, preprocessed as above, together with their flash codes 3–14. Feed the results to `EvidenceAccumulator` and `AdaptiveStopping` to decode live data.

## Limitations

- This is a research demonstration, not a medical device. It replays recorded EEG and does not acquire live signals.
- Codes 3–8 are treated as grid rows and 9–14 as columns. The build checks that every trial's two target codes split this way.
