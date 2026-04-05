# Arabic Speech Recognition (ASR)

A CTC-based Arabic Automatic Speech Recognition system built with TensorFlow/Keras.

## Project Overview

This project provides a complete pipeline for training and running an Arabic ASR model using Connectionist Temporal Classification (CTC).

## Tech Stack

- **Language:** Python 3.12
- **Deep Learning:** TensorFlow 2.x / Keras (CTC model)
- **Web Interface:** Flask (browser-based UI replacing tkinter GUI)
- **Audio Processing:** TensorFlow signal processing (Mel spectrograms)
- **Visualization:** Matplotlib, Web Audio API (in browser)

## Project Structure

```
├── web_app.py       # Flask web server (browser UI, port 5000)
├── app.py           # Original tkinter desktop GUI (not used in Replit)
├── predict.py       # CLI inference script
├── train.py         # Training script
├── model.py         # CTC model architecture (Conv1D → BiGRU → Dense)
├── data_loader.py   # Data pipeline, Mel spectrogram extraction
├── config.py        # Centralized configuration (paths, audio params, hyperparams)
├── requirements.txt # Python dependencies
├── colab/           # Google Colab support notebooks
├── data/            # (Not included) Place wav/ and lab/ subdirectories here
├── manifests/       # Generated JSONL manifest files (created during training)
└── checkpoints/     # Trained model files: model_final.keras + vocab.json
```

## Running the App

The main workflow runs `python web_app.py` on port 5000. This starts a Flask web server with a browser-based interface for audio transcription.

## Using the App

1. **Training:** Run `python train.py` after placing data in `data/wav/` and `data/lab/`
2. **Inference (web):** The web app loads the model from `checkpoints/model_final.keras` and `checkpoints/vocab.json`
3. **Inference (CLI):** Run `python predict.py --audio path/to/file.wav`

## Model Architecture

- Input: Mel spectrogram (time_steps × 80 mel bins)
- Conv1D (128 filters, kernel 11) × 2 with BatchNorm
- Bidirectional GRU (256 units) × 2 with Dropout
- Dense output layer with CTC decoding

## Configuration

All settings are in `config.py`:
- `SAMPLE_RATE = 16000` Hz
- `MAX_AUDIO_LENGTH = 10.0` seconds
- `N_MELS = 80` Mel bins
- `BATCH_SIZE = 8`, `EPOCHS = 50`
