# Arabic Speech Recognition (ASR)

A CTC-based Arabic Automatic Speech Recognition system built with TensorFlow/Keras.

## Project Overview

Complete pipeline for training and running an Arabic ASR model using Connectionist Temporal Classification (CTC). Includes a Flask web interface, a CLI inference tool, an improved training script, and a self-contained Google Colab notebook for training.

## Tech Stack

- **Language:** Python 3.12
- **Deep Learning:** TensorFlow 2.x / Keras (CTC model)
- **Web Interface:** Flask (browser-based UI on port 5000)
- **Audio Processing:** TensorFlow signal processing (log-Mel spectrograms)
- **Notebook:** Jupyter / Google Colab for training

## Project Structure

```
├── web_app.py          # Flask web server — main entry point (port 5000)
├── train.py            # Training script with gradient clipping & LR scheduling
├── predict.py          # CLI inference script
├── model.py            # CTC model: Conv1D -> BiGRU -> Dense
├── data_loader.py      # Data pipeline, Mel spectrogram extraction
├── config.py           # Centralized configuration (paths, audio params, hyperparams)
├── requirements.txt    # Python dependencies
├── scripts/
│   └── post-merge.sh   # Post-merge setup script (pip install)
├── colab/
│   └── arabic_asr_training.ipynb  # Self-contained Colab training notebook
├── data/               # Place wav/ and lab/ subdirectories here
├── manifests/          # Generated JSONL manifests (auto-created during training)
└── checkpoints/        # Trained model files: model_final.keras + vocab.json
```

## Running the App

The main workflow runs `python web_app.py` on port 5000.

## Model Architecture

- Input: log-Mel spectrogram (time_steps × 80 mel bins)
- Conv1D (128 filters, kernel 11, stride 2) + BatchNorm
- Conv1D (128 filters, kernel 11, stride 1) + BatchNorm + Dropout
- Bidirectional GRU (256 units) × 2 with Dropout
- Dense output + CTC greedy decode

## Training Improvements (train.py)

- Gradient clipping (`tf.clip_by_global_norm`, clip norm 5.0)
- ReduceLROnPlateau scheduler (halves LR after 5 epochs without val_loss improvement, min 1e-6)
- Correct input-length computation accounting for Conv stride factor of 2

## Google Colab Notebook

`colab/arabic_asr_training.ipynb` is fully self-contained — run all cells without editing any code. Only the dataset path cell (Section 2) needs to be updated with your Google Drive path.

## Configuration

All settings are in `config.py`:

| Setting             | Default | Description                   |
|---------------------|---------|-------------------------------|
| `SAMPLE_RATE`       | 16000   | Audio sample rate (Hz)        |
| `N_MELS`            | 80      | Mel spectrogram bins          |
| `MAX_AUDIO_LENGTH`  | 10.0    | Max clip duration (seconds)   |
| `BATCH_SIZE`        | 8       | Training batch size           |
| `EPOCHS`            | 50      | Number of training epochs     |
| `LEARNING_RATE`     | 3e-4    | Initial Adam learning rate    |

## Deployment

Configured for autoscale deployment via gunicorn on port 5000.
Post-merge script: `scripts/post-merge.sh`
