# Arabic Automatic Speech Recognition (ASR)

A clean, simple Arabic ASR system built with **Python** and **TensorFlow**.

Uses a **CTC-based** model (Conv1D → Bidirectional GRU → Dense) trained on the
[Arabic Speech Corpus](https://en.arabicspeechcorpus.com/) by Nawar Halabi.

---

## Project Structure

```
ASR_project/
├── config.py          # All settings (paths, audio params, hyperparams)
├── data_loader.py     # Load audio + transcripts, build tf.data pipeline
├── model.py           # CTC model definition + decode function
├── train.py           # Training script (GPU-aware, OS-path-rebuilding)
├── app.py             # Desktop GUI for transcription (tkinter)
├── predict.py         # Command-line inference
├── requirements.txt   # Python dependencies
├── colab/             # Google Colab training notebook & README
├── data/              # Flat data structure: wav/ and lab/
├── manifests/         # Auto-generated JSONL manifests
└── checkpoints/       # Saved model weights & vocabulary
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ASR_project.git
cd ASR_project

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your dataset
#    Download the Arabic Speech Corpus and extract it into data/
```

---

## Training

```bash
python train.py
```

This will:
1. Scan the corpus and build a manifest (`manifests/train.jsonl`)
2. Build a character-level vocabulary
3. Train the CTC model for 50 epochs (configurable in `config.py`)
4. Save checkpoints every 10 epochs to `checkpoints/`

---

## Inference

```bash
# Predict on a specific file
python predict.py --audio path/to/arabic.wav

# Or use the default (first file in the corpus)
python predict.py
```

---

## Configuration

All settings are in `config.py`:

| Setting           | Default | Description                  |
|-------------------|---------|------------------------------|
| `SAMPLE_RATE`     | 16000   | Audio sample rate (Hz)       |
| `N_MELS`          | 80      | Mel spectrogram bins         |
| `MAX_AUDIO_LENGTH`| 10.0    | Max clip duration (seconds)  |
| `BATCH_SIZE`      | 8       | Training batch size          |
| `EPOCHS`          | 50      | Number of training epochs    |
| `LEARNING_RATE`   | 3e-4    | Adam learning rate           |

---

## Dataset

**Arabic Speech Corpus** (University of Southampton, v2.0)
- Author: Nawar Halabi
- 1813+ utterances of Modern Standard Arabic
- WAV files (48 kHz) with orthographic `.lab` transcriptions

---

## License

This project code is open source. The Arabic Speech Corpus dataset has its own
licensing terms — please refer to its README for details.
