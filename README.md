# Arabic Automatic Speech Recognition

A clean, professional Arabic ASR system built with Python and TensorFlow. Uses a CTC-based model trained on the [Arabic Speech Corpus](https://en.arabicspeechcorpus.com/) by Nawar Halabi.

---

## Architecture

```
Audio (WAV, 16 kHz mono)
        |
        v
  Mel Spectrogram          N_MELS=80 bins, 10 ms hop
        |
        v
  Conv1D (stride=2)        128 filters, kernel=11  ->  T/2 frames
  BatchNorm + ReLU
        |
  Conv1D (stride=1)        128 filters, kernel=11  ->  T/2 frames (unchanged)
  BatchNorm + ReLU
  Dropout(0.2)
        |
        v
  BiGRU x2                 256 units each direction  ->  512-dim per frame
  Dropout(0.2) between layers
        |
        v
  Dense (vocab_size)       raw logits
        |
        v
  CTC Loss / Greedy Decode
```

---

## Project Structure

```
ASR_project/
├── config.py          All settings: paths, audio params, hyper-parameters
├── data_loader.py     Audio I/O, Mel spectrogram, vocabulary, tf.data pipeline
├── model.py           CTC model definition + greedy decoder
├── train.py           Training script (gradient clipping, LR scheduler, GPU-aware)
├── predict.py         Command-line inference tool
├── web_app.py         Flask web UI (upload WAV -> transcription)
├── requirements.txt   Python dependencies
├── colab/
│   ├── arabic_asr_training.ipynb   Self-contained Google Colab notebook
│   └── README.md
├── data/              Dataset: data/wav/ and data/lab/  (not tracked by git)
├── manifests/         Auto-generated JSONL manifest     (not tracked by git)
└── checkpoints/       Saved model weights & vocabulary  (not tracked by git)
```

---

## Dataset Setup

Download the **Arabic Speech Corpus** (Nawar Halabi, University of Southampton):  
<https://en.arabicspeechcorpus.com/>

Extract and organise the files as follows:

```
data/
  wav/   <- all .wav files  (e.g. ARA_001.wav, ARA_002.wav, ...)
  lab/   <- all .lab files  (e.g. ARA_001.lab, ARA_002.lab, ...)
```

The corpus ships at 48 kHz. TensorFlow's `tf.audio.decode_wav` will read them;
the model expects 16 kHz mono. If your files are already at 48 kHz, set
`SAMPLE_RATE = 48000` in `config.py` or resample them beforehand with:

```bash
for f in data/wav/*.wav; do
    sox "$f" -r 16000 -c 1 "data/wav/$(basename $f)"
done
```

---

## Local Training

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train
python train.py
```

The script will:
1. Scan `data/` and generate `manifests/train.jsonl`.
2. Build a character-level vocabulary and save it to `checkpoints/vocab.json`.
3. Train for `EPOCHS` epochs with gradient clipping and LR scheduling.
4. Save `checkpoints/model_best.keras` (best validation loss) and
   `checkpoints/model_final.keras` (last epoch).

---

## Google Colab Training

Open `colab/arabic_asr_training.ipynb` in Colab, select a **GPU runtime**
(*Runtime > Change runtime type > GPU*), upload your dataset to Google Drive,
set the paths in **Section 2**, then click **Run All**.

No code editing is required beyond Section 2.

---

## Inference

### Command-line

```bash
# Transcribe a specific file
python predict.py --audio path/to/arabic.wav

# Transcribe the first file in the corpus (default)
python predict.py
```

### Web interface

```bash
python web_app.py
# Open http://localhost:5000 in a browser
```

Upload a WAV file and click **Transcribe**. The result is displayed in Arabic
(RTL) below the waveform preview.

---

## Configuration

All settings live in `config.py`. The table below lists the most commonly
changed values.

| Setting            | Default  | Description                               |
|--------------------|----------|-------------------------------------------|
| `SAMPLE_RATE`      | `16000`  | Audio sample rate (Hz)                    |
| `N_MELS`           | `80`     | Mel spectrogram bins                      |
| `N_FFT`            | `512`    | STFT window size (samples)                |
| `HOP_LENGTH`       | `160`    | STFT hop size (10 ms at 16 kHz)           |
| `MAX_AUDIO_LENGTH` | `10.0`   | Maximum clip duration in seconds          |
| `BATCH_SIZE`       | `8`      | Training batch size                       |
| `EPOCHS`           | `50`     | Number of training epochs                 |
| `LEARNING_RATE`    | `3e-4`   | Initial Adam learning rate                |
| `VALIDATION_SPLIT` | `0.1`    | Fraction of data held out for validation  |

---

## Dataset

**Arabic Speech Corpus** (v2.0)  
- Author: Nawar Halabi (University of Southampton)  
- ~1,813 utterances of Modern Standard Arabic  
- WAV files with orthographic `.lab` transcriptions  
- License: see the corpus README at the download URL above

---

## License

The project code is released under the MIT License.  
The Arabic Speech Corpus dataset has its own licensing terms — refer to the
corpus README for details.
