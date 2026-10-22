# Arabic Speech Recognition

A browser-based Arabic speech recognition app powered by [OpenAI Whisper](https://github.com/openai/whisper). Upload a WAV file and get an accurate Arabic transcription instantly — no training, no setup, no API keys required.

---

## Features

- Automatic transcription of Arabic speech from WAV audio files
- Pre-trained Whisper model — works out of the box on first run
- Waveform preview rendered in the browser before transcription
- Right-to-left Arabic output display
- Clean, responsive web interface

---

## How It Works

At startup the app loads OpenAI's `whisper-base` model (downloaded automatically on first run, ~140 MB). When you upload a WAV file, Whisper processes it with the Arabic language hint and returns the transcription as Arabic text.

---

## Project Structure

```
├── web_app.py           Flask web server — main entry point (port 5000)
├── requirements.txt     Python dependencies
├── README.md            This file
├── .gitignore
├── .replit
└── scripts/
    └── post-merge.sh    Post-merge dependency installer
```

---

## Getting Started

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Run the server**

```bash
python web_app.py
```

Open `http://localhost:5000` in your browser. The status banner turns green once the model is ready. On the first run, Whisper downloads the model weights automatically (~140 MB).

---

## Usage

1. Click the upload area or drag and drop a WAV file onto it.
2. A waveform preview is drawn automatically.
3. Click **Transcribe**.
4. The Arabic transcription appears in the result box (right-to-left).

**Supported audio:** WAV format, mono or stereo, any sample rate (Whisper resamples internally). Files up to 50 MB are accepted.

---

## Dependencies

| Package          | Purpose                              |
|------------------|--------------------------------------|
| `openai-whisper` | Pre-trained multilingual ASR model   |
| `flask`          | Web framework and HTTP server        |
| `numpy`          | Numerical array operations           |

---

## License

Project code is released under the MIT License.
The Whisper model weights are released by OpenAI under the MIT License.
