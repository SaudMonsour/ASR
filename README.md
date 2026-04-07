# Arabic Speech Recognition

A browser-based Arabic speech recognition app powered by [OpenAI Whisper](https://github.com/openai/whisper). Upload a WAV file, click **Transcribe**, and get an Arabic transcription immediately — no training required.

---

## How it works

The app loads the pre-trained `whisper-base` model at startup. When you upload a WAV file, it is passed to Whisper with the language hint `ar` (Arabic) and the transcription is returned as Arabic text displayed right-to-left.

---

## Project structure

```
├── web_app.py          Flask web server — main entry point (port 5000)
├── requirements.txt    Python dependencies
├── README.md           This file
├── .gitignore
├── .replit
└── scripts/
    └── post-merge.sh   Post-merge setup script
```

---

## Running the app

```bash
pip install -r requirements.txt
python web_app.py
```

Open `http://localhost:5000` in a browser.

The status banner turns green once the Whisper model has finished loading (usually a few seconds on first run). On the first run Whisper will download the model weights (~140 MB) automatically.

---

## Usage

1. Click the drop zone (or drag and drop) to select a WAV file.
2. A waveform preview is drawn automatically.
3. Click **Transcribe**.
4. The Arabic transcription appears in the result box (right-to-left).

---

## Dependencies

| Package          | Purpose                        |
|------------------|--------------------------------|
| `openai-whisper` | Pre-trained ASR model          |
| `flask`          | Web framework                  |
| `numpy`          | Numerical array support        |

---

## License

Project code is released under the MIT License.
