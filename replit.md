# Project Overview

This project is a Flask web app for Arabic speech recognition. Users upload audio or video files in the browser, and the server transcribes them with OpenAI Whisper.

# Architecture

- `web_app.py` contains the Flask server, rendered HTML interface, model loading thread, status endpoint, and transcription endpoint.
- The web app runs on `0.0.0.0:5000` for Replit preview compatibility.
- Whisper runs server-side only; the client uploads media to `/transcribe` and never receives model internals or credentials.
- `.replit` defines the Replit workflow and deployment command.
- `requirements.txt` pins CPU-only PyTorch before installing Whisper to avoid large GPU dependency downloads in Replit.

# Dependencies

- Python 3.12
- Flask
- OpenAI Whisper
- CPU-only PyTorch
- NumPy
- Gunicorn for deployment
- FFmpeg provided through Replit/Nix configuration

# Notes

- Accepted uploads are any audio/video files FFmpeg can decode, up to 200 MB.
- Whisper model weights download on first startup and are cached locally.
