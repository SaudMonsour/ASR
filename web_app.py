"""
web_app.py — Flask web interface for Arabic Speech Recognition.

Provides a browser-based UI that mirrors the functionality of app.py (tkinter GUI).
Allows users to upload WAV files and get transcriptions.

Usage:
    python web_app.py
"""

import os
import json
import tempfile

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template_string

import config
from data_loader import load_wav, wav_to_mel
from model import ctc_greedy_decode

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max upload

# Global model state
model = None
id_to_char = None
model_status = "loading"
model_message = "Loading model..."


def load_model():
    global model, id_to_char, model_status, model_message

    model_path = os.path.join(config.CHECKPOINT_DIR, "model_final.keras")
    vocab_path = os.path.join(config.CHECKPOINT_DIR, "vocab.json")

    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        model_status = "not_found"
        model_message = (
            "Model not found. Place model_final.keras and vocab.json "
            "in the checkpoints/ folder, then restart the server."
        )
        return

    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            char_to_id = json.load(f)
        id_to_char = {int(v): k for k, v in char_to_id.items()}
        model = tf.keras.models.load_model(model_path)
        model_status = "ready"
        model_message = f"Model ready ({len(char_to_id)} characters in vocabulary)"
    except Exception as e:
        model_status = "error"
        model_message = f"Error loading model: {e}"


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Arabic Speech Recognition</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: #f5f7fa;
    color: #333;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 40px 20px;
  }
  .card {
    background: white;
    border-radius: 16px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    padding: 40px;
    width: 100%;
    max-width: 700px;
  }
  h1 {
    font-size: 28px;
    font-weight: 700;
    color: #222;
    margin-bottom: 6px;
    text-align: center;
  }
  .subtitle {
    text-align: center;
    color: #777;
    font-size: 14px;
    margin-bottom: 30px;
  }
  .status-bar {
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 24px;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .status-ready   { background: #e8f5e9; color: #2e7d32; }
  .status-loading { background: #fff8e1; color: #f57f17; }
  .status-error   { background: #fce4ec; color: #c62828; }
  .status-not_found { background: #fce4ec; color: #c62828; }

  .upload-area {
    border: 2px dashed #c5cae9;
    border-radius: 12px;
    padding: 40px 20px;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    margin-bottom: 20px;
    background: #fafbff;
  }
  .upload-area:hover { border-color: #4a90d9; background: #f0f4ff; }
  .upload-area.drag  { border-color: #4a90d9; background: #e8f0ff; }
  .upload-icon { font-size: 48px; margin-bottom: 12px; }
  .upload-label { font-size: 16px; color: #555; margin-bottom: 6px; }
  .upload-hint { font-size: 13px; color: #999; }
  #file-input { display: none; }

  .waveform-container { margin-bottom: 20px; }
  .waveform-label { font-size: 13px; font-weight: 600; color: #555; margin-bottom: 6px; }
  canvas#waveform {
    width: 100%;
    height: 100px;
    background: #f0f4f8;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
    display: block;
  }

  .file-info {
    font-size: 13px;
    color: #555;
    margin-bottom: 16px;
    min-height: 20px;
    text-align: center;
  }

  .btn {
    display: block;
    width: 100%;
    padding: 14px;
    background: #4a90d9;
    color: white;
    font-size: 16px;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    cursor: pointer;
    transition: background 0.2s;
    margin-bottom: 24px;
  }
  .btn:hover:not(:disabled) { background: #3a7bc8; }
  .btn:disabled { background: #b0c4de; cursor: not-allowed; }

  .result-label { font-size: 13px; font-weight: 600; color: #555; margin-bottom: 8px; }
  .result-box {
    background: #f9f9f9;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 16px;
    min-height: 80px;
    font-size: 22px;
    direction: rtl;
    text-align: right;
    color: #222;
    line-height: 1.6;
    word-break: break-word;
  }
  .result-placeholder { color: #bbb; font-size: 15px; }

  .spinner {
    display: inline-block;
    width: 18px; height: 18px;
    border: 3px solid #fff;
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    vertical-align: middle;
    margin-left: 8px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="card">
  <h1>&#x1F5E3; Arabic Speech Recognition</h1>
  <p class="subtitle">Upload a WAV audio file to transcribe it using the trained CTC model</p>

  <div id="status-bar" class="status-bar status-{{ status }}">
    <span id="status-icon">{{ '✅' if status == 'ready' else '⏳' if status == 'loading' else '❌' }}</span>
    <span id="status-msg">{{ message }}</span>
  </div>

  <div class="upload-area" id="drop-zone" onclick="document.getElementById('file-input').click()">
    <div class="upload-icon">&#x1F4C2;</div>
    <div class="upload-label">Click or drag &amp; drop a WAV file here</div>
    <div class="upload-hint">16 kHz mono WAV, max 10 seconds recommended</div>
    <input type="file" id="file-input" accept=".wav,audio/wav" />
  </div>

  <div class="file-info" id="file-info">No file selected</div>

  <div class="waveform-container">
    <div class="waveform-label">Waveform</div>
    <canvas id="waveform"></canvas>
  </div>

  <button class="btn" id="transcribe-btn" onclick="transcribe()" disabled>
    &#x25B6; Transcribe
  </button>

  <div class="result-label">Result</div>
  <div class="result-box" id="result-box">
    <span class="result-placeholder">Transcription will appear here...</span>
  </div>
</div>

<script>
let selectedFile = null;

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const transcribeBtn = document.getElementById('transcribe-btn');
const resultBox = document.getElementById('result-box');
const waveformCanvas = document.getElementById('waveform');
const ctx = waveformCanvas.getContext('2d');

// Drag and drop
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  selectedFile = file;
  fileInfo.textContent = `${file.name}  (${(file.size / 1024).toFixed(1)} KB)`;
  transcribeBtn.disabled = false;
  drawWaveform(file);
}

function drawWaveform(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    audioCtx.decodeAudioData(e.target.result, (buffer) => {
      const data = buffer.getChannelData(0);
      renderWaveform(data);
    });
  };
  reader.readAsArrayBuffer(file);
}

function renderWaveform(samples) {
  const W = waveformCanvas.offsetWidth;
  const H = waveformCanvas.offsetHeight;
  waveformCanvas.width = W;
  waveformCanvas.height = H;
  ctx.clearRect(0, 0, W, H);

  const mid = H / 2;
  ctx.strokeStyle = '#cccccc';
  ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(W, mid); ctx.stroke();

  const step = Math.max(1, Math.floor(samples.length / W));
  ctx.strokeStyle = '#4a90d9';
  ctx.lineWidth = 1;
  for (let i = 0; i < W; i++) {
    const start = i * step;
    const chunk = samples.slice(start, start + step);
    let min = Infinity, max = -Infinity;
    for (const v of chunk) { if (v < min) min = v; if (v > max) max = v; }
    const yTop = mid - max * (mid - 4);
    const yBot = mid - min * (mid - 4);
    ctx.beginPath();
    ctx.moveTo(i, yTop);
    ctx.lineTo(i, yBot);
    ctx.stroke();
  }
}

async function transcribe() {
  if (!selectedFile) return;

  transcribeBtn.disabled = true;
  transcribeBtn.innerHTML = '&#x23F3; Processing... <span class="spinner"></span>';
  resultBox.innerHTML = '<span class="result-placeholder">Processing audio...</span>';

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const resp = await fetch('/transcribe', { method: 'POST', body: formData });
    const data = await resp.json();
    if (data.error) {
      resultBox.innerHTML = `<span style="color:#c62828">Error: ${data.error}</span>`;
    } else {
      resultBox.textContent = data.text || '(empty result)';
    }
  } catch (err) {
    resultBox.innerHTML = `<span style="color:#c62828">Network error: ${err}</span>`;
  } finally {
    transcribeBtn.disabled = false;
    transcribeBtn.innerHTML = '&#x25B6; Transcribe';
  }
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        status=model_status,
        message=model_message,
    )


@app.route("/status")
def status():
    return jsonify({"status": model_status, "message": model_message})


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if model is None:
        return jsonify({"error": model_message}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".wav"):
        return jsonify({"error": "Only WAV files are supported"}), 400

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        file.save(tmp_path)

    try:
        audio = load_wav(tmp_path)
        mel = wav_to_mel(audio)
        mel_batch = tf.expand_dims(mel, axis=0)
        prediction = model.predict(mel_batch, verbose=0)
        decoded = ctc_greedy_decode(prediction, id_to_char)
        text = decoded[0] if decoded else ""
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
