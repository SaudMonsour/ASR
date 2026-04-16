"""
web_app.py — Flask web interface for Arabic Speech Recognition.

Provides a browser-based UI for uploading media files and getting transcriptions
using OpenAI's Whisper model (whisper-base).

Usage:
    python web_app.py
"""

import os
import tempfile
import threading

import whisper
from flask import Flask, request, jsonify, render_template_string
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

# Global model state
model = None
model_status = "loading"
model_message = "Loading model..."


def load_model():
    global model, model_status, model_message
    try:
        model = whisper.load_model("base")
        model_status = "ready"
        model_message = "Model ready"
    except Exception as e:
        model_status = "error"
        model_message = f"Error loading model: {e}"


threading.Thread(target=load_model, daemon=True).start()


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" dir="ltr">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Arabic Speech Recognition</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --accent:      #3b6fd4;
    --accent-dark: #2f5ab8;
    --accent-pale: #eef2fb;
    --surface:     #ffffff;
    --bg:          #f3f5f9;
    --border:      #dde1ea;
    --text-primary:   #1a1d23;
    --text-secondary: #5a6074;
    --text-muted:     #9399ab;
    --success-bg:  #edf7ed;
    --success-fg:  #1e6b24;
    --warning-bg:  #fff8e6;
    --warning-fg:  #7a5300;
    --error-bg:    #fdedf0;
    --error-fg:    #9b1c2e;
    --radius-sm:   6px;
    --radius-md:   10px;
    --radius-lg:   16px;
    --shadow:      0 2px 16px rgba(0,0,0,0.07);
    --transition:  0.18s ease;
  }

  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text-primary);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 48px 16px 64px;
  }

  /* ── Header ── */
  .page-header {
    text-align: center;
    margin-bottom: 32px;
  }
  .page-header h1 {
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: var(--text-primary);
  }
  .page-header p {
    margin-top: 6px;
    font-size: 14px;
    color: var(--text-secondary);
  }

  /* ── Card ── */
  .card {
    background: var(--surface);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow);
    padding: 36px 32px;
    width: 100%;
    max-width: 680px;
  }

  /* ── Status banner ── */
  .status-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    border-radius: var(--radius-md);
    padding: 11px 14px;
    font-size: 13px;
    font-weight: 500;
    margin-bottom: 28px;
    transition: background var(--transition);
  }
  .status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .status-ready   { background: var(--success-bg); color: var(--success-fg); }
  .status-ready   .status-dot { background: var(--success-fg); }
  .status-loading { background: var(--warning-bg); color: var(--warning-fg); }
  .status-loading .status-dot { background: var(--warning-fg); animation: pulse 1.2s infinite; }
  .status-error   { background: var(--error-bg); color: var(--error-fg); }
  .status-error   .status-dot { background: var(--error-fg); }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
  }

  /* ── Section label ── */
  .section-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-muted);
    margin-bottom: 8px;
  }

  /* ── Drop zone ── */
  .drop-zone {
    border: 2px dashed var(--border);
    border-radius: var(--radius-md);
    padding: 36px 20px;
    text-align: center;
    cursor: pointer;
    transition: border-color var(--transition), background var(--transition);
    margin-bottom: 20px;
    background: var(--bg);
    position: relative;
  }
  .drop-zone:hover { border-color: var(--accent); background: var(--accent-pale); }
  .drop-zone.drag  { border-color: var(--accent); background: var(--accent-pale); }

  .drop-icon {
    width: 44px; height: 44px;
    margin: 0 auto 12px;
    color: var(--text-muted);
  }
  .drop-icon svg { width: 100%; height: 100%; }

  .drop-primary {
    font-size: 15px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 4px;
  }
  .drop-secondary { font-size: 13px; color: var(--text-muted); }

  #file-input { display: none; }

  /* ── File info chip ── */
  .file-info {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: var(--text-secondary);
    min-height: 24px;
    margin-bottom: 20px;
  }
  .file-info.has-file { color: var(--text-primary); font-weight: 500; }

  /* ── Waveform ── */
  .waveform-wrap { margin-bottom: 24px; }
  canvas#waveform {
    width: 100%;
    height: 88px;
    background: var(--bg);
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
    display: block;
  }

  /* ── Transcribe button ── */
  .btn-transcribe {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    width: 100%;
    padding: 13px 20px;
    background: var(--accent);
    color: #fff;
    font-size: 15px;
    font-weight: 600;
    border: none;
    border-radius: var(--radius-md);
    cursor: pointer;
    transition: background var(--transition), opacity var(--transition), transform 0.1s;
    margin-bottom: 28px;
    letter-spacing: 0.1px;
  }
  .btn-transcribe:hover:not(:disabled) {
    background: var(--accent-dark);
    transform: translateY(-1px);
  }
  .btn-transcribe:active:not(:disabled) { transform: translateY(0); }
  .btn-transcribe:disabled { opacity: 0.45; cursor: not-allowed; }

  /* ── Result ── */
  .result-box {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 18px 16px;
    min-height: 80px;
    font-size: 22px;
    direction: rtl;
    text-align: right;
    color: var(--text-primary);
    line-height: 1.65;
    word-break: break-word;
    transition: background var(--transition);
  }
  .result-box.has-result { background: var(--surface); }
  .result-placeholder { color: var(--text-muted); font-size: 14px; font-style: italic; }

  /* ── Divider ── */
  .divider {
    height: 1px;
    background: var(--border);
    margin: 24px 0;
  }

  /* ── Spinner ── */
  .spinner {
    width: 16px; height: 16px;
    border: 2px solid rgba(255,255,255,0.4);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.65s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Footer ── */
  .page-footer {
    margin-top: 28px;
    font-size: 12px;
    color: var(--text-muted);
    text-align: center;
  }

  /* ── Responsive ── */
  @media (max-width: 480px) {
    body { padding: 24px 12px 48px; }
    .card { padding: 24px 16px; }
    .page-header h1 { font-size: 21px; }
    .result-box { font-size: 18px; }
  }
</style>
</head>
<body>

<header class="page-header">
  <h1>Arabic Speech Recognition</h1>
  <p>Upload an audio or video file to transcribe spoken Arabic with Whisper</p>
</header>

<main class="card">

  <div id="status-banner" class="status-banner status-{{ status }}">
    <span class="status-dot"></span>
    <span id="status-msg">{{ message }}</span>
  </div>

  <div class="drop-zone" id="drop-zone" onclick="document.getElementById('file-input').click()" role="button" tabindex="0" aria-label="Upload audio or video file">
    <div class="drop-icon">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
        <polyline points="17 8 12 3 7 8"/>
        <line x1="12" y1="3" x2="12" y2="15"/>
      </svg>
    </div>
    <div class="drop-primary">Click or drag any audio/video file here</div>
    <div class="drop-secondary">MP3, MP4, WAV, M4A, WebM, and other FFmpeg-supported files up to 200 MB</div>
    <input type="file" id="file-input" />
  </div>

  <div class="file-info" id="file-info">No file selected</div>

  <div class="waveform-wrap">
    <div class="section-label">Waveform</div>
    <canvas id="waveform"></canvas>
  </div>

  <button class="btn-transcribe" id="transcribe-btn" onclick="transcribe()" disabled>
    <span id="btn-label">Transcribe</span>
  </button>

  <div class="section-label">Transcription</div>
  <div class="result-box" id="result-box">
    <span class="result-placeholder">The transcription will appear here after you upload and process a file.</span>
  </div>

</main>

<footer class="page-footer">
  Powered by OpenAI Whisper (whisper-base)
</footer>

<script>
"use strict";
let selectedFile = null;

const dropZone     = document.getElementById('drop-zone');
const fileInput    = document.getElementById('file-input');
const fileInfo     = document.getElementById('file-info');
const transcribeBtn = document.getElementById('transcribe-btn');
const resultBox    = document.getElementById('result-box');
const waveformCanvas = document.getElementById('waveform');
const wCtx         = waveformCanvas.getContext('2d');

/* ── keyboard activation for drop zone ── */
dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});

/* ── drag and drop ── */
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag'); });
['dragleave', 'dragend'].forEach(ev => dropZone.addEventListener(ev, () => dropZone.classList.remove('drag')));
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
  const kb = (file.size / 1024).toFixed(1);
  fileInfo.textContent = file.name + '  \u00b7  ' + kb + ' KB';
  fileInfo.classList.add('has-file');
  transcribeBtn.disabled = false;
  drawWaveform(file);
}

/* ── waveform rendering ── */
function drawWaveform(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    audioCtx.decodeAudioData(e.target.result).then((buffer) => {
      renderWaveform(buffer.getChannelData(0));
    }).catch(() => {
      clearWaveform();
    });
  };
  reader.readAsArrayBuffer(file);
}

function clearWaveform() {
  const W = waveformCanvas.offsetWidth || 600;
  const H = 88;
  waveformCanvas.width = W;
  waveformCanvas.height = H;
  wCtx.clearRect(0, 0, W, H);
}

function renderWaveform(samples) {
  const W = waveformCanvas.offsetWidth || 600;
  const H = 88;
  waveformCanvas.width  = W * (window.devicePixelRatio || 1);
  waveformCanvas.height = H * (window.devicePixelRatio || 1);
  waveformCanvas.style.width  = W + 'px';
  waveformCanvas.style.height = H + 'px';
  wCtx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);

  wCtx.clearRect(0, 0, W, H);

  const mid = H / 2;
  wCtx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--border').trim() || '#dde1ea';
  wCtx.lineWidth = 1;
  wCtx.beginPath(); wCtx.moveTo(0, mid); wCtx.lineTo(W, mid); wCtx.stroke();

  const step = Math.max(1, Math.floor(samples.length / W));
  const accentColor = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#3b6fd4';

  for (let i = 0; i < W; i++) {
    const start = i * step;
    const chunk = samples.slice(start, start + step);
    let min = Infinity, max = -Infinity;
    for (const v of chunk) { if (v < min) min = v; if (v > max) max = v; }
    const yTop = mid - max * (mid - 4);
    const yBot = mid - min * (mid - 4);
    wCtx.strokeStyle = accentColor;
    wCtx.lineWidth = 1;
    wCtx.beginPath();
    wCtx.moveTo(i, yTop);
    wCtx.lineTo(i, Math.max(yBot, yTop + 1));
    wCtx.stroke();
  }
}

/* ── transcription request ── */
async function transcribe() {
  if (!selectedFile) return;

  transcribeBtn.disabled = true;
  document.getElementById('btn-label').textContent = 'Processing...';
  transcribeBtn.insertAdjacentHTML('beforeend', '<span class="spinner" id="spinner"></span>');

  resultBox.innerHTML = '<span class="result-placeholder">Processing file\u2026</span>';
  resultBox.classList.remove('has-result');

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const resp = await fetch('/transcribe', { method: 'POST', body: formData });
    const data = await resp.json().catch(() => ({ error: 'The server returned an unreadable response.' }));
    if (data.error) {
      resultBox.innerHTML = '<span style="color:var(--error-fg);font-size:14px">Error: ' + escapeHtml(data.error) + '</span>';
    } else {
      resultBox.textContent = data.text || '(no output)';
      resultBox.classList.add('has-result');
    }
  } catch (err) {
    resultBox.innerHTML = '<span style="color:var(--error-fg);font-size:14px">Network error: ' + escapeHtml(String(err)) + '</span>';
  } finally {
    transcribeBtn.disabled = false;
    document.getElementById('btn-label').textContent = 'Transcribe';
    const sp = document.getElementById('spinner');
    if (sp) sp.remove();
  }
}

function escapeHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/* ── auto-poll status until model is ready ── */
(function pollStatus() {
  const banner = document.getElementById('status-banner');
  const msg    = document.getElementById('status-msg');
  if (banner.classList.contains('status-loading')) {
    setTimeout(() => {
      fetch('/status').then(r => r.json()).then(d => {
        msg.textContent = d.message;
        banner.className = 'status-banner status-' + d.status;
        if (d.status === 'loading') {
          pollStatus();
        }
      }).catch(() => pollStatus());
    }, 2000);
  }
})();
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


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.errorhandler(RequestEntityTooLarge)
def file_too_large(error):
    return jsonify({"error": "File is too large. Upload a file up to 200 MB."}), 413

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if model_status != "ready":
        return jsonify({"error": f"Model not ready: {model_message}"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio_file = request.files["file"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(audio_file.filename)
    suffix = os.path.splitext(filename)[1].lower() or ".upload"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        audio_file.save(tmp_path)

    try:
        result = model.transcribe(tmp_path, language="ar", fp16=False)
        text = result["text"].strip()
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": f"Could not transcribe this file. Upload an audio/video file FFmpeg can read. Details: {e}"}), 500
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
