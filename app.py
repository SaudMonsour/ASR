"""
app.py — Simple GUI for Arabic Speech Recognition.

Uses Python's built-in tkinter library.
Loads a trained model from checkpoints/ and transcribes WAV audio files.

Usage:
    python app.py
"""

import os
import json
import threading
import wave

import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox

import config
from data_loader import load_wav, wav_to_mel
from model import ctc_greedy_decode

# Try importing sounddevice for microphone recording (optional)
try:
    import sounddevice as sd
    HAS_MIC = True
except ImportError:
    HAS_MIC = False


# ────────────────────────────────────────────
# Simple Waveform Drawing
# ────────────────────────────────────────────

class WaveformCanvas(tk.Canvas):
    """Draws audio waveform on a tkinter Canvas."""

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("bg", "#f0f0f0")
        kwargs.setdefault("height", 120)
        kwargs.setdefault("highlightthickness", 1)
        kwargs.setdefault("highlightbackground", "#cccccc")
        super().__init__(parent, **kwargs)
        self.audio = None
        self.bind("<Configure>", lambda e: self.draw())

    def set_audio(self, samples):
        """Set audio data (1D numpy array) and redraw."""
        self.audio = samples
        self.draw()

    def clear(self):
        self.audio = None
        self.draw()

    def draw(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        mid = h / 2

        # Centre line
        self.create_line(0, mid, w, mid, fill="#cccccc")

        if self.audio is None or len(self.audio) == 0:
            self.create_text(w / 2, mid, text="No audio loaded",
                             fill="#999999", font=("Segoe UI", 9))
            return

        # Downsample audio to fit canvas width
        step = max(1, len(self.audio) // w)
        for i in range(0, len(self.audio), step):
            chunk = self.audio[i:i + step]
            x = int(i / len(self.audio) * w)
            y_top = mid - float(np.max(chunk)) * (mid - 4)
            y_bot = mid - float(np.min(chunk)) * (mid - 4)
            self.create_line(x, y_top, x, y_bot, fill="#4a90d9", width=1)


# ────────────────────────────────────────────
# Main Application
# ────────────────────────────────────────────

class ASRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Arabic ASR")
        self.root.geometry("650x580")
        self.root.configure(bg="white")

        # State variables
        self.audio_data = None      # numpy array of audio samples
        self.audio_path = None      # path to the WAV file
        self.model = None           # loaded Keras model
        self.id_to_char = None      # vocab mapping (int -> char)
        self.is_recording = False
        self.mic_frames = []

        # --- UI Attributes (initialized here for linter/clarity) ---
        self.status_label = None
        self.browse_btn = None
        self.record_btn = None
        self.file_label = None
        self.waveform = None
        self.transcribe_btn = None
        self.result_text = None
        self.mic_stream = None

        self.build_ui()

        # Load model in background so the window appears immediately
        threading.Thread(target=self.load_model, daemon=True).start()

    # ── Build the interface ────────────────

    def build_ui(self):

        # --- Title ---
        tk.Label(self.root, text="Arabic Speech Recognition",
                 font=("Segoe UI", 18, "bold"), bg="white", fg="#333333"
                 ).pack(pady=(16, 4))

        # --- Status label ---
        self.status_label = tk.Label(self.root, text="⏳ Loading model...",
                                     font=("Segoe UI", 10), bg="white", fg="#888888")
        self.status_label.pack()

        # --- Buttons row ---
        btn_frame = tk.Frame(self.root, bg="white")
        btn_frame.pack(pady=(16, 4))

        self.browse_btn = tk.Button(
            btn_frame, text="📂 Open WAV File",
            font=("Segoe UI", 11), padx=12, pady=6,
            bg="#f0f0f0", relief="groove", cursor="hand2",
            command=self.browse_file,
        )
        self.browse_btn.pack(side="left", padx=6)

        self.record_btn = tk.Button(
            btn_frame, text="🎤 Record",
            font=("Segoe UI", 11), padx=12, pady=6,
            bg="#f0f0f0", relief="groove", cursor="hand2",
            command=self.toggle_record,
        )
        self.record_btn.pack(side="left", padx=6)
        if not HAS_MIC:
            self.record_btn.config(state="disabled", text="🎤 Record (no sounddevice)")

        # --- File info ---
        self.file_label = tk.Label(self.root, text="No file selected",
                                   font=("Segoe UI", 9), bg="white", fg="#999999")
        self.file_label.pack(pady=(4, 8))

        # --- Waveform ---
        tk.Label(self.root, text="Waveform", font=("Segoe UI", 11, "bold"),
                 bg="white", fg="#555555").pack(anchor="w", padx=24)

        self.waveform = WaveformCanvas(self.root)
        self.waveform.pack(fill="x", padx=24, pady=(2, 12))

        # --- Transcribe button ---
        self.transcribe_btn = tk.Button(
            self.root, text="▶ Transcribe",
            font=("Segoe UI", 13, "bold"), padx=20, pady=8,
            bg="#4a90d9", fg="white", activebackground="#3a7bc8",
            relief="flat", cursor="hand2",
            command=self.transcribe,
        )
        self.transcribe_btn.pack(pady=(0, 12))

        # --- Result area ---
        tk.Label(self.root, text="Result", font=("Segoe UI", 11, "bold"),
                 bg="white", fg="#555555").pack(anchor="w", padx=24)

        self.result_text = tk.Text(
            self.root, font=("Segoe UI", 16), height=4, wrap="word",
            bg="#f9f9f9", fg="#222222", relief="groove", padx=10, pady=8,
        )
        self.result_text.tag_configure("rtl", justify="right")
        self.result_text.pack(fill="both", expand=True, padx=24, pady=(2, 16))

    # ── Load the trained model ─────────────

    def load_model(self):
        model_path = os.path.join(config.CHECKPOINT_DIR, "model_final.keras")
        vocab_path = os.path.join(config.CHECKPOINT_DIR, "vocab.json")

        if not os.path.exists(model_path) or not os.path.exists(vocab_path):
            self.root.after(0, self.set_status,
                            "❌ Model not found — place model_final.keras and vocab.json in checkpoints/",
                            "#cc0000")
            return

        try:
            # Load vocabulary
            with open(vocab_path, "r", encoding="utf-8") as f:
                char_to_id = json.load(f)
            self.id_to_char = {int(v): k for k, v in char_to_id.items()}

            # Load model
            self.model = tf.keras.models.load_model(model_path)

            self.root.after(0, self.set_status, "✅ Model loaded and ready", "#228B22")
        except Exception as e:
            self.root.after(0, self.set_status, f"❌ Error: {e}", "#cc0000")

    def set_status(self, text, color):
        self.status_label.config(text=text, fg=color)

    # ── Browse for a WAV file ──────────────

    def browse_file(self):
        path = filedialog.askopenfilename(
            title="Select a WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if not path:
            return

        self.audio_path = path
        self.file_label.config(text=os.path.basename(path), fg="#333333")

        try:
            audio_tensor = load_wav(path)
            self.audio_data = audio_tensor.numpy()
            self.waveform.set_audio(self.audio_data)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load audio:\n{e}")
            self.audio_data = None
            self.waveform.clear()

    # ── Record from microphone ─────────────

    def toggle_record(self):
        if not HAS_MIC:
            return

        if not self.is_recording:
            # Start recording
            self.is_recording = True
            self.mic_frames = []
            self.record_btn.config(text="⏹ Stop", bg="#ffcccc")
            self.file_label.config(text="Recording...", fg="#cc0000")
            self.waveform.clear()

            self.mic_stream = sd.InputStream(
                samplerate=config.SAMPLE_RATE, channels=1, dtype="float32",
                callback=self.mic_callback,
            )
            self.mic_stream.start()
        else:
            # Stop recording
            self.is_recording = False
            self.mic_stream.stop()
            self.mic_stream.close()
            self.record_btn.config(text="🎤 Record", bg="#f0f0f0")

            if self.mic_frames:
                samples = np.concatenate(self.mic_frames).flatten()
                self.audio_data = samples
                self.waveform.set_audio(samples)
                self.file_label.config(
                    text=f"Recorded {len(samples)/config.SAMPLE_RATE:.1f}s",
                    fg="#333333",
                )
                # Save to temp file so TensorFlow can read it
                self.save_temp_wav(samples)
            else:
                self.file_label.config(text="No audio captured", fg="#cc0000")

    def mic_callback(self, indata, frames, time_info, status):
        if self.is_recording:
            self.mic_frames.append(indata.copy())

    def save_temp_wav(self, samples):
        """Save recorded audio to a temporary WAV file."""
        path = os.path.join(config.PROJECT_ROOT, "temp_recording.wav")
        pcm = (samples * 32767).astype(np.int16)
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(config.SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        self.audio_path = path

    # ── Run transcription ──────────────────

    def transcribe(self):
        if self.model is None:
            messagebox.showwarning(
                "Model Not Loaded",
                "The model is not loaded.\n\n"
                "Put these files in the checkpoints/ folder:\n"
                "  • model_final.keras\n"
                "  • vocab.json",
            )
            return

        if self.audio_data is None or self.audio_path is None:
            messagebox.showinfo("No Audio", "Please open a WAV file or record audio first.")
            return

        self.transcribe_btn.config(text="⏳ Processing...", state="disabled")
        self.root.update()

        # Run in background thread
        threading.Thread(target=self.run_inference, daemon=True).start()

    def run_inference(self):
        try:
            # Load and convert audio to mel spectrogram
            audio = load_wav(self.audio_path)
            mel = wav_to_mel(audio)                     # shape: (time, n_mels)
            mel_batch = tf.expand_dims(mel, axis=0)     # shape: (1, time, n_mels)

            # Run model prediction
            prediction = self.model.predict(mel_batch, verbose=0)
            text = ctc_greedy_decode(prediction, self.id_to_char)
            result = text[0] if text else ""

            self.root.after(0, self.show_result, result)
        except Exception as e:
            self.root.after(0, self.show_result, f"Error: {e}")

    def show_result(self, text):
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text, "rtl")
        self.transcribe_btn.config(text="▶ Transcribe", state="normal")


# ────────────────────────────────────────────
# Run the app
# ────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app = ASRApp(root)
    root.mainloop()
