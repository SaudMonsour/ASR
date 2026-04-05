"""
predict.py — Run inference on a single audio file.

Usage:
    python predict.py --audio path/to/file.wav
    python predict.py                              # uses a default test file
"""

import os
import json
import argparse

import tensorflow as tf

import config
from data_loader import load_wav, wav_to_mel
from model import ctc_greedy_decode


def main():
    parser = argparse.ArgumentParser(description="Arabic ASR — Predict")
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to a .wav file (16 kHz, mono).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(config.CHECKPOINT_DIR, "model_final.keras"),
        help="Path to a saved Keras model.",
    )
    args = parser.parse_args()

    # ── 1. Load vocabulary ─────────────────────
    vocab_path = os.path.join(config.CHECKPOINT_DIR, "vocab.json")
    if not os.path.exists(vocab_path):
        print("ERROR: Vocabulary file not found. Train the model first (python train.py).")
        return

    with open(vocab_path, "r", encoding="utf-8") as f:
        char_to_id = json.load(f)
    id_to_char = {int(v): k for k, v in char_to_id.items()}
    print(f"Vocabulary loaded ({len(char_to_id)} characters)")

    # ── 2. Load model ──────────────────────────
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        print("   Train the model first:  python train.py")
        return

    model = tf.keras.models.load_model(args.model)
    print(f"Model loaded: {args.model}")

    # ── 3. Pick an audio file ──────────────────
    audio_path = args.audio
    if audio_path is None:
        # Default: use the first WAV in the corpus
        import glob
        wavs = sorted(glob.glob(os.path.join(config.WAV_DIR, "*.wav")))
        if not wavs:
            print("ERROR: No WAV files found. Specify --audio path/to/file.wav")
            return
        audio_path = wavs[0]

    print(f"\nAudio file: {audio_path}")

    # ── 4. Process audio ───────────────────────
    audio = load_wav(audio_path)
    mel = wav_to_mel(audio)                        # (time, n_mels)
    mel_batch = tf.expand_dims(mel, axis=0)        # (1, time, n_mels)

    # ── 5. Predict ─────────────────────────────
    y_pred = model.predict(mel_batch, verbose=0)
    decoded = ctc_greedy_decode(y_pred, id_to_char)

    print(f"\nPrediction:\n   {decoded[0]}\n")


if __name__ == "__main__":
    main()
