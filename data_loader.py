"""
data_loader.py — Load the Arabic Speech Corpus and build tf.data pipelines.

Reads .lab transcription files + .wav audio files, builds a character-level
vocabulary, converts audio to Mel spectrograms, and returns padded batches
ready for CTC training.

Run standalone to generate/verify the manifest:
    python data_loader.py
"""

import os
import json
import glob

import numpy as np
import tensorflow as tf

import config


# ═══════════════════════════════════════════════
# 1.  Read transcriptions from .lab files
# ═══════════════════════════════════════════════

def read_lab_files(lab_dir: str) -> dict:
    """
    Read all .lab files and return a dict:  { utterance_id : transcript_text }
    Each .lab file contains one line of Arabic orthographic text.
    """
    transcripts = {}
    for path in sorted(glob.glob(os.path.join(lab_dir, "*.lab"))):
        utt_id = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            transcripts[utt_id] = text
    return transcripts


# ═══════════════════════════════════════════════
# 2.  Build manifest  (list of dicts saved as JSONL)
# ═══════════════════════════════════════════════

def build_manifest(wav_dir: str, lab_dir: str, output_path: str) -> list:
    """
    Pair each .wav file with its .lab transcript, compute duration,
    and write a JSONL manifest.  Returns the list of entries.
    """
    transcripts = read_lab_files(lab_dir)
    entries = []

    for wav_path in sorted(glob.glob(os.path.join(wav_dir, "*.wav"))):
        utt_id = os.path.splitext(os.path.basename(wav_path))[0]
        text = transcripts.get(utt_id)
        if text is None:
            continue

        # Quick duration check via file size  (PCM 16-bit mono @ 16 kHz)
        file_bytes = os.path.getsize(wav_path)
        duration = file_bytes / (config.SAMPLE_RATE * 2)  # rough estimate

        if duration > config.MAX_AUDIO_LENGTH:
            continue

        entries.append({
            "wav": wav_path,
            "text": text,
            "duration": round(duration, 3),
        })

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Manifest saved: {output_path}  ({len(entries)} utterances)")
    return entries


def load_manifest(manifest_path: str) -> list:
    """Read an existing JSONL manifest back into a list of dicts."""
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


# ═══════════════════════════════════════════════
# 3.  Character vocabulary
# ═══════════════════════════════════════════════

def build_vocab(texts: list) -> tuple:
    """
    Build a character-level vocabulary from a list of text strings.
    Returns:
        char_to_id : dict mapping each character → integer
        id_to_char : dict mapping integer → character
    Index 0 is reserved for the CTC blank token.
    """
    chars = sorted(set("".join(texts)))
    char_to_id = {c: i + 1 for i, c in enumerate(chars)}
    char_to_id["<blank>"] = 0
    id_to_char = {v: k for k, v in char_to_id.items()}
    return char_to_id, id_to_char


def encode_text(text: str, char_to_id: dict) -> list:
    """Convert a text string into a list of integer IDs."""
    return [char_to_id[c] for c in text if c in char_to_id]


# ═══════════════════════════════════════════════
# 4.  Audio → Mel spectrogram  (pure TensorFlow)
# ═══════════════════════════════════════════════

def load_wav(path: str) -> tf.Tensor:
    """Load a WAV file and return a 1-D float32 tensor at SAMPLE_RATE."""
    raw = tf.io.read_file(path)
    audio, sr = tf.audio.decode_wav(raw, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    return audio


def wav_to_mel(audio: tf.Tensor) -> tf.Tensor:
    """Convert a 1-D audio tensor to a log-Mel spectrogram  (time, n_mels)."""
    # STFT
    stft = tf.signal.stft(
        audio,
        frame_length=config.N_FFT,
        frame_step=config.HOP_LENGTH,
    )
    magnitude = tf.abs(stft)

    # Mel filter bank
    num_spectrogram_bins = magnitude.shape[-1] or (config.N_FFT // 2 + 1)
    mel_weight = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=config.N_MELS,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=config.SAMPLE_RATE,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0,
    )
    mel = tf.matmul(magnitude, mel_weight)

    # Log scale
    log_mel = tf.math.log(mel + 1e-6)
    return log_mel  # shape: (time_steps, N_MELS)


# ═══════════════════════════════════════════════
# 5.  tf.data pipeline
# ═══════════════════════════════════════════════

def create_dataset(entries: list, char_to_id: dict, batch_size: int):
    """
    Build a tf.data.Dataset that yields:
        inputs  : padded Mel spectrograms   (batch, time, n_mels)
        labels  : padded integer sequences   (batch, max_label_len)
        input_lengths  : number of time-steps per spectrogram
        label_lengths  : number of characters per label
    """
    wav_paths = [e["wav"] for e in entries]
    texts     = [e["text"] for e in entries]
    labels    = [encode_text(t, char_to_id) for t in texts]

    def generator():
        for wp, lab in zip(wav_paths, labels):
            audio = load_wav(wp)
            mel   = wav_to_mel(audio)
            yield mel, np.array(lab, dtype=np.int32)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, config.N_MELS), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        ),
    )

    # Pad, batch, prefetch
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=([None, config.N_MELS], [None]),
        padding_values=(0.0, 0),
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# ═══════════════════════════════════════════════
# 6.  Standalone: generate manifest
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    entries = build_manifest(config.WAV_DIR, config.LAB_DIR, config.MANIFEST_PATH)
    if entries:
        texts = [e["text"] for e in entries]
        char_to_id, id_to_char = build_vocab(texts)
        print(f"   Vocabulary size: {len(char_to_id)} characters")
        print(f"   Sample text:     {texts[0][:60]}...")
