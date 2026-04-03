"""
config.py — All project settings in one place.
Change these values to match your setup.
"""

import os

# ──────────────────────────────────────────────
# Paths  (relative to project root)
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data directory (should contain 'wav' and 'lab' folders)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
WAV_DIR = os.path.join(DATA_DIR, "wav")
LAB_DIR = os.path.join(DATA_DIR, "lab")

# Where to save the generated manifest
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "manifests", "train.jsonl")

# Model checkpoints
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# ──────────────────────────────────────────────
# Audio settings
# ──────────────────────────────────────────────
SAMPLE_RATE = 16_000          # 16 kHz
MAX_AUDIO_LENGTH = 10.0       # seconds — clips longer than this are skipped
N_FFT = 512                   # FFT window size
HOP_LENGTH = 160              # 10 ms hop  (SAMPLE_RATE * 0.01)
N_MELS = 80                   # Mel spectrogram bins

# ──────────────────────────────────────────────
# Training hyper-parameters
# ──────────────────────────────────────────────
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 3e-4
VALIDATION_SPLIT = 0.1        # 10% for validation during training
