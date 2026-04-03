"""
train.py — Train the Arabic ASR model.

Usage:
    python train.py
"""

import os
import json
import random
from tqdm import tqdm

import tensorflow as tf

import config
from data_loader import (
    build_manifest,
    load_manifest,
    build_vocab,
    encode_text,
    create_dataset,
)
from model import build_model, ctc_loss, ctc_greedy_decode


def compute_input_lengths(spectrograms):
    """
    After Conv1D with stride=2, the time dimension is halved.
    We compute the actual (non-padded) length of each spectrogram.
    """
    # Sum across mel bins; non-zero frames are actual data
    mask = tf.reduce_sum(tf.abs(spectrograms), axis=-1)   # (batch, time)
    lengths = tf.reduce_sum(tf.cast(mask != 0, tf.int32), axis=-1)
    # Account for conv stride=2
    lengths = (lengths + 1) // 2
    return lengths


def compute_label_lengths(labels):
    """Count the number of non-zero (non-padding) tokens per label."""
    return tf.reduce_sum(tf.cast(labels != 0, tf.int32), axis=-1)


# ═══════════════════════════════════════════════
# Main training loop
# ═══════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Arabic ASR — Training")
    print("=" * 60)

    # ── 0. Check for GPU ────────────────────────
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
    else:
        print("⚠️ NO GPU FOUND. Training will be extremely slow on CPU.")
    print("-" * 60)

    # ── 1. Build or load manifest ──────────────
    rebuild = not os.path.exists(config.MANIFEST_PATH)

    if not rebuild:
        # Look at the first entry; if the file doesn't exist, the paths are likely
        # from a different machine (e.g. Windows paths on Colab).
        test_entries = load_manifest(config.MANIFEST_PATH)
        if test_entries and not os.path.exists(test_entries[0]["wav"]):
            print("⚠️ Manifest contains invalid paths (likely from a different OS). Rebuilding...")
            rebuild = True
        else:
            entries = test_entries

    if rebuild:
        print("\n📦 Building manifest from corpus...")
        entries = build_manifest(config.WAV_DIR, config.LAB_DIR, config.MANIFEST_PATH)
    else:
        print(f"\n📦 Loading existing manifest: {config.MANIFEST_PATH}")

    if not entries:
        print("❌ No data found. Check your data paths in config.py")
        return

    # Shuffle and Split (Train / Val)
    random.seed(42)
    random.shuffle(entries)
    num_val = int(len(entries) * config.VALIDATION_SPLIT)
    val_entries = entries[:num_val]
    train_entries = entries[num_val:]

    print(f"   Total utterances: {len(entries)}")
    print(f"   Training:         {len(train_entries)}")
    print(f"   Validation:       {len(val_entries)}")

    # ── 2. Build vocabulary ────────────────────
    texts = [e["text"] for e in entries]
    char_to_id, id_to_char = build_vocab(texts)
    vocab_size = len(char_to_id)
    print(f"   Vocabulary size:  {vocab_size}")

    # Save vocabulary for later use in predict.py
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    vocab_path = os.path.join(config.CHECKPOINT_DIR, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(char_to_id, f, ensure_ascii=False, indent=2)
    print(f"   Vocabulary saved: {vocab_path}")

    # ── 3. Create datasets ─────────────────────
    train_ds = create_dataset(train_entries, char_to_id, config.BATCH_SIZE)
    val_ds   = create_dataset(val_entries, char_to_id, config.BATCH_SIZE)
    print(f"   Batch size:       {config.BATCH_SIZE}")

    # ── 4. Build model ─────────────────────────
    model = build_model(n_mels=config.N_MELS, vocab_size=vocab_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.summary()

    # ── 5. Training loop ───────────────────────
    print(f"\n🚀 Starting training for {config.EPOCHS} epochs...\n")

    best_val_loss = float("inf")

    for epoch in range(1, config.EPOCHS + 1):
        # --- Training phase ---
        train_loss = 0.0
        num_batches = 0
        
        progress = tqdm(train_ds, desc=f"  Epoch {epoch:3d}/{config.EPOCHS} [Train]", unit="batch")
        for spectrograms, labels in progress:
            input_lengths = compute_input_lengths(spectrograms)
            label_lengths  = compute_label_lengths(labels)

            with tf.GradientTape() as tape:
                y_pred = model(spectrograms, training=True)
                loss = ctc_loss(labels, y_pred, input_lengths, label_lengths)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            current_loss = loss.numpy()
            train_loss += current_loss
            num_batches += 1
            progress.set_postfix({"loss": f"{current_loss:.4f}"})

        avg_train_loss = train_loss / max(num_batches, 1)

        # --- Validation phase ---
        val_loss = 0.0
        val_batches = 0
        for spectrograms, labels in val_ds:
            input_lengths = compute_input_lengths(spectrograms)
            label_lengths  = compute_label_lengths(labels)
            
            y_pred = model(spectrograms, training=False)
            loss = ctc_loss(labels, y_pred, input_lengths, label_lengths)
            
            val_loss += loss.numpy()
            val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        
        print(f"  ✨ Results: train_loss: {avg_train_loss:.4f}  —  val_loss: {avg_val_loss:.4f}")

        # Save "best" model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(config.CHECKPOINT_DIR, "model_best.keras")
            model.save(best_path)
            print(f"  ⭐ New best validation loss! Saved: {best_path}")

        # Save checkpoint periodically
        if epoch % 10 == 0:
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"model_epoch_{epoch}.keras")
            model.save(ckpt_path)
            print(f"  💾 Checkpoint saved: {ckpt_path}")

    # ── 6. Save final model ────────────────────
    final_path = os.path.join(config.CHECKPOINT_DIR, "model_final.keras")
    model.save(final_path)
    print(f"\n✅ Training complete. Final model saved: {final_path}")


if __name__ == "__main__":
    main()
