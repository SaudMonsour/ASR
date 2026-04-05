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


# ===================================================
# Input-length helper
# ===================================================

def compute_input_lengths(spectrograms):
    """
    Compute the number of non-padded time frames in each spectrogram, then
    apply the cumulative downsampling factor of the Conv stack.

    Conv layer summary
    ------------------
    conv1 : kernel=11, stride=2, padding='same'
            T_out = ceil(T_in / 2)  =>  factor 2
    conv2 : kernel=11, stride=1, padding='same'
            T_out = T_in            =>  factor 1

    Combined stride factor = 2 * 1 = 2.
    Using integer arithmetic: output_len = (input_len + 1) // 2
    """
    # Sum across mel bins; frames that are all-zero are padding.
    mask = tf.reduce_sum(tf.abs(spectrograms), axis=-1)        # (batch, time)
    lengths = tf.reduce_sum(tf.cast(mask != 0, tf.int32), axis=-1)  # (batch,)
    # Apply Conv stride factor
    lengths = (lengths + 1) // 2
    return lengths


def compute_label_lengths(labels):
    """Count the number of non-zero (non-padding) tokens per label."""
    return tf.reduce_sum(tf.cast(labels != 0, tf.int32), axis=-1)


# ===================================================
# Learning-rate scheduler
# ===================================================

class ReduceLROnPlateau:
    """
    Halve the learning rate when validation loss has not improved
    for `patience` consecutive epochs.

    Args:
        optimizer  : tf.keras.optimizers instance
        patience   : number of epochs with no improvement before reducing LR
        factor     : multiplicative factor applied to LR (default 0.5)
        min_lr     : minimum allowed learning rate
        verbose    : print a message when LR is reduced
    """

    def __init__(self, optimizer, patience: int = 5, factor: float = 0.5,
                 min_lr: float = 1e-6, verbose: bool = True):
        self.optimizer = optimizer
        self.patience  = patience
        self.factor    = factor
        self.min_lr    = min_lr
        self.verbose   = verbose

        self._best      = float("inf")
        self._wait      = 0

    def step(self, val_loss: float) -> float:
        """Call once per epoch with the current validation loss.
        Returns the (possibly updated) learning rate."""
        if val_loss < self._best:
            self._best = val_loss
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                old_lr = float(self.optimizer.learning_rate)
                new_lr = max(old_lr * self.factor, self.min_lr)
                self.optimizer.learning_rate.assign(new_lr)
                self._wait = 0
                if self.verbose:
                    print(f"  [LR Scheduler] val_loss did not improve for "
                          f"{self.patience} epochs. "
                          f"LR: {old_lr:.2e} -> {new_lr:.2e}")
        return float(self.optimizer.learning_rate)


# ===================================================
# Main training loop
# ===================================================

def main():
    print("=" * 60)
    print("  Arabic ASR -- Training")
    print("=" * 60)

    # -- 0. Check for GPU --------------------------------
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"  GPU found: {[g.name for g in gpus]}")
    else:
        print("  WARNING: No GPU found. Training will be slow on CPU.")
    print("-" * 60)

    # -- 1. Build or load manifest -----------------------
    rebuild = not os.path.exists(config.MANIFEST_PATH)

    if not rebuild:
        test_entries = load_manifest(config.MANIFEST_PATH)
        if test_entries and not os.path.exists(test_entries[0]["wav"]):
            print("  Manifest contains invalid paths. Rebuilding...")
            rebuild = True
        else:
            entries = test_entries

    if rebuild:
        print("\n  Building manifest from corpus...")
        entries = build_manifest(config.WAV_DIR, config.LAB_DIR, config.MANIFEST_PATH)
    else:
        print(f"\n  Loading existing manifest: {config.MANIFEST_PATH}")

    if not entries:
        print("  ERROR: No data found. Check data paths in config.py")
        return

    # Shuffle and split (Train / Val)
    random.seed(42)
    random.shuffle(entries)
    num_val      = int(len(entries) * config.VALIDATION_SPLIT)
    val_entries  = entries[:num_val]
    train_entries = entries[num_val:]

    print(f"  Total utterances : {len(entries)}")
    print(f"  Training         : {len(train_entries)}")
    print(f"  Validation       : {len(val_entries)}")

    # -- 2. Build vocabulary -----------------------------
    texts = [e["text"] for e in entries]
    char_to_id, id_to_char = build_vocab(texts)
    vocab_size = len(char_to_id)
    print(f"  Vocabulary size  : {vocab_size}")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    vocab_path = os.path.join(config.CHECKPOINT_DIR, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(char_to_id, f, ensure_ascii=False, indent=2)
    print(f"  Vocabulary saved : {vocab_path}")

    # -- 3. Create tf.data datasets ----------------------
    train_ds = create_dataset(train_entries, char_to_id, config.BATCH_SIZE)
    val_ds   = create_dataset(val_entries,   char_to_id, config.BATCH_SIZE)
    print(f"  Batch size       : {config.BATCH_SIZE}")

    # -- 4. Build model + optimizer ----------------------
    model     = build_model(n_mels=config.N_MELS, vocab_size=vocab_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-6)
    model.summary()

    # -- 5. Training loop --------------------------------
    print(f"\n  Starting training for {config.EPOCHS} epochs...\n")

    best_val_loss = float("inf")

    for epoch in range(1, config.EPOCHS + 1):

        # Training phase
        train_loss  = 0.0
        num_batches = 0

        progress = tqdm(
            train_ds,
            desc=f"  Epoch {epoch:3d}/{config.EPOCHS} [Train]",
            unit="batch",
        )
        for spectrograms, labels in progress:
            input_lengths = compute_input_lengths(spectrograms)
            label_lengths = compute_label_lengths(labels)

            with tf.GradientTape() as tape:
                y_pred = model(spectrograms, training=True)
                loss   = ctc_loss(labels, y_pred, input_lengths, label_lengths)

            gradients = tape.gradient(loss, model.trainable_variables)

            # Gradient clipping — prevents exploding gradients in RNNs
            gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=5.0)

            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            current_loss = loss.numpy()
            train_loss  += current_loss
            num_batches += 1
            progress.set_postfix({"loss": f"{current_loss:.4f}"})

        avg_train_loss = train_loss / max(num_batches, 1)

        # Validation phase
        val_loss    = 0.0
        val_batches = 0
        for spectrograms, labels in val_ds:
            input_lengths = compute_input_lengths(spectrograms)
            label_lengths = compute_label_lengths(labels)

            y_pred   = model(spectrograms, training=False)
            loss     = ctc_loss(labels, y_pred, input_lengths, label_lengths)
            val_loss += loss.numpy()
            val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        current_lr = scheduler.step(avg_val_loss)
        print(
            f"  Epoch {epoch:3d}  train_loss: {avg_train_loss:.4f}  "
            f"val_loss: {avg_val_loss:.4f}  lr: {current_lr:.2e}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(config.CHECKPOINT_DIR, "model_best.keras")
            model.save(best_path)
            print(f"  New best val_loss={best_val_loss:.4f}  saved: {best_path}")

        # Periodic checkpoint
        if epoch % 10 == 0:
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"model_epoch_{epoch}.keras")
            model.save(ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    # -- 6. Save final model -----------------------------
    final_path = os.path.join(config.CHECKPOINT_DIR, "model_final.keras")
    model.save(final_path)
    print(f"\n  Training complete. Final model saved: {final_path}")


if __name__ == "__main__":
    main()
