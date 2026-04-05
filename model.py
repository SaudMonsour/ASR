"""
model.py — CTC-based ASR model built with TensorFlow / Keras.

Architecture:  Conv1D  ->  Bidirectional GRU  ->  Dense  ->  CTC loss
Simple, proven, and easy to read.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def build_model(n_mels: int, vocab_size: int) -> Model:
    """
    Build and return a Keras model for CTC-based speech recognition.

    Args:
        n_mels     : number of Mel spectrogram bins  (input feature dim)
        vocab_size : total characters including CTC blank (index 0)

    Returns:
        An uncompiled Keras Model.
        (Compilation is done in train.py so the optimizer is co-located
        with gradient-clipping and LR-scheduling logic.)
    """

    # --- Input ---
    # Shape: (batch, time_steps, n_mels)
    inputs = layers.Input(shape=(None, n_mels), name="audio_input")

    # --- Feature extraction (1-D convolutions) ---
    # conv1: stride=2  =>  T_out = ceil(T_in / 2)
    x = layers.Conv1D(128, kernel_size=11, strides=2, padding="same", activation="relu", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)

    # conv2: stride=1  =>  T_out = T_in  (no additional downsampling)
    x = layers.Conv1D(128, kernel_size=11, strides=1, padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)

    x = layers.Dropout(0.2, name="drop1")(x)

    # --- Recurrent layers ---
    x = layers.Bidirectional(layers.GRU(256, return_sequences=True), name="bigru1")(x)
    x = layers.Dropout(0.2, name="drop2")(x)

    x = layers.Bidirectional(layers.GRU(256, return_sequences=True), name="bigru2")(x)
    x = layers.Dropout(0.2, name="drop3")(x)

    # --- Output ---
    # activation=None because tf.nn.ctc_loss expects raw logits.
    outputs = layers.Dense(vocab_size, activation=None, name="output")(x)

    model = Model(inputs, outputs, name="ASR_CTC")
    return model


# ===================================================
# CTC Loss function
# ===================================================

def ctc_loss(y_true, y_pred, input_lengths, label_lengths):
    """
    Compute CTC loss.

    Args:
        y_true         : integer labels         (batch, max_label_len)
        y_pred         : logit outputs          (batch, time_steps, vocab_size)
        input_lengths  : actual encoder lengths (batch,)  — already
                         adjusted for Conv strides
        label_lengths  : actual label lengths   (batch,)

    Returns:
        Scalar loss.
    """
    loss = tf.nn.ctc_loss(
        labels=tf.cast(y_true, tf.int32),
        logits=y_pred,
        label_length=tf.cast(label_lengths, tf.int32),
        logit_length=tf.cast(input_lengths, tf.int32),
        logits_time_major=False,
        blank_index=0,
    )
    return tf.reduce_mean(loss)


# ===================================================
# CTC Decode (greedy)
# ===================================================

def ctc_greedy_decode(y_pred, id_to_char: dict) -> list:
    """
    Greedy-decode a batch of logit outputs into text strings.

    Args:
        y_pred     : model output  (batch, time_steps, vocab_size)
        id_to_char : mapping from integer -> character

    Returns:
        List of decoded strings.
    """
    indices = tf.argmax(y_pred, axis=-1).numpy()  # (batch, time)

    results = []
    for seq in indices:
        chars = []
        prev = -1
        for idx in seq:
            if idx != 0 and idx != prev:   # skip blanks and repeated tokens
                ch = id_to_char.get(int(idx), "")
                chars.append(ch)
            prev = idx
        results.append("".join(chars))
    return results


# ===================================================
# Quick sanity check
# ===================================================

if __name__ == "__main__":
    model = build_model(n_mels=80, vocab_size=50)
    model.summary()
