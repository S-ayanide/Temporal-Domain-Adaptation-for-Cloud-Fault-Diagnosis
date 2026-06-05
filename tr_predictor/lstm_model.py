"""
lstm_model.py — Weighted LSTM weak learner for Tr-Predictor.

Each weak learner is a small LSTM that supports per-sample importance
weights during training (passed as sample_weight to model.fit).

Architecture follows Liu et al. (2022) §3.3:
  Input → LSTM(units) → Dense(units//2) → Dense(n_targets)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_weak_learner(
    seq_len: int,
    n_features: int = 1,
    n_targets: int = 1,
    lstm_units: int = 64,
    dense_units: int = 32,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Build a small LSTM regression model.

    Parameters
    ----------
    seq_len      : input sequence length (time steps)
    n_features   : input feature dimension
    n_targets    : forecast horizon / number of outputs
    lstm_units   : LSTM hidden size
    dense_units  : dense layer size after LSTM
    dropout      : dropout rate (applied after LSTM)
    learning_rate: Adam learning rate

    Returns
    -------
    Compiled Keras model.
    """
    inp = layers.Input(shape=(seq_len, n_features), name="input")
    h = layers.LSTM(lstm_units, return_sequences=False, name="lstm")(inp)
    h = layers.Dropout(dropout, name="dropout")(h)
    h = layers.Dense(dense_units, activation="relu", name="dense1")(h)
    out = layers.Dense(n_targets, name="output")(h)

    model = keras.Model(inputs=inp, outputs=out, name="weak_learner")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        weighted_metrics=[],   # suppress duplicate metric printout
    )
    return model


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_weak_learner(
    model: keras.Model,
    X: np.ndarray,
    Y: np.ndarray,
    weights: np.ndarray,
    X_val: np.ndarray = None,
    Y_val: np.ndarray = None,
    batch_size: int = 32,
    max_epochs: int = 100,
    patience: int = 10,
    verbose: int = 0,
) -> keras.callbacks.History:
    """
    Fit the weak learner with per-sample importance weights.

    Parameters
    ----------
    model    : compiled Keras model from build_weak_learner
    X        : (N, seq_len, n_features)
    Y        : (N, n_targets)
    weights  : (N,) non-negative importance weights (will be normalised
               to sum=N so that loss scale stays stable)
    X_val, Y_val : optional validation set (not weighted)
    batch_size, max_epochs, patience : training hyper-params
    verbose  : Keras verbosity level

    Returns
    -------
    Keras History object.
    """
    # Normalise weights → mean=1 so MSE scale stays comparable
    w = np.array(weights, dtype=np.float32)
    w = w / (w.mean() + 1e-12)

    # Adapt batch size to dataset size (avoid empty batches)
    effective_batch = min(batch_size, max(1, len(X)))

    callbacks = []
    val_data = None
    has_val = (X_val is not None and Y_val is not None and len(X_val) > 0)
    if has_val:
        val_data = (X_val, Y_val)
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=0,
            )
        )

    history = model.fit(
        X, Y,
        sample_weight=w,
        validation_data=val_data,
        epochs=max_epochs,
        batch_size=effective_batch,
        callbacks=callbacks,
        verbose=verbose,
    )
    return history


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def predict_weak_learner(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """
    Return predictions, shape (N, n_targets).
    """
    return model.predict(X, verbose=0)


# ---------------------------------------------------------------------------
# Clone (for creating independent copies with same architecture)
# ---------------------------------------------------------------------------

def clone_weak_learner(model: keras.Model) -> keras.Model:
    """
    Return a new model with the same architecture (different weights).
    """
    cfg = model.get_config()
    new_model = keras.Model.from_config(cfg)
    # Re-compile with same optimizer settings
    new_model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=float(model.optimizer.learning_rate)
        ),
        loss="mse",
        weighted_metrics=[],
    )
    return new_model
