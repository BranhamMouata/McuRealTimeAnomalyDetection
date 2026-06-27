
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from typing import Tuple, Optional, List


def build_autoencoder(input_dim: int, latent_dim: int = 2, hidden_dims: Optional[List[int]] = None) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    """Builds and returns a compiled Keras autoencoder, plus encoder and decoder.

    Args:
            input_dim: Dimensionality of the input feature vector.
            latent_dim: Size of the latent representation.
            hidden_dims: List of hidden layer sizes for encoder (decoder is symmetric). If None, defaults to [128, 64].

    Returns:
            (autoencoder, encoder, decoder)
    """
    if hidden_dims is None:
        hidden_dims = [8, 4]

    # Encoder
    inputs = layers.Input(shape=(input_dim,), name="ae_input")
    x = inputs
    for i, h in enumerate(hidden_dims):
        x = layers.Dense(h, activation="relu", name=f"enc_dense_{i}")(x)
    latent = layers.Dense(latent_dim, activation="relu", name="latent")(x)
    encoder = models.Model(inputs, latent, name="encoder")

    # Decoder
    latent_inputs = layers.Input(shape=(latent_dim,), name="z_input")
    x = latent_inputs
    for i, h in enumerate(reversed(hidden_dims)):
        x = layers.Dense(h, activation="relu", name=f"dec_dense_{i}")(x)
    outputs = layers.Dense(input_dim, activation="linear",
                           name="reconstruction")(x)
    decoder = models.Model(latent_inputs, outputs, name="decoder")

    # Autoencoder = decoder(encoder(x))
    ae_outputs = decoder(encoder(inputs))
    autoencoder = models.Model(inputs, ae_outputs, name="autoencoder")
    autoencoder.compile(optimizer=optimizers.Adam(),
                        loss=losses.MeanSquaredError())

    return autoencoder, encoder, decoder


def train_autoencoder(
        autoencoder: tf.keras.Model,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        save_path: Optional[str] = None,
        verbose: int = 1,
) -> tf.keras.callbacks.History:
    """Train the autoencoder using only normal data (X_train should contain normal examples).

    Args:
            autoencoder: Compiled Keras autoencoder model.
            X_train: Normal training examples (N x input_dim).
            X_val: Optional validation examples (N_val x input_dim).
            epochs: Number of training epochs.
            batch_size: Batch size.
            save_path: Optional path to save the best model (ModelCheckpoint).
            verbose: Verbosity level for fit().

    Returns:
            Keras History object from fit().
    """
    cbs = []
    monitor = "val_loss" if X_val is not None else "loss"
    if save_path:
        cbs.append(callbacks.ModelCheckpoint(
            save_path, save_best_only=True, monitor=monitor, mode="min"))

    history = autoencoder.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, X_val) if X_val is not None else None,
        callbacks=cbs,
        verbose=verbose,
    )

    return history


def reconstruction_errors(autoencoder: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    """Compute per-sample mean squared reconstruction error."""
    recon = autoencoder.predict(X)
    errors = np.mean(np.square(X - recon), axis=1)
    return errors


def choose_threshold(errors: np.ndarray, factor: float = 3.0) -> float:
    """Choose a threshold from reconstruction errors using mean + factor * std.

    Args:
            errors: 1D array of reconstruction errors (from normal training set).
            factor: Multiplier for standard deviation (default 3.0).

    Returns:
            Threshold value above which a sample is considered anomalous.
    """
    mu = float(np.mean(errors))
    sigma = float(np.std(errors))
    return mu + factor * sigma
