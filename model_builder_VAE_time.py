"""
Model builders for 1D convolutional VAEs on time-domain inputs (256 x 4).
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    Conv1DTranspose,
    Conv2D,
    Conv2DTranspose,
    BatchNormalization,
    GaussianNoise,
    Dense,
    Reshape,
    Flatten,
)
from tensorflow.keras.optimizers import Adam

from model_builder_VAE import (
    Sampling,
    VAE,
    KLAnnealingCallback,
    KLCyclicalAnnealingCallback,
    CyclicalLRCallback,
)

__all__ = [
    "KLAnnealingCallback",
    "KLCyclicalAnnealingCallback",
    "CyclicalLRCallback",
    "build_vae_model_time",
    "build_vae_bottleneck_model_time",
    "build_vae_denoising_model_time",
    "build_vae_mae_loss_model_time",
    "custom_weights_time",
    "build_vae_custom_loss_model_time_samplewise",
    "build_vae_model_time_2d_input",
]


def build_vae_model_time(input_shape=(256, 4), learning_rate=0.001, latent_dim=32, kl_weight_initial=0.0):
    """Builds a 1D Convolutional Variational Autoencoder for time-domain inputs."""
    encoder_inputs = Input(shape=input_shape)
    x = Conv1D(16, kernel_size=40, padding="same", activation="relu", strides=2)(encoder_inputs)
    x = BatchNormalization()(x)
    x = Conv1D(32, kernel_size=20, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    x = Conv1D(64, kernel_size=10, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(32 * 64, activation="relu")(latent_inputs)
    x = Reshape((32, 64))(x)
    x = Conv1DTranspose(32, kernel_size=10, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    x = Conv1DTranspose(16, kernel_size=20, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    decoder_outputs = Conv1DTranspose(
        input_shape[-1], kernel_size=40, padding="same", activation="linear", strides=2
    )(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    vae = VAE(encoder, decoder, kl_weight=kl_weight_initial)
    vae.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=keras.losses.MeanAbsoluteError(),
    )

    return vae, True


def build_vae_bottleneck_model_time(
    input_shape=(256, 4), learning_rate=0.001, latent_dim=32, kl_weight_initial=0.0
):
    """VAE with a tighter convolutional bottleneck for time-domain inputs."""
    encoder_inputs = Input(shape=input_shape)
    x = Conv1D(16, kernel_size=40, padding="same", activation="relu", strides=2)(encoder_inputs)
    x = BatchNormalization()(x)
    x = Conv1D(32, kernel_size=20, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    x = Conv1D(32, kernel_size=10, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(32 * 32, activation="relu")(latent_inputs)
    x = Reshape((32, 32))(x)
    x = Conv1DTranspose(32, kernel_size=10, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    x = Conv1DTranspose(16, kernel_size=20, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    decoder_outputs = Conv1DTranspose(
        input_shape[-1], kernel_size=40, padding="same", activation="linear", strides=2
    )(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    vae = VAE(encoder, decoder, kl_weight=kl_weight_initial)
    vae.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=keras.losses.MeanAbsoluteError(),
    )

    return vae, True


def build_vae_denoising_model_time(
    input_shape=(256, 4), learning_rate=0.001, latent_dim=32, kl_weight_initial=0.0, noise_stddev=0.1
):
    """Denoising VAE with a tighter bottleneck for time-domain inputs."""
    encoder_inputs = Input(shape=input_shape, name="clean_input")
    noisy_inputs = GaussianNoise(stddev=noise_stddev)(encoder_inputs)

    x = Conv1D(16, kernel_size=40, padding="same", activation="relu", strides=2)(noisy_inputs)
    x = BatchNormalization()(x)
    x = Conv1D(32, kernel_size=20, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    x = Conv1D(32, kernel_size=10, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(32 * 32, activation="relu")(latent_inputs)
    x = Reshape((32, 32))(x)
    x = Conv1DTranspose(32, kernel_size=10, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    x = Conv1DTranspose(16, kernel_size=20, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    decoder_outputs = Conv1DTranspose(
        input_shape[-1], kernel_size=40, padding="same", activation="linear", strides=2
    )(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    vae = VAE(encoder, decoder, kl_weight=kl_weight_initial)
    vae.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=keras.losses.MeanAbsoluteError(),
    )

    return vae, True


def build_vae_mae_loss_model_time(
    input_shape=(256, 4), learning_rate=0.001, latent_dim=32, kl_weight_initial=0.0, noise_stddev=0.1
):
    """Denoising VAE with Huber reconstruction loss for time-domain inputs."""
    encoder_inputs = Input(shape=input_shape, name="clean_input")
    noisy_inputs = GaussianNoise(stddev=noise_stddev)(encoder_inputs)

    x = Conv1D(16, kernel_size=40, padding="same", activation="relu", strides=2)(noisy_inputs)
    x = BatchNormalization()(x)
    x = Conv1D(32, kernel_size=20, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    x = Conv1D(32, kernel_size=10, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(32 * 32, activation="relu")(latent_inputs)
    x = Reshape((32, 32))(x)
    x = Conv1DTranspose(32, kernel_size=10, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    x = Conv1DTranspose(16, kernel_size=20, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    decoder_outputs = Conv1DTranspose(
        input_shape[-1], kernel_size=40, padding="same", activation="linear", strides=2
    )(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    vae = VAE(encoder, decoder, kl_weight=kl_weight_initial)
    vae.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(),
    )

    return vae, True


custom_weights_time = np.ones(256)
custom_weights_time[0:80] = 0.5     # Deprioritize pre-trigger
custom_weights_time[80:200] = 5.0   # Prioritize during trigger

def build_vae_custom_loss_model_time_samplewise(
    input_shape=(256, 4),
    learning_rate=0.001,
    latent_dim=32,
    kl_weight_initial=0.0,
    noise_stddev=0.1,
    sample_weights=custom_weights_time,
):
    """Denoising VAE with sample-wise weighted MAE loss for time-domain inputs."""

    def create_weighted_mae_loss(weights_list):
        if weights_list is None:
            weights_list = [1.0] * input_shape[0]
        if len(weights_list) != input_shape[0]:
            raise ValueError(
                f"Length of sample_weights ({len(weights_list)}) must match the number of input samples ({input_shape[0]})"
            )

        weights_tensor = tf.constant(weights_list, dtype=tf.float32)
        weights_tensor = tf.reshape(weights_tensor, [1, len(weights_list), 1])

        def weighted_mae(y_true, y_pred):
            error = tf.abs(y_true - y_pred)
            weighted_error = error * weights_tensor
            return tf.reduce_mean(weighted_error)

        return weighted_mae

    custom_loss = create_weighted_mae_loss(sample_weights)

    encoder_inputs = Input(shape=input_shape, name="clean_input")
    noisy_inputs = GaussianNoise(stddev=noise_stddev)(encoder_inputs)

    x = Conv1D(16, kernel_size=40, padding="same", activation="relu", strides=2)(noisy_inputs)
    x = BatchNormalization()(x)
    x = Conv1D(32, kernel_size=20, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    x = Conv1D(32, kernel_size=10, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(32 * 32, activation="relu")(latent_inputs)
    x = Reshape((32, 32))(x)
    x = Conv1DTranspose(32, kernel_size=10, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    x = Conv1DTranspose(16, kernel_size=20, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    decoder_outputs = Conv1DTranspose(
        input_shape[-1], kernel_size=40, padding="same", activation="linear", strides=2
    )(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    vae = VAE(encoder, decoder, kl_weight=kl_weight_initial)
    vae.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=custom_loss,
    )

    return vae, True


def build_vae_model_time_2d_input(
    input_shape=(256, 4), learning_rate=0.001, latent_dim=32, kl_weight_initial=0.0
):
    """Builds a VAE using 2D convolutions for time-domain inputs."""
    encoder_inputs = Input(shape=input_shape)
    x = Reshape((input_shape[0], input_shape[1], 1))(encoder_inputs)

    x = Conv2D(16, kernel_size=(40, 3), padding="same", activation="relu", strides=(2, 1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, kernel_size=(20, 3), padding="same", activation="relu", strides=(2, 1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=(10, 3), padding="same", activation="relu", strides=(2, 1))(x)
    x = BatchNormalization()(x)

    pre_flatten_shape = keras.backend.int_shape(x)[1:]
    pre_flatten_dim = np.prod(pre_flatten_shape)

    x = Flatten()(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(pre_flatten_dim, activation="relu")(latent_inputs)
    x = Reshape(pre_flatten_shape)(x)

    x = Conv2DTranspose(32, kernel_size=(10, 3), padding="same", activation="relu", strides=(2, 1))(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(16, kernel_size=(20, 3), padding="same", activation="relu", strides=(2, 1))(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(1, kernel_size=(40, 3), padding="same", activation="linear", strides=(2, 1))(x)

    decoder_outputs = Reshape((input_shape[0], input_shape[1]))(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    vae = VAE(encoder, decoder, kl_weight=kl_weight_initial)
    vae.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=keras.losses.MeanAbsoluteError(),
    )

    return vae, True
