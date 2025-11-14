"""
Model builder for a 1D Variational Autoencoder that uses a 
Kymatio Wavelet Scattering Transform (WST) as the primary feature extractor.

This model is designed to work with time-domain inputs (e.g., (256, 4)).

DEPENDENCY: This file requires the 'kymatio' library.
Install with: pip install kymatio
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Layer,
    Input,
    Dense,
    Reshape,
    Flatten,
    Concatenate,
    Lambda,
    BatchNormalization
)
from tensorflow.keras.callbacks import Callback
import numpy as np

# Import the TensorFlow-compatible Scattering1D layer
try:
    from kymatio.tensorflow import Scattering1D
except ImportError:
    print("="*50)
    print("ERROR: Kymatio library not found.")
    print("Please install it with: pip install kymatio")
    print("="*50)
    raise

# --- VAE Helper Classes (Copied from model_builder_VAE.py) ---
# These are included here to make this file self-contained.

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the latent vector."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(Model):
    """
    Custom VAE Model class.
    Combines encoder and decoder into one model with a custom train_step.
    """
    def __init__(self, encoder, decoder, kl_weight=0.01, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = tf.Variable(kl_weight, trainable=False, dtype=tf.float32, name="kl_weight")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        """Standard forward pass for inference."""
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    def train_step(self, data):
        if isinstance(data, tuple):
            inputs, targets = data
        else:
            inputs, targets = data, data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(inputs)
            reconstruction = self.decoder(z)
            recon_loss = self.compiled_loss(targets, reconstruction, regularization_losses=self.losses)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = recon_loss + self.kl_weight * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            inputs, targets = data
        else:
            inputs, targets = data, data

        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        recon_loss = self.compiled_loss(targets, reconstruction, regularization_losses=self.losses)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# VAE KL Annealing Callback
class KLAnnealingCallback(Callback):
    def __init__(self, kl_weight_target, kl_anneal_epochs, kl_warmup_epochs=0, verbose=1):
        super(KLAnnealingCallback, self).__init__()
        self.kl_weight_target = float(kl_weight_target)
        self.kl_anneal_epochs = float(kl_anneal_epochs)
        self.kl_warmup_epochs = float(kl_warmup_epochs)
        self.verbose = verbose
        self.current_kl_weight = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        epoch_float = float(epoch)
        if epoch_float < self.kl_warmup_epochs:
            self.current_kl_weight = 0.0
        else:
            ramp_progress = (epoch_float - self.kl_warmup_epochs) / self.kl_anneal_epochs
            ramp_progress_clipped = np.clip(ramp_progress, 0.0, 1.0)
            self.current_kl_weight = self.kl_weight_target * ramp_progress_clipped
        K.set_value(self.model.kl_weight, self.current_kl_weight)

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs["kl_weight"] = self.current_kl_weight
        if self.verbose > 0 and (epoch == 0 or (epoch + 1) % self.verbose == 0):
            print(f" - kl_weight: {self.current_kl_weight:.6f}")

# (Other callbacks like KLCyclicalAnnealingCallback could be copied here too if needed)

# --- VAE Builder Function ---

def build_vae_model_scattering(input_shape=(256, 4), learning_rate=0.001, latent_dim=32, kl_weight_initial=0.0):
    """
    Builds a 1D VAE using a Wavelet Scattering Transform encoder.
    
    This model assumes a time-domain input (e.g., (256, 4)).
    """
    
    T = input_shape[0] # Time dimension (e.g., 256)
    N_channels = input_shape[1] # Channel dimension (e.g., 4)
    
    # --- Encoder ---
    # We will apply the scattering transform to each of the 4 channels independently
    # and then concatenate the results.
    
    encoder_inputs = Input(shape=input_shape)

    # Parameters for Scattering1D.
    # J (scale) is chosen such that 2^J <= T. 
    # T=256, so 2^7=128, 2^8=256. J=7 is a good choice.
    # Q (quality factor) controls wavelets per octave.
    J = 7 
    Q = 8
    
    # Instantiate the Scattering1D layer
    # This layer is "fixed" and not trained.
    scattering = Scattering1D(J=J, T=T, Q=Q)

    channel_outputs = []
    for i in range(N_channels):
        # Extract one channel (shape: (batch, T))
        channel_slice = Lambda(lambda x: x[..., i])(encoder_inputs)
        
        # Apply scattering transform (shape: (batch, n_coeffs, n_timesteps))
        channel_coeffs = scattering(channel_slice)
        
        # Flatten the coefficients for this channel
        channel_coeffs_flat = Flatten()(channel_coeffs)
        channel_outputs.append(channel_coeffs_flat)
    
    # Concatenate the flattened coefficients from all channels
    x = Concatenate()(channel_outputs)
    
    # This is the "MLP" part of your colleague's encoder, now in Keras.
    # We add a Dense layer to create a bottleneck before the latent space.
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    
    # Probabilistic latent space
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    # --- Decoder ---
    # Since the scattering transform is not easily invertible, we use a
    # standard fully-connected decoder to learn to reconstruct the time signal
    # from the latent space.
    
    latent_inputs = Input(shape=(latent_dim,))
    
    # Upsample through Dense layers
    x = Dense(256, activation="relu")(latent_inputs)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    
    # Final layer must have T * N_channels outputs
    x = Dense(T * N_channels, activation="linear")(x)
    
    # Reshape back to the original time-series shape
    decoder_outputs = Reshape((T, N_channels))(x)
    
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    # --- VAE ---
    vae = VAE(encoder, decoder, kl_weight=kl_weight_initial)
    
    vae.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=keras.losses.MeanAbsoluteError() # This is the reconstruction loss
    )
    
    # Return False for requires_transpose, as this model expects (Batch, 256, 4)
    return vae, False