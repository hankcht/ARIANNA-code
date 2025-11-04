import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, ReLU, Input, 
    Conv1DTranspose, MaxPooling1D, UpSampling1D,
    GaussianNoise, Dropout, Concatenate,
    Dense, Reshape, Layer, Flatten
)
from tensorflow.keras.callbacks import Callback
import numpy as np



# --- VAE Helper Classes ---

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

        # KL annealing
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
        # Denoising autoencoders pass (x_noisy, y_clean)
        # Standard autoencoders pass (x, x)
        if isinstance(data, tuple):
            inputs, targets = data
        else:
            inputs, targets = data, data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(inputs)
            reconstruction = self.decoder(z)
            
            # Use the model's compiled loss (e.g., MSE or MAE)
            recon_loss = self.compiled_loss(targets, reconstruction, regularization_losses=self.losses)
            
            # Calculate KL divergence loss
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
    """
    Callback to anneal the KL weight (beta) from 0.0 to a target value.
    """
    def __init__(self, kl_weight_target, kl_anneal_epochs, kl_warmup_epochs=0, verbose=1):
        """
        Args:
            kl_weight_target (float): The final target value for the KL weight.
            kl_anneal_epochs (int): The number of epochs over which to ramp up the weight.
            kl_warmup_epochs (int): The number of epochs to wait at weight=0.0 before starting.
            verbose (int): Set to 1 to print the weight at the end of each epoch.
        """
        super(KLAnnealingCallback, self).__init__()
        self.kl_weight_target = float(kl_weight_target)
        self.kl_anneal_epochs = float(kl_anneal_epochs)
        self.kl_warmup_epochs = float(kl_warmup_epochs)
        self.verbose = verbose
        self.current_kl_weight = 0.0 # Start at 0

    def on_epoch_begin(self, epoch, logs=None):
        epoch_float = float(epoch)
        
        if epoch_float < self.kl_warmup_epochs:
            # We are in the warmup phase
            self.current_kl_weight = 0.0
        else:
            # We are in the annealing phase
            # Calculate the progress of the ramp
            ramp_progress = (epoch_float - self.kl_warmup_epochs) / self.kl_anneal_epochs
            # Clip progress to be between 0.0 and 1.0
            ramp_progress_clipped = np.clip(ramp_progress, 0.0, 1.0)
            # Calculate the new weight
            self.current_kl_weight = self.kl_weight_target * ramp_progress_clipped
        
        # Set the model's kl_weight variable
        K.set_value(self.model.kl_weight, self.current_kl_weight)

    def on_epoch_end(self, epoch, logs=None):
        # Log the current KL weight so it appears in training history
        if logs is not None:
            logs["kl_weight"] = self.current_kl_weight
        
        # Print the current weight if verbose is set
        if self.verbose > 0 and (epoch == 0 or (epoch + 1) % self.verbose == 0):
            print(f" - kl_weight: {self.current_kl_weight:.6f}")


class KLCyclicalAnnealingCallback(Callback):
    """
    Callback to cyclically anneal the KL weight (beta).
    
    This callback implements a cyclical schedule for the KL weight, 
    often used to improve VAE training.
    """
    
    def __init__(self, kl_weight_target, cycle_length_epochs, kl_warmup_epochs=0, ramp_up_fraction=1.0, verbose=1):
        """
        Args:
            kl_weight_target (float): The final target value for the KL weight (peak of the cycle).
            cycle_length_epochs (int): The total number of epochs for one full cycle.
            kl_warmup_epochs (int): The number of epochs to wait at weight=0.0 before starting cycles.
            ramp_up_fraction (float): The fraction of the cycle used for ramping up (0.0 to 1.0).
                                      1.0 = sawtooth wave (ramp up for the whole cycle, then reset).
                                      0.5 = trapezoidal (ramp up for 1st half, hold for 2nd half).
            verbose (int): Set to 1 to print the weight at the end of each epoch.
        """
        super(KLCyclicalAnnealingCallback, self).__init__()
        self.kl_weight_target = float(kl_weight_target)
        self.cycle_length_epochs = float(cycle_length_epochs)
        self.kl_warmup_epochs = float(kl_warmup_epochs)
        self.ramp_up_fraction = float(ramp_up_fraction)
        
        # Calculate the number of epochs for the ramp-up phase
        self.ramp_up_epochs = self.cycle_length_epochs * self.ramp_up_fraction
        
        self.verbose = verbose
        self.current_kl_weight = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        epoch_float = float(epoch)
        
        if epoch_float < self.kl_warmup_epochs:
            # We are in the initial warmup phase
            self.current_kl_weight = 0.0
        else:
            # We are in the cyclical phase
            
            # 1. Find how many epochs have passed since the warmup ended
            epoch_after_warmup = epoch_float - self.kl_warmup_epochs
            
            # 2. Find where we are *within* the current cycle
            # The modulo operator (%) resets the count every 'cycle_length_epochs'
            epoch_in_cycle = epoch_after_warmup % self.cycle_length_epochs
            
            # 3. Calculate ramp progress
            if self.ramp_up_epochs > 0 and epoch_in_cycle < self.ramp_up_epochs:
                # We are in the ramp-up part of the cycle
                ramp_progress = epoch_in_cycle / self.ramp_up_epochs
            else:
                # We are in the "hold" part of the cycle (or ramp_up_epochs is 0)
                ramp_progress = 1.0 # Stay at the target
                
            # 4. Calculate the new weight
            self.current_kl_weight = self.kl_weight_target * ramp_progress
        
        # Set the model's kl_weight variable
        K.set_value(self.model.kl_weight, self.current_kl_weight)

    def on_epoch_end(self, epoch, logs=None):
        # Log the current KL weight so it appears in training history
        if logs is not None:
            logs["kl_weight"] = self.current_kl_weight
        
        # Print the current weight if verbose is set
        if self.verbose > 0 and (epoch == 0 or (epoch + 1) % self.verbose == 0):
            print(f" - kl_weight: {self.current_kl_weight:.6f}")

class CyclicalLRCallback(Callback):
    """
    Callback to cyclically anneal the Learning Rate.
    
    This callback implements a cyclical schedule for the learning rate,
    designed to be synchronized with the KLCyclicalAnnealingCallback.
    """
    
    def __init__(self, max_lr, min_lr, cycle_length_epochs, kl_warmup_epochs=0, ramp_up_fraction=1.0, verbose=1):
        """
        Args:
            max_lr (float): The peak learning rate.
            min_lr (float): The base/minimum learning rate.
            cycle_length_epochs (int): The total number of epochs for one full cycle.
            kl_warmup_epochs (int): The number of epochs to wait at min_lr before starting cycles.
            ramp_up_fraction (float): The fraction of the cycle used for ramping up (0.0 to 1.0).
            verbose (int): Set to 1 to print the LR at the end of each epoch.
        """
        super(CyclicalLRCallback, self).__init__()
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.cycle_length_epochs = float(cycle_length_epochs)
        self.kl_warmup_epochs = float(kl_warmup_epochs)
        self.ramp_up_fraction = float(ramp_up_fraction)
        
        # Calculate the number of epochs for the ramp-up phase
        self.ramp_up_epochs = self.cycle_length_epochs * self.ramp_up_fraction
        
        self.verbose = verbose
        self.current_lr = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        epoch_float = float(epoch)
        
        if epoch_float < self.kl_warmup_epochs:
            # We are in the initial warmup phase
            self.current_lr = self.min_lr
        else:
            # We are in the cyclical phase
            
            # 1. Find how many epochs have passed since the warmup ended
            epoch_after_warmup = epoch_float - self.kl_warmup_epochs
            
            # 2. Find where we are *within* the current cycle
            epoch_in_cycle = epoch_after_warmup % self.cycle_length_epochs
            
            # 3. Calculate ramp progress
            if self.ramp_up_epochs > 0 and epoch_in_cycle < self.ramp_up_epochs:
                # We are in the ramp-up part of the cycle
                ramp_progress = epoch_in_cycle / self.ramp_up_epochs
            else:
                # We are in the "hold" part of the cycle
                ramp_progress = 1.0 # Stay at the peak
                
            # 4. Calculate the new LR using linear interpolation
            self.current_lr = self.min_lr + (self.max_lr - self.min_lr) * ramp_progress
        
        # Set the model's optimizer learning rate
        K.set_value(self.model.optimizer.learning_rate, self.current_lr)

    def on_epoch_end(self, epoch, logs=None):
        # Log the current LR so it appears in training history
        if logs is not None:
            logs["lr"] = self.current_lr
        
        # Print the current LR if verbose is set
        if self.verbose > 0 and (epoch == 0 or (epoch + 1) % self.verbose == 0):
            print(f" - lr: {self.current_lr:.6f}")


# --- VAE Builder Function ---

def build_vae_model_freq(input_shape=(129, 4), learning_rate=0.001, latent_dim=16, kl_weight_initial=0.0):
    """
    Builds a 1D Convolutional Variational Autoencoder.
    """
    
    # --- Encoder ---
    encoder_inputs = Input(shape=input_shape)
    # (129, 4) -> (64, 16)
    x = Conv1D(16, kernel_size=3, padding="valid", activation="relu", strides=2)(encoder_inputs)
    x = BatchNormalization()(x)
    # (64, 16) -> (32, 32)
    x = Conv1D(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    # (32, 32) -> (16, 64)
    x = Conv1D(64, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # Flatten features and get probabilistic latent space
    x = Flatten()(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    # --- Decoder ---
    latent_inputs = Input(shape=(latent_dim,))
    
    # Project back to the shape before flattening
    x = Dense(16 * 64, activation="relu")(latent_inputs)
    x = Reshape((16, 64))(x)
    
    # (16, 64) -> (32, 32)
    x = Conv1DTranspose(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    # (32, 32) -> (64, 16)
    x = Conv1DTranspose(16, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    # (64, 16) -> (129, 4)
    decoder_outputs = Conv1DTranspose(
        input_shape[-1], kernel_size=3, padding="valid", activation="linear", strides=2
    )(x)
    
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    # --- VAE ---
    vae = VAE(encoder, decoder, kl_weight=kl_weight_initial)
    
    vae.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=keras.losses.MeanSquaredError() # This is the reconstruction loss
    )
    
    return vae, True

def build_vae_bottleneck_model_freq(input_shape=(129, 4), learning_rate=0.001, latent_dim=8, kl_weight_initial=0.0):
    """
    Step 1: VAE with a Tighter Convolutional Bottleneck (32 filters).
    """
    
    # --- Encoder ---
    encoder_inputs = Input(shape=input_shape)
    # (129, 4) -> (64, 16)
    x = Conv1D(16, kernel_size=3, padding="valid", activation="relu", strides=2)(encoder_inputs)
    x = BatchNormalization()(x)
    # (64, 16) -> (32, 32)
    x = Conv1D(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (32, 32) -> (16, 32)  <--- STEP 1 CHANGE
    # Tighter convolutional bottleneck: 32 filters
    x = Conv1D(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # Flatten features and get probabilistic latent space
    x = Flatten()(x) # Shape will be (16 * 32 = 512)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    # --- Decoder ---
    latent_inputs = Input(shape=(latent_dim,))
    
    # Project back to the shape before flattening (16, 32)
    x = Dense(16 * 32, activation="relu")(latent_inputs)
    x = Reshape((16, 32))(x)
    
    # (16, 32) -> (32, 32) <--- STEP 1 CHANGE
    x = Conv1DTranspose(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    # (32, 32) -> (64, 16)
    x = Conv1DTranspose(16, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    # (64, 16) -> (129, 4)
    decoder_outputs = Conv1DTranspose(
        input_shape[-1], kernel_size=3, padding="valid", activation="linear", strides=2
    )(x)
    
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    # --- VAE ---
    vae = VAE(encoder, decoder, kl_weight=kl_weight_initial)
    
    vae.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=keras.losses.MeanSquaredError()
    )
    
    return vae, True

def build_vae_denoising_model_freq(input_shape=(129, 4), learning_rate=0.001, latent_dim=8, kl_weight_initial=0.0, noise_stddev=0.1):
    """
    Step 2: Denoising VAE with Tighter Bottleneck.
    Includes a GaussianNoise layer in the encoder.
    """
    
    # --- Encoder ---
    encoder_inputs = Input(shape=input_shape, name="clean_input")
    
    # --- STEP 2 CHANGE ---
    noisy_inputs = GaussianNoise(stddev=noise_stddev)(encoder_inputs)
    # ---------------------

    # (129, 4) -> (64, 16)
    x = Conv1D(16, kernel_size=3, padding="valid", activation="relu", strides=2)(noisy_inputs)
    x = BatchNormalization()(x)
    # (64, 16) -> (32, 32)
    x = Conv1D(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    # (32, 32) -> (16, 32)
    x = Conv1D(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    x = Flatten()(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    
    # Encoder model maps from CLEAN inputs to latent space
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    # --- Decoder ---
    # (Decoder is identical to Step 1)
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(16 * 32, activation="relu")(latent_inputs)
    x = Reshape((16, 32))(x)
    x = Conv1DTranspose(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    x = Conv1DTranspose(16, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    decoder_outputs = Conv1DTranspose(
        input_shape[-1], kernel_size=3, padding="valid", activation="linear", strides=2
    )(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    # --- VAE ---
    vae = VAE(encoder, decoder, kl_weight=kl_weight_initial)
    
    vae.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=keras.losses.MeanSquaredError()
    )
    
    return vae, True

def build_vae_mae_loss_model_freq(input_shape=(129, 4), learning_rate=0.001, latent_dim=8, kl_weight_initial=0.0, noise_stddev=0.1):
    """
    Step 3: Denoising VAE Bottleneck model compiled with MAE loss.
    """
    
    # --- Encoder ---
    # (Identical to Step 2)
    encoder_inputs = Input(shape=input_shape, name="clean_input")
    noisy_inputs = GaussianNoise(stddev=noise_stddev)(encoder_inputs)
    x = Conv1D(16, kernel_size=3, padding="valid", activation="relu", strides=2)(noisy_inputs)
    x = BatchNormalization()(x)
    x = Conv1D(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    x = Conv1D(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # --- Decoder ---
    # (Identical to Step 2)
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(16 * 32, activation="relu")(latent_inputs)
    x = Reshape((16, 32))(x)
    x = Conv1DTranspose(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    x = Conv1DTranspose(16, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    decoder_outputs = Conv1DTranspose(
        input_shape[-1], kernel_size=3, padding="valid", activation="linear", strides=2
    )(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")

    # --- VAE ---
    vae = VAE(encoder, decoder, kl_weight=kl_weight_initial)
    
    # --- STEP 3 CHANGE ---
    vae.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=keras.losses.MeanAbsoluteError() # Use MAE for reconstruction
    )
    # ---------------------
    
    return vae, True