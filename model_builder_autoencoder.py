"""
Model Builder for 1D Convolutional Autoencoder
"""

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, ReLU, Input, 
    Conv1DTranspose, MaxPooling1D, UpSampling1D,
    GaussianNoise, Dropout, Concatenate
)

def build_autoencoder_model(input_shape=(256, 4), learning_rate=0.001):
    """
    Builds and compiles a 1D Convolutional Autoencoder.
    
    The model is trained on background (data) and is expected to have
    HIGH reconstruction error for signal (sim) and LOW error for background.

    Args:
        input_shape (tuple): Input shape (samples, channels). E.g., (256, 4).
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        tuple: (compiled keras.Model, bool requires_transpose)
               The model is the autoencoder. requires_transpose is True.
    """
    
    inputs = Input(shape=input_shape)

    # --- Encoder ---
    # (256, 4) -> (128, 16)
    x = Conv1D(16, kernel_size=5, padding="same", activation="relu", strides=2)(inputs)
    x = BatchNormalization()(x)
    
    # (128, 16) -> (64, 32)
    x = Conv1D(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (64, 32) -> (32, 64)
    x = Conv1D(64, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization(name="latent_space")(x)
    
    # Bottleneck
    # (32, 64)
    
    # --- Decoder ---
    # (32, 64) -> (64, 32)
    x = Conv1DTranspose(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (64, 32) -> (128, 16)
    x = Conv1DTranspose(16, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (128, 16) -> (256, 4)
    # Use Conv1DTranspose with stride 2 to get back to 256
    # Final layer uses 'linear' activation to reconstruct the input values
    outputs = Conv1DTranspose(input_shape[-1], kernel_size=5, padding="same", activation="linear", strides=2)(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs, name="Conv1D_Autoencoder")

    # Compile with Mean Squared Error loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae'] # Mean Absolute Error
    )
    
    # This model expects (samples, channels), so transpose is needed.
    return model, True

def build_autoencoder_mae_model(input_shape=(256, 4), learning_rate=0.001):
    """
    Builds and compiles a 1D Convolutional Autoencoder.
    
    The model is trained on background (data) and is expected to have
    HIGH reconstruction error for signal (sim) and LOW error for background.

    Args:
        input_shape (tuple): Input shape (samples, channels). E.g., (256, 4).
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        tuple: (compiled keras.Model, bool requires_transpose)
               The model is the autoencoder. requires_transpose is True.
    """
    
    inputs = Input(shape=input_shape)

    # --- Encoder ---
    # (256, 4) -> (128, 16)
    x = Conv1D(16, kernel_size=5, padding="same", activation="relu", strides=2)(inputs)
    x = BatchNormalization()(x)
    
    # (128, 16) -> (64, 32)
    x = Conv1D(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (64, 32) -> (32, 64)
    x = Conv1D(64, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization(name="latent_space")(x)
    
    # Bottleneck
    # (32, 64)
    
    # --- Decoder ---
    # (32, 64) -> (64, 32)
    x = Conv1DTranspose(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (64, 32) -> (128, 16)
    x = Conv1DTranspose(16, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (128, 16) -> (256, 4)
    # Use Conv1DTranspose with stride 2 to get back to 256
    # Final layer uses 'linear' activation to reconstruct the input values
    outputs = Conv1DTranspose(input_shape[-1], kernel_size=5, padding="same", activation="linear", strides=2)(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs, name="Conv1D_Autoencoder")

    # Compile with Mean Squared Error loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mae', # Replace MSE with MAE to reduce sensitivity to outliers
        metrics=['mae'] # Mean Absolute Error
    )
    
    # This model expects (samples, channels), so transpose is needed.
    return model, True

def build_autoencoder_freq_model(input_shape=(129, 4), learning_rate=0.001):
    """
    Builds and compiles a 1D Convolutional Autoencoder tailored for 129-sample inputs.
    """
    inputs = Input(shape=input_shape)

    # --- Encoder ---
    # (129, 4) -> (64, 16)
    x = Conv1D(16, kernel_size=3, padding="valid", activation="relu", strides=2)(inputs)
    x = BatchNormalization()(x)
    
    # (64, 16) -> (32, 32)
    x = Conv1D(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (32, 32) -> (16, 64)
    x = Conv1D(64, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization(name="latent_space")(x)
    
    # --- Decoder ---
    # (16, 64) -> (32, 32)
    x = Conv1DTranspose(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (32, 32) -> (64, 16)
    x = Conv1DTranspose(16, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (64, 16) -> (129, 4)
    outputs = Conv1DTranspose(
        input_shape[-1], kernel_size=3, padding="valid", activation="linear", strides=2
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name="Conv1D_Freq_Autoencoder")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model, True

def build_autoencoder_dropout_model(input_shape=(256, 4), learning_rate=0.001):
    """
    Builds and compiles a 1D Convolutional Autoencoder.
    
    The model is trained on background (data) and is expected to have
    HIGH reconstruction error for signal (sim) and LOW error for background.

    Args:
        input_shape (tuple): Input shape (samples, channels). E.g., (256, 4).
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        tuple: (compiled keras.Model, bool requires_transpose)
               The model is the autoencoder. requires_transpose is True.
    """
    
    inputs = Input(shape=input_shape)

    # --- Encoder ---
    # (256, 4) -> (128, 16)
    x = Conv1D(16, kernel_size=5, padding="same", activation="relu", strides=2)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    
    # (128, 16) -> (64, 32)
    x = Conv1D(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    
    # (64, 32) -> (32, 64)
    x = Conv1D(64, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization(name="latent_space")(x)
    x = Dropout(0.1)(x)
    
    # Bottleneck
    # (32, 64)
    
    # --- Decoder ---
    # (32, 64) -> (64, 32)
    x = Conv1DTranspose(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (64, 32) -> (128, 16)
    x = Conv1DTranspose(16, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (128, 16) -> (256, 4)
    # Use Conv1DTranspose with stride 2 to get back to 256
    # Final layer uses 'linear' activation to reconstruct the input values
    outputs = Conv1DTranspose(input_shape[-1], kernel_size=5, padding="same", activation="linear", strides=2)(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs, name="Conv1D_Autoencoder")

    # Compile with Mean Squared Error loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae'] # Mean Absolute Error
    )
    
    # This model expects (samples, channels), so transpose is needed.
    return model, True

def build_autoencoder_tightneck_model(input_shape=(256, 4), learning_rate=0.001):
    """
    Builds and compiles a 1D Convolutional Autoencoder.
    
    The model is trained on background (data) and is expected to have
    HIGH reconstruction error for signal (sim) and LOW error for background.

    Args:
        input_shape (tuple): Input shape (samples, channels). E.g., (256, 4).
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        tuple: (compiled keras.Model, bool requires_transpose)
               The model is the autoencoder. requires_transpose is True.
    """
    
    inputs = Input(shape=input_shape)

    # --- Encoder ---
    # (256, 4) -> (128, 16)
    x = Conv1D(16, kernel_size=5, padding="same", activation="relu", strides=2)(inputs)
    x = BatchNormalization()(x)
    
    # (128, 16) -> (64, 32)
    x = Conv1D(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
        
    # (64, 32) -> (32, 32)  <-- Tighter bottleneck
    x = Conv1D(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization(name="latent_space")(x)
    
    # Bottleneck
    # (32, 32)
    
    # --- Decoder ---
    # (32, 32) -> (64, 32)
    x = Conv1DTranspose(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (64, 32) -> (128, 16)
    x = Conv1DTranspose(16, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (128, 16) -> (256, 4)
    # Use Conv1DTranspose with stride 2 to get back to 256
    # Final layer uses 'linear' activation to reconstruct the input values
    outputs = Conv1DTranspose(input_shape[-1], kernel_size=5, padding="same", activation="linear", strides=2)(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs, name="Conv1D_Autoencoder")

    # Compile with Mean Squared Error loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae'] # Mean Absolute Error
    )
    
    # This model expects (samples, channels), so transpose is needed.
    return model, True


def build_autoencoder_denoising_model(input_shape=(256, 4), learning_rate=0.001):
    """
    Builds and compiles a 1D Convolutional Autoencoder.
    
    The model is trained on background (data) and is expected to have
    HIGH reconstruction error for signal (sim) and LOW error for background.

    Args:
        input_shape (tuple): Input shape (samples, channels). E.g., (256, 4).
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        tuple: (compiled keras.Model, bool requires_transpose)
               The model is the autoencoder. requires_transpose is True.
    """
    
    inputs = Input(shape=input_shape)

    # Add noise to the input, but the model will reconstruct the original (y)
    noisy_inputs = GaussianNoise(stddev=0.1)(inputs) # Tune stddev

    # --- Encoder ---
    # (256, 4) -> (128, 16)
    x = Conv1D(16, kernel_size=5, padding="same", activation="relu", strides=2)(noisy_inputs)
    x = BatchNormalization()(x)
    
    # (128, 16) -> (64, 32)
    x = Conv1D(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (64, 32) -> (32, 64)
    x = Conv1D(64, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization(name="latent_space")(x)
    
    # Bottleneck
    # (32, 64)
    
    # --- Decoder ---
    # (32, 64) -> (64, 32)
    x = Conv1DTranspose(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (64, 32) -> (128, 16)
    x = Conv1DTranspose(16, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (128, 16) -> (256, 4)
    # Use Conv1DTranspose with stride 2 to get back to 256
    # Final layer uses 'linear' activation to reconstruct the input values
    outputs = Conv1DTranspose(input_shape[-1], kernel_size=5, padding="same", activation="linear", strides=2)(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs, name="Conv1D_Autoencoder")

    # Compile with Mean Squared Error loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae'] # Mean Absolute Error
    )
    
    # This model expects (samples, channels), so transpose is needed.
    return model, True


def build_autoencoder_parallel_model(input_shape=(256, 4), learning_rate=0.001):
    """
    Builds and compiles a 1D Convolutional Autoencoder.
    
    The model is trained on background (data) and is expected to have
    HIGH reconstruction error for signal (sim) and LOW error for background.

    Args:
        input_shape (tuple): Input shape (samples, channels). E.g., (256, 4).
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        tuple: (compiled keras.Model, bool requires_transpose)
               The model is the autoencoder. requires_transpose is True.
    """
    
    inputs = Input(shape=input_shape)


    # --- Encoder ---
    
    # 1. Parallel Multi-Scale Input Module
    # (256, 4) -> (256, F1+F2+F3)
    branch_5  = Conv1D(8, kernel_size=5, padding="same", activation="relu")(inputs)
    branch_15 = Conv1D(8, kernel_size=15, padding="same", activation="relu")(inputs)
    branch_31 = Conv1D(8, kernel_size=31, padding="same", activation="relu")(inputs)
    
    # Concatenate the features from all scales
    # 8+8+8 = 24 filters total
    x = Concatenate()([branch_5, branch_15, branch_31])
    x = BatchNormalization()(x)
    
    # Now, proceed with downsampling this rich feature map
    # (256, 24) -> (128, 32)
    x = Conv1D(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (128, 32) -> (64, 64)
    x = Conv1D(64, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)

    # (64, 64) -> (32, 128)
    x = Conv1D(128, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization(name="latent_space")(x)
    
    # Bottleneck
    # (32, 128)
    
    # --- Decoder ---
    # (32, 128) -> (64, 64)
    x = Conv1DTranspose(64, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (64, 64) -> (128, 32)
    x = Conv1DTranspose(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (128, 32) -> (256, 24) <-- Note: filters match the *concatenated* layer
    x = Conv1DTranspose(24, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)

    # 2. Final Output Layer
    # (256, 24) -> (256, 4)
    # We use a 1x1 conv to map the 24 filters back to the 4 original channels
    outputs = Conv1D(input_shape[-1], kernel_size=1, padding="same", activation="linear")(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs, name="MultiScale_AE")

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    # This model expects (samples, channels), so transpose is needed.
    return model, True


def build_autoencoder_sequential_model(input_shape=(256, 4), learning_rate=0.001):
    """
    Builds and compiles a 1D Convolutional Autoencoder.
    
    The model is trained on background (data) and is expected to have
    HIGH reconstruction error for signal (sim) and LOW error for background.

    Args:
        input_shape (tuple): Input shape (samples, channels). E.g., (256, 4).
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        tuple: (compiled keras.Model, bool requires_transpose)
               The model is the autoencoder. requires_transpose is True.
    """
    
    inputs = Input(shape=input_shape)

    # --- Encoder ---
    # (256, 4) -> (128, 16)
    # Use a LARGE kernel to find long-range patterns
    x = Conv1D(16, kernel_size=31, padding="same", activation="relu", strides=2)(inputs)
    x = BatchNormalization()(x)
    
    # (128, 16) -> (64, 32)
    # Use a MEDIUM kernel
    x = Conv1D(32, kernel_size=15, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (64, 32) -> (32, 64)
    # Use a SMALL kernel for fine-grained details in the compressed space
    x = Conv1D(64, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization(name="latent_space")(x)
    
    # Bottleneck
    # (32, 64)
    
    # --- Decoder ---
    # (32, 64) -> (64, 32)
    # Mirror the encoder: use the SMALL kernel first
    x = Conv1DTranspose(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (64, 32) -> (128, 16)
    # MEDIUM kernel
    x = Conv1DTranspose(16, kernel_size=15, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (128, 16) -> (256, 4)
    # LARGE kernel
    outputs = Conv1DTranspose(input_shape[-1], kernel_size=31, padding="same", activation="linear", strides=2)(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs, name="Conv1D_Autoencoder")

    # Compile with Mean Squared Error loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae'] # Mean Absolute Error
    )
    
    # This model expects (samples, channels), so transpose is needed.
    return model, True


def build_autoencoder_freq_bottleneck_model(input_shape=(129, 4), learning_rate=0.001):
    """
    Step 1: Combines the FFT-based model with a Tighter Bottleneck.
    
    The latent space filter count is reduced from 64 to 32 to force
    a more aggressive compression.
    """
    inputs = Input(shape=input_shape)

    # --- Encoder ---
    # (129, 4) -> (64, 16)
    x = Conv1D(16, kernel_size=3, padding="valid", activation="relu", strides=2)(inputs)
    x = BatchNormalization()(x)
    
    # (64, 16) -> (32, 32)
    x = Conv1D(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (32, 32) -> (16, 32)  <--- STEP 1 CHANGE
    # Tighter bottleneck: Reduced filters from 64 to 32
    x = Conv1D(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization(name="latent_space")(x)
    
    # --- Decoder ---
    # (16, 32) -> (32, 32)
    # This layer's filter count (32) matches the corresponding encoder layer (E2)
    x = Conv1DTranspose(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (32, 32) -> (64, 16)
    # This layer's filter count (16) matches the corresponding encoder layer (E1)
    x = Conv1DTranspose(16, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (64, 16) -> (129, 4)
    outputs = Conv1DTranspose(
        input_shape[-1], kernel_size=3, padding="valid", activation="linear", strides=2
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name="Step1_Bottleneck_AE")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model, True

def build_autoencoder_freq_denoising_model(input_shape=(129, 4), learning_rate=0.001, noise_stddev=0.1):
    """
    Step 2: Adds Denoising to the Step 1 (Bottleneck) model.
    
    A GaussianNoise layer is added after the input. The model is
    fed noisy data but must reconstruct the original, clean data,
    forcing it to learn the true underlying manifold.
    """
    inputs = Input(shape=input_shape, name="clean_input")

    # --- STEP 2 CHANGE ---
    # Add noise to the inputs. The encoder will see this.
    noisy_inputs = GaussianNoise(stddev=noise_stddev)(inputs)
    # ---------------------

    # --- Encoder ---
    # Encoder starts from the NOISY inputs
    # (129, 4) -> (64, 16)
    x = Conv1D(16, kernel_size=3, padding="valid", activation="relu", strides=2)(noisy_inputs)
    x = BatchNormalization()(x)
    
    # (64, 16) -> (32, 32)
    x = Conv1D(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (32, 32) -> (16, 32)
    # Tighter bottleneck (from Step 1)
    x = Conv1D(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization(name="latent_space")(x)
    
    # --- Decoder ---
    # (16, 32) -> (32, 32)
    x = Conv1DTranspose(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (32, 32) -> (64, 16)
    x = Conv1DTranspose(16, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (64, 16) -> (129, 4)
    # The final output
    outputs = Conv1DTranspose(
        input_shape[-1], kernel_size=3, padding="valid", activation="linear", strides=2
    )(x)

    # Note: The model's input is the CLEAN data (inputs)
    # The model's output is the reconstruction (outputs)
    # The noise is applied internally.
    model = Model(inputs=inputs, outputs=outputs, name="Step2_Denoising_AE")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model, True

def build_autoencoder_freq_mae_loss_model(input_shape=(129, 4), learning_rate=0.001, noise_stddev=0.1):
    """
    Step 3: Combines the Denoising Bottleneck model with MAE loss.
    
    This is the full model from Step 2, but compiled with
    Mean Absolute Error ('mae') as the loss function, which can be
    more robust to outliers (i.e., less likely to try and learn
    the signal's "spiky" frequency features).
    """
    inputs = Input(shape=input_shape, name="clean_input")

    # Add noise to the inputs
    noisy_inputs = GaussianNoise(stddev=noise_stddev)(inputs)

    # --- Encoder ---
    # (129, 4) -> (64, 16)
    x = Conv1D(16, kernel_size=3, padding="valid", activation="relu", strides=2)(noisy_inputs)
    x = BatchNormalization()(x)
    
    # (64, 16) -> (32, 32)
    x = Conv1D(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (32, 32) -> (16, 32)
    # Tighter bottleneck
    x = Conv1D(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization(name="latent_space")(x)
    
    # --- Decoder ---
    # (16, 32) -> (32, 32)
    x = Conv1DTranspose(32, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (32, 32) -> (64, 16)
    x = Conv1DTranspose(16, kernel_size=5, padding="same", activation="relu", strides=2)(x)
    x = BatchNormalization()(x)
    
    # (64, 16) -> (129, 4)
    outputs = Conv1DTranspose(
        input_shape[-1], kernel_size=3, padding="valid", activation="linear", strides=2
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name="Step3_MAE_Denoise_AE")

    # --- STEP 3 CHANGE ---
    # Compile with 'mae' loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mae',
        metrics=['mse', 'mae'] # Monitor both
    )
    # ---------------------
    
    return model, True