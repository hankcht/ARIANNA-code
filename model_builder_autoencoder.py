"""
Model Builder for 1D Convolutional Autoencoder
"""

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, ReLU, Input, 
    Conv1DTranspose, MaxPooling1D, UpSampling1D
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
