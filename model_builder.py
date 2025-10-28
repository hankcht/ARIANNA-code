"""
Model Builder Module
Contains functions to build different CNN architectures for the ARIANNA project.
"""

from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv1D, Conv2D, BatchNormalization, ReLU, 
    GlobalAveragePooling1D, Dense, Input, Concatenate, 
    Dropout, Flatten
)


def build_cnn_model(input_shape=(4, 256, 1), learning_rate=0.001):
    """
    Builds and compiles the Astrid CNN model architecture (2D CNN).

    Args:
        input_shape (tuple): Input shape for the model (channels, samples, 1).
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        tuple: (compiled keras.Model, bool requires_transpose)
    """
    model = Sequential(name="Astrid_CNN")
    model.add(Conv2D(20, (4, 10), activation='relu', input_shape=input_shape, groups=1))
    model.add(Conv2D(10, (1, 10), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model, False


def build_1d_model(input_shape=(256, 4), learning_rate=0.001):
    """
    Builds and compiles a 1D CNN model with sequential multi-scale convolutions.

    Args:
        input_shape (tuple): Input shape for the model (samples, channels).
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        tuple: (compiled keras.Model, bool requires_transpose)
    """
    model = Sequential(name="1D_CNN")

    # Multi-scale idea with stacked Conv1D layers scanning across the sequence.
    model.add(Conv1D(32, kernel_size=5, padding="valid", activation="relu", input_shape=input_shape))
    model.add(Conv1D(32, kernel_size=15, padding="valid", activation="relu"))
    model.add(Conv1D(32, kernel_size=31, padding="valid", activation="relu"))
    model.add(BatchNormalization())
    model.add(ReLU())

    # Downstream feature extractor
    model.add(Conv1D(64, kernel_size=7, padding="valid", activation="relu"))

    # Collapse across time
    model.add(GlobalAveragePooling1D())

    # Dense classification head
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model, True


def build_cnn_model_freq(input_shape=(4, 129, 1), learning_rate=0.001):
    """Frequency-domain variant of Astrid CNN with reduced sequence length."""

    model = Sequential(name="Astrid_CNN")
    model.add(Conv2D(20, (4, 10), activation='relu', input_shape=input_shape, groups=1))
    model.add(Conv2D(10, (1, 10), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model, False


def build_1d_model_freq(input_shape=(129, 4), learning_rate=0.001):
    """Frequency-domain variant of the 1D CNN with reduced sequence length."""

    model = Sequential(name="1D_CNN")

    model.add(Conv1D(32, kernel_size=5, padding="valid", activation="relu", input_shape=input_shape))
    model.add(Conv1D(32, kernel_size=15, padding="valid", activation="relu"))
    model.add(Conv1D(32, kernel_size=31, padding="valid", activation="relu"))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Conv1D(64, kernel_size=7, padding="valid", activation="relu"))

    model.add(GlobalAveragePooling1D())

    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model, True


def build_parallel_model(input_shape=(256, 4), learning_rate=0.001):
    """
    Builds and compiles a Keras model with parallel Conv1D branches.

    Args:
        input_shape (tuple): Input shape for the model (samples, channels).
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        tuple: (compiled keras.Model, bool requires_transpose)
    """
    # 1. Define the input layer - this is the entry point for your data
    inputs = Input(shape=input_shape)

    # 2. Create the three parallel "specialist" branches.
    # Each branch takes the valid `inputs` tensor.
    
    # Branch A: Specialist with small kernel
    branch_a = Conv1D(32, kernel_size=5, padding="same", activation="relu")(inputs)
    
    # Branch B: Specialist with medium kernel
    branch_b = Conv1D(32, kernel_size=15, padding="same", activation="relu")(inputs)
    
    # Branch C: Specialist with large kernel
    branch_c = Conv1D(32, kernel_size=31, padding="same", activation="relu")(inputs)

    # 3. Combine the "reports" from the specialists.
    # The Concatenate layer stacks the feature maps along the channel axis.
    # Output shape will be (256, 32+32+32) -> (256, 96)
    concatenated = Concatenate()([branch_a, branch_b, branch_c])
    
    # --- This is the "Manager" part of the model ---
    
    # We apply Batch Norm and ReLU to the combined feature map
    x = BatchNormalization()(concatenated)
    x = ReLU()(x)
    
    # Downstream feature extractor
    x = Conv1D(64, kernel_size=7, padding="valid", activation="relu")(x)
    
    # Collapse across time
    x = GlobalAveragePooling1D()(x)
    
    # Dense classification head
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    
    # 4. Create the final Model
    # The model is defined by its inputs and outputs.
    model = Model(inputs=inputs, outputs=outputs, name="Parallel_CNN")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model, True


def build_parallel_model_freq(input_shape=(129, 4), learning_rate=0.001):
    """Frequency-domain variant of the parallel Conv1D model."""

    inputs = Input(shape=input_shape)

    branch_a = Conv1D(32, kernel_size=5, padding="same", activation="relu")(inputs)
    branch_b = Conv1D(32, kernel_size=15, padding="same", activation="relu")(inputs)
    branch_c = Conv1D(32, kernel_size=31, padding="same", activation="relu")(inputs)

    concatenated = Concatenate()([branch_a, branch_b, branch_c])

    x = BatchNormalization()(concatenated)
    x = ReLU()(x)

    x = Conv1D(64, kernel_size=7, padding="valid", activation="relu")(x)

    x = GlobalAveragePooling1D()(x)

    x = Dense(32, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Parallel_CNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model, True


def build_strided_model(input_shape=(256, 4), learning_rate=0.001):
    """
    Builds and compiles a 1D CNN model using strided convolutions in the first 3 layers.
    This reduces the sequence length progressively while extracting features.

    Args:
        input_shape (tuple): Input shape for the model (samples, channels).
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        tuple: (compiled keras.Model, bool requires_transpose)
    """
    model = Sequential(name="Strided_CNN")

    # First 3 layers use stride to downsample the temporal dimension
    # stride=2 reduces sequence length by half at each layer
    model.add(Conv1D(32, kernel_size=5, strides=5, padding="same", activation="relu", input_shape=input_shape))
    model.add(Conv1D(32, kernel_size=15, strides=15, padding="same", activation="relu"))
    model.add(Conv1D(32, kernel_size=31, strides=31, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(ReLU())

    # Downstream feature extractor (no stride)
    model.add(Conv1D(64, kernel_size=7, padding="same", activation="relu"))

    # Collapse across time
    model.add(GlobalAveragePooling1D())

    # Dense classification head
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model, True


def build_strided_model_freq(input_shape=(129, 4), learning_rate=0.001):
    """Frequency-domain variant of the strided Conv1D model."""

    model = Sequential(name="Strided_CNN")

    model.add(Conv1D(32, kernel_size=5, strides=5, padding="same", activation="relu", input_shape=input_shape))
    model.add(Conv1D(32, kernel_size=15, strides=15, padding="same", activation="relu"))
    model.add(Conv1D(32, kernel_size=31, strides=31, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Conv1D(64, kernel_size=7, padding="same", activation="relu"))

    model.add(GlobalAveragePooling1D())

    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model, True


def build_parallel_strided_model(input_shape=(256, 4), learning_rate=0.001):
    """
    Builds and compiles a parallel Conv1D model with strided convolutions.
    Combines the parallel branch architecture with strided downsampling.

    Args:
        input_shape (tuple): Input shape for the model (samples, channels).
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        tuple: (compiled keras.Model, bool requires_transpose)
    """
    # Define the input layer
    inputs = Input(shape=input_shape)

    # Create three parallel branches with strided convolutions
    # Branch A: Small kernel with stride
    branch_a = Conv1D(32, kernel_size=5, strides=5, padding="same", activation="relu")(inputs)
    
    # Branch B: Medium kernel with stride
    branch_b = Conv1D(32, kernel_size=15, strides=15, padding="same", activation="relu")(inputs)
    
    # Branch C: Large kernel with stride
    branch_c = Conv1D(32, kernel_size=31, strides=31, padding="same", activation="relu")(inputs)

    # Printing shapes
    print(f"Branch A shape: {branch_a.shape}")
    print(f"Branch B shape: {branch_b.shape}")
    print(f"Branch C shape: {branch_c.shape}")

    # Combine the parallel branches
    concatenated = Concatenate()([branch_a, branch_b, branch_c])
    
    # Apply batch normalization and activation
    x = BatchNormalization()(concatenated)
    x = ReLU()(x)
    
    # Additional strided convolution layer
    x = Conv1D(64, kernel_size=15, strides=2, padding="same", activation="relu")(x)
    
    # Downstream feature extractor
    x = Conv1D(64, kernel_size=7, padding="same", activation="relu")(x)
    
    # Collapse across time
    x = GlobalAveragePooling1D()(x)
    
    # Dense classification head
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    
    # Create the final Model
    model = Model(inputs=inputs, outputs=outputs, name="Parallel_Strided_CNN")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model, True


def build_parallel_strided_model_freq(input_shape=(129, 4), learning_rate=0.001):
    """Frequency-domain variant of the parallel strided Conv1D model."""

    inputs = Input(shape=input_shape)

    branch_a = Conv1D(32, kernel_size=5, strides=5, padding="same", activation="relu")(inputs)
    branch_b = Conv1D(32, kernel_size=15, strides=15, padding="same", activation="relu")(inputs)
    branch_c = Conv1D(32, kernel_size=31, strides=31, padding="same", activation="relu")(inputs)

    print(f"Branch A shape: {branch_a.shape}")
    print(f"Branch B shape: {branch_b.shape}")
    print(f"Branch C shape: {branch_c.shape}")

    concatenated = Concatenate()([branch_a, branch_b, branch_c])

    x = BatchNormalization()(concatenated)
    x = ReLU()(x)

    x = Conv1D(64, kernel_size=15, strides=2, padding="same", activation="relu")(x)

    x = Conv1D(64, kernel_size=7, padding="same", activation="relu")(x)

    x = GlobalAveragePooling1D()(x)

    x = Dense(32, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Parallel_Strided_CNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model, True
