import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, GlobalAveragePooling1D, Dense, Input, Concatenate, Conv2D, Dropout, Flatten

# --- Configuration & Setup ---

# 1. Define Model & Data Parameters
# The user specified a shape of [n_training, 256, 4] after correction.
# The input shape for one sample is (sequence_length, num_channels).
n_samples = 256    # Sequence length (time steps)
n_channels = 4     # Number of features per time step
learning_rate = 0.001

# 2. Create a directory for saving plots
if not os.path.exists('plots'):
    os.makedirs('plots')
    print("Created 'plots/' directory for output images.")

# --- Astrid CNN Model ---
def build_cnn_model(input_shape):
    """
    Builds and compiles the CNN model architecture.

    Args:
        n_channels (int): Number of input channels.
        n_samples (int): Number of samples per trace.
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        keras.Model: The compiled Keras model.
    """
    model = Sequential()
    model.add(Conv2D(20, (4, 10), activation='relu', input_shape=input_shape, groups=1))
    model.add(Conv2D(10, (1, 10), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# --- 1D CNN Model Definition ---
def build_1d_model(input_shape):
    """Builds and compiles the Keras Sequential model."""
    model = Sequential(name="Visualized_CNN")

    # Multi-scale idea with stacked Conv1D layers scanning across the 256-step sequence.
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

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# --- Parallel Model Definition using the Functional API ---
def build_parallel_model(input_shape):
    """Builds and compiles a Keras model with parallel Conv1D branches."""
    
    # 1. Define the input layer - this is the entry point for your data
    inputs = Input(shape=input_shape)

    # 2. Create the three parallel "specialist" branches.
    # Each branch takes the valid `inputs` tensor.
    
    # Branch A: Specialist with small kernel
    branch_a = Conv1D(32, kernel_size=5, padding="valid", activation="relu")(inputs)
    
    # Branch B: Specialist with medium kernel
    branch_b = Conv1D(32, kernel_size=15, padding="valid", activation="relu")(inputs)
    
    # Branch C: Specialist with large kernel
    branch_c = Conv1D(32, kernel_size=31, padding="valid", activation="relu")(inputs)
    
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
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# --- Main Visualization Logic ---
if __name__ == "__main__":
    # Build the models
    astrid_cnn = build_cnn_model(input_shape=(n_channels, n_samples, 1))
    model = build_1d_model(input_shape=(n_samples, n_channels))
    parallel_model = build_parallel_model(input_shape=(n_samples, n_channels))

    models = [("1D Model", model), ("Parallel Model", parallel_model), ("Astrid CNN Model", astrid_cnn)]

    for model_name, model in models:
        print(f"\n\n{'='*60}\nVISUALIZING: {model_name}\n{'='*60}")
        # --- Method 1: Print Text Summary ---
        print("\n" + "="*50)
        print(f"METHOD 1: MODEL SUMMARY {model_name}")
        print("="*50)
        model.summary()


        # --- Method 2: Save Static Graph Plot ---
        print("\n" + "="*50)
        print(f"METHOD 2: PLOTTING MODEL ARCHITECTURE {model_name}")
        print("="*50)
        try:
            plot_path = os.path.join('plots', f'model_architecture_{model_name}.png')
            keras.utils.plot_model(
                model,
                to_file=plot_path,
                show_shapes=True,
                show_layer_names=True,
                show_dtype=False,
                show_layer_activations=True
            )
            print(f"Successfully saved model diagram to '{plot_path}'")
            print("NOTE: This requires pydot and graphviz to be installed (`pip install pydot graphviz`).")
        except ImportError as e:
            print(f"\nCould not generate model plot. Error: {e}")
            print("Please install pydot and graphviz (`pip install pydot graphviz`) to use `plot_model`.")


        # --- Method 3: Visualize and Save Layer Activations ---
        print("\n" + "="*50)
        print("METHOD 3: VISUALIZING LAYER ACTIVATIONS")
        print("="*50)

        # 1. Create a sample input tensor (e.g., random noise)
        # Shape is (num_samples_to_predict, sequence_length, num_channels)
        sample_input = np.random.rand(1, n_samples, n_channels)
        print(f"Generated a sample input tensor of shape: {sample_input.shape}")

        # 2. Identify the convolutional layers to inspect
        conv_layer_names = [layer.name for layer in model.layers if 'conv1d' in layer.name]
        print(f"Found Conv1D layers to inspect: {conv_layer_names}")

        # 3. Create a new model that outputs the activations from these layers
        layer_outputs = [model.get_layer(name).output for name in conv_layer_names]
        activation_model = Model(inputs=model.inputs, outputs=layer_outputs)

        # 4. Get the activations by running prediction
        activations = activation_model.predict(sample_input)
        print("Extracted activations from each convolutional layer.")

        # 5. Plot the results and save the figure
        num_layers_to_plot = len(activations)
        fig, axes = plt.subplots(num_layers_to_plot + 1, 1, figsize=(15, 12), dpi=100)
        plt.style.use('seaborn-v0_8-whitegrid')

        # Plot all channels of the original input
        ax = axes[0]
        for i in range(n_channels):
            ax.plot(sample_input[0, :, i], label=f'Channel {i+1}', linestyle='-')
        ax.set_title(f'Original Input Signal (Showing All {n_channels} Channels)')
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.legend(loc='upper right')

        # Plot the activation maps for each convolutional layer
        for i, (name, act) in enumerate(zip(conv_layer_names, activations)):
            ax = axes[i + 1]
            # The shape of `act` is (1, n_samples, num_filters), e.g., (1, 256, 32)
            # We plot the output of the first few filters on the valid axes for clarity.
            num_filters_to_show = 5
            num_filters_total = act.shape[-1]
            ax.plot(act[0, :, :num_filters_to_show], alpha=0.7)
            ax.set_title(f'Activations from: "{name}" (Showing {num_filters_to_show}/{num_filters_total} Filters)')
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Activation")

        fig.suptitle("CNN Layer Activation Visualization", fontsize=18, weight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle

        # Save the plot
        activation_plot_path = os.path.join('plots', f'layer_activations_{model_name}.png')
        plt.savefig(activation_plot_path)
        # plt.show()
        print(f"\nSuccessfully saved layer activation plot to '{activation_plot_path}'")

