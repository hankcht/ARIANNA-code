import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Model

# Import model building functions from model_builder module
from model_builder import (
    build_cnn_model,
    build_1d_model,
    build_parallel_model,
    build_strided_model,
    build_parallel_strided_model
)

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


# --- Main Visualization Logic ---
if __name__ == "__main__":
    # Build the models
    astrid_cnn = build_cnn_model(input_shape=(n_channels, n_samples, 1), learning_rate=learning_rate)
    model_1d = build_1d_model(input_shape=(n_samples, n_channels), learning_rate=learning_rate)
    parallel_model = build_parallel_model(input_shape=(n_samples, n_channels), learning_rate=learning_rate)
    strided_model = build_strided_model(input_shape=(n_samples, n_channels), learning_rate=learning_rate)
    parallel_strided_model = build_parallel_strided_model(input_shape=(n_samples, n_channels), learning_rate=learning_rate)

    models = [
        ("1D Model", model_1d),
        ("Parallel Model", parallel_model),
        ("Strided Model", strided_model),
        ("Parallel Strided Model", parallel_strided_model),
        ("Astrid CNN Model", astrid_cnn)
    ]

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
        # Only visualize activations for 1D CNN models (skip 2D CNN)
        if "Astrid CNN" in model_name:
            print("\n" + "="*50)
            print("METHOD 3: SKIPPING ACTIVATION VISUALIZATION FOR 2D CNN")
            print("="*50)
            print("Layer activation visualization is configured for 1D Conv layers only.")
            continue

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

