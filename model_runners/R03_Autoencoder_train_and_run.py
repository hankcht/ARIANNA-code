"""
Trains a Convolutional Autoencoder for Anomaly Detection.

1. Trains *only* on real background data (Backlobe).
2. The model learns to reconstruct "normal" background.
3. Evaluates by measuring reconstruction error (MSE).
   - Signal (RCR) should have HIGH error.
   - Background (Backlobe) should have LOW error.
"""

import os
import sys
import re
import pickle
import argparse
from datetime import datetime
from glob import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Model
from sklearn.manifold import TSNE # Added for latent space plots
from NuRadioReco.utilities import units
from pathlib import Path

# --- Local Imports from your project structure ---
sys.path.append(str(Path(__file__).resolve().parents[1] / '200s_time'))
from A0_Utilities import load_sim_rcr, load_data, pT, load_config

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_builder_autoencoder import build_autoencoder_model # Import AE builder
from data_channel_cycling import cycle_channels
# Import original data loading functions
from R01_1D_CNN_train_and_run import (
    load_and_prep_data_for_training, 
    _compute_frequency_magnitude, 
    _apply_frequency_edge_filter,
    save_and_plot_training_history, # Will plot MSE/MAE loss
    load_2016_backlobe_templates,
    load_new_coincidence_data
)

# --- Model Selection (Autoencoders) ---
MODEL_BUILDERS = {
    '1d_autoencoder': build_autoencoder_model
    # '1d_autoencoder_freq': build_autoencoder_model_freq (if you create this)
}


def train_autoencoder_model(training_backlobe, config, learning_rate, model_type):
    """
    Trains the Autoencoder model.
    Trains ONLY on background data (training_backlobe).
    """
    
    if model_type not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model_type: {model_type}.")
        
    model_builder = MODEL_BUILDERS[model_type]
    
    # 1. Get input shape and builder
    is_freq_model = model_type.endswith('_freq')
    if is_freq_model:
        input_seq_len = 129
    else:
        input_seq_len = 256

    # 2. Prepare data
    # We only use background data for training!
    x_train = training_backlobe
    
    # The model learns to reconstruct its own input
    y_train = x_train 

    # Get model and transpose flag
    model, requires_transpose = model_builder(
        input_shape=(input_seq_len, 4), # Hard-coded for 1D models
        learning_rate=learning_rate
    )

    if requires_transpose:
        x_train = x_train.transpose(0, 2, 1)
        y_train = y_train.transpose(0, 2, 1)
    else:
        # (This AE model builder always requires transpose, but good to check)
        if x_train.ndim == 3:
            x_train = np.expand_dims(x_train, axis=-1)
            y_train = np.expand_dims(y_train, axis=-1)
    
    print(f"Training data shape (x): {x_train.shape}, (y): {y_train.shape}")

    # 3. Train the model
    callbacks_list = [keras.callbacks.EarlyStopping(
        monitor='val_loss', # Monitor validation MSE
        patience=config['early_stopping_patience']
    )]

    model.summary()

    history = model.fit(x_train, y_train,
                        validation_split=0.25,
                        epochs=config['keras_epochs'],
                        batch_size=config['keras_batch_size'],
                        verbose=config['verbose_fit'],
                        callbacks=callbacks_list)

    return model, history, requires_transpose


def evaluate_model_performance_autoencoder(model, sim_rcr_all, data_backlobe_traces_rcr_all, output_cut_value, config, model_type, requires_transpose):
    """
    Evaluates the Autoencoder model using per-event reconstruction error (MSE).
    
    Signal (RCR) should have HIGH error.
    Background (Backlobe) should have LOW error.
    
    `output_cut_value` is the MSE threshold.
    """
    # Prepare data based on model type
    if requires_transpose:
        sim_rcr_prepped = sim_rcr_all.transpose(0, 2, 1)
        data_backlobe_prepped = data_backlobe_traces_rcr_all.transpose(0, 2, 1)
    else:
        sim_rcr_prepped = sim_rcr_all[..., np.newaxis] if sim_rcr_all.ndim == 3 else sim_rcr_all
        data_backlobe_prepped = data_backlobe_traces_rcr_all[..., np.newaxis] if data_backlobe_traces_rcr_all.ndim == 3 else data_backlobe_traces_rcr_all
    
    print(f"Predicting on RCR (signal) set of shape {sim_rcr_prepped.shape}...")
    reconstructed_rcr = model.predict(sim_rcr_prepped, batch_size=config['keras_batch_size'])
    print(f"Predicting on Backlobe (background) set of shape {data_backlobe_prepped.shape}...")
    reconstructed_backlobe = model.predict(data_backlobe_prepped, batch_size=config['keras_batch_size'])
    
    # Calculate per-event MSE
    # (reconstructed_shape) - (original_shape)
    # Axis (1, 2) are (samples, channels)
    axis_to_average = tuple(range(1, sim_rcr_prepped.ndim)) # (1, 2)
    
    prob_rcr = np.mean(np.square(reconstructed_rcr - sim_rcr_prepped), axis=axis_to_average)
    prob_backlobe = np.mean(np.square(reconstructed_backlobe - data_backlobe_prepped), axis=axis_to_average)
    
    print(f"Mean RCR (Signal) Reconstruction MSE: {np.mean(prob_rcr):.4g}")
    print(f"Mean Backlobe (BG) Reconstruction MSE: {np.mean(prob_backlobe):.4g}")

    # Efficiency: Signal (RCR) is events WITH HIGH ERROR
    rcr_efficiency = (np.sum(prob_rcr > output_cut_value) / len(prob_rcr)) * 100
    # Background (Backlobe) retained are events WITH HIGH ERROR
    backlobe_efficiency = (np.sum(prob_backlobe > output_cut_value) / len(prob_backlobe)) * 100

    print(f'RCR efficiency (Signal > Cut): {rcr_efficiency:.2f}%')
    print(f'Backlobe efficiency (BG > Cut): {backlobe_efficiency:.4f}%')
    print(f'Lengths: RCR {len(prob_rcr)}, Backlobe {len(prob_backlobe)}')

    return prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency


def plot_network_output_histogram_autoencoder(prob_rcr, prob_backlobe, rcr_efficiency,
                                              backlobe_efficiency, config, timestamp, learning_rate, model_type,
                                              dataset_name_suffix="main"):
    """
    Plots the histogram of network outputs (Reconstruction MSE)
    for RCR and Backlobe events.
    
    Args:
        dataset_name_suffix (str): Suffix for filename and title (e.g., "main", "validation").
    """
    amp = config['amp']
    prefix = config['prefix']
    output_cut_value = config['output_cut_value']
    lr_str = f"lr{learning_rate:.0e}".replace('-', '')
    plot_path = config['base_plot_path']
    domain_label = config.get('domain_label', 'time')
    domain_suffix = config.get('domain_suffix', '')
    os.makedirs(plot_path, exist_ok=True)

    plt.figure(figsize=(8, 6))
    
    # Determine plot range
    max_val = max(np.percentile(prob_rcr, 99), np.percentile(prob_backlobe, 99.9))
    min_val = min(np.min(prob_rcr), np.min(prob_backlobe))
    plot_range = (min_val, max_val * 1.1)
    
    plt.hist(prob_backlobe, bins=50, range=plot_range, histtype='step', color='blue', linestyle='solid', label=f'Backlobe {len(prob_backlobe)}', density=True)
    plt.hist(prob_rcr, bins=50, range=plot_range, histtype='step', color='red', linestyle='solid', label=f'RCR {len(prob_rcr)}', density=True)

    plt.xlabel('Reconstruction Loss (MSE)', fontsize=18)
    plt.ylabel('Normalized Density', fontsize=18)
    plt.yscale('log')
    
    title_suffix = f"({dataset_name_suffix} set)"
    plt.title(f'{amp}_{domain_label} {model_type} Loss {title_suffix} (LR={learning_rate:.0e})', fontsize=14)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper right', fontsize=12)

    ax = plt.gca()
    ax.text(0.45, 0.75, f'RCR efficiency (Signal > Cut): {rcr_efficiency:.2f}%', fontsize=12, transform=ax.transAxes)
    ax.text(0.45, 0.70, f'Backlobe efficiency (BG > Cut): {backlobe_efficiency:.4f}%', fontsize=12, transform=ax.transAxes)
    ax.text(0.45, 0.60, f'LR: {learning_rate:.0e}', fontsize=12, transform=ax.transAxes)
    ax.text(0.45, 0.55, f'Model: {model_type}', fontsize=12, transform=ax.transAxes)
    plt.axvline(x=output_cut_value, color='y', label=f'Cut = {output_cut_value:.2g}', linestyle='--')
    plt.legend()
    
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.9)

    hist_file = os.path.join(plot_path, f'{timestamp}_{amp}_{model_type}_network_output_{prefix}_{lr_str}{domain_suffix}_{dataset_name_suffix}.png')
    print(f'saving {hist_file}')
    plt.savefig(hist_file)
    plt.close()


# --- NEW FUNCTION: Original vs. Reconstructed ---
def plot_original_vs_reconstructed(model, sim_rcr_all, data_backlobe_traces_rcr_all,
                                   prob_rcr, prob_backlobe, requires_transpose,
                                   timestamp, config, model_type, learning_rate,
                                   dataset_name_suffix="main"):
    """
    Plots the original vs. reconstructed traces for the:
    1. WORST reconstructed RCR (signal) event (max MSE).
    2. BEST reconstructed Backlobe (background) event (min MSE).
    
    Args:
        prob_rcr (np.ndarray): Array of MSEs for the RCR (signal) set.
        prob_backlobe (np.ndarray): Array of MSEs for the Backlobe (BG) set.
        dataset_name_suffix (str): Suffix for filename and title.
    """
    amp = config['amp']
    prefix = config['prefix']
    lr_str = f"lr{learning_rate:.0e}".replace('-', '')
    plot_path = config['base_plot_path']
    domain_label = config.get('domain_label', 'time')
    domain_suffix = config.get('domain_suffix', '')
    os.makedirs(plot_path, exist_ok=True)

    # --- 1. Find best/worst events ---
    # RCR (Signal): Find the WORST reconstruction (max MSE)
    worst_rcr_idx = np.argmax(prob_rcr)
    worst_rcr_mse = prob_rcr[worst_rcr_idx]
    rcr_original = sim_rcr_all[worst_rcr_idx] # Shape (4, 256)
    
    # Backlobe (BG): Find the BEST reconstruction (min MSE)
    best_bl_idx = np.argmin(prob_backlobe)
    best_bl_mse = prob_backlobe[best_bl_idx]
    bl_original = data_backlobe_traces_rcr_all[best_bl_idx] # Shape (4, 256)

    # --- 2. Prepare traces for model ---
    rcr_prepped = rcr_original.transpose(1, 0) if requires_transpose else rcr_original
    bl_prepped = bl_original.transpose(1, 0) if requires_transpose else bl_original
    
    # Add batch dimension
    rcr_prepped = rcr_prepped[np.newaxis, ...]
    bl_prepped = bl_prepped[np.newaxis, ...]

    # --- 3. Get reconstructions ---
    rcr_reconstructed = model.predict(rcr_prepped)[0] # Get first (only) batch item
    bl_reconstructed = model.predict(bl_prepped)[0]

    # --- 4. Transpose back if needed ---
    # Model output is (256, 4) if transpose was used, plot needs (4, 256)
    if requires_transpose:
        rcr_reconstructed = rcr_reconstructed.transpose(1, 0)
        bl_reconstructed = bl_reconstructed.transpose(1, 0)

    # --- 5. Plot RCR (Signal) ---
    fig, axes = plt.subplots(4, 2, figsize=(15, 10), sharex=True, sharey=True)
    fig.suptitle(f'{amp} {model_type} - WORST Reconstructed RCR (Signal) - MSE: {worst_rcr_mse:.4g}\n(Dataset: {dataset_name_suffix})', fontsize=16)
    
    axes[0, 0].set_title('Original Trace')
    axes[0, 1].set_title('Reconstructed Trace')
    
    for i in range(4):
        axes[i, 0].plot(rcr_original[i, :], color='red', label=f'Channel {i}')
        axes[i, 0].set_ylabel(f'Channel {i}')
        axes[i, 1].plot(rcr_reconstructed[i, :], color='black', label=f'Recon Chan {i}')
    
    axes[-1, 0].set_xlabel('Sample')
    axes[-1, 1].set_xlabel('Sample')
    
    plot_file_rcr = os.path.join(plot_path, f'{timestamp}_{amp}_{model_type}_recon_RCR_{prefix}_{lr_str}{domain_suffix}_{dataset_name_suffix}.png')
    plt.savefig(plot_file_rcr)
    plt.close(fig)
    print(f'Saved worst RCR reconstruction plot to: {plot_file_rcr}')

    # --- 6. Plot Backlobe (BG) ---
    fig, axes = plt.subplots(4, 2, figsize=(15, 10), sharex=True, sharey=True)
    fig.suptitle(f'{amp} {model_type} - BEST Reconstructed Backlobe (BG) - MSE: {best_bl_mse:.4g}\n(Dataset: {dataset_name_suffix})', fontsize=16)
    
    axes[0, 0].set_title('Original Trace')
    axes[0, 1].set_title('Reconstructed Trace')
    
    for i in range(4):
        axes[i, 0].plot(bl_original[i, :], color='blue', label=f'Channel {i}')
        axes[i, 0].set_ylabel(f'Channel {i}')
        axes[i, 1].plot(bl_reconstructed[i, :], color='black', label=f'Recon Chan {i}')
    
    axes[-1, 0].set_xlabel('Sample')
    axes[-1, 1].set_xlabel('Sample')
    
    plot_file_bl = os.path.join(plot_path, f'{timestamp}_{amp}_{model_type}_recon_Backlobe_{prefix}_{lr_str}{domain_suffix}_{dataset_name_suffix}.png')
    plt.savefig(plot_file_bl)
    plt.close(fig)
    print(f'Saved best Backlobe reconstruction plot to: {plot_file_bl}')


# --- NEW FUNCTION: Latent Space ---
def plot_latent_space(model, sim_rcr_all, data_backlobe_traces_rcr_all,
                      requires_transpose, timestamp, config, model_type, learning_rate,
                      dataset_name_suffix="main"):
    """
    Generates a t-SNE plot of the autoencoder's latent space.
    
    Colors points by class (RCR=Signal, Backlobe=Background).
    Assumes the bottleneck layer is named 'latent_space' in the model definition.
    
    Args:
        dataset_name_suffix (str): Suffix for filename and title.
    """
    amp = config['amp']
    prefix = config['prefix']
    lr_str = f"lr{learning_rate:.0e}".replace('-', '')
    plot_path = config['base_plot_path']
    domain_label = config.get('domain_label', 'time')
    domain_suffix = config.get('domain_suffix', '')
    os.makedirs(plot_path, exist_ok=True)
    
    # --- 1. Create Encoder-only model ---
    try:
        # Assumes your bottleneck layer is named 'latent_space' in model_builder_autoencoder.py
        latent_layer = model.get_layer('latent_space')
    except ValueError:
        print("\n--- LATENT SPACE PLOTTER ERROR ---")
        print("Error: Layer 'latent_space' not found.")
        print("Please name your bottleneck layer 'latent_space' in model_builder_autoencoder.py")
        print("Trying to find smallest layer as a fallback...")
        try:
            # Find layer with minimum number of units/neurons in its output shape
            # (batch_size, ...dims...)
            smallest_layer = min(
                [l for l in model.layers if 'input' not in l.name.lower()], 
                key=lambda l: np.prod(l.output_shape[1:])
            )
            latent_layer = smallest_layer
            print(f"Using layer '{latent_layer.name}' as latent space (shape: {latent_layer.output_shape})")
        except Exception as e:
            print(f"Could not find fallback layer: {e}. Skipping latent space plot.")
            return
            
    encoder = Model(inputs=model.input, outputs=latent_layer.output)

    # --- 2. Subsample data (t-SNE is slow) ---
    n_samples = min(2000, len(sim_rcr_all), len(data_backlobe_traces_rcr_all))
    if n_samples == 0:
        print(f"Skipping latent space plot for {dataset_name_suffix}: not enough data.")
        return
        
    print(f"Subsampling {n_samples} events from each class for t-SNE plot...")
    rcr_indices = np.random.choice(len(sim_rcr_all), n_samples, replace=False)
    bl_indices = np.random.choice(len(data_backlobe_traces_rcr_all), n_samples, replace=False)
    
    rcr_subset = sim_rcr_all[rcr_indices]
    bl_subset = data_backlobe_traces_rcr_all[bl_indices]

    # --- 3. Prepare and combine data ---
    rcr_prepped = rcr_subset.transpose(0, 2, 1) if requires_transpose else rcr_subset
    bl_prepped = bl_subset.transpose(0, 2, 1) if requires_transpose else bl_subset
    
    if rcr_prepped.ndim == 3: rcr_prepped = rcr_prepped[..., np.newaxis]
    if bl_prepped.ndim == 3: bl_prepped = bl_prepped[..., np.newaxis]
        
    combined_data = np.vstack([rcr_prepped, bl_prepped])
    labels = np.array([1] * n_samples + [0] * n_samples) # 1=RCR, 0=Backlobe

    # --- 4. Get latent vectors ---
    print("Running encoder to get latent vectors...")
    latent_vectors = encoder.predict(combined_data)
    
    # Flatten if latent space is not already 1D
    if latent_vectors.ndim > 2:
        latent_vectors = latent_vectors.reshape(latent_vectors.shape[0], -1)
    
    print(f"Latent vector shape: {latent_vectors.shape}")

    # --- 5. Run t-SNE ---
    print("Running t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(latent_vectors)

    # --- 6. Plot ---
    print("Plotting t-SNE results...")
    plt.figure(figsize=(10, 8))
    
    rcr_points = tsne_results[labels==1]
    bl_points = tsne_results[labels==0]
    
    plt.scatter(bl_points[:, 0], bl_points[:, 1], label=f'Backlobe (BG) - {n_samples}', alpha=0.5, c='blue')
    plt.scatter(rcr_points[:, 0], rcr_points[:, 1], label=f'RCR (Signal) - {n_samples}', alpha=0.5, c='red')
    
    plt.legend(loc='upper right')
    plt.title(f't-SNE Latent Space Visualization ({dataset_name_suffix} set)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    plot_file_tsne = os.path.join(plot_path, f'{timestamp}_{amp}_{model_type}_latent_space_tSNE_{prefix}_{lr_str}{domain_suffix}_{dataset_name_suffix}.png')
    plt.savefig(plot_file_tsne)
    plt.close()
    print(f'Saved t-SNE plot to: {plot_file_tsne}')


def main():
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train Autoencoder model.')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate (e.g., 0.0001)')
    parser.add_argument('--model_type', type=str, required=True, choices=list(MODEL_BUILDERS.keys()),
                        help=f'Type of AE model to train. Choose from: {list(MODEL_BUILDERS.keys())}')

    args = parser.parse_args()
    
    learning_rate = args.learning_rate
    model_type = args.model_type
    
    config = load_config()
    amp = config['amp']
    prefix = config['prefix']
    
    # --- Setup Paths ---
    is_freq_model = model_type.endswith('_freq')
    config['is_freq_model'] = is_freq_model
    config['domain_label'] = 'freq' if is_freq_model else 'time'
    config['domain_suffix'] = '_freq' if is_freq_model else ''
    
    base_model_root = config['base_model_path']
    base_plot_root = config['base_plot_path']

    lr_folder = f'lr_{learning_rate:.0e}'.replace('-', '')
    model_folder = model_type # e.g., "1d_autoencoder"
    timestamp = datetime.now().strftime('%m.%d.%y_%H-%M')
    
    config['base_model_path'] = os.path.join(base_model_root, model_folder, lr_folder)
    config['base_plot_path'] = os.path.join(base_plot_root, f"{timestamp}", model_folder, lr_folder)
    
    # This cut value is now an MSE threshold. You may need to adjust it in your config.
    output_cut_value = config['output_cut_value']
    print(f"--- Starting Autoencoder training at {timestamp} ---")
    print(f"Model type: {model_type}, LR: {learning_rate}")
    print(f"NOTE: 'output_cut_value' from config is used as MSE threshold: {output_cut_value}")
    
    # Data load & prep
    # We load ALL data, but will only use background (backlobe) for training
    data = load_and_prep_data_for_training(config)
    training_backlobe = data['training_backlobe']
    sim_rcr_all = data['sim_rcr_all']
    data_backlobe_traces_rcr_all = data['data_backlobe_tracesRCR']

    # Apply channel cycling (augmentation) TO BACKGROUND TRAINING DATA
    print("Applying channel cycling augmentation to Backlobe training data...")
    training_backlobe_aug = cycle_channels(training_backlobe.copy(), channel_axis=1)
    
    # Train model
    model, history, requires_transpose = train_autoencoder_model(
        training_backlobe_aug, config, learning_rate, model_type
    )
    print('------> Autoencoder Training is Done!')

    # Save model
    domain_suffix = config['domain_suffix']
    lr_str = f"lr{learning_rate:.0e}".replace('-', '')
    model_filename = f'{timestamp}_{amp}_{model_type}_model_{prefix}_{lr_str}{domain_suffix}.h5'
    model_save_path = os.path.join(config['base_model_path'], model_filename)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f'Model saved to: {model_save_path}')

    # Save training history & plots (will show MSE loss)
    save_and_plot_training_history(history, config['base_model_path'], config['base_plot_path'], 
                                   timestamp, amp, config, learning_rate, model_type)

    # --- Run evaluation on MAIN training/test split ---
    print("\n--- Running evaluation on MAIN dataset (sim RCR vs. data Backlobe) ---")
    
    # 1. Evaluate & plot network output histogram
    prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency = \
        evaluate_model_performance_autoencoder(model, sim_rcr_all, data_backlobe_traces_rcr_all, 
                                               output_cut_value, config, model_type, requires_transpose)

    plot_network_output_histogram_autoencoder(prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency, 
                                              config, timestamp, learning_rate, model_type,
                                              dataset_name_suffix="main")

    # 2. Plot Original vs. Reconstructed
    print("\n--- Generating Original vs. Reconstructed plots for MAIN dataset ---")
    plot_original_vs_reconstructed(
        model, sim_rcr_all, data_backlobe_traces_rcr_all, 
        prob_rcr, prob_backlobe, # Pass in the calculated MSEs
        requires_transpose, timestamp, config, model_type, learning_rate,
        dataset_name_suffix="main"
    )

    # 3. Plot Latent Space
    print("\n--- Generating Latent Space plot for MAIN dataset ---")
    plot_latent_space(
        model, sim_rcr_all, data_backlobe_traces_rcr_all, 
        requires_transpose, timestamp, config, model_type, learning_rate,
        dataset_name_suffix="main"
    )

    # --- NEW: Run evaluation on VALIDATION dataset ---
    print("\n--- Running evaluation on VALIDATION dataset (2016 BL, Coincidence) ---")

    # Load 2016 backlobe templates
    template_dir = "/pub/tangch3/ARIANNA/DeepLearning/refactor/confirmed_2016_templates/"
    template_paths = sorted(glob(os.path.join(template_dir, "filtered_Event2016_Stn*.npy")))
    
    if not (os.path.exists(template_dir) and template_paths):
        print(f"Warning: 2016 backlobe template directory not found at {template_dir}. Skipping validation run.")
        print("Script finished.")
        return

    all_2016_backlobes, dict_2016 = load_2016_backlobe_templates(template_paths, amp_type=amp)
    print(f"Loaded {len(all_2016_backlobes)} 2016 backlobe traces.")
    
    # Load new coincidence data
    pkl_path = "/dfs8/s_barwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/9.24.25_CoincidenceDatetimes_passing_cuts_with_all_params_recalcZenAzi_calcPol.pkl"
    if not os.path.exists(pkl_path):
        print(f"Warning: Coincidence PKL file not found at {pkl_path}. Skipping validation run.")
        print("Script finished.")
        return

    passing_event_ids = [3047, 3432, 10195, 10231, 10273, 10284, 10444, 10449, 10466, 10471, 10554, 11197, 11220, 11230, 11236, 11243]
    special_event_id = 11230 # Suspected RCR (Signal)
    special_station_id = 13
    
    passing_traces, raw_traces, special_traces, _, _, _ = \
        load_new_coincidence_data(pkl_path, passing_event_ids, special_event_id, special_station_id)

    if len(passing_traces) == 0 or len(raw_traces) == 0:
        print(f"Warning: No traces loaded from coincidence data. Skipping validation run.")
        print("Script finished.")
        return
        
    # --- Prepare Validation Datasets ---
    # Validation Background = 2016 Backlobes + Passing Coincidence + Raw Coincidence
    val_background_traces = np.vstack([all_2016_backlobes, passing_traces, raw_traces])
    
    # Validation Signal = Special "Suspected RCR" event
    val_signal_traces = special_traces

    if len(val_signal_traces) == 0:
        print(f"Warning: No special signal traces (Evt {special_event_id}, Stn {special_station_id}) loaded. Validation run will have no signal.")
        # Create an empty array with the correct shape to avoid errors
        val_signal_traces = np.empty((0, val_background_traces.shape[1], val_background_traces.shape[2]), dtype=val_background_traces.dtype)
    
    print(f"Validation Background traces shape: {val_background_traces.shape}")
    print(f"Validation Signal traces shape: {val_signal_traces.shape}")
    
    # Apply freq/filtering if necessary (though this AE is time-domain)
    if is_freq_model:
        sampling_rate = float(config.get('frequency_sampling_rate', 2.0))
        use_filtering = bool(config.get('use_filtering', False))
        
        print("Converting validation data to frequency domain...")
        val_background_traces = _compute_frequency_magnitude(val_background_traces, sampling_rate)
        val_signal_traces = _compute_frequency_magnitude(val_signal_traces, sampling_rate)

        if use_filtering:
            print("Applying frequency edge filtering to validation data...")
            val_background_traces = _apply_frequency_edge_filter(val_background_traces)
            val_signal_traces = _apply_frequency_edge_filter(val_signal_traces)

    # --- Run All Tests on Validation Data ---
    
    # 1. Evaluate & plot network output histogram
    print("\n--- Running evaluation on VALIDATION dataset ---")
    prob_rcr_val, prob_backlobe_val, rcr_eff_val, bl_eff_val = \
        evaluate_model_performance_autoencoder(model, val_signal_traces, val_background_traces, 
                                               output_cut_value, config, model_type, requires_transpose)

    plot_network_output_histogram_autoencoder(prob_rcr_val, prob_backlobe_val, rcr_eff_val, bl_eff_val, 
                                              config, timestamp, learning_rate, model_type,
                                              dataset_name_suffix="validation")

    # 2. Plot Original vs. Reconstructed
    print("\n--- Generating Original vs. Reconstructed plots for VALIDATION dataset ---")
    if len(val_signal_traces) > 0 and len(val_background_traces) > 0:
        plot_original_vs_reconstructed(
            model, val_signal_traces, val_background_traces, 
            prob_rcr_val, prob_backlobe_val,
            requires_transpose, timestamp, config, model_type, learning_rate,
            dataset_name_suffix="validation"
        )
    else:
        print("Skipping validation recon plot (missing signal or background data).")

    # 3. Plot Latent Space
    print("\n--- Generating Latent Space plot for VALIDATION dataset ---")
    if len(val_signal_traces) > 0 and len(val_background_traces) > 0:
        plot_latent_space(
            model, val_signal_traces, val_background_traces, 
            requires_transpose, timestamp, config, model_type, learning_rate,
            dataset_name_suffix="validation"
        )
    else:
        print("Skipping validation latent space plot (missing signal or background data).")

    print(f"\nScript finished successfully for Autoencoder model {model_type}, lr {learning_rate}")

if __name__ == "__main__":
    main()
