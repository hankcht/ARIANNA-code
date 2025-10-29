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
import argparse
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Model
from sklearn.manifold import TSNE # Added for latent space plots
from pathlib import Path

# --- Local Imports from your project structure ---
sys.path.append(str(Path(__file__).resolve().parents[1] / '200s_time'))
from A0_Utilities import load_config

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_builder_autoencoder import build_autoencoder_model # Import AE builder
from data_channel_cycling import cycle_channels
# Import original data loading functions
from R01_1D_CNN_train_and_run import (
    load_and_prep_data_for_training,
    save_and_plot_training_history,  # Will plot MSE/MAE loss
    load_new_coincidence_data,
    _compute_frequency_magnitude,
    _apply_frequency_edge_filter
)

# --- Model Selection (Autoencoders) ---
MODEL_BUILDERS = {
    '1d_autoencoder': build_autoencoder_model
    # '1d_autoencoder_freq': build_autoencoder_model_freq (if you create this)
}


# --- Validation Defaults & Plotting Helpers ---
DEFAULT_VALIDATION_PKL_PATH = \
    "/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/9.24.25_CoincidenceDatetimes_passing_cuts_with_all_params_recalcZenAzi_calcPol.pkl"
DEFAULT_VALIDATION_PASSING_EVENT_IDS = [
    3047, 3432, 10195, 10231, 10273, 10284, 10444, 10449,
    10466, 10471, 10554, 11197, 11220, 11230, 11236, 11243
]
DEFAULT_VALIDATION_SPECIAL_EVENT_ID = 11230
DEFAULT_VALIDATION_SPECIAL_STATION_ID = 13

LATENT_COLOR_MAP = {
    'Backlobe (BG)': 'blue',
    'RCR (Signal)': 'red',
    'Validation Passing': 'green',
    'Validation Raw': 'purple',
    'Validation Special Event': 'orange'
}

LATENT_MARKER_MAP = {
    'Backlobe (BG)': 'o',
    'RCR (Signal)': 'o',
    'Validation Passing': 's',
    'Validation Raw': '^',
    'Validation Special Event': '*'
}

HIST_COLOR_MAP = {
    'Backlobe (BG)': 'blue',
    'RCR (Signal)': 'red',
    'Validation Passing': 'green',
    'Validation Raw': 'purple',
    'Validation Special Event': 'orange'
}


def _sanitize_label_for_filename(label):
    """Return a filesystem-friendly string derived from a label."""

    safe_chars = set("abcdefghijklmnopqrstuvwxyz0123456789-_")
    sanitized = ''.join(ch_lower if ch_lower in safe_chars else '_' for ch_lower in label.lower())
    # Collapse repeated underscores for readability
    while '__' in sanitized:
        sanitized = sanitized.replace('__', '_')
    return sanitized.strip('_') or 'label'


def prepare_data_for_model(input_data, requires_transpose):
    """Prepare traces for inference based on the model's expected layout."""

    data_array = np.asarray(input_data)
    if data_array.size == 0:
        return data_array

    if data_array.ndim != 3:
        raise ValueError(
            f"Expected data with shape (events, channels, samples), got {data_array.shape}"
        )

    if requires_transpose:
        return data_array.transpose(0, 2, 1)

    # Channels-first layout requires an explicit feature axis
    return data_array[..., np.newaxis]


def calculate_reconstruction_mse(model, prepared_data, config):
    """Run model inference and compute per-event reconstruction MSE."""

    if prepared_data.size == 0 or prepared_data.shape[0] == 0:
        return np.array([]), np.empty_like(prepared_data)

    reconstructed = model.predict(prepared_data, batch_size=config['keras_batch_size'])
    axis = tuple(range(1, reconstructed.ndim))
    mse = np.mean(np.square(reconstructed - prepared_data), axis=axis)
    return mse, reconstructed


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

    if len(prob_rcr) == 0 or len(prob_backlobe) == 0:
        print(f"Skipping network output plot for {dataset_name_suffix}: missing data.")
        return

    plt.figure(figsize=(8, 6))
    
    # Determine plot range
    max_val = max(np.percentile(prob_rcr, 99), np.percentile(prob_backlobe, 99.9))
    min_val = min(np.min(prob_rcr), np.min(prob_backlobe))
    upper_bound = max_val * 1.1 if max_val > 0 else max_val + 1e-6
    plot_range = (min_val, upper_bound)
    
    plt.hist(prob_backlobe, bins=50, range=plot_range, histtype='step', color='blue', linestyle='solid', linewidth=1.5,
             label=f'Backlobe {len(prob_backlobe)}')
    plt.hist(prob_rcr, bins=50, range=plot_range, histtype='step', color='red', linestyle='solid', linewidth=1.5,
             label=f'RCR {len(prob_rcr)}')
    plt.yscale('log')

    plt.xlabel('Reconstruction Loss (MSE)', fontsize=18)
    plt.ylabel('Count', fontsize=18)
    
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
    # plt.axvline(x=output_cut_value, color='y', label=f'Cut = {output_cut_value:.2g}', linestyle='--')
    plt.legend()
    
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.9)

    hist_file = os.path.join(plot_path, f'{timestamp}_{amp}_{model_type}_network_output_{prefix}_{lr_str}{domain_suffix}_{dataset_name_suffix}.png')
    print(f'saving {hist_file}')
    plt.savefig(hist_file)
    plt.close()


def plot_normalized_network_output_histogram_autoencoder(prob_rcr, prob_backlobe, config, timestamp,
                                                         learning_rate, model_type, dataset_name_suffix="main",
                                                         cut_value=0.9):
    """Normalizes network outputs to [0, 1], plots histograms, and reports efficiencies."""

    amp = config['amp']
    prefix = config['prefix']
    domain_label = config.get('domain_label', 'time')
    domain_suffix = config.get('domain_suffix', '')
    lr_str = f"lr{learning_rate:.0e}".replace('-', '')
    plot_path = config['base_plot_path']
    os.makedirs(plot_path, exist_ok=True)

    if len(prob_rcr) == 0 or len(prob_backlobe) == 0:
        print(f"Skipping normalized network output plot for {dataset_name_suffix}: missing data.")
        return

    combined = np.concatenate([prob_rcr, prob_backlobe])
    min_val = float(np.min(combined))
    max_val = float(np.max(combined))

    if np.isclose(max_val, min_val):
        print(f"Warning: network outputs identical for {dataset_name_suffix}; normalized plot will be zero.")
        norm_rcr = np.zeros_like(prob_rcr, dtype=float)
        norm_backlobe = np.zeros_like(prob_backlobe, dtype=float)
    else:
        denom = max_val - min_val
        norm_rcr = np.clip((prob_rcr - min_val) / denom, 0.0, 1.0)
        norm_backlobe = np.clip((prob_backlobe - min_val) / denom, 0.0, 1.0)

    rcr_eff_norm = float(np.mean(norm_rcr > cut_value) * 100) if norm_rcr.size else 0.0
    backlobe_eff_norm = float(np.mean(norm_backlobe > cut_value) * 100) if norm_backlobe.size else 0.0

    print(f'Normalized RCR efficiency (>{cut_value:.2f}): {rcr_eff_norm:.2f}%')
    print(f'Normalized Backlobe efficiency (>{cut_value:.2f}): {backlobe_eff_norm:.4f}%')

    plt.figure(figsize=(8, 6))
    plt.hist(norm_backlobe, bins=50, range=(0.0, 1.0), histtype='step', color='blue', linestyle='solid', linewidth=1.5,
             label=f'Backlobe {len(norm_backlobe)}')
    plt.hist(norm_rcr, bins=50, range=(0.0, 1.0), histtype='step', color='red', linestyle='solid', linewidth=1.5,
             label=f'RCR {len(norm_rcr)}')

    plt.xlabel('Normalized Reconstruction Loss', fontsize=18)
    plt.ylabel('Count', fontsize=18)
    title_suffix = f"({dataset_name_suffix} set)"
    plt.title(f'{amp}_{domain_label} {model_type} Normalized Loss {title_suffix} (LR={learning_rate:.0e})', fontsize=14)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.yscale('log')

    ax = plt.gca()
    ax.text(0.45, 0.75, f'RCR eff (> {cut_value:.2f}): {rcr_eff_norm:.2f}%', fontsize=12, transform=ax.transAxes)
    ax.text(0.45, 0.70, f'Backlobe eff (> {cut_value:.2f}): {backlobe_eff_norm:.4f}%', fontsize=12, transform=ax.transAxes)
    ax.text(0.45, 0.60, f'LR: {learning_rate:.0e}', fontsize=12, transform=ax.transAxes)
    ax.text(0.45, 0.55, f'Model: {model_type}', fontsize=12, transform=ax.transAxes)
    plt.axvline(x=cut_value, color='y', linestyle='--', label=f'Cut = {cut_value:.2f}')
    plt.legend(loc='upper center')

    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.9)

    hist_file = os.path.join(
        plot_path,
        f'{timestamp}_{amp}_{model_type}_network_output_normalized_{prefix}_{lr_str}{domain_suffix}_{dataset_name_suffix}.png'
    )
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
                      dataset_name_suffix="main", extra_datasets=None):
    """Generate a t-SNE latent space visualization with optional overlays."""

    amp = config['amp']
    prefix = config['prefix']
    lr_str = f"lr{learning_rate:.0e}".replace('-', '')
    plot_path = config['base_plot_path']
    domain_label = config.get('domain_label', 'time')
    domain_suffix = config.get('domain_suffix', '')
    os.makedirs(plot_path, exist_ok=True)

    extra_datasets = extra_datasets or []

    # --- 1. Create Encoder-only model ---
    try:
        latent_layer = model.get_layer('latent_space')
    except ValueError:
        print("\n--- LATENT SPACE PLOTTER WARNING ---")
        print("Layer 'latent_space' not found; searching for smallest layer as fallback.")
        try:
            candidates = [l for l in model.layers if 'input' not in l.name.lower()]
            latent_layer = min(candidates, key=lambda l: np.prod(l.output_shape[1:]))
            print(f"Using layer '{latent_layer.name}' as latent space (shape: {latent_layer.output_shape})")
        except Exception as exc:
            print(f"Could not determine latent layer ({exc}). Skipping latent space plot for {dataset_name_suffix}.")
            return

    encoder = Model(inputs=model.input, outputs=latent_layer.output)

    # --- 2. Collect datasets ---
    dataset_entries = []
    n_samples = min(2000, len(sim_rcr_all), len(data_backlobe_traces_rcr_all))
    if n_samples == 0:
        print(f"Skipping latent space plot for {dataset_name_suffix}: not enough RCR/Backlobe data.")
        return

    print(f"Subsampling {n_samples} events from RCR and Backlobe for latent space plot.")
    rcr_indices = np.random.choice(len(sim_rcr_all), n_samples, replace=False)
    bl_indices = np.random.choice(len(data_backlobe_traces_rcr_all), n_samples, replace=False)

    dataset_entries.append({
        'label': 'RCR (Signal)',
        'data': sim_rcr_all[rcr_indices],
        'color': LATENT_COLOR_MAP.get('RCR (Signal)', 'red'),
        'marker': LATENT_MARKER_MAP.get('RCR (Signal)', 'o'),
        'alpha': 0.5
    })

    dataset_entries.append({
        'label': 'Backlobe (BG)',
        'data': data_backlobe_traces_rcr_all[bl_indices],
        'color': LATENT_COLOR_MAP.get('Backlobe (BG)', 'blue'),
        'marker': LATENT_MARKER_MAP.get('Backlobe (BG)', 'o'),
        'alpha': 0.5
    })

    for extra in extra_datasets:
        label = extra.get('label', 'Validation')
        data_array = np.asarray(extra.get('data', []))
        if data_array.size == 0 or data_array.ndim != 3:
            print(f"Skipping latent overlay '{label}': data missing or wrong shape {data_array.shape}.")
            continue

        max_samples = extra.get('max_samples', min(2000, data_array.shape[0]))
        sample_count = min(max_samples, data_array.shape[0])
        if sample_count == 0:
            print(f"Skipping latent overlay '{label}': no samples available.")
            continue

        if sample_count < data_array.shape[0]:
            indices = np.random.choice(data_array.shape[0], sample_count, replace=False)
            data_subset = data_array[indices]
        else:
            data_subset = data_array

        color = extra.get('color')
        if color is None:
            color = LATENT_COLOR_MAP.get(label, 'gray')

        marker = extra.get('marker', LATENT_MARKER_MAP.get(label, 'o'))

        dataset_entries.append({
            'label': label,
            'data': data_subset,
            'color': color,
            'marker': marker,
            'alpha': extra.get('alpha', 0.8),
            'size': extra.get('size', 60)
        })

    # --- 3. Prepare inputs for encoder ---
    combined_batches = []
    for entry in dataset_entries:
        try:
            prepared = prepare_data_for_model(entry['data'], requires_transpose)
        except ValueError as exc:
            print(f"Skipping dataset '{entry['label']}' in latent plotting: {exc}")
            entry['prepared'] = None
            continue

        if prepared.size == 0 or prepared.shape[0] == 0:
            print(f"Skipping dataset '{entry['label']}' in latent plotting: empty after preparation.")
            entry['prepared'] = None
            continue

        entry['prepared'] = prepared
        combined_batches.append(prepared)

    if not combined_batches:
        print(f"No datasets available for latent space plot ({dataset_name_suffix}).")
        return

    combined_data = np.concatenate(combined_batches, axis=0)

    # --- 4. Get latent vectors ---
    print(f"Running encoder for latent space ({combined_data.shape[0]} samples total)...")
    latent_vectors = encoder.predict(combined_data)
    if latent_vectors.ndim > 2:
        latent_vectors = latent_vectors.reshape(latent_vectors.shape[0], -1)

    print(f"Latent vector shape: {latent_vectors.shape}")

    # --- 5. Run t-SNE ---
    print("Running t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(latent_vectors)

    # --- 6. Plot ---
    print("Plotting t-SNE projection with validation overlays...")
    plt.figure(figsize=(10, 8))

    start_idx = 0
    for entry in dataset_entries:
        prepared = entry.get('prepared')
        if prepared is None:
            continue

        end_idx = start_idx + prepared.shape[0]
        points = tsne_results[start_idx:end_idx]
        start_idx = end_idx

        color = entry.get('color') or LATENT_COLOR_MAP.get(entry['label'], 'gray')
        marker = entry.get('marker', 'o')
        alpha = entry.get('alpha', 0.6)
        size = entry.get('size', 40)
        if entry['label'].startswith('Validation Special Event'):
            size = 120
            alpha = entry.get('alpha', 1.0)

        plt.scatter(points[:, 0], points[:, 1],
                    label=f"{entry['label']} ({prepared.shape[0]})",
                    color=color,
                    marker=marker,
                    alpha=alpha,
                    s=size,
                    edgecolors='none' if marker != '*' else 'k')

    plt.legend(loc='upper right')
    plt.title(f't-SNE Latent Space Visualization ({domain_label}, {dataset_name_suffix} set)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    plot_file_tsne = os.path.join(
        plot_path,
        f'{timestamp}_{amp}_{model_type}_latent_space_tSNE_{prefix}_{lr_str}{domain_suffix}_{dataset_name_suffix}.png'
    )
    plt.savefig(plot_file_tsne)
    plt.close()
    print(f'Saved t-SNE plot to: {plot_file_tsne}')


def plot_validation_loss_histogram(loss_entries, config, timestamp, learning_rate, model_type,
                                   normalized=False, dataset_name_suffix="validation", cut_value=0.9):
    """Plot reconstruction loss histograms for multiple datasets (raw or normalized)."""

    amp = config['amp']
    prefix = config['prefix']
    domain_label = config.get('domain_label', 'time')
    domain_suffix = config.get('domain_suffix', '')
    lr_str = f"lr{learning_rate:.0e}".replace('-', '')
    plot_path = config['base_plot_path']
    os.makedirs(plot_path, exist_ok=True)

    valid_entries = []
    for label, values in loss_entries:
        if values is None:
            continue
        values_array = np.asarray(values).flatten()
        if values_array.size == 0:
            continue
        values_array = values_array[np.isfinite(values_array)]
        if values_array.size == 0:
            continue
        valid_entries.append((label, values_array))

    if not valid_entries:
        mode = "normalized" if normalized else "raw"
        print(f"Skipping validation loss histogram ({mode}): no data available.")
        return

    if normalized:
        combined = np.concatenate([vals for _, vals in valid_entries])
        min_val = float(np.min(combined))
        max_val = float(np.max(combined))

        if np.isclose(max_val, min_val):
            normalized_entries = [(label, np.zeros_like(vals)) for label, vals in valid_entries]
        else:
            denom = max_val - min_val
            normalized_entries = [
                (label, np.clip((vals - min_val) / denom, 0.0, 1.0))
                for label, vals in valid_entries
            ]

        plot_entries = normalized_entries
        bins = 50
        range_vals = (0.0, 1.0)
        xlabel = 'Normalized Reconstruction Loss'
        filename_suffix = 'normalized'
    else:
        plot_entries = valid_entries
        combined = np.concatenate([vals for _, vals in plot_entries])
        min_val = float(np.min(combined))
        upper_percentile = max(float(np.percentile(vals, 99.9)) for _, vals in plot_entries)
        upper_bound = upper_percentile * 1.1 if upper_percentile > 0 else upper_percentile + 1e-6
        bins = 50
        range_vals = (min_val, upper_bound)
        xlabel = 'Reconstruction Loss (MSE)'
        filename_suffix = 'raw'

    plt.figure(figsize=(8, 6))
    max_hist = 0.0
    for label, vals in plot_entries:
        base_label = label.split(' (')[0]
        color = HIST_COLOR_MAP.get(label, HIST_COLOR_MAP.get(base_label, None))
        counts, _, _ = plt.hist(
            vals,
            bins=bins,
            range=range_vals,
            histtype='step',
            linewidth=1.6,
            color=color,
            label=f'{label} ({len(vals)})'
        )
        if counts.size:
            max_hist = max(max_hist, counts.max())

    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel('Count', fontsize=18)
    title_bits = 'Normalized ' if normalized else ''
    plt.title(f'{amp}_{domain_label} {model_type} Validation {title_bits}Loss (LR={learning_rate:.0e})', fontsize=14)
    plt.yscale('log')

    if max_hist > 0:
        upper = max(10 ** (np.ceil(np.log10(max_hist * 1.1))), 10)
        plt.ylim(7e-1, upper)

    if normalized:
        plt.axvline(cut_value, color='y', linestyle='--', linewidth=1.2, label=f'Cut = {cut_value:.2f}')
    else:
        plt.axvline(cut_value, color='y', linestyle='--', linewidth=1.2, label=f'Cut = {cut_value:.3g}')

    plt.legend(loc='upper right', fontsize=10)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    filename = (
        f'{timestamp}_{amp}_{model_type}_validation_reconstruction_loss_{filename_suffix}_{prefix}_{lr_str}'
        f'{domain_suffix}_{dataset_name_suffix}.png'
    )
    out_path = os.path.join(plot_path, filename)
    plt.savefig(out_path)
    plt.close()
    print(f'Saved validation {filename_suffix} loss histogram to: {out_path}')


def plot_special_event_reconstruction(original_trace, reconstructed_trace, mse_value, timestamp,
                                      config, model_type, learning_rate, dataset_name_suffix,
                                      event_label):
    """Plot original vs reconstructed traces for a single special event."""

    if original_trace is None or reconstructed_trace is None:
        print("No special event traces provided; skipping special reconstruction plot.")
        return

    original = np.asarray(original_trace)
    reconstructed = np.asarray(reconstructed_trace)

    if original.ndim != 2 or reconstructed.ndim != 2:
        print(f"Skipping special reconstruction plot: expected 2D traces, got {original.shape} & {reconstructed.shape}.")
        return

    if original.shape != reconstructed.shape:
        print(f"Skipping special reconstruction plot: shape mismatch {original.shape} vs {reconstructed.shape}.")
        return

    amp = config['amp']
    prefix = config['prefix']
    domain_suffix = config.get('domain_suffix', '')
    domain_label = config.get('domain_label', 'time')
    lr_str = f"lr{learning_rate:.0e}".replace('-', '')
    plot_path = config['base_plot_path']
    os.makedirs(plot_path, exist_ok=True)

    num_channels, num_samples = original.shape
    fig, axes = plt.subplots(num_channels, 2, figsize=(15, 3 * num_channels), sharex=True, sharey=True)
    if num_channels == 1:
        axes = np.expand_dims(axes, axis=0)

    title_label = event_label or 'Validation Special Event'
    fig.suptitle(
        f'{amp} {model_type} - {title_label}\nReconstruction MSE: {mse_value:.4g} ({domain_label}, {dataset_name_suffix})',
        fontsize=16
    )

    for idx in range(num_channels):
        axes[idx, 0].plot(original[idx, :], color='orange')
        axes[idx, 0].set_ylabel(f'Channel {idx}')
        axes[idx, 1].plot(reconstructed[idx, :], color='black')
        if idx == 0:
            axes[idx, 0].set_title('Original Trace')
            axes[idx, 1].set_title('Reconstructed Trace')

    axes[-1, 0].set_xlabel('Sample')
    axes[-1, 1].set_xlabel('Sample')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename_label = _sanitize_label_for_filename(title_label)
    filename = (
        f'{timestamp}_{amp}_{model_type}_recon_{filename_label}_{prefix}_{lr_str}{domain_suffix}_{dataset_name_suffix}.png'
    )
    out_path = os.path.join(plot_path, filename)
    plt.savefig(out_path)
    plt.close(fig)
    print(f'Saved validation special event reconstruction plot to: {out_path}')


def run_validation_checks(model, requires_transpose, config, timestamp, learning_rate, model_type,
                          sim_rcr_all, data_backlobe_traces_rcr_all,
                          prob_rcr_main, prob_backlobe_main):
    """Run validation-only analysis using the configured coincidence PKL dataset."""

    print("\n--- Running validation checks with coincidence PKL dataset ---")

    validation_pkl_path = config.get('validation_pkl_path', DEFAULT_VALIDATION_PKL_PATH)
    if not validation_pkl_path:
        print("Validation PKL path not provided; skipping validation checks.")
        return

    if not os.path.exists(validation_pkl_path):
        print(f"Validation PKL not found at {validation_pkl_path}; skipping validation checks.")
        return

    passing_event_ids = config.get('validation_passing_event_ids', DEFAULT_VALIDATION_PASSING_EVENT_IDS)
    special_event_id = config.get('validation_special_event_id', DEFAULT_VALIDATION_SPECIAL_EVENT_ID)
    special_station_id = config.get('validation_special_station_id', DEFAULT_VALIDATION_SPECIAL_STATION_ID)

    passing_traces, raw_traces, special_traces, passing_metadata, raw_metadata, special_metadata = (
        load_new_coincidence_data(validation_pkl_path, passing_event_ids, special_event_id, special_station_id)
    )

    reference_channels = sim_rcr_all.shape[1] if sim_rcr_all.ndim == 3 else 4
    reference_samples = sim_rcr_all.shape[2] if sim_rcr_all.ndim == 3 else 256

    def coerce_traces(traces):
        arr = np.asarray(traces)
        if arr.size == 0:
            return np.empty((0, reference_channels, reference_samples), dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        if arr.ndim != 3:
            raise ValueError(f"Expected traces with 3 dims, got shape {arr.shape}")
        return arr.astype(np.float32, copy=False)

    try:
        passing_traces = coerce_traces(passing_traces)
        raw_traces = coerce_traces(raw_traces)
        special_traces = coerce_traces(special_traces)
    except ValueError as exc:
        print(f"Skipping validation checks: {exc}")
        return

    is_freq_model = bool(config.get('is_freq_model', False))
    sampling_rate = float(config.get('frequency_sampling_rate', 2.0))
    use_filtering = bool(config.get('use_filtering', False))

    def transform(traces):
        if traces.shape[0] == 0:
            return traces
        transformed = traces.copy()
        if is_freq_model:
            transformed = _compute_frequency_magnitude(transformed, sampling_rate)
            if use_filtering:
                transformed = _apply_frequency_edge_filter(transformed)
        return transformed.astype(np.float32, copy=False)

    passing_traces_proc = transform(passing_traces)
    raw_traces_proc = transform(raw_traces)
    special_traces_proc = transform(special_traces)

    special_label = 'Validation Special Event'
    if special_traces_proc.shape[0] > 0:
        meta = special_metadata.get(0, {})
        evt = meta.get('event_id')
        stn = meta.get('station_id')
        if evt is not None and stn is not None:
            special_label = f'Validation Special Event (Evt {evt}, Stn {stn})'

    # --- Reconstruction losses ---
    passing_mse = np.array([])
    raw_mse = np.array([])
    special_mse = np.array([])
    special_trace_for_plot = None
    special_recon_for_plot = None

    if passing_traces_proc.shape[0] > 0:
        prepared = prepare_data_for_model(passing_traces_proc, requires_transpose)
        passing_mse, _ = calculate_reconstruction_mse(model, prepared, config)
        print(f"Validation passing set size: {len(passing_mse)}, mean MSE: {np.mean(passing_mse):.4g}")

    if raw_traces_proc.shape[0] > 0:
        prepared = prepare_data_for_model(raw_traces_proc, requires_transpose)
        raw_mse, _ = calculate_reconstruction_mse(model, prepared, config)
        print(f"Validation raw set size: {len(raw_mse)}, mean MSE: {np.mean(raw_mse):.4g}")

    if special_traces_proc.shape[0] > 0:
        prepared = prepare_data_for_model(special_traces_proc, requires_transpose)
        special_mse, special_recon = calculate_reconstruction_mse(model, prepared, config)
        special_trace_for_plot = special_traces_proc[0]
        recon_first = special_recon[0]
        if requires_transpose:
            special_recon_for_plot = recon_first.transpose(1, 0)
        else:
            special_recon_for_plot = np.squeeze(recon_first, axis=-1)
        print(f"Validation special event MSE values: {special_mse}")

    # --- Latent space overlay ---
    validation_overlays = []
    if passing_traces_proc.shape[0] > 0:
        validation_overlays.append({
            'label': 'Validation Passing',
            'data': passing_traces_proc,
            'color': LATENT_COLOR_MAP.get('Validation Passing', 'green'),
            'marker': LATENT_MARKER_MAP.get('Validation Passing', 's'),
            'alpha': 0.8,
            'size': 60
        })
    if raw_traces_proc.shape[0] > 0:
        validation_overlays.append({
            'label': 'Validation Raw',
            'data': raw_traces_proc,
            'color': LATENT_COLOR_MAP.get('Validation Raw', 'purple'),
            'marker': LATENT_MARKER_MAP.get('Validation Raw', '^'),
            'alpha': 0.8,
            'size': 60
        })
    if special_traces_proc.shape[0] > 0:
        validation_overlays.append({
            'label': special_label,
            'data': special_traces_proc,
            'color': LATENT_COLOR_MAP.get('Validation Special Event', 'orange'),
            'marker': LATENT_MARKER_MAP.get('Validation Special Event', '*'),
            'alpha': 1.0,
            'size': 160
        })

    if validation_overlays:
        plot_latent_space(
            model,
            sim_rcr_all,
            data_backlobe_traces_rcr_all,
            requires_transpose,
            timestamp,
            config,
            model_type,
            learning_rate,
            dataset_name_suffix="validation",
            extra_datasets=validation_overlays
        )
    else:
        print("No validation traces available for latent space overlay.")

    # --- Loss histograms ---
    loss_entries = [
        ('RCR (Signal)', prob_rcr_main),
        ('Backlobe (BG)', prob_backlobe_main),
        ('Validation Passing', passing_mse),
        ('Validation Raw', raw_mse),
        (special_label, special_mse)
    ]

    output_cut_value = config.get('output_cut_value', 0.9)
    plot_validation_loss_histogram(
        loss_entries,
        config,
        timestamp,
        learning_rate,
        model_type,
        normalized=False,
        dataset_name_suffix="validation",
        cut_value=output_cut_value
    )

    plot_validation_loss_histogram(
        loss_entries,
        config,
        timestamp,
        learning_rate,
        model_type,
        normalized=True,
        dataset_name_suffix="validation",
        cut_value=0.9
    )

    # --- Special event reconstruction plot ---
    if special_trace_for_plot is not None and special_recon_for_plot is not None and special_mse.size > 0:
        plot_special_event_reconstruction(
            special_trace_for_plot,
            special_recon_for_plot,
            float(special_mse.flatten()[0]),
            timestamp,
            config,
            model_type,
            learning_rate,
            dataset_name_suffix="validation",
            event_label=special_label
        )
    else:
        print("No special event reconstruction plot generated (missing data).")

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

    plot_normalized_network_output_histogram_autoencoder(
        prob_rcr,
        prob_backlobe,
        config,
        timestamp,
        learning_rate,
        model_type,
        dataset_name_suffix="main",
        cut_value=0.9
    )

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

    # --- Validation-only checks ---
    run_validation_checks(
        model,
        requires_transpose,
        config,
        timestamp,
        learning_rate,
        model_type,
        sim_rcr_all,
        data_backlobe_traces_rcr_all,
        prob_rcr,
        prob_backlobe
    )

    print(f"\nScript finished successfully for Autoencoder model {model_type}, lr {learning_rate}")

if __name__ == "__main__":
    main()
