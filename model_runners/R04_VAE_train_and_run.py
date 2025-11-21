"""
Trains a 1D Variational Autoencoder (VAE) for anomaly detection.

1. Trains only on real background data (Backlobe).
2. The model learns to reconstruct "normal" background behaviour.
3. Evaluates by measuring reconstruction error (MSE).
   - Signal (RCR) should have HIGH reconstruction error.
   - Background (Backlobe) should have LOW reconstruction error.
"""

import os
import sys
import argparse
from datetime import datetime
import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from tensorflow import keras
from tensorflow.keras.models import Model
from sklearn.manifold import TSNE
from pathlib import Path
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

import umap.umap_ as umap
import hdbscan

# --- Local Imports from project structure ---
sys.path.append(str(Path(__file__).resolve().parents[1] / '200s_time'))
from A0_Utilities import load_config

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_builder_VAE import (
    build_vae_model_freq,
    build_vae_bottleneck_model_freq,
    build_vae_denoising_model_freq,
    build_vae_mae_loss_model_freq,
    build_vae_custom_loss_model_freq_samplewise,
    build_vae_model_freq_2d_input,
    KLAnnealingCallback,
    KLCyclicalAnnealingCallback,
    CyclicalLRCallback,
    build_vae_independent_channel_freq
)
from model_builder_VAE_time import (
    build_vae_model_time,
    build_vae_bottleneck_model_time,
    build_vae_denoising_model_time,
    build_vae_mae_loss_model_time,
    build_vae_custom_loss_model_time_samplewise,
    build_vae_model_time_2d_input,
)
from model_builder_scattering_VAE import (
    build_vae_model_scattering
)
from data_channel_cycling import cycle_channels
from R01_1D_CNN_train_and_run import (
    load_and_prep_data_for_training,
    save_and_plot_training_history,
    load_new_coincidence_data,
    load_combined_backlobe_data,
    _compute_frequency_magnitude,
    _apply_frequency_edge_filter,
    convert_to_db_scale,
)

MODEL_BUILDERS = {
    '1d_vae_freq': build_vae_model_freq,
    '1d_vae_bottleneck_freq': build_vae_bottleneck_model_freq,
    '1d_vae_denoising_freq': build_vae_denoising_model_freq,
    '1d_vae_mae_loss_freq': build_vae_mae_loss_model_freq,
    '1d_vae_custom_loss_freq': build_vae_custom_loss_model_freq_samplewise,
    '2d_vae_freq': build_vae_model_freq_2d_input,
    '1d_vae': build_vae_model_time,
    '1d_vae_bottleneck': build_vae_bottleneck_model_time,
    '1d_vae_denoising': build_vae_denoising_model_time,
    '1d_vae_mae_loss': build_vae_mae_loss_model_time,
    '1d_vae_custom_loss': build_vae_custom_loss_model_time_samplewise,
    '2d_vae': build_vae_model_time_2d_input,
    '1d_vae_scattering': build_vae_model_scattering,
    '1d_vae_independent_channel_freq': build_vae_independent_channel_freq
}

DEFAULT_VALIDATION_PKL_PATH = (
    "/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/"
    "9.24.25_CoincidenceDatetimes_passing_cuts_with_all_params_recalcZenAzi_calcPol.pkl"
)
DEFAULT_VALIDATION_PASSING_EVENT_IDS = [
    3047,
    3432,
    10195,
    10231,
    10273,
    10284,
    10444,
    10449,
    10466,
    10471,
    10554,
    11197,
    11220,
    11230,
    11236,
    11243,
]
DEFAULT_VALIDATION_SPECIAL_EVENT_ID = 11230
DEFAULT_VALIDATION_SPECIAL_STATION_ID = 13

DEFAULT_ABOVE_CURVE_PKL_PATH = (
    "/dfs8/sbarwick_lab/ariannaproject/tangch3/station_data/above_curve_data/"
    "5000evt_10.17.25/above_curve_combined.pkl"
)

LATENT_COLOR_MAP = {
    'Backlobe (BG)': 'blue',
    'RCR (Signal)': 'red',
    'Validation Passing': 'green',
    'Validation Raw': 'purple',
    'Validation Special Event': 'orange',
    'Above Curve': 'cyan',
}

LATENT_MARKER_MAP = {
    'Backlobe (BG)': 'o',
    'RCR (Signal)': 'o',
    'Validation Passing': 's',
    'Validation Raw': '^',
    'Validation Special Event': '*',
    'Above Curve': 'd',
}

HIST_COLOR_MAP = {
    'Backlobe (BG)': 'blue',
    'RCR (Signal)': 'red',
    'Validation Passing': 'green',
    'Validation Raw': 'purple',
    'Validation Special Event': 'orange',
    'Above Curve': 'cyan',
}


def _sanitize_label_for_filename(label):
    """Return a filesystem-friendly string derived from a label."""

    safe_chars = set("abcdefghijklmnopqrstuvwxyz0123456789-_")
    sanitized = ''.join(
        ch_lower if (ch_lower := ch.lower()) in safe_chars else '_' for ch in label
    )
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

    return data_array[..., np.newaxis]


def calculate_reconstruction_mse(model, prepared_data, config):
    """Run model inference and compute per-event reconstruction MSE."""

    if prepared_data.size == 0 or prepared_data.shape[0] == 0:
        return np.array([]), np.empty_like(prepared_data)

    reconstructed = model.predict(prepared_data, batch_size=config['keras_batch_size'])
    axis = tuple(range(1, reconstructed.ndim))
    mse = np.mean(np.square(reconstructed - prepared_data), axis=axis)
    return mse, reconstructed


def _run_hdbscan_with_retries(latent_vectors, config, dataset_label='Above Curve'):
    """Cluster latent vectors with HDBSCAN, retrying with varied parameters until a cluster forms."""

    sample_count = latent_vectors.shape[0]
    if sample_count == 0:
        return np.empty((0,), dtype=int), None, {'reason': 'empty_dataset'}

    if hdbscan is None:
        print(
            f"hdbscan package not available; assigning a single fallback cluster for {dataset_label}."
        )
        return np.zeros(sample_count, dtype=int), None, {'reason': 'missing_package'}

    default_min_cluster_size = int(config.get('hdbscan_min_cluster_size', 4))
    default_min_cluster_size = max(2, default_min_cluster_size)
    default_min_cluster_size = min(default_min_cluster_size, max(2, sample_count - 1))

    configured_min_samples = config.get('hdbscan_min_samples', None)
    if configured_min_samples is not None:
        configured_min_samples = max(4, int(configured_min_samples))

    candidate_sizes = set()
    upscale = [1.0, 1.5, 2.0, 3.0, 4.0]
    downscale = [0.75, 0.5, 0.33, 0.25]
    for factor in upscale + downscale:
        candidate_sizes.add(int(round(default_min_cluster_size * factor)))
    candidate_sizes.update(
        [
            default_min_cluster_size,
            max(2, sample_count // 2),
            max(2, sample_count // 3),
            max(2, sample_count // 4),
            max(2, sample_count // 5),
            max(2, int(np.sqrt(sample_count))),
            sample_count - 1,
        ]
    )

    sanitized_sizes = []
    for size in candidate_sizes:
        size = max(2, size)
        size = min(size, max(2, sample_count - 1))
        sanitized_sizes.append(size)
    candidate_sizes = sorted(set(sanitized_sizes), reverse=True)

    tried = set()
    for min_cluster_size in candidate_sizes:
        min_samples_options = []
        if configured_min_samples is not None:
            min_samples_options.append(min(configured_min_samples, min_cluster_size))
        min_samples_options.extend(
            [
                None,
                max(1, min_cluster_size // 2),
                max(1, int(np.sqrt(min_cluster_size))),
                1,
            ]
        )

        seen_ms = set()
        ordered_min_samples = []
        for option in min_samples_options:
            if option is not None and option > min_cluster_size:
                option = min_cluster_size
            key = option if option is not None else 'auto'
            if key in seen_ms:
                continue
            seen_ms.add(key)
            ordered_min_samples.append(option)

        for min_samples in ordered_min_samples:
            attempt_key = (min_cluster_size, min_samples if min_samples is not None else 'auto')
            if attempt_key in tried:
                continue
            tried.add(attempt_key)

            kwargs = {'min_cluster_size': min_cluster_size}
            if min_samples is not None:
                kwargs['min_samples'] = min_samples

            try:
                clusterer = hdbscan.HDBSCAN(**kwargs)
                labels = clusterer.fit_predict(latent_vectors)
                clusters_found = int(np.sum(labels >= 0))
                noise_points = int(np.sum(labels < 0))

                if clusters_found > 0:
                    print(
                        f"HDBSCAN clustering succeeded with min_cluster_size={min_cluster_size},"
                        f" min_samples={kwargs.get('min_samples', 'auto')}"
                        f" (clusters={clusters_found}, noise={noise_points})."
                    )
                    return labels, clusterer, kwargs

                print(
                    f"HDBSCAN attempt produced only noise (min_cluster_size={min_cluster_size},"
                    f" min_samples={kwargs.get('min_samples', 'auto')})."
                )
            except Exception as exc:
                print(
                    f"HDBSCAN attempt failed (min_cluster_size={min_cluster_size},"
                    f" min_samples={kwargs.get('min_samples', 'auto')}): {exc}"
                )

    print(
        f"HDBSCAN could not identify clusters for {dataset_label} after {len(tried)} attempts;"
        " assigning a single fallback cluster."
    )
    return np.zeros(sample_count, dtype=int), None, {'reason': 'fallback'}


def train_vae_model(training_backlobe, config, learning_rate, model_type):
    """Train the selected VAE using only background data."""

    if model_type not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model_type: {model_type}.")

    model_builder = MODEL_BUILDERS[model_type]
    is_freq_model = model_type.endswith('_freq')
    input_seq_len = 129 if is_freq_model else 256

    x_train = training_backlobe
    y_train = x_train

    model, requires_transpose = model_builder(
        input_shape=(input_seq_len, 4),
        learning_rate=learning_rate,
    )

    if requires_transpose:
        x_train = x_train.transpose(0, 2, 1)
        y_train = y_train.transpose(0, 2, 1)
    elif x_train.ndim == 3:
        x_train = np.expand_dims(x_train, axis=-1)
        y_train = np.expand_dims(y_train, axis=-1)

    # callbacks_list = [
    #     keras.callbacks.EarlyStopping(
    #         monitor='val_loss',
    #         patience=config['early_stopping_patience'],
    #     )
    # ]
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=25,
        verbose=1,
        min_lr=1e-7
    )

    early_stopper = EarlyStopping(
        monitor='val_loss',
        patience=config['early_stopping_patience'],
        restore_best_weights=False,
    )

    # Requires rewriting my VAE model, so commenting out for now
    # model_checkpoint = ModelCheckpoint(
    #     'best_model.keras',
    #     monitor='val_loss',
    #     save_best_only=True
    # )

    # KL Annealing Callback
    # kl_annealing_callback = KLAnnealingCallback(
    #     kl_weight_target=2.0,   # Final weight
    #     kl_anneal_epochs=50,    # Number of epochs to reach target weight
    #     kl_warmup_epochs=20     # Number of epochs to wait before starting annealing
    # )
    WARMUP_EPOCHS = 20
    CYCLE_LENGTH = 50
    RAMP_FRACTION = 0.5

    kl_cyclical_callback = KLCyclicalAnnealingCallback(
        kl_weight_target=2.0,   # Peak weight (beta)
        cycle_length_epochs=CYCLE_LENGTH, # Number of epochs for full cycle
        kl_warmup_epochs=WARMUP_EPOCHS,    # Number of epochs to wait at 0
        ramp_up_fraction=RAMP_FRACTION    # % of cycle to ramp up, rest at peak
    )
    # lr_cyclical_callback = CyclicalLRCallback(
    #     max_lr=learning_rate,
    #     min_lr=max(learning_rate*0.01, 1e-7),
    #     cycle_length_epochs=CYCLE_LENGTH,
    #     kl_warmup_epochs=WARMUP_EPOCHS,
    #     ramp_up_fraction=RAMP_FRACTION
    # )


    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=config['keras_epochs'],
        batch_size=config['keras_batch_size'],
        verbose=config['verbose_fit'],
        callbacks=[lr_scheduler, kl_cyclical_callback, early_stopper],
        # callbacks=callbacks_list,
    )

    return model, history, requires_transpose


def evaluate_model_performance_vae(
    model,
    sim_rcr_all,
    data_backlobe_traces_rcr_all,
    output_cut_value,
    config,
    model_type,
    requires_transpose,
):
    """Evaluate the VAE using reconstruction error as anomaly score."""

    if requires_transpose:
        sim_rcr_prepped = sim_rcr_all.transpose(0, 2, 1)
        data_backlobe_prepped = data_backlobe_traces_rcr_all.transpose(0, 2, 1)
    else:
        sim_rcr_prepped = (
            sim_rcr_all[..., np.newaxis] if sim_rcr_all.ndim == 3 else sim_rcr_all
        )
        data_backlobe_prepped = (
            data_backlobe_traces_rcr_all[..., np.newaxis]
            if data_backlobe_traces_rcr_all.ndim == 3
            else data_backlobe_traces_rcr_all
        )

    print(
        f"Predicting on RCR (signal) set of shape {sim_rcr_prepped.shape}..."
    )
    reconstructed_rcr = model.predict(
        sim_rcr_prepped, batch_size=config['keras_batch_size']
    )
    print(
        f"Predicting on Backlobe (background) set of shape {data_backlobe_prepped.shape}..."
    )
    reconstructed_backlobe = model.predict(
        data_backlobe_prepped, batch_size=config['keras_batch_size']
    )

    axis_to_average = tuple(range(1, sim_rcr_prepped.ndim))
    prob_rcr = np.mean(
        np.square(reconstructed_rcr - sim_rcr_prepped), axis=axis_to_average
    )
    prob_backlobe = np.mean(
        np.square(reconstructed_backlobe - data_backlobe_prepped),
        axis=axis_to_average,
    )

    print(f"Mean RCR (Signal) Reconstruction MSE: {np.mean(prob_rcr):.4g}")
    print(f"Mean Backlobe (BG) Reconstruction MSE: {np.mean(prob_backlobe):.4g}")

    rcr_efficiency = (np.sum(prob_rcr > output_cut_value) / len(prob_rcr)) * 100
    backlobe_efficiency = (
        np.sum(prob_backlobe > output_cut_value) / len(prob_backlobe)
    ) * 100

    print(f'RCR efficiency (Signal > Cut): {rcr_efficiency:.2f}%')
    print(f'Backlobe efficiency (BG > Cut): {backlobe_efficiency:.4f}%')
    print(f'Lengths: RCR {len(prob_rcr)}, Backlobe {len(prob_backlobe)}')

    return prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency


def plot_network_output_histogram_vae(
    prob_rcr,
    prob_backlobe,
    rcr_efficiency,
    backlobe_efficiency,
    config,
    timestamp,
    learning_rate,
    model_type,
    dataset_name_suffix="main",
):
    """Plot reconstruction MSE distributions for RCR and Backlobe events."""

    amp = config['amp']
    prefix = config['prefix']
    lr_str = f"lr{learning_rate:.0e}".replace('-', '')
    plot_path = config['base_plot_path']
    domain_label = config.get('domain_label', 'time')
    domain_suffix = config.get('domain_suffix', '')
    os.makedirs(plot_path, exist_ok=True)

    if len(prob_rcr) == 0 or len(prob_backlobe) == 0:
        print(
            f"Skipping network output plot for {dataset_name_suffix}: missing data."
        )
        return

    plt.figure(figsize=(8, 6))

    max_val = max(
        np.percentile(prob_rcr, 99), np.percentile(prob_backlobe, 99.9)
    )
    min_val = min(np.min(prob_rcr), np.min(prob_backlobe))
    upper_bound = max_val * 1.1 if max_val > 0 else max_val + 1e-6
    plot_range = (min_val, upper_bound)

    plt.hist(
        prob_backlobe,
        bins=50,
        range=plot_range,
        histtype='step',
        color='blue',
        linestyle='solid',
        linewidth=1.5,
        label=f'Backlobe {len(prob_backlobe)}',
    )
    plt.hist(
        prob_rcr,
        bins=50,
        range=plot_range,
        histtype='step',
        color='red',
        linestyle='solid',
        linewidth=1.5,
        label=f'RCR {len(prob_rcr)}',
    )
    plt.yscale('log')

    plt.xlabel('Reconstruction Loss (MSE)', fontsize=18)
    plt.ylabel('Count', fontsize=18)
    title_suffix = f"({dataset_name_suffix} set)"
    plt.title(
        f'{amp}_{domain_label} {model_type} Loss {title_suffix} (LR={learning_rate:.0e})',
        fontsize=14,
    )
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper right', fontsize=12)

    ax = plt.gca()
    ax.text(
        0.45,
        0.75,
        f'RCR efficiency (Signal > Cut): {rcr_efficiency:.2f}%',
        fontsize=12,
        transform=ax.transAxes,
    )
    ax.text(
        0.45,
        0.70,
        f'Backlobe efficiency (BG > Cut): {backlobe_efficiency:.4f}%',
        fontsize=12,
        transform=ax.transAxes,
    )
    ax.text(
        0.45,
        0.60,
        f'LR: {learning_rate:.0e}',
        fontsize=12,
        transform=ax.transAxes,
    )
    ax.text(
        0.45,
        0.55,
        f'Model: {model_type}',
        fontsize=12,
        transform=ax.transAxes,
    )
    plt.legend()

    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.9)

    hist_file = os.path.join(
        plot_path,
        f'{timestamp}_{amp}_{model_type}_network_output_{prefix}_{lr_str}{domain_suffix}_{dataset_name_suffix}.png',
    )
    print(f'saving {hist_file}')
    plt.savefig(hist_file)
    plt.close()


def plot_normalized_network_output_histogram_vae(
    prob_rcr,
    prob_backlobe,
    config,
    timestamp,
    learning_rate,
    model_type,
    dataset_name_suffix="main",
    cut_value=0.9,
):
    """Normalize reconstruction losses to [0, 1] and plot histograms."""

    amp = config['amp']
    prefix = config['prefix']
    domain_label = config.get('domain_label', 'time')
    domain_suffix = config.get('domain_suffix', '')
    lr_str = f"lr{learning_rate:.0e}".replace('-', '')
    plot_path = config['base_plot_path']
    os.makedirs(plot_path, exist_ok=True)

    if len(prob_rcr) == 0 or len(prob_backlobe) == 0:
        print(
            f"Skipping normalized network output plot for {dataset_name_suffix}: missing data."
        )
        return

    combined = np.concatenate([prob_rcr, prob_backlobe])
    min_val = float(np.min(combined))
    max_val = float(np.max(combined))

    if np.isclose(max_val, min_val):
        print(
            f"Warning: network outputs identical for {dataset_name_suffix}; normalized plot will be zero."
        )
        norm_rcr = np.zeros_like(prob_rcr, dtype=float)
        norm_backlobe = np.zeros_like(prob_backlobe, dtype=float)
    else:
        denom = max_val - min_val
        norm_rcr = np.clip((prob_rcr - min_val) / denom, 0.0, 1.0)
        norm_backlobe = np.clip((prob_backlobe - min_val) / denom, 0.0, 1.0)

    rcr_eff_norm = float(np.mean(norm_rcr > cut_value) * 100) if norm_rcr.size else 0.0
    backlobe_eff_norm = (
        float(np.mean(norm_backlobe > cut_value) * 100) if norm_backlobe.size else 0.0
    )

    print(
        f'Normalized RCR efficiency (>{cut_value:.2f}): {rcr_eff_norm:.2f}%'
    )
    print(
        f'Normalized Backlobe efficiency (>{cut_value:.2f}): {backlobe_eff_norm:.4f}%'
    )

    plt.figure(figsize=(8, 6))
    plt.hist(
        norm_backlobe,
        bins=50,
        range=(0.0, 1.0),
        histtype='step',
        color='blue',
        linestyle='solid',
        linewidth=1.5,
        label=f'Backlobe {len(norm_backlobe)}',
    )
    plt.hist(
        norm_rcr,
        bins=50,
        range=(0.0, 1.0),
        histtype='step',
        color='red',
        linestyle='solid',
        linewidth=1.5,
        label=f'RCR {len(norm_rcr)}',
    )

    plt.xlabel('Normalized Reconstruction Loss', fontsize=18)
    plt.ylabel('Count', fontsize=18)
    title_suffix = f"({dataset_name_suffix} set)"
    plt.title(
        f'{amp}_{domain_label} {model_type} Normalized Loss {title_suffix} (LR={learning_rate:.0e})',
        fontsize=14,
    )
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.yscale('log')

    ax = plt.gca()
    ax.text(
        0.45,
        0.75,
        f'RCR eff (> {cut_value:.2f}): {rcr_eff_norm:.2f}%',
        fontsize=12,
        transform=ax.transAxes,
    )
    ax.text(
        0.45,
        0.70,
        f'Backlobe eff (> {cut_value:.2f}): {backlobe_eff_norm:.4f}%',
        fontsize=12,
        transform=ax.transAxes,
    )
    ax.text(
        0.45,
        0.60,
        f'LR: {learning_rate:.0e}',
        fontsize=12,
        transform=ax.transAxes,
    )
    ax.text(
        0.45,
        0.55,
        f'Model: {model_type}',
        fontsize=12,
        transform=ax.transAxes,
    )
    plt.axvline(x=cut_value, color='y', linestyle='--', label=f'Cut = {cut_value:.2f}')
    plt.legend(loc='upper center')

    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.9)

    hist_file = os.path.join(
        plot_path,
        f'{timestamp}_{amp}_{model_type}_network_output_normalized_{prefix}_{lr_str}{domain_suffix}_{dataset_name_suffix}.png',
    )
    print(f'saving {hist_file}')
    plt.savefig(hist_file)
    plt.close()


def plot_original_vs_reconstructed(
    model,
    sim_rcr_all,
    data_backlobe_traces_rcr_all,
    prob_rcr,
    prob_backlobe,
    requires_transpose,
    timestamp,
    config,
    model_type,
    learning_rate,
    dataset_name_suffix="main",
):
    """Plot original vs reconstructed traces for worst/best examples."""

    amp = config['amp']
    prefix = config['prefix']
    lr_str = f"lr{learning_rate:.0e}".replace('-', '')
    plot_path = config['base_plot_path']
    domain_label = config.get('domain_label', 'time')
    domain_suffix = config.get('domain_suffix', '')
    os.makedirs(plot_path, exist_ok=True)

    worst_rcr_idx = np.argmax(prob_rcr)
    worst_rcr_mse = prob_rcr[worst_rcr_idx]
    rcr_original = sim_rcr_all[worst_rcr_idx]

    best_rcr_idx = np.argmin(prob_rcr)
    best_rcr_mse = prob_rcr[best_rcr_idx]
    rcr_original_best = sim_rcr_all[best_rcr_idx]

    best_bl_idx = np.argmin(prob_backlobe)
    best_bl_mse = prob_backlobe[best_bl_idx]
    bl_original = data_backlobe_traces_rcr_all[best_bl_idx]

    worst_bl_idx = np.argmax(prob_backlobe)
    worst_bl_mse = prob_backlobe[worst_bl_idx]
    bl_original_worst = data_backlobe_traces_rcr_all[worst_bl_idx]

    worst_rcr_prepped = rcr_original.transpose(1, 0) if requires_transpose else rcr_original
    best_bl_prepped = bl_original.transpose(1, 0) if requires_transpose else bl_original
    best_rcr_prepped = rcr_original_best.transpose(1, 0) if requires_transpose else rcr_original_best
    worst_bl_prepped = bl_original_worst.transpose(1, 0) if requires_transpose else bl_original_worst

    worst_rcr_prepped = worst_rcr_prepped[np.newaxis, ...]
    best_bl_prepped = best_bl_prepped[np.newaxis, ...]
    best_rcr_prepped = best_rcr_prepped[np.newaxis, ...]
    worst_bl_prepped = worst_bl_prepped[np.newaxis, ...]

    worst_rcr_reconstructed = model.predict(worst_rcr_prepped)[0]
    best_bl_reconstructed = model.predict(best_bl_prepped)[0]
    best_rcr_reconstructed = model.predict(best_rcr_prepped)[0]
    worst_bl_reconstructed = model.predict(worst_bl_prepped)[0]

    if requires_transpose:
        worst_rcr_reconstructed = worst_rcr_reconstructed.transpose(1, 0)
        best_bl_reconstructed = best_bl_reconstructed.transpose(1, 0)
        best_rcr_reconstructed = best_rcr_reconstructed.transpose(1, 0)
        worst_bl_reconstructed = worst_bl_reconstructed.transpose(1, 0)

    fig, axes = plt.subplots(4, 2, figsize=(15, 10), sharex=True, sharey=True)
    fig.suptitle(
        f'{amp} {model_type} - WORST Reconstructed RCR (Signal) - MSE: {worst_rcr_mse:.4g}\n(Dataset: {dataset_name_suffix})',
        fontsize=16,
    )

    axes[0, 0].set_title('Original Trace')
    axes[0, 1].set_title('Reconstructed Trace')

    for i in range(4):
        axes[i, 0].plot(rcr_original[i, :], color='red', label=f'Channel {i}')
        axes[i, 0].set_ylabel(f'Channel {i}')
        axes[i, 1].plot(worst_rcr_reconstructed[i, :], color='black', label=f'Recon Chan {i}')

    axes[-1, 0].set_xlabel('Sample')
    axes[-1, 1].set_xlabel('Sample')

    plot_file_rcr = os.path.join(
        plot_path,
        f'{timestamp}_{amp}_{model_type}_recon_worst_RCR_{prefix}_{lr_str}{domain_suffix}_{dataset_name_suffix}.png',
    )
    plt.savefig(plot_file_rcr)
    plt.close(fig)
    print(f'Saved worst RCR reconstruction plot to: {plot_file_rcr}')

    fig, axes = plt.subplots(4, 2, figsize=(15, 10), sharex=True, sharey=True)
    fig.suptitle(
        f'{amp} {model_type} - BEST Reconstructed Backlobe (BG) - MSE: {best_bl_mse:.4g}\n(Dataset: {dataset_name_suffix})',
        fontsize=16,
    )

    axes[0, 0].set_title('Original Trace')
    axes[0, 1].set_title('Reconstructed Trace')

    for i in range(4):
        axes[i, 0].plot(bl_original[i, :], color='blue', label=f'Channel {i}')
        axes[i, 0].set_ylabel(f'Channel {i}')
        axes[i, 1].plot(best_bl_reconstructed[i, :], color='black', label=f'Recon Chan {i}')

    axes[-1, 0].set_xlabel('Sample')
    axes[-1, 1].set_xlabel('Sample')

    plot_file_bl = os.path.join(
        plot_path,
        f'{timestamp}_{amp}_{model_type}_recon_best_Backlobe_{prefix}_{lr_str}{domain_suffix}_{dataset_name_suffix}.png',
    )
    plt.savefig(plot_file_bl)
    plt.close(fig)
    print(f'Saved best Backlobe reconstruction plot to: {plot_file_bl}')

    fig, axes = plt.subplots(4, 2, figsize=(15, 10), sharex=True, sharey=True)
    fig.suptitle(
        f'{amp} {model_type} - BEST Reconstructed RCR (Signal) - MSE: {best_rcr_mse:.4g}\n(Dataset: {dataset_name_suffix})',
        fontsize=16,
    )

    axes[0, 0].set_title('Original Trace')
    axes[0, 1].set_title('Reconstructed Trace')

    for i in range(4):
        axes[i, 0].plot(rcr_original_best[i, :], color='red', label=f'Channel {i}')
        axes[i, 0].set_ylabel(f'Channel {i}')
        axes[i, 1].plot(best_rcr_reconstructed[i, :], color='black', label=f'Recon Chan {i}')

    axes[-1, 0].set_xlabel('Sample')
    axes[-1, 1].set_xlabel('Sample')

    plot_file_rcr_best = os.path.join(
        plot_path,
        f'{timestamp}_{amp}_{model_type}_recon_best_RCR_{prefix}_{lr_str}{domain_suffix}_{dataset_name_suffix}.png',
    )
    plt.savefig(plot_file_rcr_best)
    plt.close(fig)
    print(f'Saved best RCR reconstruction plot to: {plot_file_rcr_best}')

    fig, axes = plt.subplots(4, 2, figsize=(15, 10), sharex=True, sharey=True)
    fig.suptitle(
        f'{amp} {model_type} - WORST Reconstructed Backlobe (BG) - MSE: {worst_bl_mse:.4g}\n(Dataset: {dataset_name_suffix})',
        fontsize=16,
    )   

    axes[0, 0].set_title('Original Trace')
    axes[0, 1].set_title('Reconstructed Trace')

    for i in range(4):
        axes[i, 0].plot(bl_original_worst[i, :], color='blue', label=f'Channel {i}')
        axes[i, 0].set_ylabel(f'Channel {i}')
        axes[i, 1].plot(worst_bl_reconstructed[i, :], color='black', label=f'Recon Chan {i}')

    axes[-1, 0].set_xlabel('Sample')
    axes[-1, 1].set_xlabel('Sample')

    plot_file_bl_worst = os.path.join(
        plot_path,
        f'{timestamp}_{amp}_{model_type}_recon_worst_Backlobe_{prefix}_{lr_str}{domain_suffix}_{dataset_name_suffix}.png',
    )
    plt.savefig(plot_file_bl_worst)
    plt.close(fig)
    print(f'Saved worst Backlobe reconstruction plot to: {plot_file_bl_worst}')

def _extract_latent_vectors(model, prepared_batch):
    """Return deterministic latent representations for VAE inputs."""

    encoder_outputs = model.encoder.predict(prepared_batch)
    if isinstance(encoder_outputs, (list, tuple)) and len(encoder_outputs) >= 1:
        return encoder_outputs[0]
    return encoder_outputs


def plot_latent_space(
    model,
    sim_rcr_all,
    data_backlobe_traces_rcr_all,
    requires_transpose,
    timestamp,
    config,
    model_type,
    learning_rate,
    dataset_name_suffix="main",
    extra_datasets=None,
):
    """Generate an n×n latent-space pair plot with histograms on the diagonal."""

    amp = config['amp']
    prefix = config['prefix']
    lr_str = f"lr{learning_rate:.0e}".replace('-', '')
    plot_path = config['base_plot_path']
    domain_label = config.get('domain_label', 'time')
    domain_suffix = config.get('domain_suffix', '')
    os.makedirs(plot_path, exist_ok=True)

    extra_datasets = extra_datasets or []

    dataset_entries = []
    n_samples = min(2000, len(sim_rcr_all), len(data_backlobe_traces_rcr_all))
    if n_samples == 0:
        print(
            f"Skipping latent space plot for {dataset_name_suffix}: not enough RCR/Backlobe data."
        )
        return

    print(f"Subsampling {n_samples} events from RCR and Backlobe for latent space plot.")
    rcr_indices = np.random.choice(len(sim_rcr_all), n_samples, replace=False)
    bl_indices = np.random.choice(len(data_backlobe_traces_rcr_all), n_samples, replace=False)

    dataset_entries.append(
        {
            'label': 'RCR (Signal)',
            'data': sim_rcr_all[rcr_indices],
            'color': LATENT_COLOR_MAP.get('RCR (Signal)', 'red'),
            'marker': LATENT_MARKER_MAP.get('RCR (Signal)', 'o'),
            'alpha': 0.5,
            'size': 40,
            'category': 'simulation',
        }
    )

    dataset_entries.append(
        {
            'label': 'Backlobe (BG)',
            'data': data_backlobe_traces_rcr_all[bl_indices],
            'color': LATENT_COLOR_MAP.get('Backlobe (BG)', 'blue'),
            'marker': LATENT_MARKER_MAP.get('Backlobe (BG)', 'o'),
            'alpha': 0.5,
            'size': 40,
            'category': 'background',
        }
    )

    for extra in extra_datasets:
        label = extra.get('label', 'Validation')
        data_array = np.asarray(extra.get('data', []))
        if data_array.size == 0 or data_array.ndim != 3:
            print(
                f"Skipping latent overlay '{label}': data missing or wrong shape {data_array.shape}."
            )
            continue

        category = extra.get('category', 'overlay')
        if category == 'above_curve':
            max_samples = extra.get('max_samples', data_array.shape[0])
        else:
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

        color = extra.get('color', LATENT_COLOR_MAP.get(label, 'gray'))
        marker = extra.get('marker', LATENT_MARKER_MAP.get(label, 'o'))

        dataset_entries.append(
            {
                'label': label,
                'data': data_subset,
                'color': color,
                'marker': marker,
                'alpha': extra.get('alpha', 0.8),
                'size': extra.get('size', 60),
                'category': category,
                'source_meta': extra,
            }
        )

    plottable_entries = []
    for entry in dataset_entries:
        try:
            prepared = prepare_data_for_model(entry['data'], requires_transpose)
        except ValueError as exc:
            print(f"Skipping dataset '{entry['label']}' in latent plotting: {exc}")
            continue

        if prepared.size == 0 or prepared.shape[0] == 0:
            print(
                f"Skipping dataset '{entry['label']}' in latent plotting: empty after preparation."
            )
            continue

        latent_vectors = _extract_latent_vectors(model, prepared)
        if latent_vectors.ndim > 2:
            latent_vectors = latent_vectors.reshape(latent_vectors.shape[0], -1)

        if latent_vectors.size == 0 or latent_vectors.shape[0] == 0:
            print(
                f"Skipping dataset '{entry['label']}' in latent plotting: encoder returned empty output."
            )
            continue

        entry.setdefault('category', 'overlay')
        entry['prepared'] = prepared
        entry['latent'] = latent_vectors
        entry['legend_label'] = f"{entry['label']} ({prepared.shape[0]})"
        plottable_entries.append(entry)

    if not plottable_entries:
        print(f"No datasets available for latent space plot ({dataset_name_suffix}).")
        return

    latent_dim = plottable_entries[0]['latent'].shape[1]
    if latent_dim == 0:
        print("Latent space has zero dimensions; skipping plot.")
        return

        # --- Now plotting UMAP ---
    print("Generating UMAP plot...")
    all_latent_vectors = np.concatenate(
        [entry['latent'] for entry in plottable_entries], axis=0
    )

    if all_latent_vectors.shape[0] < 5:
        print(
            f"Not enough total samples ({all_latent_vectors.shape[0]}) for UMAP. Skipping."
        )
        return

    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        random_state=42,
        min_dist=0.1,
    )
    embedding = reducer.fit_transform(all_latent_vectors)

    start_idx = 0
    for entry in plottable_entries:
        n_samples_in_entry = entry['latent'].shape[0]
        end_idx = start_idx + n_samples_in_entry
        entry['embedding'] = embedding[start_idx:end_idx, :]
        start_idx = end_idx

    # *** ERROR WAS HERE ***
    # The loop 'for group in all_plot_groups:' was here,
    # but 'all_plot_groups' isn't created until later.

    for entry in plottable_entries:
        if entry['latent'].shape[1] != latent_dim:
            print(
                f"Dataset '{entry['label']}' latent dimension mismatch; expected {latent_dim}, got {entry['latent'].shape[1]}."
            )
            return

    if latent_dim > 10:
        print(
            f"Latent dimension {latent_dim} is large; resulting plot may be hard to interpret."
        )

    def _category_priority(category):
        mapping = {
            'simulation': 0,
            'background': 0,
            'above_curve': 1,
            'validation': 2,
        }
        return mapping.get(category, 1)

    def _color_to_hex(color_value):
        try:
            return mcolors.to_hex(color_value)
        except (ValueError, TypeError):
            try:
                return mcolors.to_hex(mcolors.to_rgba(color_value))
            except Exception:
                return color_value

    above_curve_entry = next(
        (entry for entry in plottable_entries if entry['label'] == 'Above Curve'),
        None,
    )

    if above_curve_entry is not None:
        # Check if the embedding was successfully created
        if 'embedding' in above_curve_entry and above_curve_entry['embedding'].size > 0:
            print("Running HDBSCAN on the 2D UMAP embedding of 'Above Curve' data...")
            cluster_labels, clusterer, _ = _run_hdbscan_with_retries(
                above_curve_entry['embedding'],  # <-- *** This is the fix ***
                config,
                dataset_label=above_curve_entry['label'],
            )
            above_curve_entry['cluster_labels'] = cluster_labels
            above_curve_entry['clusterer'] = clusterer
        else:
            print("Skipping HDBSCAN for 'Above Curve': no UMAP embedding found.")
    else:
        print("No 'Above Curve' dataset found; assigning fallback single cluster.")

    all_plot_groups = []

    for entry in plottable_entries:
        entry_category = entry.get('category', 'overlay')
        base_color = _color_to_hex(entry.get('color', 'gray'))
        base_marker = entry.get('marker', 'o')
        base_alpha = entry.get('alpha', 0.7)
        base_size = entry.get('size', 40)
        highlight_marker = base_marker == '*'
        latent = entry['latent']
        num_points = latent.shape[0]

        labels = entry.get('cluster_labels') if entry['label'] == 'Above Curve' else None
        if labels is None or labels.size != num_points:
            labels = None

        if labels is None:
            group = {
                'name': entry['label'],
                'legend_text': entry['legend_label'],
                'color': base_color,
                'marker': base_marker,
                'alpha': base_alpha,
                'size': base_size,
                'latent': latent,
                'indices': np.arange(num_points),
                'count': num_points,
                'category': entry_category,
                'priority': _category_priority(entry_category),
                'edge_highlight': highlight_marker,
                'entry': entry,
            }
            entry['plot_groups'] = [group]
            all_plot_groups.append(group)
        else:
            unique_labels = np.unique(labels)
            positive_labels = sorted(lbl for lbl in unique_labels if lbl >= 0)
            cluster_id_map = {lbl: idx for idx, lbl in enumerate(positive_labels)}
            cmap_size = max(len(cluster_id_map), 1)
            cmap = plt.cm.get_cmap('tab20', cmap_size if cmap_size > 0 else 1)
            entry_groups = []
            for lbl in sorted(unique_labels, key=lambda value: (value < 0, value)):
                mask = labels == lbl
                count = int(np.sum(mask))
                if count == 0:
                    continue
                if lbl == -1:
                    display_name = 'Above Curve Noise'
                    color = '#808080'
                else:
                    display_idx = cluster_id_map.get(lbl, lbl)
                    color = _color_to_hex(cmap(display_idx % cmap.N))
                    display_name = f'Above Curve Cluster {display_idx}'
                group = {
                    'name': display_name,
                    'legend_text': f'{display_name} (N={count})',
                    'color': color,
                    'marker': base_marker,
                    'alpha': base_alpha,
                    'size': base_size,
                    'latent': latent[mask],
                    'indices': np.where(mask)[0],
                    'count': count,
                    'category': 'above_curve',
                    'priority': _category_priority('above_curve'),
                    'edge_highlight': highlight_marker,
                    'entry': entry,
                    'cluster_label': int(lbl),
                }
                entry_groups.append(group)
                all_plot_groups.append(group)

            if entry_groups:
                entry['plot_groups'] = entry_groups
            else:
                group = {
                    'name': entry['label'],
                    'legend_text': entry['legend_label'],
                    'color': base_color,
                    'marker': base_marker,
                    'alpha': base_alpha,
                    'size': base_size,
                    'latent': latent,
                    'indices': np.arange(num_points),
                    'count': num_points,
                    'category': entry_category,
                    'priority': _category_priority(entry_category),
                    'edge_highlight': highlight_marker,
                    'entry': entry,
                }
                entry['plot_groups'] = [group]
                all_plot_groups.append(group)

    if not all_plot_groups:
        print("No plottable groups available after processing; skipping latent plots.")
        return

    dim_ranges = []
    for dim in range(latent_dim):
        dim_values = np.concatenate([entry['latent'][:, dim] for entry in plottable_entries], axis=0)
        dim_min = float(np.min(dim_values))
        dim_max = float(np.max(dim_values))
        span = dim_max - dim_min
        pad = 0.05 * span if span > 0 else 1.0
        dim_ranges.append((dim_min - pad, dim_max + pad))

    groups_by_priority = {}
    for group in all_plot_groups:
        groups_by_priority.setdefault(group['priority'], []).append(group)

    groups_for_plot = []
    for priority in sorted(groups_by_priority.keys()):
        groups_for_plot.extend(groups_by_priority[priority])

    legend_groups = groups_for_plot

    # *** FIX IS TO MOVE THE LOOP HERE ***
    # This loop populates the 'embedding' key for each group,
    # which the UMAP plot will need.
    for group in all_plot_groups:
        indices = group['indices']
        if isinstance(indices, (list, tuple)):
            indices = np.asarray(indices)
        group['indices'] = indices
        if indices.size == 0:
            group['embedding'] = np.empty((0, 2))
        else:
            group['embedding'] = group['entry']['embedding'][indices, :]


    base_size = 3.0
    fig, axes = plt.subplots(
        latent_dim,
        latent_dim,
        figsize=(base_size * latent_dim, base_size * latent_dim),
    )
    if latent_dim == 1:
        axes = np.array([[axes]])

    for row in range(latent_dim):
        for col in range(latent_dim):
            ax = axes[row, col]
            if row == col:
                for group in groups_for_plot:
                    data = group['latent'][:, col]
                    if data.size == 0:
                        continue
                    ax.hist(
                        data,
                        bins=40,
                        histtype='step',
                        color=group['color'],
                        alpha=group['alpha'],
                        linewidth=1.5,
                    )
                ax.set_xlim(dim_ranges[col])
                ax.set_title(f'Latent {col}', fontsize=10)
                ax.set_yscale('log')
                if col == 0:
                    ax.set_ylabel('Count')
            elif row > col:
                for group in groups_for_plot:
                    data = group['latent']
                    if data.size == 0:
                        continue
                    ax.scatter(
                        data[:, col],
                        data[:, row],
                        color=group['color'],
                        marker=group['marker'],
                        alpha=group['alpha'],
                        s=group['size'],
                        edgecolors='k' if group['edge_highlight'] else 'none',
                        linewidths=0.6 if group['edge_highlight'] else 0.0,
                        zorder=group['priority'] + 1,
                    )
                ax.set_xlim(dim_ranges[col])
                ax.set_ylim(dim_ranges[row])
                if row == latent_dim - 1:
                    ax.set_xlabel(f'Latent {col}')
                if col == 0:
                    ax.set_ylabel(f'Latent {row}')
            else:
                ax.axis('off')

    fig.suptitle(
        f'Latent Space Pair Plot ({domain_label}, {dataset_name_suffix} set)',
        fontsize=16,
    )

    marker_symbol_map = {
        'o': '\u25CF',
        's': '\u25A0',
        '^': '\u25B2',
        'v': '\u25BC',
        '*': '\u2605',
        'd': '\u25C6',
        'D': '\u25C6',
    }
    text_y = 0.95
    line_height = 0.04
    for group in legend_groups:
        marker_symbol = marker_symbol_map.get(group['marker'], '\u25CF')
        fig.text(
            0.98,
            text_y,
            f'{marker_symbol} {group["legend_text"]}',
            color=group['color'],
            ha='right',
            va='top',
            fontsize=10,
        )
        text_y -= line_height

    plt.tight_layout(rect=[0.02, 0.02, 0.9, 0.9])

    plot_file = os.path.join(
        plot_path,
        f'{timestamp}_{amp}_{model_type}_latent_space_pairplot_{prefix}_{lr_str}{domain_suffix}_{dataset_name_suffix}.png',
    )
    plt.savefig(plot_file, dpi=200)
    plt.close(fig)
    print(f'Saved latent space pair plot to: {plot_file}')


    fig_umap, ax_umap = plt.subplots(figsize=(11, 9))
    used_labels = set()
    for group in groups_for_plot:
        # This line will now work, because the loop above populated 'embedding'
        embed = group.get('embedding', np.empty((0, 2)))
        if embed.size == 0:
            continue
        label = group['legend_text']
        label_to_use = label if label not in used_labels else None
        if label_to_use:
            used_labels.add(label_to_use)
        ax_umap.scatter(
            embed[:, 0],
            embed[:, 1],
            color=group['color'],
            marker=group['marker'],
            alpha=group['alpha'],
            s=group['size'],
            label=label_to_use,
            edgecolors='k' if group['edge_highlight'] else 'none',
            linewidths=0.6 if group['edge_highlight'] else 0.0,
            zorder=group['priority'] + 1,
        )

    ax_umap.set_title(
        f'UMAP Projection of Latent Space ({domain_label}, {dataset_name_suffix} set)',
        fontsize=16,
    )
    ax_umap.set_xlabel('UMAP Component 1')
    ax_umap.set_ylabel('UMAP Component 2')
    ax_umap.grid(True, linestyle='--', alpha=0.3)
    ax_umap.legend(loc='best', fontsize=10)
    plt.tight_layout()

    umap_plot_file = os.path.join(
        plot_path,
        f'{timestamp}_{amp}_{model_type}_latent_space_UMAP_{prefix}_{lr_str}{domain_suffix}_{dataset_name_suffix}.png',
    )
    plt.savefig(umap_plot_file, dpi=200)
    plt.close(fig_umap)
    print(f'Saved UMAP plot to: {umap_plot_file}')


def plot_validation_loss_histogram(
    loss_entries,
    config,
    timestamp,
    learning_rate,
    model_type,
    normalized=False,
    dataset_name_suffix="validation",
    cut_value=0.9,
):
    """Plot reconstruction loss histograms for multiple datasets."""

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
            normalized_entries = [
                (label, np.zeros_like(vals)) for label, vals in valid_entries
            ]
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
        upper_percentile = max(
            float(np.percentile(vals, 99.9)) for _, vals in plot_entries
        )
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
            label=f'{label} ({len(vals)})',
        )
        if counts.size:
            max_hist = max(max_hist, counts.max())

    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel('Count', fontsize=18)
    title_bits = 'Normalized ' if normalized else ''
    plt.title(
        f'{amp}_{domain_label} {model_type} Validation {title_bits}Loss (LR={learning_rate:.0e})',
        fontsize=14,
    )
    plt.yscale('log')

    if max_hist > 0:
        upper = max(10 ** (np.ceil(np.log10(max_hist * 1.1))), 10)
        plt.ylim(7e-1, upper)

    if normalized:
        plt.axvline(
            cut_value,
            color='y',
            linestyle='--',
            linewidth=1.2,
            label=f'Cut = {cut_value:.2f}',
        )
    else:
        plt.axvline(
            cut_value,
            color='y',
            linestyle='--',
            linewidth=1.2,
            label=f'Cut = {cut_value:.3g}',
        )

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


def plot_special_event_reconstruction(
    original_trace,
    reconstructed_trace,
    mse_value,
    timestamp,
    config,
    model_type,
    learning_rate,
    dataset_name_suffix,
    event_label,
):
    """Plot original vs reconstructed traces for a single special event."""

    if original_trace is None or reconstructed_trace is None:
        print("No special event traces provided; skipping special reconstruction plot.")
        return

    original = np.asarray(original_trace)
    reconstructed = np.asarray(reconstructed_trace)

    if original.ndim != 2 or reconstructed.ndim != 2:
        print(
            f"Skipping special reconstruction plot: expected 2D traces, got {original.shape} & {reconstructed.shape}."
        )
        return

    if original.shape != reconstructed.shape:
        print(
            f"Skipping special reconstruction plot: shape mismatch {original.shape} vs {reconstructed.shape}."
        )
        return

    amp = config['amp']
    prefix = config['prefix']
    domain_suffix = config.get('domain_suffix', '')
    domain_label = config.get('domain_label', 'time')
    lr_str = f"lr{learning_rate:.0e}".replace('-', '')
    plot_path = config['base_plot_path']
    os.makedirs(plot_path, exist_ok=True)

    num_channels, _ = original.shape
    fig, axes = plt.subplots(num_channels, 2, figsize=(15, 3 * num_channels), sharex=True, sharey=True)
    if num_channels == 1:
        axes = np.expand_dims(axes, axis=0)

    title_label = event_label or 'Validation Special Event'
    fig.suptitle(
        f'{amp} {model_type} - {title_label}\nReconstruction MSE: {mse_value:.4g} ({domain_label}, {dataset_name_suffix})',
        fontsize=16,
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


def run_validation_checks(
    model,
    requires_transpose,
    config,
    timestamp,
    learning_rate,
    model_type,
    sim_rcr_all,
    data_backlobe_traces_rcr_all,
    prob_rcr_main,
    prob_backlobe_main,
):
    """Run validation-only analysis using the configured coincidence dataset."""

    print("\n--- Running validation checks with coincidence PKL dataset ---")

    validation_pkl_path = config.get('validation_pkl_path', DEFAULT_VALIDATION_PKL_PATH)
    if not validation_pkl_path:
        print("Validation PKL path not provided; skipping validation checks.")
        return

    if not os.path.exists(validation_pkl_path):
        print(f"Validation PKL not found at {validation_pkl_path}; skipping validation checks.")
        return

    passing_event_ids = config.get(
        'validation_passing_event_ids', DEFAULT_VALIDATION_PASSING_EVENT_IDS
    )
    special_event_id = config.get(
        'validation_special_event_id', DEFAULT_VALIDATION_SPECIAL_EVENT_ID
    )
    special_station_id = config.get(
        'validation_special_station_id', DEFAULT_VALIDATION_SPECIAL_STATION_ID
    )

    (
        passing_traces,
        raw_traces,
        special_traces,
        passing_metadata,
        raw_metadata,
        special_metadata,
    ) = load_new_coincidence_data(
        validation_pkl_path,
        passing_event_ids,
        special_event_id,
        special_station_id,
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
            if config.get('convert_to_db_scale', False):
                transformed = convert_to_db_scale(transformed)
        return transformed.astype(np.float32, copy=False)

    passing_traces_proc = transform(passing_traces)
    raw_traces_proc = transform(raw_traces)
    special_traces_proc = transform(special_traces)

    above_curve_traces_proc = np.empty((0, reference_channels, reference_samples), dtype=np.float32)
    above_curve_pkl_path = config.get('above_curve_pkl_path', DEFAULT_ABOVE_CURVE_PKL_PATH)
    if above_curve_pkl_path:
        if os.path.exists(above_curve_pkl_path):
            try:
                (
                    _,
                    _,
                    _,
                    _,
                    above_curve_traces2016,
                    above_curve_tracesRCR,
                    _,
                    _,
                ) = load_combined_backlobe_data(above_curve_pkl_path)

                processed_above_curve = []
                for subset in (above_curve_traces2016, above_curve_tracesRCR):
                    if subset is None:
                        continue
                    try:
                        coerced = coerce_traces(subset)
                    except ValueError as exc:
                        print(
                            f"Skipping Above Curve subset due to incompatible shape: {exc}"
                        )
                        continue
                    if coerced.size == 0:
                        continue
                    if (
                        coerced.shape[1] != reference_channels
                        and coerced.shape[2] == reference_channels
                    ):
                        coerced = coerced.transpose(0, 2, 1)
                    if coerced.shape[1] != reference_channels:
                        print(
                            f"Skipping Above Curve subset: expected {reference_channels} channels, got {coerced.shape[1]}."
                        )
                        continue
                    if coerced.shape[2] != reference_samples and not is_freq_model:
                        print(
                            f"Skipping Above Curve subset: expected {reference_samples} samples, got {coerced.shape[2]}."
                        )
                        continue

                    coerced = coerced.astype(np.float32, copy=False)
                    transformed_subset = transform(coerced)
                    if transformed_subset.shape[1] != reference_channels:
                        print(
                            "Skipping Above Curve subset after transform: channel count mismatch "
                            f"{transformed_subset.shape[1]} (expected {reference_channels})."
                        )
                        continue
                    if transformed_subset.shape[2] != reference_samples:
                        print(
                            "Skipping Above Curve subset after transform: sample count mismatch "
                            f"{transformed_subset.shape[2]} (expected {reference_samples})."
                        )
                        continue
                    processed_above_curve.append(transformed_subset)

                if processed_above_curve:
                    above_curve_traces_proc = np.concatenate(processed_above_curve, axis=0)
                else:
                    print(
                        "Above Curve dataset contained no usable traces after preprocessing; skipping."
                    )
            except Exception as exc:
                print(
                    f"Failed to load Above Curve dataset from {above_curve_pkl_path}: {exc}"
                )
                quit()
        else:
            print(
                f"Above Curve PKL not found at {above_curve_pkl_path}; quitting."
            )
            quit()

    special_label = 'Validation Special Event'
    if special_traces_proc.shape[0] > 0:
        meta = special_metadata.get(0, {})
        evt = meta.get('event_id')
        stn = meta.get('station_id')
        if evt is not None and stn is not None:
            special_label = f'Validation Special Event (Evt {evt}, Stn {stn})'

    passing_mse = np.array([])
    raw_mse = np.array([])
    special_mse = np.array([])
    above_curve_mse = np.array([])
    special_trace_for_plot = None
    special_recon_for_plot = None

    if passing_traces_proc.shape[0] > 0:
        prepared = prepare_data_for_model(passing_traces_proc, requires_transpose)
        passing_mse, _ = calculate_reconstruction_mse(model, prepared, config)
        print(
            f"Validation passing set size: {len(passing_mse)}, mean MSE: {np.mean(passing_mse):.4g}"
        )

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

    if above_curve_traces_proc.shape[0] > 0:
        prepared = prepare_data_for_model(above_curve_traces_proc, requires_transpose)
        above_curve_mse, _ = calculate_reconstruction_mse(model, prepared, config)
        if above_curve_mse.size > 0:
            print(
                f"Above Curve set size: {len(above_curve_mse)}, mean MSE: {np.mean(above_curve_mse):.4g}"
            )

    validation_overlays = []
    if passing_traces_proc.shape[0] > 0:
        validation_overlays.append(
            {
                'label': 'Validation Passing',
                'data': passing_traces_proc,
                'color': LATENT_COLOR_MAP.get('Validation Passing', 'green'),
                'marker': LATENT_MARKER_MAP.get('Validation Passing', 's'),
                'alpha': 0.8,
                'size': 60,
                'category': 'validation',
            }
        )
    if raw_traces_proc.shape[0] > 0:
        validation_overlays.append(
            {
                'label': 'Validation Raw',
                'data': raw_traces_proc,
                'color': LATENT_COLOR_MAP.get('Validation Raw', 'purple'),
                'marker': LATENT_MARKER_MAP.get('Validation Raw', '^'),
                'alpha': 0.8,
                'size': 60,
                'category': 'validation',
            }
        )
    if special_traces_proc.shape[0] > 0:
        validation_overlays.append(
            {
                'label': special_label,
                'data': special_traces_proc,
                'color': LATENT_COLOR_MAP.get('Validation Special Event', 'orange'),
                'marker': LATENT_MARKER_MAP.get('Validation Special Event', '*'),
                'alpha': 1.0,
                'size': 160,
                'category': 'validation',
            }
        )
    if above_curve_traces_proc.shape[0] > 0:
        validation_overlays.append(
            {
                'label': 'Above Curve',
                'data': above_curve_traces_proc,
                'color': LATENT_COLOR_MAP.get('Above Curve', 'cyan'),
                'marker': LATENT_MARKER_MAP.get('Above Curve', 'd'),
                'alpha': 0.7,
                'size': 60,
                'category': 'above_curve',
                'max_samples': above_curve_traces_proc.shape[0],
            }
        )

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
            extra_datasets=validation_overlays,
        )
    else:
        print("No validation traces available for latent space overlay.")

    loss_entries = [
        ('RCR (Signal)', prob_rcr_main),
        ('Backlobe (BG)', prob_backlobe_main),
        ('Validation Passing', passing_mse),
        ('Validation Raw', raw_mse),
        (special_label, special_mse),
        ('Above Curve', above_curve_mse),
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
        cut_value=output_cut_value,
    )

    plot_validation_loss_histogram(
        loss_entries,
        config,
        timestamp,
        learning_rate,
        model_type,
        normalized=True,
        dataset_name_suffix="validation",
        cut_value=0.9,
    )

    if (
        special_trace_for_plot is not None
        and special_recon_for_plot is not None
        and special_mse.size > 0
    ):
        plot_special_event_reconstruction(
            special_trace_for_plot,
            special_recon_for_plot,
            float(special_mse.flatten()[0]),
            timestamp,
            config,
            model_type,
            learning_rate,
            dataset_name_suffix="validation",
            event_label=special_label,
        )
    else:
        print("No special event reconstruction plot generated (missing data).")


def _save_vae_model_artifacts(
    model,
    config,
    timestamp,
    amp,
    model_type,
    learning_rate,
):
    """Persist the full VAE plus separate encoder/decoder artifacts."""

    domain_suffix = config.get('domain_suffix', '')
    prefix = config['prefix']
    lr_str = f"lr{learning_rate:.0e}".replace('-', '')

    model_dir = os.path.join(
        config['base_model_path'],
        f'{timestamp}_{amp}_{model_type}_model_{prefix}_{lr_str}{domain_suffix}',
    )
    encoder_dir = os.path.join(model_dir, 'encoder')
    decoder_dir = os.path.join(model_dir, 'decoder')
    os.makedirs(model_dir, exist_ok=True)

    # print(f'Saving full VAE to: {model_dir}')
    # model.save(model_dir)

    print(f'Saving encoder to: {encoder_dir}')
    # model.encoder.save(encoder_dir)
    model.encoder.save_weights(encoder_dir)

    print(f'Saving decoder to: {decoder_dir}')
    # model.decoder.save(decoder_dir)
    model.decoder.save_weights(decoder_dir)

    weights_path = os.path.join(model_dir, 'weights.h5')
    model.save_weights(weights_path)
    print(f'Saved VAE weights to: {weights_path}')


def main():
    parser = argparse.ArgumentParser(description='Train VAE model.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        required=True,
        help='Learning rate (e.g., 0.0001)',
    )
    parser.add_argument(
        '--model_type',
        type=str,
        required=True,
        choices=list(MODEL_BUILDERS.keys()),
        help=f'Type of VAE model to train. Choose from: {list(MODEL_BUILDERS.keys())}',
    )

    args = parser.parse_args()

    learning_rate = args.learning_rate
    model_type = args.model_type

    config = load_config()
    amp = config['amp']
    prefix = config['prefix']

    is_freq_model = model_type.endswith('_freq')
    config['is_freq_model'] = is_freq_model
    config['domain_label'] = 'freq' if is_freq_model else 'time'
    config['domain_suffix'] = '_freq' if is_freq_model else ''

    base_model_root = config['base_model_path']
    base_plot_root = config['base_plot_path']

    lr_folder = f'lr_{learning_rate:.0e}'.replace('-', '')
    model_folder = model_type
    timestamp = datetime.now().strftime('%m.%d.%y_%H-%M')

    config['base_model_path'] = os.path.join(base_model_root, model_folder, lr_folder)
    config['base_plot_path'] = os.path.join(
        base_plot_root, f"{timestamp}", model_folder, lr_folder
    )

    output_cut_value = config['output_cut_value']
    print(f"--- Starting VAE training at {timestamp} ---")
    print(f"Model type: {model_type}, LR: {learning_rate}")
    print(
        "NOTE: 'output_cut_value' from config is used as MSE threshold for reconstruction loss."
    )

    data = load_and_prep_data_for_training(config)

    # Alternative loading of all station data

    # file_path = '/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/9.1.25/'
    # training_backlobe = []
    # for file in os.listdir(file_path):
    #     # load all numpy files with the substring "Traces" that don't have "_0evts" in their name
    #     if "Traces" in file and "_0evts" not in file:
    #         data_array = np.load(os.path.join(file_path, file))
    #         training_backlobe.append(data_array)
    # training_backlobe = np.concatenate(training_backlobe, axis=0)
    # print(f"Training data shape: {training_backlobe.shape}")
    # if is_freq_model:
    #     training_backlobe = _compute_frequency_magnitude(
    #         training_backlobe, config.get('frequency_sampling_rate', 2.0)
    #     )
    #     if config.get('use_filtering', False):
    #         training_backlobe = _apply_frequency_edge_filter(training_backlobe)
    #     if config.get('convert_to_db_scale', False):
    #         training_backlobe = convert_to_db_scale(training_backlobe)
    #     print(f"Is freq model, training data shape: {training_backlobe.shape}")
    # training_backlobe = np.concatenate(training_backlobe, axis=0)

    training_backlobe = data['training_backlobe']
    sim_rcr_all = data['sim_rcr_all']
    data_backlobe_traces_rcr_all = data['data_backlobe_tracesRCR']

    # print("Applying channel cycling augmentation to Backlobe training data...")
    # training_backlobe_aug = cycle_channels(training_backlobe.copy(), channel_axis=1)


    model, history, requires_transpose = train_vae_model(
        training_backlobe, config, learning_rate, model_type
        # training_backlobe_aug, config, learning_rate, model_type
    )
    print('------> VAE Training is Done!')

    _save_vae_model_artifacts(model, config, timestamp, amp, model_type, learning_rate)

    save_and_plot_training_history(
        history,
        config['base_model_path'],
        config['base_plot_path'],
        timestamp,
        amp,
        config,
        learning_rate,
        model_type,
    )

    print("\n--- Running evaluation on MAIN dataset (sim RCR vs. data Backlobe) ---")
    (
        prob_rcr,
        prob_backlobe,
        rcr_efficiency,
        backlobe_efficiency,
    ) = evaluate_model_performance_vae(
        model,
        sim_rcr_all,
        data_backlobe_traces_rcr_all,
        output_cut_value,
        config,
        model_type,
        requires_transpose,
    )

    plot_network_output_histogram_vae(
        prob_rcr,
        prob_backlobe,
        rcr_efficiency,
        backlobe_efficiency,
        config,
        timestamp,
        learning_rate,
        model_type,
        dataset_name_suffix="main",
    )

    plot_normalized_network_output_histogram_vae(
        prob_rcr,
        prob_backlobe,
        config,
        timestamp,
        learning_rate,
        model_type,
        dataset_name_suffix="main",
        cut_value=0.9,
    )

    print("\n--- Generating Original vs. Reconstructed plots for MAIN dataset ---")
    plot_original_vs_reconstructed(
        model,
        sim_rcr_all,
        data_backlobe_traces_rcr_all,
        prob_rcr,
        prob_backlobe,
        requires_transpose,
        timestamp,
        config,
        model_type,
        learning_rate,
        dataset_name_suffix="main",
    )

    print("\n--- Generating Latent Space plot for MAIN dataset ---")
    plot_latent_space(
        model,
        sim_rcr_all,
        data_backlobe_traces_rcr_all,
        requires_transpose,
        timestamp,
        config,
        model_type,
        learning_rate,
        dataset_name_suffix="main",
    )

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
        prob_backlobe,
    )

    print(
        f"\nScript finished successfully for VAE model {model_type}, lr {learning_rate}"
    )


if __name__ == "__main__":
    main()


