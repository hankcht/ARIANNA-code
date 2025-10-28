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
    # Note: We will write new evaluation and plotting functions
)

# --- Model Selection (Autoencoders) ---
# We only have one AE model, but we can have time/freq versions
# This script will only support the time-domain (256, 4) for now
# To support freq, new AE architecture for (129, 4) is needed
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
    
    print(f"Predicting on RCR (signal) set...")
    reconstructed_rcr = model.predict(sim_rcr_prepped, batch_size=config['keras_batch_size'])
    print(f"Predicting on Backlobe (background) set...")
    reconstructed_backlobe = model.predict(data_backlobe_prepped, batch_size=config['keras_batch_size'])
    
    # Calculate per-event MSE
    # (reconstructed_shape) - (original_shape)
    # Axis (1, 2) are (samples, channels)
    axis_to_average = tuple(range(1, sim_rcr_prepped.ndim)) # (1, 2)
    
    prob_rcr = np.mean(np.square(reconstructed_rcr - sim_rcr_prepped), axis=axis_to_average)
    prob_backlobe = np.mean(np.square(reconstructed_backlobe - data_backlobe_prepped), axis=axis_to_average)
    
    print(f"Mean RCR (Signal) Reconstruction MSE: {np.mean(prob_rcr):.4f}")
    print(f"Mean Backlobe (BG) Reconstruction MSE: {np.mean(prob_backlobe):.4f}")

    # Efficiency: Signal (RCR) is events WITH HIGH ERROR
    rcr_efficiency = (np.sum(prob_rcr > output_cut_value) / len(prob_rcr)) * 100
    # Background (Backlobe) retained are events WITH HIGH ERROR
    backlobe_efficiency = (np.sum(prob_backlobe > output_cut_value) / len(prob_backlobe)) * 100

    print(f'RCR efficiency (Signal > Cut): {rcr_efficiency:.2f}%')
    print(f'Backlobe efficiency (BG > Cut): {backlobe_efficiency:.4f}%')
    print(f'Lengths: RCR {len(prob_rcr)}, Backlobe {len(prob_backlobe)}')

    return prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency


def plot_network_output_histogram_autoencoder(prob_rcr, prob_backlobe, rcr_efficiency,
                                              backlobe_efficiency, config, timestamp, learning_rate, model_type):
    """
    Plots the histogram of network outputs (Reconstruction MSE)
    for RCR and Backlobe events.
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
    max_val = max(np.percentile(prob_rcr, 99), np.percentile(prob_backlobe, 99))
    min_val = min(np.min(prob_rcr), np.min(prob_backlobe))
    plot_range = (min_val, max_val * 1.1)
    
    plt.hist(prob_backlobe, bins=50, range=plot_range, histtype='step', color='blue', linestyle='solid', label=f'Backlobe {len(prob_backlobe)}', density=True)
    plt.hist(prob_rcr, bins=50, range=plot_range, histtype='step', color='red', linestyle='solid', label=f'RCR {len(prob_rcr)}', density=True)

    plt.xlabel('Reconstruction Loss (MSE)', fontsize=18)
    plt.ylabel('Normalized Density', fontsize=18)
    plt.yscale('log')
    plt.title(f'{amp}_{domain_label} {model_type} Reconstruction Loss (LR={learning_rate:.0e})', fontsize=14)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper right', fontsize=12)

    ax = plt.gca()
    ax.text(0.45, 0.75, f'RCR efficiency (Signal > Cut): {rcr_efficiency:.2f}%', fontsize=12, transform=ax.transAxes)
    ax.text(0.45, 0.70, f'Backlobe efficiency (BG > Cut): {backlobe_efficiency:.4f}%', fontsize=12, transform=ax.transAxes)
    ax.text(0.45, 0.60, f'LR: {learning_rate:.0e}', fontsize=12, transform=ax.transAxes)
    ax.text(0.45, 0.55, f'Model: {model_type}', fontsize=12, transform=ax.transAxes)
    plt.axvline(x=output_cut_value, color='y', label=f'Cut = {output_cut_value:.2f}', linestyle='--')
    plt.legend()
    
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.9)

    hist_file = os.path.join(plot_path, f'{timestamp}_{amp}_{model_type}_network_output_{prefix}_{lr_str}{domain_suffix}.png')
    print(f'saving {hist_file}')
    plt.savefig(hist_file)
    plt.close()


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

    # Evaluate & plot network output histogram
    prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency = \
        evaluate_model_performance_autoencoder(model, sim_rcr_all, data_backlobe_traces_rcr_all, 
                                               output_cut_value, config, model_type, requires_transpose)

    plot_network_output_histogram_autoencoder(prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency, 
                                              config, timestamp, learning_rate, model_type)

    # --- We don't run the 'check histogram' as it's not applicable here ---
    # The 'check' events (2016 BL, passing, raw) are all types of 'background'
    # and should all have low reconstruction loss.
         
    print(f"Script finished successfully for Autoencoder model {model_type}, lr {learning_rate}")

if __name__ == "__main__":
    main()
