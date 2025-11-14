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
from tensorflow.keras.models import Model  # Import Model for activation plotting
from NuRadioReco.utilities import units
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / '200s_time'))
from A0_Utilities import load_sim_rcr, load_data, pT, load_config

# Add parent directory to path to import model_builder and data_channel_cycling
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_builder import (
    build_cnn_model,
    build_cnn_model_freq,
    build_1d_model,
    build_1d_model_freq,
    build_parallel_model,
    build_parallel_model_freq,
    build_strided_model,
    build_strided_model_freq,
    build_parallel_strided_model,
    build_parallel_strided_model_freq
)
from data_channel_cycling import cycle_channels

# --- Model Selection and Building ---
MODEL_TYPES = {
    '1d_cnn': build_1d_model,
    '1d_cnn_freq': build_1d_model_freq,
    'parallel': build_parallel_model,
    'parallel_freq': build_parallel_model_freq,
    'strided': build_strided_model,
    'strided_freq': build_strided_model_freq,
    'parallel_strided': build_parallel_strided_model,
    'parallel_strided_freq': build_parallel_strided_model_freq,
    'astrid_2d': build_cnn_model,
    'astrid_2d_freq': build_cnn_model_freq
}


def _compute_frequency_magnitude(traces, sampling_rate):
    """Return scaled magnitude of the real FFT along the final axis."""

    traces_array = np.asarray(traces)
    if traces_array.size == 0:
        return traces_array

    freq = np.fft.rfft(traces_array, axis=-1)
    magnitude = np.abs(freq)
    if sampling_rate > 0:
        magnitude = magnitude / sampling_rate * np.sqrt(2.0)
    if np.issubdtype(traces_array.dtype, np.floating):
        magnitude = magnitude.astype(traces_array.dtype, copy=False)
    return magnitude


def _apply_frequency_edge_filter(freq_array, num_bins=10):
    """Zero out low/high frequency bins in-place to suppress edge artifacts."""

    freq_array = np.asarray(freq_array)
    if freq_array.size == 0:
        return freq_array

    if freq_array.shape[-1] <= num_bins * 2:
        return freq_array

    freq_array[..., :num_bins] = 0
    freq_array[..., -num_bins:] = 0
    return freq_array

def convert_to_db_scale(array, min_value=1e-2):
    """Convert magnitude values to dB, guarding against log(0)."""

    arr = np.asarray(array)
    if arr.size == 0:
        return arr.astype(np.float32, copy=False)

    safe = np.maximum(arr, min_value)
    db = 20.0 * np.log10(safe)
    return db.astype(np.float32, copy=False)

def load_combined_backlobe_data(combined_pkl_path):
    """
    Load the combined backlobe data saved by refactor_converter.py.
    
    This function loads a pickle file containing backlobe data from all stations
    that has been processed through chi and bin cuts.
    
    Args:
        combined_pkl_path (str): Path to the combined pickle file containing all stations' data.
                                Default location from refactor_converter.py:
                                '/dfs8/sbarwick_lab/ariannaproject/tangch3/station_data/above_curve_data/5000evt_10.17.25/above_curve_combined.pkl'
                                '/dfs8/sbarwick_lab/ariannaproject/tangch3/station_data/above_curve_data/toy_data_11.6.25/toy_data_combined.pkl'
    
    Returns:
        tuple: (snr2016, snrRCR, chi2016, chiRCR, traces2016, tracesRCR, unix2016, unixRCR)
               All as numpy arrays containing data from all stations combined.
    """
    print(f"Loading combined backlobe data from: {combined_pkl_path}")
    
    with open(combined_pkl_path, 'rb') as f:
        combined_data = pickle.load(f)
    
    # Extract all arrays from the combined dictionary
    snr2016 = combined_data['snr2016']
    snrRCR = combined_data['snrRCR']
    chi2016 = combined_data['chi2016']
    chiRCR = combined_data['chiRCR']
    traces2016 = combined_data['traces2016']
    tracesRCR = combined_data['tracesRCR']
    unix2016 = combined_data['unix2016']
    unixRCR = combined_data['unixRCR']
    
    print(f"Loaded combined data:")
    print(f"  SNR2016: {len(snr2016)} events")
    print(f"  SNRRCR: {len(snrRCR)} events")
    print(f"  Chi2016: {len(chi2016)} events")
    print(f"  ChiRCR: {len(chiRCR)} events")
    print(f"  Traces2016 shape: {traces2016.shape}")
    print(f"  TracesRCR shape: {tracesRCR.shape}")
    print(f"  Unix2016: {len(unix2016)} events")
    print(f"  UnixRCR: {len(unixRCR)} events")
    
    return snr2016, snrRCR, chi2016, chiRCR, traces2016, tracesRCR, unix2016, unixRCR

# --- Data Loading and Preparation ---
def load_and_prep_data_for_training(config):
    """
    Loads sim RCR and data Backlobe from the combined pickle file saved by refactor_converter.py.
    Selects random subset for training.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        dict: A dictionary containing prepared data arrays and indices.
    """
    amp = config['amp']
    train_cut = config['train_cut']
    is_freq_model = config.get('is_freq_model', False)
    sampling_rate = float(config.get('frequency_sampling_rate', 2.0))
    use_filtering = bool(config.get('use_filtering', False))
    print(f"Loading data for amplifier type: {amp}")

    # Load simulation RCR data
    sim_folder = os.path.join(config['base_sim_rcr_folder'], amp, config['sim_rcr_date'])
    sim_rcr = load_sim_rcr(sim_folder, noise_enabled=config['noise_enabled'], filter_enabled=use_filtering, amp=amp)
    
    # Add 8.14.25 sim RCR
    if amp == '200s': 
        sim_rcr_814 = np.load(f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/8.14.25/200s/all_traces_200s_RCR_part0_4473events.npy') 
        print('Loading /dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/8.14.25/200s/all_traces_200s_RCR_part0_4473events.npy')
    elif amp == '100s':
        sim_rcr_814 = np.load(f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/8.14.25/100s/all_traces_100s_RCR_part0_4200events.npy')
        print('Loading /dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/8.14.25/100s/all_traces_100s_RCR_part0_4200events.npy') 
    else:
        print(f'amp not found')
    sim_rcr = np.vstack([sim_rcr, sim_rcr_814])

    # Load combined backlobe data from the pickle file saved by refactor_converter.py
    # This file contains data from all stations after chi and bin cuts
    combined_pkl_path = f'/dfs8/sbarwick_lab/ariannaproject/tangch3/station_data/above_curve_data/5000evt_10.17.25/above_curve_combined.pkl'
    # combined_pkl_path = f'/dfs8/sbarwick_lab/ariannaproject/tangch3/station_data/above_curve_data/toy_data_11.6.25/toy_data_combined.pkl'
    snr2016, snrRCR, chi2016, chiRCR, traces2016, tracesRCR, unix2016, unixRCR = load_combined_backlobe_data(combined_pkl_path)

    # Convert to numpy arrays (they should already be arrays from the pickle, but ensure consistency)
    backlobe_traces_2016 = np.array(traces2016)
    backlobe_traces_rcr = np.array(tracesRCR)

    if is_freq_model:
        print('Converting training and evaluation data to frequency-domain magnitude...')
        # Only transform traces to frequency space when explicitly training frequency-domain models.
        sim_rcr = _compute_frequency_magnitude(sim_rcr, sampling_rate)
        backlobe_traces_2016 = _compute_frequency_magnitude(backlobe_traces_2016, sampling_rate)
        backlobe_traces_rcr = _compute_frequency_magnitude(backlobe_traces_rcr, sampling_rate)

        if use_filtering:
            print('Applying frequency edge filtering (zeroing first/last 10 bins).')
            sim_rcr = _apply_frequency_edge_filter(sim_rcr)
            backlobe_traces_2016 = _apply_frequency_edge_filter(backlobe_traces_2016)
            backlobe_traces_rcr = _apply_frequency_edge_filter(backlobe_traces_rcr)

        if config.get('convert_to_db_scale', False):
            print('Converting frequency magnitudes to dB scale.')
            sim_rcr = convert_to_db_scale(sim_rcr)
            backlobe_traces_2016 = convert_to_db_scale(backlobe_traces_2016)
            backlobe_traces_rcr = convert_to_db_scale(backlobe_traces_rcr)

    print(f'RCR shape: {sim_rcr.shape}, Backlobe 2016 shape: {backlobe_traces_2016.shape}, Backlobe RCR shape: {backlobe_traces_rcr.shape}')

    # Pick random subsets for training
    train_cut = int(min(sim_rcr.shape[0], backlobe_traces_2016.shape[0]))
    rcr_training_indices = np.random.choice(sim_rcr.shape[0], size=train_cut, replace=False)
    bl_training_indices = np.random.choice(backlobe_traces_2016.shape[0], size=train_cut, replace=False)

    training_rcr = sim_rcr[rcr_training_indices, :]
    training_backlobe = backlobe_traces_2016[bl_training_indices, :]  # Using traces2016 for training

    rcr_non_training_indices = np.setdiff1d(np.arange(sim_rcr.shape[0]), rcr_training_indices)
    bl_non_training_indices = np.setdiff1d(np.arange(backlobe_traces_2016.shape[0]), bl_training_indices)

    print(f'Training shape RCR {training_rcr.shape}, Training Shape Backlobe {training_backlobe.shape}, TrainCut {train_cut}')
    print(f'Non-training RCR count {len(rcr_non_training_indices)}, Non-training Backlobe count {len(bl_non_training_indices)}')

    return {
        'training_rcr': training_rcr,
        'training_backlobe': training_backlobe,  # This is traces2016
        'sim_rcr_all': sim_rcr,
        'data_backlobe_traces2016': backlobe_traces_2016,
        'data_backlobe_tracesRCR': backlobe_traces_rcr,
        'data_backlobe_unix2016_all': unix2016,
        'data_backlobe_chi2016_all': chi2016,
        'rcr_non_training_indices': rcr_non_training_indices,
        'bl_non_training_indices': bl_non_training_indices
    } 




def get_model_builder(model_type):
    """
    Returns the appropriate model building function based on model_type.
    
    Args:
        model_type (str): Type of model to build.
        
    Returns:
        function: Model building function.
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(MODEL_TYPES.keys())}")
    return MODEL_TYPES[model_type]


def train_cnn_model(training_rcr, training_backlobe, config, learning_rate, model_type):
    """
    Trains the CNN model based on the specified model_type.

    Args:
        training_rcr (np.ndarray): RCR training data.
        training_backlobe (np.ndarray): Backlobe training data.
        config (dict): Configuration dictionary.
        learning_rate (float): Learning rate for the optimizer.
        model_type (str): Type of model to build.

    Returns:
        tuple: (keras.Model, keras.callbacks.History, bool) The trained model, history, and transpose flag.
    """
    model_builder = get_model_builder(model_type)
    model, requires_transpose = model_builder(learning_rate=learning_rate)

    x = np.vstack((training_rcr, training_backlobe))
    if requires_transpose:
        x = x.transpose(0, 2, 1)
    else:
        if x.ndim == 3:
            x = np.expand_dims(x, axis=-1)

    y = np.vstack((np.ones((training_rcr.shape[0], 1)), np.zeros((training_backlobe.shape[0], 1)))) # 1s for RCR (signal)
    s = np.arange(x.shape[0])
    np.random.shuffle(s)
    x = x[s]
    y = y[s]
    print(f"Training data shape: {x.shape}, label shape: {y.shape}, requires_transpose: {requires_transpose}")

    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=config['early_stopping_patience'])]

    model.summary()

    history = model.fit(x, y,
                        validation_split=0.25,
                        epochs=config['keras_epochs'],
                        batch_size=config['keras_batch_size'],
                        verbose=config['verbose_fit'],
                        callbacks=callbacks_list)

    return model, history, requires_transpose


def save_and_plot_training_history(history, model_path, plot_path, timestamp, amp, config, learning_rate, model_type):
    """
    Saves the training history and plots loss and accuracy curves.

    Args:
        history (keras.callbacks.History): The history object returned by model.fit.
        model_path (str): Directory to save the history pickle.
        plot_path (str): Directory to save the plots.
        timestamp (str): Timestamp for filenames.
        amp (str): Amplifier type for plot filenames.
        config (dict): Configuration dictionary.
        learning_rate (float): Learning rate used for training.
        model_type (str): Type of model being trained.
    """
    prefix = config['prefix']
    lr_str = f"lr{learning_rate:.0e}".replace('-', '')
    domain_suffix = config.get('domain_suffix', '')
    domain_label = config.get('domain_label', 'time')

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)

    history_file = os.path.join(model_path, f'{timestamp}_{amp}_{model_type}_history_{prefix}_{lr_str}{domain_suffix}.pkl')
    with open(history_file, 'wb') as f: 
        pickle.dump(history.history, f)
    print(f'Training history saved to: {history_file}')

    # Plot loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title(f'{model_type} Training vs Validation Loss ({domain_label}, LR={learning_rate:.0e})')
    plt.legend()
    loss_plot_file = os.path.join(plot_path, f'{timestamp}_{amp}_{model_type}_loss_{prefix}_{lr_str}{domain_suffix}.png')
    plt.savefig(loss_plot_file)
    plt.close()
    print(f'Loss plot saved to: {loss_plot_file}')

    # Plot accuracy
    if 'accuracy' not in history.history:
        print('No accuracy data found in history; skipping accuracy plot.')
        return
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_type} Training vs Validation Accuracy ({domain_label}, LR={learning_rate:.0e})')
    plt.legend()
    accuracy_plot_file = os.path.join(plot_path, f'{timestamp}_{amp}_{model_type}_accuracy_{prefix}_{lr_str}{domain_suffix}.png')
    plt.savefig(accuracy_plot_file)
    plt.close()
    print(f'Accuracy plot saved to: {accuracy_plot_file}')


# --- Model Evaluation ---
def evaluate_model_performance(model, sim_rcr_all, data_backlobe_traces_rcr_all, output_cut_value, config, model_type, requires_transpose):
    """
    Evaluates the model on above curve Backlobe in RCR template.

    Args:
        model (keras.Model): The trained Keras model.
        sim_rcr_all (np.ndarray): All RCR simulation data.
        data_backlobe_traces_rcr_all (np.ndarray): All Backlobe data (TracesRCR).
        output_cut_value (float): The threshold for classification.
    model_type (str): Type of model being evaluated.
    requires_transpose (bool): Whether inputs require axis transpose to match the model.

    Returns:
        tuple: (prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency)
    """
    # Prepare data based on model type
    if requires_transpose:
        sim_rcr_expanded = sim_rcr_all.transpose(0, 2, 1)
        data_backlobe_expanded = data_backlobe_traces_rcr_all.transpose(0, 2, 1)
    else:
        sim_rcr_expanded = sim_rcr_all if sim_rcr_all.ndim == 4 else sim_rcr_all[..., np.newaxis]
        data_backlobe_expanded = data_backlobe_traces_rcr_all if data_backlobe_traces_rcr_all.ndim == 4 else data_backlobe_traces_rcr_all[..., np.newaxis]

    prob_rcr = model.predict(sim_rcr_expanded, batch_size=config['keras_batch_size'])
    prob_backlobe = model.predict(data_backlobe_expanded, batch_size=config['keras_batch_size'])

    rcr_efficiency = (np.sum(prob_rcr > output_cut_value) / len(prob_rcr)) * 100
    backlobe_efficiency = (np.sum(prob_backlobe > output_cut_value) / len(prob_backlobe)) * 100

    print(f'RCR efficiency: {rcr_efficiency:.2f}%')
    print(f'Backlobe efficiency: {backlobe_efficiency:.4f}%')
    print(f'Lengths: RCR {len(prob_rcr)}, Backlobe {len(prob_backlobe)}')

    return prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency


# --- Plotting Network Output Histogram ---
def plot_network_output_histogram(prob_rcr, prob_backlobe, rcr_efficiency,
                                  backlobe_efficiency, config, timestamp, learning_rate, model_type):
    """
    Plots the histogram of network outputs for RCR and Backlobe events.

    Args:
        prob_rcr (np.ndarray): Network output probabilities for RCR.
        prob_backlobe (np.ndarray): Network output probabilities for Backlobe.
        rcr_efficiency (float): Calculated RCR efficiency.
        backlobe_efficiency (float): Calculated Backlobe efficiency.
        config (dict): Configuration dictionary.
        timestamp (str): Timestamp for filename.
        learning_rate (float): Learning rate used for training.
        model_type (str): Type of model being plotted.
    """
    amp = config['amp']
    prefix = config['prefix']
    output_cut_value = config['output_cut_value']
    lr_str = f"lr{learning_rate:.0e}".replace('-', '')
    plot_path = config['base_plot_path']
    domain_label = config.get('domain_label', 'time')
    domain_suffix = config.get('domain_suffix', '')
    os.makedirs(plot_path, exist_ok=True)

    dense_val = False
    plt.figure(figsize=(8, 6))

    if "autoencoder" in model_type:
        # If autoencoder, renormalize prob values from [0, max_of_both] to [0, 1]
        all_probs = np.concatenate([prob_rcr, prob_backlobe])
        max_prob = np.max(all_probs)
        prob_rcr = prob_rcr / max_prob
        prob_backlobe = prob_backlobe / max_prob


    plt.hist(prob_backlobe, bins=20, range=(0, 1), histtype='step', color='blue', linestyle='solid', label=f'Backlobe {len(prob_backlobe)}', density=dense_val)
    plt.hist(prob_rcr, bins=20, range=(0, 1), histtype='step', color='red', linestyle='solid', label=f'RCR {len(prob_rcr)}', density=dense_val)

    plt.xlabel('Network Output', fontsize=18)
    plt.ylabel('Number of Events', fontsize=18)
    plt.yscale('log')
    plt.title(f'{amp}_{domain_label} {model_type} RCR-Backlobe network output (LR={learning_rate:.0e})', fontsize=14)
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
    plt.yticks(fontsize=18)

    hist_values_bl, _ = np.histogram(prob_backlobe, bins=20, range=(0, 1))
    hist_values_rcr, _ = np.histogram(prob_rcr, bins=20, range=(0, 1))
    max_overall_hist = max(np.max(hist_values_bl), np.max(hist_values_rcr))

    plt.ylim(0, max(10 ** (np.ceil(np.log10(max_overall_hist * 1.1))), 10))
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(loc='upper left', fontsize=12)

    ax = plt.gca()
    ax.text(0.25, 0.75, f'RCR efficiency: {rcr_efficiency:.2f}%', fontsize=12, transform=ax.transAxes)
    ax.text(0.25, 0.70, f'Backlobe efficiency: {backlobe_efficiency:.4f}%', fontsize=12, transform=ax.transAxes)
    # ax.text(0.25, 0.65, f'TrainCut: {train_cut}', fontsize=12, transform=ax.transAxes)
    ax.text(0.25, 0.60, f'LR: {learning_rate:.0e}', fontsize=12, transform=ax.transAxes)
    ax.text(0.25, 0.55, f'Model: {model_type}', fontsize=12, transform=ax.transAxes)
    plt.axvline(x=output_cut_value, color='y', label='cut', linestyle='--')
    ax.annotate('BL', xy=(0.0, -0.1), xycoords='axes fraction', ha='left', va='center', fontsize=12, color='blue')
    ax.annotate('RCR', xy=(1.0, -0.1), xycoords='axes fraction', ha='right', va='center', fontsize=12, color='red')
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.9)

    hist_file = os.path.join(plot_path, f'{timestamp}_{amp}_{model_type}_network_output_{prefix}_{lr_str}{domain_suffix}.png')
    print(f'saving {hist_file}')
    plt.savefig(hist_file)
    plt.close()


# --- Additional Check Functions from refactor_checks.py ---
def load_2016_backlobe_templates(file_paths, amp_type='200s'):
    """
    Load 2016 backlobe templates from file paths.
    
    Args:
        file_paths (list): List of file paths to load.
        amp_type (str): Amplifier type ('200s' or '100s').
        
    Returns:
        tuple: (arrays, metadata_dict)
    """
    station_groups = {
        '200s': [14, 17, 19, 30],
        '100s': [13, 15, 18]
    }

    allowed_stations = station_groups.get(amp_type, [])
    arrays = []
    metadata = {}

    for path in file_paths:
        match = re.search(r'filtered_Event2016_Stn(\d+)_(\d+\.\d+)_Chi(\d+\.\d+)_SNR(\d+\.\d+)\.npy', path)
        if match:
            print(f'found match {match}')
            station_id = match.group(1)
            unix_timestamp = match.group(2)
            chi = match.group(3)
            snr = match.group(4)

            if int(station_id) in allowed_stations:
                arr = np.load(path)
                arrays.append(arr)
                index = len(arrays) - 1

                plot_filename = f"Event2016_Stn{station_id}_{unix_timestamp}_Chi{chi}_SNR{snr}.png"

                metadata[index] = {
                    "station": station_id,
                    "chi": chi,
                    "snr": snr,
                    "trace": arr,
                    "plot_filename": plot_filename
                }

    return np.stack(arrays, axis=0), metadata


def load_new_coincidence_data(pkl_path, passing_event_ids, special_event_id=None, special_station_id=None):
    """
    Load coincidence data from the new PKL file and separate into passing and raw events.
    
    Parameters
    ----------
    pkl_path : str
        Path to the PKL file containing coincidence data.
    passing_event_ids : list
        List of event IDs that pass cuts.
    special_event_id : int, optional
        Event ID to separate out for special analysis.
    special_station_id : int, optional
        Station ID to separate out for special analysis (must be combined with special_event_id).
    
    Returns
    -------
    passing_traces : np.ndarray
        Traces from events that pass cuts, shape (n_events, n_channels, n_samples).
    raw_traces : np.ndarray
        Traces from all other events, shape (n_events, n_channels, n_samples).
    special_traces : np.ndarray
        Traces from the special event/station combination.
    passing_metadata : dict
        Metadata for passing traces.
    raw_metadata : dict
        Metadata for raw traces.
    special_metadata : dict
        Metadata for special traces.
    """
    with open(pkl_path, "rb") as f:
        coinc_dict = pickle.load(f)
    
    passing_traces = []
    raw_traces = []
    special_traces = []
    passing_metadata = {}
    raw_metadata = {}
    special_metadata = {}
    passing_idx = 0
    raw_idx = 0
    special_idx = 0
    
    print(f"Loading coincidence data from {pkl_path}")
    print(f"Passing event IDs: {passing_event_ids}")
    if special_event_id and special_station_id:
        print(f"Special event: ID={special_event_id}, Station={special_station_id}")
    
    # Iterate through all events in the dictionary
    for event_id, event_data in coinc_dict.items():
        # Check if this event passes cuts
        is_passing = event_id in passing_event_ids
        
        # Get all stations for this event
        if 'stations' in event_data:
            for station_id, station_data in event_data['stations'].items():
                # Get traces for this station
                if 'Traces' in station_data:
                    traces = station_data['Traces']
                    if traces is not None and len(traces) > 0:
                        traces = np.array(traces)
                        
                        # If traces is a single trace (4, n_samples), wrap it in a list
                        if traces.ndim == 2:
                            traces = [traces]
                        
                        # Add each trace
                        for trace in traces:
                            # Check if this is the special event/station combination
                            is_special = (special_event_id is not None and 
                                         special_station_id is not None and
                                         event_id == special_event_id and 
                                         station_id == special_station_id)
                            
                            if is_special:
                                special_traces.append(trace)
                                special_metadata[special_idx] = {
                                    'event_id': event_id,
                                    'station_id': station_id,
                                }
                                special_idx += 1
                            elif is_passing:
                                passing_traces.append(trace)
                                passing_metadata[passing_idx] = {
                                    'event_id': event_id,
                                    'station_id': station_id,
                                }
                                passing_idx += 1
                            else:
                                raw_traces.append(trace)
                                raw_metadata[raw_idx] = {
                                    'event_id': event_id,
                                    'station_id': station_id,
                                }
                                raw_idx += 1
    
    # Convert to numpy arrays
    passing_traces = np.stack(passing_traces, axis=0) if passing_traces else np.array([])
    raw_traces = np.stack(raw_traces, axis=0) if raw_traces else np.array([])
    special_traces = np.stack(special_traces, axis=0) if special_traces else np.array([])
    
    print(f"Loaded {len(passing_traces)} traces from {len(passing_event_ids)} passing events")
    print(f"Loaded {len(raw_traces)} traces from raw coincidence events")
    if len(special_traces) > 0:
        print(f"Loaded {len(special_traces)} special traces (Event {special_event_id}, Station {special_station_id})")
    
    return passing_traces, raw_traces, special_traces, passing_metadata, raw_metadata, special_metadata


def plot_check_histogram(prob_2016, prob_passing, prob_raw, prob_special, amp, timestamp, prefix, 
                         learning_rate, model_type, config):
    """
    Plot histogram comparing 2016 backlobes, passing cuts events, raw coincidence events, and special suspected RCR.
    
    Args:
        prob_2016 (np.ndarray): Network output for 2016 backlobe events.
        prob_passing (np.ndarray): Network output for events passing cuts.
        prob_raw (np.ndarray): Network output for raw coincidence events.
        prob_special (np.ndarray): Network output for special suspected RCR event.
        amp (str): Amplifier type.
        timestamp (str): Timestamp for filename.
        prefix (str): Prefix for filename.
        learning_rate (float): Learning rate used for training.
        model_type (str): Type of model.
        config (dict): Configuration dictionary.
    """
    plt.figure(figsize=(8, 6))
    bins = 20
    range_vals = (0, 1)

    plt.hist(prob_2016, bins=bins, range=range_vals, histtype='step', color='orange', linestyle='solid',
             label=f'2016-Backlobes {len(prob_2016)}', density=False)
    plt.hist(prob_passing, bins=bins, range=range_vals, histtype='step', color='green', linestyle='solid',
             label=f'Passing cuts {len(prob_passing)}', density=False)
    plt.hist(prob_raw, bins=bins, range=range_vals, histtype='step', color='black', linestyle='solid',
             label=f'Raw coincidence {len(prob_raw)}', density=False)
    
    # Plot special suspected RCR if it exists
    if len(prob_special) > 0:
        mean_output = np.mean(prob_special)
        plt.hist(prob_special, bins=bins, range=range_vals, histtype='step', color='red', linestyle='solid',
                 linewidth=2, label=f'Suspected RCR (Evt 11230, Stn 13) - Output: {mean_output:.3f}', density=False)

    plt.xlabel('Network Output', fontsize=18)
    plt.ylabel('Number of Events', fontsize=18)
    plt.yscale('log')

    hist_values_2016, _ = np.histogram(prob_2016, bins=20, range=(0, 1))
    hist_values_passing, _ = np.histogram(prob_passing, bins=20, range=(0, 1))
    hist_values_raw, _ = np.histogram(prob_raw, bins=20, range=(0, 1))
    max_overall_hist = max(np.max(hist_values_2016), np.max(hist_values_passing), np.max(hist_values_raw))
    if len(prob_special) > 0:
        hist_values_special, _ = np.histogram(prob_special, bins=20, range=(0, 1))
        max_overall_hist = max(max_overall_hist, np.max(hist_values_special))
    plt.ylim(7*1e-1, max(10 ** (np.ceil(np.log10(max_overall_hist * 1.1))), 10))

    domain_label = config.get('domain_label', 'time')
    domain_suffix = config.get('domain_suffix', '')

    plt.title(f'{amp}-{domain_label} 2016 BL and Coincidence Events Network Output', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)

    lr_str = f"lr{learning_rate:.0e}".replace('-', '')
    filename = f'{timestamp}_{amp}_{model_type}_check_histogram_{prefix}_{lr_str}{domain_suffix}.png'
    out_path = os.path.join(config['base_plot_path'], filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.savefig(out_path)
    print(f"Saved check histogram to {out_path}")
    plt.close()


# --- NEW: Activation Visualization Function ---

def plot_layer_activations(model, event_trace_original, model_type, save_path):
    """
    Visualizes the activations of convolutional layers for a single event.

    Plots the 4 input channels and heatmaps of activations for each conv layer.

    Args:
        model (keras.Model): The trained Keras model.
        event_trace_original (np.ndarray): The event trace, shape (4, n_samples).
        model_type (str): The type of model ('1d_cnn', 'astrid_2d', etc.).
        save_path (str): Full path to save the output plot.
    """
    domain_desc = 'Frequency Domain' if model_type.endswith('_freq') else 'Time Domain'
    print(f"Generating activation map for model {model.name} ({domain_desc})...")

    event_array = np.asarray(event_trace_original)
    event_for_plot = np.abs(event_array) if np.iscomplexobj(event_array) else event_array
    
    # --- 1. Prepare input for the model ---
    if model_type.startswith('astrid_2d'):
        # Expected shape: (1, 4, samples, 1)
        event_for_model = event_array[np.newaxis, ..., np.newaxis]
        layer_keyword = 'conv2d'
    else:
        # Expected shape: (1, samples, 4)
        event_for_model = event_array.transpose(1, 0)
        event_for_model = event_for_model[np.newaxis, ...]
        layer_keyword = 'conv1d'

    # --- 2. Create activation model ---
    layer_outputs = []
    layer_names = []
    for layer in model.layers:
        # Find conv layers, but ignore the parallel branches in 'Parallel_Strided_CNN'
        # as their outputs are not the same length
        if 'parallel_strided' in model.name.lower() and 'conv1d_' in layer.name.lower():
            print(f'Skipping parallel branch layer: {layer.name}')
            continue
             
        if layer_keyword in layer.name.lower():
            layer_outputs.append(layer.output)
            layer_names.append(layer.name)
    
    if not layer_outputs:
        print(f"No '{layer_keyword}' layers found in model {model.name} to visualize.")
        return

    activation_model = Model(inputs=model.input, outputs=layer_outputs)

    # --- 3. Get activations ---
    try:
        activations = activation_model.predict(event_for_model)
        if not isinstance(activations, list):
            activations = [activations]
    except Exception as e:
        print(f"Error during model prediction for activations: {e}")
        print(f"Model input shape: {model.input.shape}, Data shape: {event_for_model.shape}")
        quit(1)
        return

    num_layers = len(activations)
    
    # --- 4. Plot ---
    fig, axes = plt.subplots(4 + num_layers, 1, figsize=(20, 4 + 3 * num_layers), sharex=True)
    fig.suptitle(f'Layer Activations for {model.name} ({model_type}, {domain_desc}) on Special Event', fontsize=16)

    sequence_length = event_for_plot.shape[1]
    sample_indices = np.arange(sequence_length)

    # Plot Traces (Axes 0-3)
    for i in range(4):
        axes[i].plot(sample_indices, event_for_plot[i, :], color='black', linewidth=1)
        axes[i].set_ylabel(f'Channel {i}\n(Amplitude)')
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].set_xlim(0, sequence_length - 1)

    # Plot Heatmaps (Axes 4...)
    for i in range(num_layers):
        act = activations[i][0] # Get batch 0
        ax = axes[4 + i]
        
        # Squeeze out any dimensions of size 1 (e.g., from Conv2D height)
        act = np.squeeze(act) 
        
        if act.ndim != 2:
            print(f"Warning: Activation for layer {layer_names[i]} has unexpected shape {act.shape}. Trying to reduce.")
            # Try to average over first dimension if 3D
            if act.ndim == 3:
                act = np.mean(act, axis=0)
            else:
                ax.set_ylabel(f'{layer_names[i]}\n(Error: Shape {act.shape})')
                continue
        
        # We want (filters, seq_len). Current shape is (seq_len, filters).
        act_heatmap = act.T
        
        # Get sequence length for extent
        seq_len = act_heatmap.shape[1]
        
        # We set extent so the x-axis matches the trace plot's samples
        # Note: seq_len might not be equal to the input length due to padding/stride
        x_start = 0
        x_end = sequence_length - 1
        
        # If sequence length is different, we can't perfectly align.
        # For simplicity, we'll stretch the heatmap to align with the input length.
        # A more complex approach would involve calculating sample mapping.
        im = ax.imshow(act_heatmap, aspect='auto', interpolation='nearest', cmap='viridis', 
                       extent=[x_start, x_end, -0.5, act_heatmap.shape[0] - 0.5])
        
        ax.set_ylabel(f'{layer_names[i]}\n(Filters: {act_heatmap.shape[0]})')
        fig.colorbar(im, ax=ax, orientation='vertical', pad=0.01)

    axes[-1].set_xlabel('Frequency Bin' if model_type.endswith('_freq') else 'Time Sample')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    try:
        plt.savefig(save_path)
        print(f"Saved activation map to {save_path}")
    except Exception as e:
        print(f"Error saving activation plot: {e}")
        
    plt.close(fig)


def main(enable_sim_bl_814):
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train CNN model with specified learning rate and model type.')
    parser.add_argument('--learning_rate', type=float, required=True, 
                        help='Learning rate for the Adam optimizer (e.g., 0.001, 0.0001)')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=list(MODEL_TYPES.keys()),
                        help=f'Type of model to train. Choose from: {list(MODEL_TYPES.keys())}')
    args = parser.parse_args()
    
    learning_rate = args.learning_rate
    model_type = args.model_type
    lr_str = f"lr{learning_rate:.0e}".replace('-', '')
    
    config = load_config()
    amp = config['amp']
    prefix = config['prefix']
    use_filtering = bool(config.get('use_filtering', False))
    config['use_filtering'] = use_filtering

    is_freq_model = model_type.endswith('_freq')
    config['is_freq_model'] = is_freq_model
    config['frequency_sampling_rate'] = float(config.get('frequency_sampling_rate', 2.0))
    config['domain_label'] = 'freq' if is_freq_model else 'time'
    config['domain_suffix'] = '_freq' if is_freq_model else ''

    base_model_root = config['base_model_path']
    base_plot_root = config['base_plot_path']

    # Create learning rate and model type specific directories
    lr_folder = f'lr_{learning_rate:.0e}'.replace('-', '')
    model_folder = model_type
    timestamp = datetime.now().strftime('%m.%d.%y_%H-%M')
    config['base_model_path'] = os.path.join(base_model_root, model_folder, lr_folder)
    config['base_plot_path'] = os.path.join(base_plot_root, f"{timestamp}", model_folder, lr_folder)

    print(f"Starting CNN training at {timestamp} for {amp} amplifier")
    print(f"Model type: {model_type}, Learning rate: {learning_rate}")
    print(f"Training domain: {config['domain_label']}")
    
    # Data load & prep
    data = load_and_prep_data_for_training(config)
    training_rcr_original = data['training_rcr']
    training_backlobe_original = data['training_backlobe']

    # Keep copies of original training data before augmentation
    # (used for evaluation to ensure histogram proportions are correct)
    training_rcr = training_rcr_original.copy()
    training_backlobe = training_backlobe_original.copy()

    # Apply channel cycling to RCR training data to augment the dataset
    print(f"\n--- Applying channel cycling to RCR training data ---")
    print(f"Original RCR training shape: {training_rcr.shape}")
    # training_rcr has shape [n_events, 4, n_samples], so channel_axis=1
    training_rcr = cycle_channels(training_rcr, channel_axis=1)
    print(f"Augmented RCR training shape: {training_rcr.shape}")
    print(f"RCR training data multiplied by factor of 7\n")

    # Apply channel cycling to Backlobe training data to augment the dataset
    print(f"\n--- Applying channel cycling to Backlobe training data ---")
    print(f"Original Backlobe training shape: {training_backlobe.shape}")
    # training_backlobe has shape [n_events, 4, n_samples], so channel_axis=1
    training_backlobe = cycle_channels(training_backlobe, channel_axis=1)
    print(f"Augmented Backlobe training shape: {training_backlobe.shape}")
    print(f"Backlobe training data multiplied by factor of 7\n")

    sim_rcr_all = data['sim_rcr_all']
    data_backlobe_traces_rcr_all = data['data_backlobe_tracesRCR']
    
    # Train model
    model, history, requires_transpose = train_cnn_model(training_rcr, training_backlobe, config, learning_rate, model_type)
    print('------> Training is Done!')

    # Save model
    domain_suffix = config['domain_suffix']
    model_filename = f'{timestamp}_{amp}_{model_type}_model_{prefix}_{lr_str}{domain_suffix}.h5'
    model_save_path = os.path.join(config['base_model_path'], model_filename)
    model.save(model_save_path)
    print(f'Model saved to: {model_save_path}')

    # Save training history & plots
    save_and_plot_training_history(history, config['base_model_path'], config['base_plot_path'], 
                                   timestamp, amp, config, learning_rate, model_type)

    # Evaluate & plot network output histogram ON RCR-like TRACES!
    prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency = \
        evaluate_model_performance(model, sim_rcr_all, data_backlobe_traces_rcr_all, 
                                 config['output_cut_value'], config, model_type, requires_transpose)

    plot_network_output_histogram(prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency, 
                                 config, timestamp, learning_rate, model_type)

    # --- Additional Checks: Load and evaluate on 2016 backlobes and coincidence events ---
    print("\n--- Running additional checks on 2016 backlobes and coincidence events ---")
    
    # Load 2016 backlobe templates
    template_dir = "/pub/tangch3/ARIANNA/DeepLearning/refactor/confirmed_2016_templates/"
    template_paths = sorted(glob(os.path.join(template_dir, "filtered_Event2016_Stn*.npy")))
    
    if os.path.exists(template_dir) and template_paths:
        all_2016_backlobes, dict_2016 = load_2016_backlobe_templates(template_paths, amp_type=amp)
        print(f"Loaded {len(all_2016_backlobes)} 2016 backlobe traces.")
        
        # Load new coincidence data
        pkl_path = "/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/9.24.25_CoincidenceDatetimes_passing_cuts_with_all_params_recalcZenAzi_calcPol.pkl"
        
        # Event IDs that pass cuts
        passing_event_ids = [3047, 3432, 10195, 10231, 10273, 10284, 10444, 10449, 10466, 10471, 10554, 11197, 11220, 11230, 11236, 11243]
        
        # Special event/station to check separately (suspected RCR)
        special_event_id = 11230
        special_station_id = 13
        
        if os.path.exists(pkl_path):
            passing_traces, raw_traces, special_traces, passing_metadata, raw_metadata, special_metadata = \
                load_new_coincidence_data(pkl_path, passing_event_ids, special_event_id, special_station_id)
            
            special_trace_to_plot = special_traces[0].copy() if len(special_traces) > 0 else None

            if len(passing_traces) > 0 and len(raw_traces) > 0:
                # Ensure proper shape for model prediction
                all_2016_backlobes = np.array(all_2016_backlobes)
                passing_traces = np.array(passing_traces)
                raw_traces = np.array(raw_traces)
                special_traces = np.array(special_traces) if len(special_traces) > 0 else special_traces

                sampling_rate = float(config.get('frequency_sampling_rate', 2.0))
                domain_suffix_local = config.get('domain_suffix', '')

                if is_freq_model:
                    all_2016_backlobes = _compute_frequency_magnitude(all_2016_backlobes, sampling_rate)
                    passing_traces = _compute_frequency_magnitude(passing_traces, sampling_rate)
                    raw_traces = _compute_frequency_magnitude(raw_traces, sampling_rate)
                    if len(special_traces) > 0:
                        special_traces = _compute_frequency_magnitude(special_traces, sampling_rate)
                    if special_trace_to_plot is not None:
                        special_trace_to_plot = _compute_frequency_magnitude(special_trace_to_plot[np.newaxis, ...], sampling_rate)[0]

                    if use_filtering:
                        print('Applying frequency edge filtering to coincidence validation sets.')
                        all_2016_backlobes = _apply_frequency_edge_filter(all_2016_backlobes)
                        passing_traces = _apply_frequency_edge_filter(passing_traces)
                        raw_traces = _apply_frequency_edge_filter(raw_traces)
                        if len(special_traces) > 0:
                            special_traces = _apply_frequency_edge_filter(special_traces)
                        if special_trace_to_plot is not None:
                            special_trace_to_plot = _apply_frequency_edge_filter(special_trace_to_plot)


                print(f"2016 backlobes shape: {all_2016_backlobes.shape}")
                print(f"Passing cuts traces shape: {passing_traces.shape}")
                print(f"Raw coincidence traces shape: {raw_traces.shape}")
                if len(special_traces) > 0:
                    print(f"Special traces shape: {special_traces.shape}")
                
                # Prepare data based on model requirements
                if requires_transpose:
                    all_2016_backlobes_prepped = all_2016_backlobes.transpose(0, 2, 1)
                    passing_traces_prepped = passing_traces.transpose(0, 2, 1)
                    raw_traces_prepped = raw_traces.transpose(0, 2, 1)

                    prob_2016_backlobe = model.predict(all_2016_backlobes_prepped)
                    prob_passing = model.predict(passing_traces_prepped)
                    prob_raw = model.predict(raw_traces_prepped)

                    if len(special_traces) > 0:
                        special_traces_prepped = special_traces.transpose(0, 2, 1)
                        prob_special = model.predict(special_traces_prepped)
                    else:
                        prob_special = np.array([])
                else:
                    if all_2016_backlobes.ndim == 3:
                        all_2016_backlobes = all_2016_backlobes[..., np.newaxis]
                        print(f'Changed 2016 backlobes to shape {all_2016_backlobes.shape}')
                    if passing_traces.ndim == 3:
                        passing_traces = passing_traces[..., np.newaxis]
                        print(f'Changed passing traces to shape {passing_traces.shape}')
                    if raw_traces.ndim == 3:
                        raw_traces = raw_traces[..., np.newaxis]
                        print(f'Changed raw traces to shape {raw_traces.shape}')
                    if len(special_traces) > 0 and special_traces.ndim == 3:
                        special_traces = special_traces[..., np.newaxis]
                        print(f'Changed special traces to shape {special_traces.shape}')

                    prob_2016_backlobe = model.predict(all_2016_backlobes)
                    prob_passing = model.predict(passing_traces)
                    prob_raw = model.predict(raw_traces)
                    prob_special = model.predict(special_traces) if len(special_traces) > 0 else np.array([])
                
                # Flatten probabilities
                prob_2016_backlobe = prob_2016_backlobe.flatten()
                prob_passing = prob_passing.flatten()
                prob_raw = prob_raw.flatten()
                prob_special = prob_special.flatten() if len(prob_special) > 0 else np.array([])
                
                print(f'Mean network output - 2016 BL: {np.mean(prob_2016_backlobe):.3f}, Passing cuts: {np.mean(prob_passing):.3f}, Raw: {np.mean(prob_raw):.3f}')
                if len(prob_special) > 0:
                    print(f'Mean network output - Suspected RCR (Evt 11230, Stn 13): {np.mean(prob_special):.3f}')
                
                # Plot the check histogram
                plot_check_histogram(prob_2016_backlobe, prob_passing, prob_raw, prob_special,
                                   amp, timestamp, prefix, learning_rate, model_type, config)
                                   
                # --- ADDED: Plot layer activations for the first special event ---
                if special_trace_to_plot is not None:
                    print(f"\n--- Generating activation plot for special event ---")
                    plot_save_dir = os.path.join(config['base_plot_path'], 'activation_maps')
                    os.makedirs(plot_save_dir, exist_ok=True)
                    plot_save_path = os.path.join(plot_save_dir, f'{timestamp}_{amp}_{model_type}_special_event_activations{domain_suffix_local}.png')
                    
                    plot_layer_activations(model, special_trace_to_plot, model_type, plot_save_path)
                else:
                    print(f"\n--- No special event trace found, skipping activation plot ---")
                    quit(1)

            else:
                print(f"Warning: No traces loaded from coincidence data")
                quit(1)
        else:
            print(f"Warning: Coincidence PKL file not found at {pkl_path}")
            quit(1)
    else:
        print(f"Warning: 2016 backlobe template directory not found at {template_dir}")
        quit(1)

    # --- ADDED: Plot layer activations for max output events from training sets ---
    print("\n--- Generating activation plots for max output training events ---")
    domain_suffix_local = config.get('domain_suffix', '')
    plot_save_dir = os.path.join(config['base_plot_path'], 'activation_maps')
    os.makedirs(plot_save_dir, exist_ok=True)
    
    # Get network predictions on the ORIGINAL training sets (before augmentation)
    # training_rcr_original and training_backlobe_original have shape (n_events, 4, n_samples)
    if requires_transpose:
        training_rcr_for_pred = training_rcr_original.transpose(0, 2, 1)
        training_backlobe_for_pred = training_backlobe_original.transpose(0, 2, 1)
    else:
        training_rcr_for_pred = training_rcr_original[..., np.newaxis]
        training_backlobe_for_pred = training_backlobe_original[..., np.newaxis]
    
    print(f"Predicting on RCR training set ({training_rcr_original.shape[0]} events)...")
    prob_training_rcr = model.predict(training_rcr_for_pred, batch_size=config['keras_batch_size']).flatten()
    
    print(f"Predicting on Backlobe training set ({training_backlobe_original.shape[0]} events)...")
    prob_training_backlobe = model.predict(training_backlobe_for_pred, batch_size=config['keras_batch_size']).flatten()
    
    # Find indices of maximum network output
    max_rcr_idx = np.argmax(prob_training_rcr)
    max_backlobe_idx = np.argmin(prob_training_backlobe)
    
    print(f"Max RCR training output: {prob_training_rcr[max_rcr_idx]:.4f} at index {max_rcr_idx}")
    print(f"Max Backlobe training output: {prob_training_backlobe[max_backlobe_idx]:.4f} at index {max_backlobe_idx}")
    
    # Get the traces (shape: (4, n_samples))
    max_rcr_trace = training_rcr_original[max_rcr_idx]
    max_backlobe_trace = training_backlobe_original[max_backlobe_idx]
    
    # Generate activation plots
    rcr_activation_path = os.path.join(plot_save_dir, f'{timestamp}_{amp}_{model_type}_max_rcr_training_activations{domain_suffix_local}.png')
    print(f"Generating activation plot for max RCR training event (index {max_rcr_idx})...")
    plot_layer_activations(model, max_rcr_trace, model_type, rcr_activation_path)
    
    backlobe_activation_path = os.path.join(plot_save_dir, f'{timestamp}_{amp}_{model_type}_max_backlobe_training_activations{domain_suffix_local}.png')
    print(f"Generating activation plot for max Backlobe training event (index {max_backlobe_idx})...")
    plot_layer_activations(model, max_backlobe_trace, model_type, backlobe_activation_path)


    # Plotting individual traces if needed 
    # indices = np.where(prob_backlobe.flatten() > config['output_cut_value'])[0]
    # for index in indices:
    #     plot_traces_save_path = os.path.join(config['base_plot_path'], 'traces', f'{timestamp}_plot_pot_rcr_{amp}_{index}.png')
    #     pT(data['data_backlobe_tracesRCR'][index], f'Backlobe Trace {index} (Output > {config["output_cut_value"]:.2f})', plot_traces_save_path)
    #     print(f"Saved trace plot for Backlobe event {index} to {plot_traces_save_path}")
         
    print(f"Script finished successfully. Completion for {prefix} with model {model_type} and learning rate {learning_rate}")

if __name__ == "__main__":
    main(enable_sim_bl_814=False)
