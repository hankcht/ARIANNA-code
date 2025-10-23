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
from NuRadioReco.utilities import units
from A0_Utilities import load_sim_rcr, load_data, pT, load_config

# Add parent directory to path to import model_builder and data_channel_cycling
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_builder import (
    build_cnn_model,
    build_1d_model,
    build_parallel_model,
    build_strided_model,
    build_parallel_strided_model
)
from data_channel_cycling import cycle_channels


def load_combined_backlobe_data(combined_pkl_path):
    """
    Load the combined backlobe data saved by refactor_converter.py.
    
    This function loads a pickle file containing backlobe data from all stations
    that has been processed through chi and bin cuts.
    
    Args:
        combined_pkl_path (str): Path to the combined pickle file containing all stations' data.
                                Default location from refactor_converter.py:
                                '/dfs8/sbarwick_lab/ariannaproject/tangch3/station_data/above_curve_data/5000evt_10.17.25/above_curve_combined.pkl'
    
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

    print(f"Loading data for amplifier type: {amp}")

    # Load simulation RCR data
    sim_folder = os.path.join(config['base_sim_rcr_folder'], amp, config['sim_rcr_date'])
    sim_rcr = load_sim_rcr(sim_folder, noise_enabled=config['noise_enabled'], filter_enabled=True, amp=amp)
    
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
    
    snr2016, snrRCR, chi2016, chiRCR, traces2016, tracesRCR, unix2016, unixRCR = load_combined_backlobe_data(combined_pkl_path)

    # Convert to numpy arrays (they should already be arrays from the pickle, but ensure consistency)
    backlobe_traces_2016 = np.array(traces2016)
    backlobe_traces_rcr = np.array(tracesRCR)

    print(f'RCR shape: {sim_rcr.shape}, Backlobe 2016 shape: {backlobe_traces_2016.shape}, Backlobe RCR shape: {backlobe_traces_rcr.shape}')

    # Pick random subsets for training
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


# --- Model Selection and Building ---
MODEL_TYPES = {
    '1d_cnn': build_1d_model,
    'parallel': build_parallel_model,
    'strided': build_strided_model,
    'parallel_strided': build_parallel_strided_model,
    'astrid_2d': build_cnn_model
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
        tuple: (keras.Model, keras.callbacks.History) The trained model and training history.
    """
    model_builder = get_model_builder(model_type)
    
    # Prepare data based on model type
    if model_type == 'astrid_2d':
        # For 2D CNN: shape should be (n_events, 4, 256, 1)
        x = np.vstack((training_rcr, training_backlobe))
        # Add channel dimension for 2D conv
        x = np.expand_dims(x, axis=-1)  # (n_events, 4, 256, 1)
        input_shape = (x.shape[1], x.shape[2], 1)  # (4, 256, 1)
    else:
        # For 1D CNNs: transpose from (n_events, 4, 256) to (n_events, 256, 4)
        x = np.vstack((training_rcr, training_backlobe))
        x = x.transpose(0, 2, 1)
        input_shape = (x.shape[1], x.shape[2])  # (256, 4)
    
    y = np.vstack((np.ones((training_rcr.shape[0], 1)), np.zeros((training_backlobe.shape[0], 1)))) # 1s for RCR (signal)
    s = np.arange(x.shape[0])
    np.random.shuffle(s)
    x = x[s]
    y = y[s]
    print(f"Training data shape: {x.shape}, label shape: {y.shape}, input_shape: {input_shape}")

    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=config['early_stopping_patience'])]

    model = model_builder(input_shape=input_shape, learning_rate=learning_rate)
    model.summary()

    history = model.fit(x, y,
                        validation_split=0.25,
                        epochs=config['keras_epochs'],
                        batch_size=config['keras_batch_size'],
                        verbose=config['verbose_fit'],
                        callbacks=callbacks_list)

    return model, history


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

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)

    history_file = os.path.join(model_path, f'{timestamp}_{amp}_{model_type}_history_{prefix}_{lr_str}.pkl')
    with open(history_file, 'wb') as f: 
        pickle.dump(history.history, f)
    print(f'Training history saved to: {history_file}')

    # Plot loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_type} Training vs Validation Loss (LR={learning_rate:.0e})')
    plt.legend()
    loss_plot_file = os.path.join(plot_path, f'{timestamp}_{amp}_{model_type}_loss_{prefix}_{lr_str}.png')
    plt.savefig(loss_plot_file)
    plt.close()
    print(f'Loss plot saved to: {loss_plot_file}')

    # Plot accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_type} Training vs Validation Accuracy (LR={learning_rate:.0e})')
    plt.legend()
    accuracy_plot_file = os.path.join(plot_path, f'{timestamp}_{amp}_{model_type}_accuracy_{prefix}_{lr_str}.png')
    plt.savefig(accuracy_plot_file)
    plt.close()
    print(f'Accuracy plot saved to: {accuracy_plot_file}')


# --- Model Evaluation ---
def evaluate_model_performance(model, sim_rcr_all, data_backlobe_traces_rcr_all, output_cut_value, config, model_type):
    """
    Evaluates the model on above curve Backlobe in RCR template.

    Args:
        model (keras.Model): The trained Keras model.
        sim_rcr_all (np.ndarray): All RCR simulation data.
        data_backlobe_traces_rcr_all (np.ndarray): All Backlobe data (TracesRCR).
        output_cut_value (float): The threshold for classification.
        model_type (str): Type of model being evaluated.

    Returns:
        tuple: (prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency)
    """
    # Prepare data based on model type
    if model_type == 'astrid_2d':
        # For 2D CNN: add channel dimension (n_events, 4, 256, 1)
        sim_rcr_expanded = np.expand_dims(sim_rcr_all, axis=-1)
        data_backlobe_expanded = np.expand_dims(data_backlobe_traces_rcr_all, axis=-1)
    else:
        # For 1D CNNs: transpose from (n_events, 4, 256) to (n_events, 256, 4)
        sim_rcr_expanded = sim_rcr_all.transpose(0, 2, 1)
        data_backlobe_expanded = data_backlobe_traces_rcr_all.transpose(0, 2, 1)

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
    train_cut = config['train_cut']
    lr_str = f"lr{learning_rate:.0e}".replace('-', '')
    plot_path = config['base_plot_path']
    os.makedirs(plot_path, exist_ok=True)

    dense_val = False
    plt.figure(figsize=(8, 6))

    plt.hist(prob_backlobe, bins=20, range=(0, 1), histtype='step', color='blue', linestyle='solid', label=f'Backlobe {len(prob_backlobe)}', density=dense_val)
    plt.hist(prob_rcr, bins=20, range=(0, 1), histtype='step', color='red', linestyle='solid', label=f'RCR {len(prob_rcr)}', density=dense_val)

    plt.xlabel('Network Output', fontsize=18)
    plt.ylabel('Number of Events', fontsize=18)
    plt.yscale('log')
    plt.title(f'{amp}_time {model_type} RCR-Backlobe network output (LR={learning_rate:.0e})', fontsize=14)
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
    ax.text(0.25, 0.65, f'TrainCut: {train_cut}', fontsize=12, transform=ax.transAxes)
    ax.text(0.25, 0.60, f'LR: {learning_rate:.0e}', fontsize=12, transform=ax.transAxes)
    ax.text(0.25, 0.55, f'Model: {model_type}', fontsize=12, transform=ax.transAxes)
    plt.axvline(x=output_cut_value, color='y', label='cut', linestyle='--')
    ax.annotate('BL', xy=(0.0, -0.1), xycoords='axes fraction', ha='left', va='center', fontsize=12, color='blue')
    ax.annotate('RCR', xy=(1.0, -0.1), xycoords='axes fraction', ha='right', va='center', fontsize=12, color='red')
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.9)

    hist_file = os.path.join(plot_path, f'{timestamp}_{amp}_{model_type}_network_output_{prefix}_{lr_str}.png')
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


def load_all_coincidence_traces(pkl_path, trace_key):
    """
    Load coincidence traces from a PKL file.

    Parameters
    ----------
    pkl_path : str
        Path to the PKL file.
    trace_key : str
        Key in each station dictionary from which to load traces. "Traces" or "Filtered_Traces".

    Returns
    -------
    coinc_dict : dict
        The full coincidence dictionary.
    X : np.ndarray
        Stacked traces of shape (n_events, n_channels, n_samples).
    metadata : dict
        Mapping of global trace index to metadata.
    """
    with open(pkl_path, "rb") as f:
        coinc_dict = pickle.load(f)

    all_traces = []
    metadata = {}
    idx = 0

    print(f'loading {trace_key}')

    for master_id, master_data in coinc_dict.items():
        for station_id, station_dict in master_data['stations'].items():
            traces = station_dict.get(trace_key)
            if traces is None or len(traces) == 0:
                continue
            traces = np.array(traces)
            n_traces = len(traces)

            for i in range(n_traces):
                all_traces.append(traces[i])
                metadata[idx] = {
                    'master_id': master_id,
                    'station_id': station_id,
                    'index': station_dict['indices'][i],
                    'event_id': station_dict['event_ids'][i],
                    'SNR': station_dict['SNR'][i],
                    'ChiRCR': station_dict['ChiRCR'][i],
                    'Chi2016': station_dict['Chi2016'][i],
                    'ChiBad': station_dict['ChiBad'][i],
                    'Zen': station_dict['Zen'][i],
                    'Azi': station_dict['Azi'][i],
                    'Times': station_dict['Times'][i],
                    'PolAngle': station_dict['PolAngle'][i],
                    'PolAngleErr': station_dict['PolAngleErr'][i],
                    'ExpectedPolAngle': station_dict['ExpectedPolAngle'][i],
                }
                idx += 1

    all_Traces = np.stack(all_traces, axis=0)
    return coinc_dict, all_Traces, metadata


def plot_check_histogram(prob_2016, prob_coincidence, prob_coincidence_rcr, amp, timestamp, prefix, 
                         learning_rate, model_type, config):
    """
    Plot histogram comparing 2016 backlobes and coincidence events.
    
    Args:
        prob_2016 (np.ndarray): Network output for 2016 backlobe events.
        prob_coincidence (np.ndarray): Network output for coincidence events.
        prob_coincidence_rcr (np.ndarray): Network output for coincidence RCR trace.
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
    plt.hist(prob_coincidence, bins=bins, range=range_vals, histtype='step', color='black', linestyle='solid',
             label=f'Coincidence-Events {len(prob_coincidence)}', density=False)

    plt.xlabel('Network Output', fontsize=18)
    plt.ylabel('Number of Events', fontsize=18)
    plt.yscale('log')

    hist_values_2016, _ = np.histogram(prob_2016, bins=20, range=(0, 1))
    hist_values_coincidence, _ = np.histogram(prob_coincidence, bins=20, range=(0, 1))
    max_overall_hist = max(np.max(hist_values_2016), np.max(hist_values_coincidence))
    plt.ylim(7*1e-1, max(10 ** (np.ceil(np.log10(max_overall_hist * 1.1))), 10))

    plt.text(0.00, 0.85, f'Coincidence RCR network Output is: {prob_coincidence_rcr.item():.2f}',
             fontsize=12, verticalalignment='top', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
    plt.title(f'{amp}-time 2016 BL and Coincidence Events Network Output', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)

    lr_str = f"lr{learning_rate:.0e}".replace('-', '')
    filename = f'{timestamp}_{amp}_{model_type}_check_histogram_{prefix}_{lr_str}.png'
    out_path = os.path.join(config['base_plot_path'], filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.savefig(out_path)
    print(f"Saved check histogram to {out_path}")
    plt.close()


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

    # Create learning rate and model type specific directories
    lr_folder = f'lr_{learning_rate:.0e}'.replace('-', '')
    model_folder = model_type
    timestamp = datetime.now().strftime('%m.%d.%y_%H-%M')
    config['base_model_path'] = os.path.join(config['base_model_path'], model_folder, lr_folder)
    config['base_plot_path'] = os.path.join(config['base_plot_path'], f"{timestamp}", model_folder, lr_folder)

    print(f"Starting CNN training at {timestamp} for {amp} amplifier")
    print(f"Model type: {model_type}, Learning rate: {learning_rate}")
    
    # Data load & prep
    data = load_and_prep_data_for_training(config)
    training_rcr = data['training_rcr']
    training_backlobe = data['training_backlobe']

    # Apply channel cycling to RCR training data to augment the dataset
    print(f"\n--- Applying channel cycling to RCR training data ---")
    print(f"Original RCR training shape: {training_rcr.shape}")
    # training_rcr has shape [n_events, 4, 256], so channel_axis=1
    training_rcr = cycle_channels(training_rcr, channel_axis=1)
    print(f"Augmented RCR training shape: {training_rcr.shape}")
    print(f"RCR training data multiplied by factor of 7\n")

    # Apply channel cycling to Backlobe training data to augment the dataset
    print(f"\n--- Applying channel cycling to Backlobe training data ---")
    print(f"Original Backlobe training shape: {training_backlobe.shape}")
    # training_backlobe has shape [n_events, 4, 256], so channel_axis=1
    training_backlobe = cycle_channels(training_backlobe, channel_axis=1)
    print(f"Augmented Backlobe training shape: {training_backlobe.shape}")
    print(f"Backlobe training data multiplied by factor of 7\n")

    sim_rcr_all = data['sim_rcr_all']
    data_backlobe_traces_rcr_all = data['data_backlobe_tracesRCR']
    
    # Train model
    model, history = train_cnn_model(training_rcr, training_backlobe, config, learning_rate, model_type)
    print('------> Training is Done!')

    # Save model
    model_save_path = os.path.join(config['base_model_path'], f'{timestamp}_{amp}_{model_type}_model_{prefix}_{lr_str}.h5')
    model.save(model_save_path)
    print(f'Model saved to: {model_save_path}')

    # Save training history & plots
    save_and_plot_training_history(history, config['base_model_path'], config['base_plot_path'], 
                                   timestamp, amp, config, learning_rate, model_type)

    # Evaluate & plot network output histogram ON RCR-like TRACES!
    prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency = \
        evaluate_model_performance(model, sim_rcr_all, data_backlobe_traces_rcr_all, 
                                 config['output_cut_value'], config, model_type)

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
        
        # Load coincidence events
        pkl_path = "/pub/tangch3/ARIANNA/DeepLearning/refactor/other/test_bandpass_on_coinc/filtered_coinc.pkl"
        
        if os.path.exists(pkl_path):
            coinc_dict, all_coincidence_events, metadata = load_all_coincidence_traces(pkl_path, "Filtered_Traces")
            print(f"Loaded {len(all_coincidence_events)} coincidence traces.")
            
            # Ensure proper shape for model prediction
            all_2016_backlobes = np.array(all_2016_backlobes)
            all_coincidence_events = np.array(all_coincidence_events)
            print(f"2016 backlobes shape: {all_2016_backlobes.shape}")
            print(f"Coincidence events shape: {all_coincidence_events.shape}")
            
            # Prepare data based on model type
            if model_type == 'astrid_2d':
                # For 2D CNN: add channel dimension if needed
                if all_2016_backlobes.ndim == 3:
                    all_2016_backlobes = all_2016_backlobes[..., np.newaxis]
                    print(f'Changed 2016 backlobes to shape {all_2016_backlobes.shape}')
                if all_coincidence_events.ndim == 3:
                    all_coincidence_events = all_coincidence_events[..., np.newaxis]
                    print(f'Changed coincidence events to shape {all_coincidence_events.shape}')
                
                # Predict
                prob_2016_backlobe = model.predict(all_2016_backlobes)
                prob_coincidence = model.predict(all_coincidence_events)
                
                # Get specific coincidence RCR trace (index 1297)
                coinc_rcr_idx = 1297
                if coinc_rcr_idx < len(all_coincidence_events):
                    prob_coincidence_rcr = model.predict(np.expand_dims(all_coincidence_events[coinc_rcr_idx], axis=0))
                else:
                    prob_coincidence_rcr = np.array([0.0])
                    print(f"Warning: coinc_rcr_idx {coinc_rcr_idx} out of bounds")
            else:
                # For 1D CNNs: transpose from (n_events, 4, 256) to (n_events, 256, 4)
                all_2016_backlobes_transpose = all_2016_backlobes.transpose(0, 2, 1)
                all_coincidence_events_transpose = all_coincidence_events.transpose(0, 2, 1)
                
                # Predict
                prob_2016_backlobe = model.predict(all_2016_backlobes_transpose)
                prob_coincidence = model.predict(all_coincidence_events_transpose)
                
                # Get specific coincidence RCR trace (index 1297)
                coinc_rcr_idx = 1297
                if coinc_rcr_idx < len(all_coincidence_events):
                    coinc_rcr = all_coincidence_events[coinc_rcr_idx]
                    coinc_rcr_transpose = coinc_rcr.transpose(1, 0)
                    prob_coincidence_rcr = model.predict(np.expand_dims(coinc_rcr_transpose, axis=0))
                else:
                    prob_coincidence_rcr = np.array([0.0])
                    print(f"Warning: coinc_rcr_idx {coinc_rcr_idx} out of bounds")
            
            # Flatten probabilities
            prob_2016_backlobe = prob_2016_backlobe.flatten()
            prob_coincidence = prob_coincidence.flatten()
            prob_coincidence_rcr = prob_coincidence_rcr.flatten()
            
            print(f'Coincidence RCR network output: {prob_coincidence_rcr}')
            
            # Plot the check histogram
            plot_check_histogram(prob_2016_backlobe, prob_coincidence, prob_coincidence_rcr, 
                               amp, timestamp, prefix, learning_rate, model_type, config)
        else:
            print(f"Warning: Coincidence PKL file not found at {pkl_path}")
    else:
        print(f"Warning: 2016 backlobe template directory not found at {template_dir}")

    # Plotting individual traces if needed 
    # indices = np.where(prob_backlobe.flatten() > config['output_cut_value'])[0]
    # for index in indices:
    #     plot_traces_save_path = os.path.join(config['base_plot_path'], 'traces', f'{timestamp}_plot_pot_rcr_{amp}_{index}.png')
    #     pT(data['data_backlobe_tracesRCR'][index], f'Backlobe Trace {index} (Output > {config["output_cut_value"]:.2f})', plot_traces_save_path)
    #     print(f"Saved trace plot for Backlobe event {index} to {plot_traces_save_path}")
         
    print(f"Script finished successfully. Completion for {prefix} with model {model_type} and learning rate {learning_rate}")

if __name__ == "__main__":
    main(enable_sim_bl_814=False)