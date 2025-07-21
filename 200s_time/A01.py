# A1_TrainCNN.py

# Standard library imports
import os
import pickle
from datetime import datetime

# Third-party imports
import numpy as np
import matplotlib

# Set backend before importing pyplot to avoid GUI issues
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential

# Local application/library specific imports
from NuRadioReco.utilities import units
from A0_Utilities import load_sim_rcr, load_data, pT


# --- Configuration ---
def get_config():
    """Returns a dictionary of configuration parameters."""
    return {
        'amp': '200s',
        'output_cut_value': 0.6,
        'train_cut': 995,
        'noise_rms_200s': 22.53 * units.mV,
        'noise_rms_100s': 20 * units.mV,
        'station_ids_200s': [14, 17, 19, 30],
        'station_ids_100s': [13, 15, 18],
        'base_sim_folder': '/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/',
        'sim_date': '5.28.25',
        'base_model_path': '/pub/tangch3/ARIANNA/DeepLearning/models/',
        'base_plot_path': '/pub/tangch3/ARIANNA/DeepLearning/plots/A1_Training/',
        'loading_data_type': 'new_chi_above_curve',
        'model_filename_template': '{timestamp}_RCR_Backlobe_model_2Layer.h5',
        'history_filename_template': '{timestamp}_RCR_Backlobe_model_2Layer_history.pkl',
        'loss_plot_filename_template': '{timestamp}_loss_plot_RCR_Backlobe_model_2Layer_{amp}.png',
        'accuracy_plot_filename_template': '{timestamp}_accuracy_plot_RCR_Backlobe_model_2Layer_{amp}.png',
        'histogram_filename_template': '{timestamp}_{amp}_histogram.png',
        'verbose_fit': 1,
        'keras_epochs': 100,
        'keras_batch_size': 32,
        'early_stopping_patience': 2
    }


# --- Data Loading and Preparation ---
def load_and_prepare_data(config):
    """
    Loads simulation and experimental data, and prepares it for training/testing.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        dict: A dictionary containing prepared data arrays and indices.
    """
    amp = config['amp']
    train_cut = config['train_cut']
    # Already determined in main based on amp, so directly use 'station_ids'
    station_ids = config['station_ids']
    sim_folder = os.path.join(config['base_sim_folder'], amp, config['sim_date'])

    print(f"Loading data for amplifier type: {amp}")

    sim_rcr = load_sim_rcr(sim_folder, noise_enabled=False,
                           filter_enabled=True, amp=amp)

    backlobe_data = {
        'snr': [], 'chi2016': [], 'chiRCR': [],
        'traces2016': [], 'tracesRCR': [], 'unix': []
    }
    for s_id in station_ids:
        snr, chi2016, chiRCR, traces2016, tracesRCR, unix = load_data(
            config['loading_data_type'], amp_type=amp, station_id=s_id
        )
        backlobe_data['snr'].extend(snr)
        backlobe_data['chi2016'].extend(chi2016)
        backlobe_data['chiRCR'].extend(chiRCR)
        backlobe_data['traces2016'].extend(traces2016)
        backlobe_data['tracesRCR'].extend(tracesRCR)
        backlobe_data['unix'].extend(unix)

    sim_rcr = np.array(sim_rcr)
    backlobe_traces_2016 = np.array(backlobe_data['traces2016'])
    backlobe_traces_rcr = np.array(backlobe_data['tracesRCR'])

    # pick random subsets for training
    rcr_training_indices = np.random.choice(
        sim_rcr.shape[0], size=train_cut, replace=False)
    bl_training_indices = np.random.choice(
        backlobe_traces_2016.shape[0], size=train_cut, replace=False)

    training_rcr = sim_rcr[rcr_training_indices, :]
    training_backlobe = backlobe_traces_2016[bl_training_indices, :]

    rcr_non_training_indices = np.setdiff1d(
        np.arange(sim_rcr.shape[0]), rcr_training_indices)
    bl_non_training_indices = np.setdiff1d(
        np.arange(backlobe_traces_2016.shape[0]), bl_training_indices)

    print(f'RCR shape: {sim_rcr.shape}, Backlobe shape: {backlobe_traces_2016.shape}')
    print(f'Training shape RCR {training_rcr.shape}, '
          f'Training Shape Backlobe {training_backlobe.shape}, TrainCut {train_cut}')
    print(f'Non-training RCR count {len(rcr_non_training_indices)}, '
          f'Non-training Backlobe count {len(bl_non_training_indices)}')

    return {
        'training_rcr': training_rcr,
        'training_backlobe': training_backlobe,
        'sim_rcr_all': sim_rcr,
        'data_backlobe_all': backlobe_traces_2016,
        'data_backlobe_tracesRCR_all': backlobe_traces_rcr,
        'data_backlobe_unix_all': np.array(backlobe_data['unix']),
        'data_backlobe_chi2016_all': np.array(backlobe_data['chi2016']),
        'rcr_non_training_indices': rcr_non_training_indices,
        'bl_non_training_indices': bl_non_training_indices
    }


# --- CNN Model ---
def build_cnn_model(n_channels, n_samples):
    """
    Builds and compiles the CNN model architecture.

    Args:
        n_channels (int): Number of input channels.
        n_samples (int): Number of samples per trace.

    Returns:
        keras.Model: The compiled Keras model.
    """
    model = Sequential()
    model.add(Conv2D(20, (4, 10), activation='relu', input_shape=(n_channels, n_samples, 1), groups=1))
    model.add(Conv2D(10, (1, 10), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_cnn_model(training_rcr, training_backlobe, config):
    """
    Trains the CNN model.

    Args:
        training_rcr (np.ndarray): RCR training data.
        training_backlobe (np.ndarray): Backlobe training data.
        config (dict): Configuration dictionary.

    Returns:
        tuple: (keras.Model, keras.callbacks.History) The trained model and training history.
    """
    x = np.vstack((training_rcr, training_backlobe))
    n_samples = x.shape[2]
    n_channels = x.shape[1]
    x = np.expand_dims(x, axis=-1)

    y = np.vstack((np.ones((training_rcr.shape[0], 1)), np.zeros((training_backlobe.shape[0], 1)))) # 1s for RCR (signal)
    s = np.arange(x.shape[0])
    np.random.shuffle(s)
    x = x[s]
    y = y[s]
    print(f"Training data shape: {x.shape}, label shape: {y.shape}")

    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=config['early_stopping_patience'])]

    model = build_cnn_model(n_channels, n_samples)
    model.summary()

    history = model.fit(x, y,
                        validation_split=0.25,
                        epochs=config['keras_epochs'],
                        batch_size=config['keras_batch_size'],
                        verbose=config['verbose_fit'],
                        callbacks=callbacks_list)

    return model, history


def save_and_plot_training_history(history, model_path, plot_path, timestamp, amp, config):
    """
    Saves the training history and plots loss and accuracy curves.

    Args:
        history (keras.callbacks.History): The history object returned by model.fit.
        model_path (str): Directory to save the history pickle.
        plot_path (str): Directory to save the plots.
        timestamp (str): Timestamp for filenames.
        amp (str): Amplifier type for plot filenames.
        config (dict): Configuration dictionary.
    """
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)

    history_file = os.path.join(
        model_path, config['history_filename_template'].format(timestamp=timestamp))
    with open(history_file, 'wb') as f:
        pickle.dump(history.history, f)
    print(f'Training history saved to: {history_file}')

    # Plot loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    loss_plot_file = os.path.join(plot_path, config['loss_plot_filename_template'].format(timestamp=timestamp, amp=amp))
    plt.savefig(loss_plot_file)
    plt.close()
    print(f'Loss plot saved to: {loss_plot_file}')

    # Plot accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    accuracy_plot_file = os.path.join(plot_path, config['accuracy_plot_filename_template'].format(timestamp=timestamp, amp=amp))
    plt.savefig(accuracy_plot_file)
    plt.close()
    print(f'Accuracy plot saved to: {accuracy_plot_file}')


# --- Model Evaluation ---
def evaluate_model_performance(model, sim_rcr_all, data_backlobe_traces_rcr_all,
                               output_cut_value):
    """
    Evaluates the model on full datasets and calculates efficiencies.

    Args:
        model (keras.Model): The trained Keras model.
        sim_rcr_all (np.ndarray): All RCR simulation data.
        data_backlobe_traces_rcr_all (np.ndarray): All Backlobe data (TracesRCR).
        output_cut_value (float): The threshold for classification.

    Returns:
        tuple: (prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency)
    """
    # Expand dims for prediction if not already done by previous calls
    sim_rcr_expanded = np.expand_dims(sim_rcr_all, axis=-1)
    data_backlobe_expanded = np.expand_dims(data_backlobe_traces_rcr_all, axis=-1)

    prob_rcr = model.predict(sim_rcr_expanded)
    prob_backlobe = model.predict(data_backlobe_expanded)

    rcr_efficiency = (np.sum(prob_rcr > output_cut_value) / len(prob_rcr)) * 100
    backlobe_efficiency = (np.sum(prob_backlobe > output_cut_value) / len(prob_backlobe)) * 100

    print(f'RCR efficiency: {rcr_efficiency:.2f}%')
    print(f'Backlobe efficiency: {backlobe_efficiency:.4f}%')
    print(f'Lengths: RCR {len(prob_rcr)}, Backlobe {len(prob_backlobe)}')

    return prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency


# --- Plotting Network Output Histogram ---
def plot_network_output_histogram(prob_rcr, prob_backlobe, rcr_efficiency,
                                  backlobe_efficiency, config, timestamp):
    """
    Plots the histogram of network outputs for RCR and Backlobe events.

    Args:
        prob_rcr (np.ndarray): Network output probabilities for RCR.
        prob_backlobe (np.ndarray): Network output probabilities for Backlobe.
        rcr_efficiency (float): Calculated RCR efficiency.
        backlobe_efficiency (float): Calculated Backlobe efficiency.
        config (dict): Configuration dictionary.
        timestamp (str): Timestamp for filename.
    """
    amp = config['amp']
    output_cut_value = config['output_cut_value']
    train_cut = config['train_cut']
    plot_path = os.path.join(config['base_plot_path'], 'Network_Output')
    os.makedirs(plot_path, exist_ok=True)

    dense_val = False
    plt.figure(figsize=(8, 6))

    # Split long lines for plt.hist calls
    plt.hist(prob_backlobe, bins=20, range=(0, 1), histtype='step',
             color='blue', linestyle='solid',
             label=f'Backlobe {len(prob_backlobe)}', density=dense_val)
    plt.hist(prob_rcr, bins=20, range=(0, 1), histtype='step',
             color='red', linestyle='solid',
             label=f'RCR {len(prob_rcr)}', density=dense_val)

    plt.xlabel('Network Output', fontsize=18)
    plt.ylabel('Number of Events', fontsize=18)
    plt.yscale('log')
    plt.title(f'{amp}_time RCR-Backlobe network output')
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
    plt.yticks(fontsize=18)

    hist_values_bl, _ = np.histogram(prob_backlobe, bins=20, range=(0, 1))
    hist_values_rcr, _ = np.histogram(prob_rcr, bins=20, range=(0, 1))
    max_overall_hist = max(np.max(hist_values_bl), np.max(hist_values_rcr))

    plt.ylim(0, max(10 ** (np.ceil(np.log10(max_overall_hist * 1.1))), 10))
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(loc='upper left', fontsize=8)

    ax = plt.gca()
    ax.text(0.35, 0.75, f'RCR efficiency: {rcr_efficiency:.2f}%',
            fontsize=12, transform=ax.transAxes)
    ax.text(0.35, 0.70, f'Backlobe efficiency: {backlobe_efficiency:.4f}%',
            fontsize=12, transform=ax.transAxes)
    ax.text(0.35, 0.65, f'TrainCut: {train_cut}',
            fontsize=12, transform=ax.transAxes)
    plt.axvline(x=output_cut_value, color='y', label='cut', linestyle='--')

    # Aligned arguments for better readability and PEP 8
    ax.annotate('BL', xy=(0.0, -0.1), xycoords='axes fraction',
                ha='left', va='center', fontsize=12, color='blue')
    ax.annotate('RCR', xy=(1.0, -0.1), xycoords='axes fraction',
                ha='right', va='center', fontsize=12, color='red')

    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.9)

    hist_file = os.path.join(
        plot_path, config['histogram_filename_template'].format(timestamp=timestamp, amp=amp))
    print(f'saving {hist_file}')
    plt.savefig(hist_file)
    plt.close()
    print(f'------> {amp} Done!')



# --- Main Execution Flow ---
def main():
    """
    Main function to orchestrate the CNN training and evaluation pipeline.
    """
    config = get_config()
    amp = config['amp']

    if amp == '200s':
        config['noise_rms'] = config['noise_rms_200s']
        config['station_ids'] = config['station_ids_200s']
    elif amp == '100s':
        config['noise_rms'] = config['noise_rms_100s']
        config['station_ids'] = config['station_ids_100s']
    else:
        raise ValueError(f"Unsupported amplifier type: {amp}")

    config['model_path'] = os.path.join(
        config['base_model_path'], f"{amp}_time/new_chi")
    config['loss_accuracy_plot_path'] = os.path.join(
        config['base_plot_path'], 'Loss_Accuracy')
    config['network_output_plot_path'] = os.path.join(
        config['base_plot_path'], 'Network_Output')

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(f"Starting CNN training at {timestamp} for {amp} amplifier.")

    data = load_and_prepare_data(config)
    training_rcr = data['training_rcr']
    training_backlobe = data['training_backlobe']
    sim_rcr_all = data['sim_rcr_all']
    data_backlobe_traces_rcr_all = data['data_backlobe_tracesRCR_all']

    model, history = train_cnn_model(
        training_rcr, training_backlobe, config)
    print('------> Training is Done!')

    model_save_path = os.path.join(
        config['model_path'], config['model_filename_template'].format(timestamp=timestamp))
    model.save(model_save_path)
    print(f'Model saved to: {model_save_path}')

    save_and_plot_training_history(
        history, config['model_path'], config['loss_accuracy_plot_path'], timestamp, amp, config)

    prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency = \
        evaluate_model_performance(
            model, sim_rcr_all, data_backlobe_traces_rcr_all, config['output_cut_value']
        )

    plot_network_output_histogram(
        prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency, config, timestamp
    )

    # Example for plotting individual traces if needed (currently commented out in original)
    # indices = np.where(prob_backlobe.flatten() > config['output_cut_value'])[0]
    # for index in indices:
    #     plot_traces_save_path = os.path.join(config['network_output_plot_path'],
    #                                          f'test_new_data_BL_{amp}_{index}.png')
    #     pT(data['data_backlobe_all'][index],
    #        f'Backlobe Trace {index} (Output > {config["output_cut_value"]:.2f})',
    #        plot_traces_save_path)
    #     print(f"Saved trace plot for Backlobe event {index} to {plot_traces_save_path}")

    print("Script finished successfully.")


if __name__ == "__main__":
    main()