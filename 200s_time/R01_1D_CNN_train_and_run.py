import os
import pickle
import argparse
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Dense, BatchNormalization, ReLU, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from NuRadioReco.utilities import units
from A0_Utilities import load_sim_rcr, load_data, pT, load_config

# --- Data Loading and Preparation ---
def load_and_prep_data_for_training(config):
    """
    Loads sim RCR and data Backlobe, selects random subset for training

    Args:
        config (dict): Configuration dictionary.

    Returns:
        dict: A dictionary containing prepared data arrays and indices.
    """
    amp = config['amp']
    train_cut = config['train_cut']
    # Already determined in main based on amp, so directly use 'station_ids'
    station_ids = config['station_ids']
    sim_folder = os.path.join(config['base_sim_rcr_folder'], amp, config['sim_rcr_date'])

    print(f"Loading data for amplifier type: {amp}")

    sim_rcr = load_sim_rcr(sim_folder, noise_enabled=config['noise_enabled'], filter_enabled=True, amp=amp) # currently loads 5.28.25 from config
    # add 8.14.25 sim RCR
    if amp == '200s': 
        sim_rcr_814 = np.load(f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/8.14.25/200s/all_traces_200s_RCR_part0_4473events.npy') 
        print('Loading /dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/8.14.25/200s/all_traces_200s_RCR_part0_4473events.npy')
    elif amp == '100s':
        sim_rcr_814 = np.load(f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/8.14.25/100s/all_traces_100s_RCR_part0_4200events.npy')
        print('Loading /dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/8.14.25/100s/all_traces_100s_RCR_part0_4200events.npy') 
    else:
        print(f'amp not found')
    sim_rcr = np.vstack([sim_rcr, sim_rcr_814])

    backlobe_data = {'snr2016': [], 'snrRCR': [], 'chi2016': [], 'chiRCR': [], 'traces2016': [], 'tracesRCR': [], 'unix2016': [], 'unixRCR': []}
    for s_id in station_ids:
        snr2016, snrRCR, chi2016, chiRCR, traces2016, tracesRCR, unix2016, unixRCR = load_data(config, amp_type=amp, station_id=s_id)
        backlobe_data['snr2016'].extend(snr2016)
        backlobe_data['snrRCR'].extend(snrRCR)
        backlobe_data['chi2016'].extend(chi2016)
        backlobe_data['chiRCR'].extend(chiRCR)
        backlobe_data['traces2016'].extend(traces2016)
        backlobe_data['tracesRCR'].extend(tracesRCR)
        backlobe_data['unix2016'].extend(unix2016)
        backlobe_data['unixRCR'].extend(unixRCR)

    sim_rcr = np.array(sim_rcr)
    backlobe_traces_2016 = np.array(backlobe_data['traces2016'])
    backlobe_traces_rcr = np.array(backlobe_data['tracesRCR'])

    # pick random subsets for training
    rcr_training_indices = np.random.choice(sim_rcr.shape[0], size=train_cut, replace=False)
    bl_training_indices = np.random.choice(backlobe_traces_2016.shape[0], size=train_cut, replace=False)

    training_rcr = sim_rcr[rcr_training_indices, :]
    training_backlobe = backlobe_traces_2016[bl_training_indices, :]

    rcr_non_training_indices = np.setdiff1d(np.arange(sim_rcr.shape[0]), rcr_training_indices)
    bl_non_training_indices = np.setdiff1d(np.arange(backlobe_traces_2016.shape[0]), bl_training_indices)

    print(f'RCR shape: {sim_rcr.shape}, Backlobe shape: {backlobe_traces_2016.shape}')
    print(f'Training shape RCR {training_rcr.shape}, Training Shape Backlobe {training_backlobe.shape}, TrainCut {train_cut}')
    print(f'Non-training RCR count {len(rcr_non_training_indices)}, Non-training Backlobe count {len(bl_non_training_indices)}')

    return {
        'training_rcr': training_rcr,
        'training_backlobe': training_backlobe,
        'sim_rcr_all': sim_rcr,
        'data_backlobe_traces2016': backlobe_traces_2016,
        'data_backlobe_tracesRCR': backlobe_traces_rcr,
        'data_backlobe_unix2016_all': np.array(backlobe_data['unix2016']),
        'data_backlobe_chi2016_all': np.array(backlobe_data['chi2016']),
        'rcr_non_training_indices': rcr_non_training_indices,
        'bl_non_training_indices': bl_non_training_indices
    } 


# --- CNN Model ---
def build_cnn_model(n_channels, n_samples, learning_rate):
    """
    Builds and compiles the 1D CNN model architecture.

    Args:
        n_channels (int): Number of input channels.
        n_samples (int): Number of samples per trace.
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        keras.Model: The compiled Keras model.
    """
    model = Sequential()
    
    # Multi-scale idea approximated with stacked Conv1D layers
    model.add(Conv1D(32, kernel_size=5, padding="same", activation="relu", input_shape=(n_samples, n_channels)))
    model.add(Conv1D(32, kernel_size=15, padding="same", activation="relu"))
    model.add(Conv1D(32, kernel_size=31, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(ReLU())
    
    # Downstream feature extractor
    model.add(Conv1D(64, kernel_size=7, padding="same", activation="relu"))
    
    # Collapse across time
    model.add(GlobalAveragePooling1D())
    
    # Dense classification head
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_cnn_model(training_rcr, training_backlobe, config, learning_rate):
    """
    Trains the 1D CNN model.

    Args:
        training_rcr (np.ndarray): RCR training data.
        training_backlobe (np.ndarray): Backlobe training data.
        config (dict): Configuration dictionary.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        tuple: (keras.Model, keras.callbacks.History) The trained model and training history.
    """
    x = np.vstack((training_rcr, training_backlobe))
    # Transpose from (n_events, 4, 256) to (n_events, 256, 4) for Conv1D
    x = x.transpose(0, 2, 1)
    n_samples = x.shape[1]
    n_channels = x.shape[2]

    y = np.vstack((np.ones((training_rcr.shape[0], 1)), np.zeros((training_backlobe.shape[0], 1)))) # 1s for RCR (signal)
    s = np.arange(x.shape[0])
    np.random.shuffle(s)
    x = x[s]
    y = y[s]
    print(f"Training data shape: {x.shape}, label shape: {y.shape}")

    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=config['early_stopping_patience'])]

    model = build_cnn_model(n_channels, n_samples, learning_rate)
    model.summary()

    history = model.fit(x, y,
                        validation_split=0.25,
                        epochs=config['keras_epochs'],
                        batch_size=config['keras_batch_size'],
                        verbose=config['verbose_fit'],
                        callbacks=callbacks_list)

    return model, history


def save_and_plot_training_history(history, model_path, plot_path, timestamp, amp, config, learning_rate):
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
    """
    prefix = config['prefix']
    lr_str = f"lr{learning_rate:.0e}".replace('-', '')

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)

    history_file = os.path.join(model_path, f'{timestamp}_{amp}_history_{prefix}_{lr_str}.pkl')
    with open(history_file, 'wb') as f: 
        pickle.dump(history.history, f)
    print(f'Training history saved to: {history_file}')

    # Plot loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training vs Validation Loss (LR={learning_rate:.0e})')
    plt.legend()
    loss_plot_file = os.path.join(plot_path, f'{timestamp}_{amp}_loss_{prefix}_{lr_str}.png')
    plt.savefig(loss_plot_file)
    plt.close()
    print(f'Loss plot saved to: {loss_plot_file}')

    # Plot accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Training vs Validation Accuracy (LR={learning_rate:.0e})')
    plt.legend()
    accuracy_plot_file = os.path.join(plot_path, f'{timestamp}_{amp}_accuracy_{prefix}_{lr_str}.png')
    plt.savefig(accuracy_plot_file)
    plt.close()
    print(f'Accuracy plot saved to: {accuracy_plot_file}')


# --- Model Evaluation ---
def evaluate_model_performance(model, sim_rcr_all, data_backlobe_traces_rcr_all, output_cut_value, config):
    """
    Evaluates the model on above curve Backlobe in RCR template.

    Args:
        model (keras.Model): The trained Keras model.
        sim_rcr_all (np.ndarray): All RCR simulation data.
        data_backlobe_traces_rcr_all (np.ndarray): All Backlobe data (TracesRCR).
        output_cut_value (float): The threshold for classification.

    Returns:
        tuple: (prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency)
    """
    # Transpose from (n_events, 4, 256) to (n_events, 256, 4) for Conv1D
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
                                  backlobe_efficiency, config, timestamp, learning_rate):
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
    plt.title(f'{amp}_time RCR-Backlobe network output (LR={learning_rate:.0e})', fontsize=14)
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
    plt.axvline(x=output_cut_value, color='y', label='cut', linestyle='--')
    ax.annotate('BL', xy=(0.0, -0.1), xycoords='axes fraction', ha='left', va='center', fontsize=12, color='blue')
    ax.annotate('RCR', xy=(1.0, -0.1), xycoords='axes fraction', ha='right', va='center', fontsize=12, color='red')
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.9)

    hist_file = os.path.join(plot_path, f'{timestamp}_{amp}_network_output_{prefix}_{lr_str}.png')
    print(f'saving {hist_file}')
    plt.savefig(hist_file)
    plt.close()


def main(enable_sim_bl_814):
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train CNN model with specified learning rate.')
    parser.add_argument('--learning_rate', type=float, required=True, 
                        help='Learning rate for the Adam optimizer (e.g., 0.001, 0.0001)')
    args = parser.parse_args()
    
    learning_rate = args.learning_rate
    lr_str = f"lr{learning_rate:.0e}".replace('-', '')
    
    config = load_config()
    amp = config['amp']
    prefix = config['prefix']

    # Create learning rate specific directories
    lr_folder = f'lr_{learning_rate:.0e}'.replace('-', '')
    config['base_model_path'] = os.path.join(config['base_model_path'], lr_folder)
    config['base_plot_path'] = os.path.join(config['base_plot_path'], lr_folder)

    timestamp = datetime.now().strftime('%m.%d.%y_%H-%M')
    print(f"Starting CNN training at {timestamp} for {amp} amplifier with learning rate {learning_rate}.")
    
    # Data load & prep
    data = load_and_prep_data_for_training(config)
    training_rcr = data['training_rcr']
    training_backlobe = data['training_backlobe']
    if enable_sim_bl_814:
        training_backlobe = np.load(f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedBacklobe/8.14.25/200s/all_traces_200s_part0_11239events.npy')

    sim_rcr_all = data['sim_rcr_all']
    data_backlobe_traces_rcr_all = data['data_backlobe_tracesRCR']
    
    # Train model
    model, history = train_cnn_model(training_rcr, training_backlobe, config, learning_rate)
    print('------> Training is Done!')

    # Save model
    model_save_path = os.path.join(config['base_model_path'], f'{timestamp}_{amp}_model_{prefix}_{lr_str}.h5')
    model.save(model_save_path)
    print(f'Model saved to: {model_save_path}')

    # Save training history & plots
    save_and_plot_training_history(history, config['base_model_path'], config['base_plot_path'], timestamp, amp, config, learning_rate)

    # Evaluate & plot network output histogram ON RCR-like TRACES!
    prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency = \
        evaluate_model_performance(model, sim_rcr_all, data_backlobe_traces_rcr_all, config['output_cut_value'], config)

    plot_network_output_histogram(prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency, config, timestamp, learning_rate)

    # Plotting individual traces if needed 
    # indices = np.where(prob_backlobe.flatten() > config['output_cut_value'])[0]
    # for index in indices:
    #     plot_traces_save_path = os.path.join(config['base_plot_path'], 'traces', f'{timestamp}_plot_pot_rcr_{amp}_{index}.png')
    #     pT(data['data_backlobe_tracesRCR'][index], f'Backlobe Trace {index} (Output > {config["output_cut_value"]:.2f})', plot_traces_save_path)
    #     print(f"Saved trace plot for Backlobe event {index} to {plot_traces_save_path}")
         
    print(f"Script finished successfully. Completion for {prefix} with learning rate {learning_rate}")

if __name__ == "__main__":
    main(enable_sim_bl_814=False)