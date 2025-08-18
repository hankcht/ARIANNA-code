import os
import pickle
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten
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
    sim_rcr_814 = np.load(f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/8.14.25/200s/all_traces_200s_RCR_part0_4473events.npy') 
    sim_rcr = np.vstack([sim_rcr, sim_rcr_814])

    backlobe_data = {'snr': [], 'chi2016': [], 'chiRCR': [], 'traces2016': [], 'tracesRCR': [], 'unix': []}
    for s_id in station_ids:
        snr, chi2016, chiRCR, traces2016, tracesRCR, unix = load_data(config['loading_data_type'], amp_type=amp, station_id=s_id)
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
    prefix = config['prefix']

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)

    history_file = os.path.join(model_path, config['history_filename_template'].format(timestamp=timestamp, amp=amp, prefix=prefix))
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
    loss_plot_file = os.path.join(plot_path, 'loss', config['loss_plot_filename_template'].format(timestamp=timestamp, amp=amp, prefix=prefix))
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
    accuracy_plot_file = os.path.join(plot_path, 'accuracy', config['accuracy_plot_filename_template'].format(timestamp=timestamp, amp=amp, prefix=prefix))
    plt.savefig(accuracy_plot_file)
    plt.close()
    print(f'Accuracy plot saved to: {accuracy_plot_file}')


# --- Model Evaluation ---
def evaluate_model_performance(model, sim_rcr_all, data_backlobe_traces_rcr_all, output_cut_value):
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
    prefix = config['prefix']
    output_cut_value = config['output_cut_value']
    train_cut = config['train_cut']
    plot_path = os.path.join(config['base_plot_path'], 'network_output')
    os.makedirs(plot_path, exist_ok=True)

    dense_val = False
    plt.figure(figsize=(8, 6))

    plt.hist(prob_backlobe, bins=20, range=(0, 1), histtype='step', color='blue', linestyle='solid', label=f'Backlobe {len(prob_backlobe)}', density=dense_val)
    plt.hist(prob_rcr, bins=20, range=(0, 1), histtype='step', color='red', linestyle='solid', label=f'RCR {len(prob_rcr)}', density=dense_val)

    plt.xlabel('Network Output', fontsize=18)
    plt.ylabel('Number of Events', fontsize=18)
    plt.yscale('log')
    plt.title(f'{amp}_time RCR-Backlobe network output', fontsize=14)
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
    plt.axvline(x=output_cut_value, color='y', label='cut', linestyle='--')
    ax.annotate('BL', xy=(0.0, -0.1), xycoords='axes fraction', ha='left', va='center', fontsize=12, color='blue')
    ax.annotate('RCR', xy=(1.0, -0.1), xycoords='axes fraction', ha='right', va='center', fontsize=12, color='red')
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.9)

    hist_file = os.path.join(plot_path, config['histogram_filename_template'].format(timestamp=timestamp, amp=amp, prefix=prefix))
    print(f'saving {hist_file}')
    plt.savefig(hist_file)
    plt.close()


def main():
    
    config = load_config()
    amp = config['amp']
    prefix = config['prefix']

    # Ensure all paths inside 'refactor' folder
    config['network_output_plot_path'] = os.path.join(config['base_plot_path'], 'Network_Output')

    timestamp = datetime.now().strftime('%m.%d.%y_%H-%M')
    print(f"Starting CNN training at {timestamp} for {amp} amplifier.")

    # Data load & prep
    data = load_and_prep_data_for_training(config)
    training_rcr = data['training_rcr']
    training_backlobe = data['training_backlobe']
    sim_rcr_all = data['sim_rcr_all']
    data_backlobe_traces_rcr_all = data['data_backlobe_tracesRCR']
    
    # Train model
    model, history = train_cnn_model(training_rcr, training_backlobe, config)
    print('------> Training is Done!')

    # Save model
    model_save_path = os.path.join(config['base_model_path'], config['model_filename_template'].format(timestamp=timestamp, amp=amp, prefix=prefix))
    model.save(model_save_path)
    print(f'Model saved to: {model_save_path}')

    # Save training history & plots
    save_and_plot_training_history(
        history, config['base_model_path'], config['base_plot_path'], timestamp, amp, config)

    # Evaluate & plot network output histogram
    prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency = \
        evaluate_model_performance(model, sim_rcr_all, data_backlobe_traces_rcr_all, config['output_cut_value'])

    plot_network_output_histogram(prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency, config, timestamp)

    # Plotting individual traces if needed 
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