import os
import re
import time
import pickle
import numpy as np
from glob import glob
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# --- Custom utilities ---
from NuRadioReco.utilities import units
from A0_Utilities import load_sim_rcr, load_data, pT, load_config
from refactor_train_and_run import (
    load_and_prep_data_for_training,
    evaluate_model_performance,
    plot_network_output_histogram,
    save_and_plot_training_history
)

# 1D SIMPLE CNN TRAINING


config = load_config()
amp = config['amp']

data = load_and_prep_data_for_training(config)
training_rcr = data['sim_rcr_all']
training_backlobe = data['data_backlobe_traces2016']

x = np.vstack((training_rcr, training_backlobe))
y = np.vstack((np.ones((training_rcr.shape[0], 1)), np.zeros((training_backlobe.shape[0], 1))))  # 1 for RCR

# Shuffle
s = np.arange(x.shape[0])
np.random.shuffle(s)
x = x[s]
y = y[s]

# Keras expects (samples, length, channels)
x = x.transpose(0, 2, 1)

model = Sequential([
    Conv1D(20, kernel_size=4, padding="same", activation="relu", input_shape=(256, 4)),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

timestamp = datetime.now().strftime('%m.%d.%y_%H-%M')
print(f"Starting simple CNN training at {timestamp} for {amp} amplifier.")

history = model.fit(
    x, y,
    validation_split=0.2,
    epochs=config['keras_epochs'],
    batch_size=config['keras_batch_size']
)

# Save model
sim_rcr_all = data['sim_rcr_all']
data_backlobe_traces_rcr_all = data['data_backlobe_tracesRCR']
prefix = config['prefix']

sim_rcr_expanded = sim_rcr_all.transpose(0, 2, 1)
data_backlobe_expanded = data_backlobe_traces_rcr_all.transpose(0, 2, 1)

model_save_path = os.path.join(
    config['base_model_path'],
    config['model_filename_template'].format(timestamp=timestamp, amp=amp, prefix=prefix)
)
model.save(model_save_path)
print(f'Model saved to: {model_save_path}')

save_and_plot_training_history(history, config['base_model_path'], config['base_plot_path'], timestamp, amp, config)

# NETWORK OUTPUT CHECK

def load_most_recent_model(base_model_path, amp, if_dann=False, model_prefix=None):
    pattern = re.compile(r"(\d{2}\.\d{2}\.\d{2}_\d{2}-\d{2})_.*\.h5")
    now = time.time()
    best_file = None
    smallest_diff = float('inf')
    best_timestamp = None

    for fname in os.listdir(base_model_path):
        if not fname.endswith(".h5"):
            continue
        if model_prefix and model_prefix not in fname:
            continue
        match = pattern.search(fname)
        if match:
            timestamp = match.group(1)
            model_time = datetime.strptime(timestamp, '%m.%d.%y_%H-%M').timestamp()
            diff = now - model_time
            if 0 <= diff < smallest_diff:
                smallest_diff = diff
                best_file = fname
                best_timestamp = timestamp

    if best_file:
        model_path = os.path.join(base_model_path, best_file)
        print(f"Loading most recent model: {model_path}")
        return keras.models.load_model(model_path), best_timestamp, 'CNN_checks'
    else:
        raise FileNotFoundError(f"No suitable model file found in {base_model_path}.")

def load_2016_backlobe_templates(file_paths, amp_type='200s'):
    station_groups = {'200s': [14, 17, 19, 30], '100s': [13, 15, 18]}
    allowed_stations = station_groups.get(amp_type, [])
    arrays = []
    metadata = {}
    for path in file_paths:
        match = re.search(r'filtered_Event2016_Stn(\d+)_(\d+\.\d+)_Chi(\d+\.\d+)_SNR(\d+\.\d+)\.npy', path)
        if match and int(match.group(1)) in allowed_stations:
            arr = np.load(path)
            arrays.append(arr)
            metadata[len(arrays) - 1] = {"station": match.group(1), "trace": arr}
    return np.stack(arrays, axis=0), metadata

def load_all_coincidence_traces(pkl_path, trace_key):
    with open(pkl_path, "rb") as f:
        coinc_dict = pickle.load(f)
    all_traces = []
    metadata = {}
    idx = 0
    for master_id, master_data in coinc_dict.items():
        for station_id, station_dict in master_data['stations'].items():
            traces = station_dict.get(trace_key)
            if traces is None or len(traces) == 0:
                continue
            traces = np.array(traces)
            for i in range(len(traces)):
                all_traces.append(traces[i])
                metadata[idx] = {'master_id': master_id, 'station_id': station_id}
                idx += 1
    return coinc_dict, np.stack(all_traces, axis=0), metadata

def plot_histogram(prob_2016, prob_coincidence, prob_coincidence_rcr, amp, timestamp, prefix):
    plt.figure(figsize=(8, 6))
    plt.hist(prob_2016, bins=20, range=(0,1), histtype='step', color='orange', label=f'2016-Backlobes {len(prob_2016)}')
    plt.hist(prob_coincidence, bins=20, range=(0,1), histtype='step', color='black', label=f'Coincidence {len(prob_coincidence)}')
    plt.xlabel('Network Output'); plt.ylabel('Events'); plt.yscale('log')
    plt.text(0.02, 0.85, f'Coinc RCR Output: {prob_coincidence_rcr.item():.2f}', transform=plt.gca().transAxes)
    plt.legend(); plt.title(f'{amp} 2016 BL and Coincidence')
    out_path = os.path.join(config['base_plot_path'], 'network_output', f'{timestamp}_{amp}_{prefix}_hist.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved histogram to {out_path}")
    plt.close()

# --- Run check ---
model, model_timestamp, prefix = load_most_recent_model(config['base_model_path'], amp, if_dann=False, model_prefix="CNN")

template_dir = "/pub/tangch3/ARIANNA/DeepLearning/refactor/confirmed_2016_templates/"
template_paths = sorted(glob(os.path.join(template_dir, "filtered_Event2016_Stn*.npy")))
all_2016_backlobes, _ = load_2016_backlobe_templates(template_paths, amp_type=amp)

pkl_path = "/pub/tangch3/ARIANNA/DeepLearning/refactor/other/test_bandpass_on_coinc/filtered_coinc.pkl"
_, all_coincidence_events, metadata = load_all_coincidence_traces(pkl_path, "Filtered_Traces")

# reshape for 1D CNN
all_2016_backlobes = all_2016_backlobes.transpose(0, 2, 1)
all_coincidence_events = all_coincidence_events.transpose(0, 2, 1)

print('Running 1D CNN evaluation...')
prob_backlobe = model.predict(all_2016_backlobes)
prob_coincidence = model.predict(all_coincidence_events)
coinc_rcr = all_coincidence_events[1297:1298]
prob_coincidence_rcr = model.predict(coinc_rcr)

plot_histogram(prob_backlobe.flatten(), prob_coincidence.flatten(), prob_coincidence_rcr.flatten(),
               amp, timestamp=model_timestamp, prefix=prefix)

print("Training and network output check complete.")
