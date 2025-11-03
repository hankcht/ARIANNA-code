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
from tensorflow.keras.layers import Conv1D, Dense, GlobalAveragePooling1D, Dropout
from tensorflow.keras.models import Sequential
from NuRadioReco.utilities import units
from A0_Utilities import load_sim_rcr, load_data, pT, load_config


# ============================================================
#                  Data Loading and Preparation
# ============================================================
def load_and_prep_data_for_training(config):
    amp = config['amp']
    train_cut = config['train_cut']
    station_ids = config['station_ids']
    sim_folder = os.path.join(config['base_sim_rcr_folder'], amp, config['sim_rcr_date'])

    print(f"Loading data for amplifier type: {amp}")

    sim_rcr = load_sim_rcr(sim_folder, noise_enabled=config['noise_enabled'], filter_enabled=True, amp=amp)
    if amp == '200s':
        sim_rcr_814 = np.load('/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/8.14.25/200s/all_traces_200s_RCR_part0_4473events.npy')
        print('Loaded additional 8.14.25 simulated RCRs for 200s')
    elif amp == '100s':
        sim_rcr_814 = np.load('/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/8.14.25/100s/all_traces_100s_RCR_part0_4200events.npy')
        print('Loaded additional 8.14.25 simulated RCRs for 100s')
    else:
        sim_rcr_814 = np.zeros_like(sim_rcr[:1])
        print('Unknown amplifier type.')

    sim_rcr = np.vstack([sim_rcr, sim_rcr_814])

    backlobe_data = {'traces2016': [], 'tracesRCR': []}
    for s_id in station_ids:
        _, _, _, _, traces2016, tracesRCR, _, _ = load_data(config, amp_type=amp, station_id=s_id)
        backlobe_data['traces2016'].extend(traces2016)
        backlobe_data['tracesRCR'].extend(tracesRCR)

    sim_rcr = np.array(sim_rcr)
    backlobe_traces_2016 = np.array(backlobe_data['traces2016'])
    backlobe_traces_rcr = np.array(backlobe_data['tracesRCR'])

    # Random subset for training
    rcr_idx = np.random.choice(sim_rcr.shape[0], size=train_cut, replace=False)
    bl_idx = np.random.choice(backlobe_traces_2016.shape[0], size=train_cut, replace=False)
    training_rcr = sim_rcr[rcr_idx]
    training_backlobe = backlobe_traces_2016[bl_idx]

    return {
        'training_rcr': training_rcr,
        'training_backlobe': training_backlobe,
        'sim_rcr_all': sim_rcr,
        'data_backlobe_traces2016': backlobe_traces_2016,
        'data_backlobe_tracesRCR': backlobe_traces_rcr,
    }


# ============================================================
#                        1D CNN MODEL
# ============================================================
def build_1d_cnn_model(n_samples, n_channels):
    model = Sequential()
    model.add(Conv1D(20, kernel_size=4, activation='relu', padding='same', input_shape=(n_samples, n_channels)))
    model.add(Dropout(0.5))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='Adam',  
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_cnn_model(training_rcr, training_backlobe, config):
    x = np.vstack((training_rcr, training_backlobe))
    y = np.vstack((np.ones((training_rcr.shape[0], 1)), np.zeros((training_backlobe.shape[0], 1))))
    s = np.arange(x.shape[0])
    np.random.shuffle(s)
    x = x[s]
    y = y[s]
    x = x.transpose(0, 2, 1)  # (n_events, samples, channels)

    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=config['early_stopping_patience'])]
    model = build_1d_cnn_model(x.shape[1], x.shape[2])
    model.summary()

    history = model.fit(
        x, y,
        validation_split=0.25,
        epochs=config['keras_epochs'],
        batch_size=config['keras_batch_size'],
        verbose=config['verbose_fit'],
        callbacks=callbacks_list
    )
    return model, history


# ============================================================
#                 Evaluation and Plotting
# ============================================================
def evaluate_model_performance(model, sim_rcr_all, data_backlobe_traces_rcr_all, output_cut_value, config):
    sim_rcr_expanded = sim_rcr_all.transpose(0, 2, 1)
    data_backlobe_expanded = data_backlobe_traces_rcr_all.transpose(0, 2, 1)

    prob_rcr = model.predict(sim_rcr_expanded, batch_size=config['keras_batch_size'])
    prob_backlobe = model.predict(data_backlobe_expanded, batch_size=config['keras_batch_size'])

    rcr_eff = (np.sum(prob_rcr > output_cut_value) / len(prob_rcr)) * 100
    bl_eff = (np.sum(prob_backlobe > output_cut_value) / len(prob_backlobe)) * 100
    print(f'RCR eff: {rcr_eff:.2f}% | Backlobe eff: {bl_eff:.4f}%')
    return prob_rcr, prob_backlobe, rcr_eff, bl_eff


def plot_network_output_histogram(prob_rcr, prob_backlobe, rcr_eff, bl_eff, config, timestamp):
    amp = config['amp']
    prefix = config['prefix']
    cut = config['output_cut_value']
    path = os.path.join(config['base_plot_path'], 'network_output')
    os.makedirs(path, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.hist(prob_backlobe, bins=20, range=(0, 1), histtype='step', color='blue', label=f'Backlobe {len(prob_backlobe)}')
    plt.hist(prob_rcr, bins=20, range=(0, 1), histtype='step', color='red', label=f'RCR {len(prob_rcr)}')
    plt.axvline(x=cut, color='y', linestyle='--', label='cut')
    plt.xlabel('Network Output'); plt.ylabel('Events'); plt.yscale('log')
    plt.legend()
    plt.title(f'{amp} RCR-Backlobe Output')
    plt.text(0.25, 0.8, f'RCR eff: {rcr_eff:.2f}%', transform=plt.gca().transAxes)
    plt.text(0.25, 0.75, f'Backlobe eff: {bl_eff:.4f}%', transform=plt.gca().transAxes)

    out = os.path.join(path, config['histogram_filename_template'].format(timestamp=timestamp, amp=amp, prefix=prefix))
    plt.savefig(out)
    plt.close()
    print(f'Saved histogram to {out}')


# ============================================================
#                Network Output Check Utilities
# ============================================================
def load_most_recent_model(base_model_path, amp, model_prefix=None):
    pattern = re.compile(r"(\d{2}\.\d{2}\.\d{2}_\d{2}-\d{2})_.*\.h5")
    now = time.time()
    best_file, best_ts, best_diff = None, None, float('inf')
    for fname in os.listdir(base_model_path):
        if not fname.endswith(".h5"):
            continue
        if model_prefix and model_prefix not in fname:
            continue
        match = pattern.search(fname)
        if match:
            ts = match.group(1)
            diff = now - datetime.strptime(ts, '%m.%d.%y_%H-%M').timestamp()
            if 0 <= diff < best_diff:
                best_diff, best_file, best_ts = diff, fname, ts
    if best_file:
        model_path = os.path.join(base_model_path, best_file)
        print(f"Loading model: {model_path}")
        return keras.models.load_model(model_path), best_ts
    else:
        raise FileNotFoundError("No model found.")


def load_2016_backlobe_templates(paths, amp_type='200s'):
    stn_groups = {'200s': [14, 17, 19, 30], '100s': [13, 15, 18]}
    allowed = stn_groups.get(amp_type, [])
    arrs = []
    for p in paths:
        m = re.search(r'Stn(\d+)', p)
        if m and int(m.group(1)) in allowed:
            arrs.append(np.load(p))
    return np.stack(arrs, axis=0)


def load_all_coincidence_traces(pkl_path, trace_key):
    with open(pkl_path, "rb") as f:
        coinc = pickle.load(f)
    traces = []
    for _, m in coinc.items():
        for _, st in m['stations'].items():
            tr = st.get(trace_key)
            if tr is not None:
                traces.extend(tr)
    return np.stack(traces, axis=0)


def plot_check_hist(prob_2016, prob_coinc, prob_one, amp, ts):
    plt.figure(figsize=(8, 6))
    plt.hist(prob_2016, bins=20, range=(0, 1), histtype='step', color='orange', label='2016 BL')
    plt.hist(prob_coinc, bins=20, range=(0, 1), histtype='step', color='black', label='Coincidence')
    plt.yscale('log')
    plt.xlabel('Network Output'); plt.ylabel('Events')
    plt.text(0.05, 0.85, f'Sample RCR: {prob_one.item():.2f}', transform=plt.gca().transAxes)
    plt.legend()
    out = os.path.join('/dfs8/sbarwick_lab/ariannaproject/tangch3/plots/', f'{amp}_network_output_check_{ts}.png')
    plt.savefig(out)
    plt.close()
    print(f"Saved network output check to {out}")


# ============================================================
#                          MAIN
# ============================================================
def main():
    config = load_config()
    amp = config['amp']
    prefix = config['prefix']
    timestamp = datetime.now().strftime('%m.%d.%y_%H-%M')
    print(f"Starting 1D CNN training at {timestamp} for {amp} amplifier.")

    data = load_and_prep_data_for_training(config)
    model, history = train_cnn_model(data['training_rcr'], data['training_backlobe'], config)
    print("Training complete.")

    model_save_path = os.path.join(config['base_model_path'],
                                   config['model_filename_template'].format(timestamp=timestamp, amp=amp, prefix=prefix))
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    prob_rcr, prob_bl, rcr_eff, bl_eff = evaluate_model_performance(
        model, data['sim_rcr_all'], data['data_backlobe_tracesRCR'], config['output_cut_value'], config)
    plot_network_output_histogram(prob_rcr, prob_bl, rcr_eff, bl_eff, config, timestamp)

    # === NETWORK OUTPUT CHECK ===
    model_check, ts = load_most_recent_model(config['base_model_path'], amp, model_prefix="CNN")
    template_dir = "/pub/tangch3/ARIANNA/DeepLearning/refactor/confirmed_2016_templates/"
    paths = sorted(glob(os.path.join(template_dir, "filtered_Event2016_Stn*.npy")))
    bl_2016 = load_2016_backlobe_templates(paths, amp_type=amp)

    pkl_path = "/pub/tangch3/ARIANNA/DeepLearning/refactor/other/test_bandpass_on_coinc/filtered_coinc.pkl"
    coinc = load_all_coincidence_traces(pkl_path, "Filtered_Traces")

    bl_2016 = bl_2016.transpose(0, 2, 1)
    coinc = coinc.transpose(0, 2, 1)

    prob_bl_2016 = model_check.predict(bl_2016)
    prob_coinc = model_check.predict(coinc)
    sample_prob = model_check.predict(np.expand_dims(coinc[1297], axis=0))
    plot_check_hist(prob_bl_2016.flatten(), prob_coinc.flatten(), sample_prob.flatten(), amp, ts)

    print("Script finished successfully.")


if __name__ == "__main__":
    main()
