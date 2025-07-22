import os
import re
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from datetime import datetime
from tensorflow import keras


def get_config():
    return {
        'amp': '200s',
        'base_model_path': '/pub/tangch3/ARIANNA/DeepLearning/refactor/models/',
        'base_plot_path': '/pub/tangch3/ARIANNA/DeepLearning/refactor/plots/network_output/',
        'model_filename_template': '{timestamp}_{amp}_RCR_Backlobe_model_2Layer.h5',
        'histogram_filename_template': '{timestamp}_{amp}_checks_histogram.png',
    }


def load_most_recent_model(base_model_path, amp, model_prefix=None):
    """
    Load the most recent model matching the prefix from the specified amp subfolder.
    
    Args:
        base_model_path (str): base path to the model directory.
        amp (str): amplification or timing setting (e.g., '100s', '200s').
        model_prefix (str, optional): prefix to match in model filename.

    Returns:
        tuple: (loaded_model, timestamp_str)
    """
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})_.*\.h5")

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
            model_time = datetime.strptime(timestamp, "%Y-%m-%d_%H-%M").timestamp()
            diff = now - model_time
            if 0 <= diff < smallest_diff:
                smallest_diff = diff
                best_file = fname
                best_timestamp = timestamp


    if best_file:
        model_path = os.path.join(base_model_path, best_file)
        print(f"Loading model: {model_path}")
        return keras.models.load_model(model_path), best_timestamp
    else:
        raise FileNotFoundError(f"No suitable model file found in {base_model_path}.")



def load_2016_backlobe_templates(file_paths, amp_type='200s'):
    station_groups = {
        '200s': [14, 17, 19, 30],
        '100s': [13, 15, 18]
    }

    allowed_stations = station_groups.get(amp_type, [])
    arrays = []
    metadata = {}

    for path in file_paths:
        match = re.search(r'Stn(\d+)_\d+\.\d+_Chi(\d+\.\d+)_SNR(\d+\.\d+)', path)
        if match:
            station_id = int(match.group(1))
            chi = float(match.group(2))
            snr = float(match.group(3))

            if station_id in allowed_stations:
                arr = np.load(path)
                arrays.append(arr)
                index = len(arrays) - 1
                metadata[index] = {
                    "station": station_id,
                    "chi": chi,
                    "snr": snr,
                    "trace": arr
                }

    return np.stack(arrays, axis=0), metadata


def load_all_coincidence_traces(pkl_path):
    with open(pkl_path, "rb") as f:
        coinc_dict = pickle.load(f)

    all_traces = []
    metadata = {}
    idx = 0

    for event_data in coinc_dict.values():
        stations = event_data.get('stations', {})
        for station_id, station_dict in stations.items():
            traces = station_dict.get('Traces')
            if traces is None or len(traces) == 0:
                continue

            all_traces.append(traces)
            for i in range(len(traces)):
                metadata[idx] = {
                    'index': station_dict['indices'][i],
                    'event_id': station_dict['event_ids'][i],
                    'station_id': station_id,
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

    X = np.concatenate(all_traces, axis=0)
    return X, metadata


def plot_histogram(prob_2016, prob_coincidence, amp, timestamp):
    
    plt.figure(figsize=(8, 6))
    bins = 20
    range_vals = (0, 1)

    plt.hist(prob_2016, bins=bins, range=range_vals, histtype='step', color='green', linestyle='solid',
             label=f'2016-Backlobes ({len(prob_2016)})', density=False)
    plt.hist(prob_coincidence, bins=bins, range=range_vals, histtype='step', color='black', linestyle='solid',
             label=f'Coincidence-Events ({len(prob_coincidence)})', density=False)

    plt.xlabel('Network Output', fontsize=16)
    plt.ylabel('Number of Events', fontsize=16)
    plt.yscale('log')

    hist_values_2016, _ = np.histogram(prob_2016, bins=20, range=(0, 1))
    hist_values_coincidence, _ = np.histogram(prob_coincidence, bins=20, range=(0, 1))
    max_overall_hist = max(np.max(hist_values_2016), np.max(hist_values_coincidence))
    plt.ylim(0, max(10 ** (np.ceil(np.log10(max_overall_hist * 1.1))), 10))

    plt.title(f'{amp}-time 2016 BL and Coincidence Events Network Output', fontsize=16)
    plt.legend(loc='upper left', fontsize=12)

    config = get_config()
    filename = config['histogram_filename_template'].format(timestamp=timestamp, amp=amp)
    out_path = os.path.join(config['base_plot_path'], filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.savefig(out_path)
    print(f"Saved histogram to {out_path}")
    plt.close()


def main():
    config = get_config()
    amp = config['amp']

    model, model_timestamp = load_most_recent_model(config['base_model_path'], amp, model_prefix="RCR_Backlobe")

    template_dir = "/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/templates/confirmed2016Templates"
    template_paths = glob(os.path.join(template_dir, "Event2016_Stn*.npy"))
    all_2016_backlobes, _ = load_2016_backlobe_templates(template_paths, amp_type=amp)
    print(f"[INFO] Loaded {len(all_2016_backlobes)} 2016 backlobe traces.")

    pkl_path = "/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/6.11.25_CoincidenceDatetimes_with_all_params_recalcZenAzi_calcPol.pkl"
    all_coincidence_events, _ = load_all_coincidence_traces(pkl_path)
    print(f"[INFO] Loaded {len(all_coincidence_events)} coincidence traces.")

    prob_backlobe = model.predict(all_2016_backlobes).flatten()
    prob_coincidence = model.predict(all_coincidence_events).flatten()

    plot_histogram(prob_backlobe, prob_coincidence, amp, timestamp=model_timestamp)

if __name__ == "__main__":
    main()
