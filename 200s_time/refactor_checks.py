import os
import re
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from datetime import datetime
from tensorflow import keras
from A0_Utilities import load_config


def load_most_recent_model(base_model_path, amp, if_dann, model_prefix=None):
    """
    Load the most recent model matching the prefix from the specified amp subfolder.
    
    Args:
        base_model_path (str): base path to the model directory.
        amp (str): amplification or timing setting (e.g., '100s', '200s').
        if_dann (bool): to load dann model as custom object.
        model_prefix (str, optional): prefix to match in model filename.

    Returns:
        tuple: (loaded_model, timestamp_str)
    """
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

    # Handle custom objects for DANN
    if if_dann:
        from test_DANN import gradient_reversal_operation
        custom_objects = {"gradient_reversal_operation": gradient_reversal_operation}
        prefix = 'DANN'
    else:
        custom_objects = None
        prefix = 'CNN'

    if best_file:
        model_path = os.path.join(base_model_path, best_file)
        print(f"Loading model: {model_path}")
        return keras.models.load_model(model_path, custom_objects=custom_objects), best_timestamp, prefix
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
        match = re.search(r'Event2016_Stn(\d+)_(\d+\.\d+)_Chi(\d+\.\d+)_SNR(\d+\.\d+)\.npy', path)
        if match:
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

def load_all_coincidence_traces(pkl_path):
    with open(pkl_path, "rb") as f:
        coinc_dict = pickle.load(f)

    all_traces = []
    metadata = {}

    idx = 0  # Global trace index
    for master_id, master_data in coinc_dict.items():
        for station_id, station_dict in master_data['stations'].items():
            traces = station_dict.get('Traces')
            if traces is None or len(traces) == 0:
                continue

            # Ensure traces are numpy array
            traces = np.array(traces)
            n_traces = len(traces)

            for i in range(n_traces):
                all_traces.append(traces[i])  # Append each individual trace
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

    # Stack individual traces into a big numpy array
    X = np.stack(all_traces, axis=0)  # shape: (1341, ...) for CNN input
    return X, metadata


def plot_histogram(prob_2016, prob_coincidence, amp, timestamp, prefix):
    
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

    plt.title(f'{amp}-time 2016 BL and Coincidence Events Network Output', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)

    config = load_config()
    filename = config['histogram_filename_template'].format(timestamp=timestamp, amp=amp, prefix=prefix)
    out_path = os.path.join(config['base_plot_path'], filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.savefig(out_path)
    print(f"Saved histogram to {out_path}")
    plt.close()


def main():
    config = load_config()
    amp = config['amp']

    model, model_timestamp, prefix = load_most_recent_model(config['base_model_path'], amp, if_dann=config['if_dann'], model_prefix="RCR_Backlobe")

    template_dir = "/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/templates/confirmed2016Templates"
    template_paths = glob(os.path.join(template_dir, "Event2016_Stn*.npy"))
    all_2016_backlobes, dict_2016 = load_2016_backlobe_templates(template_paths, amp_type=amp)

    print(f"[INFO] Loaded {len(all_2016_backlobes)} 2016 backlobe traces.")

    pkl_path = "/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/6.11.25_CoincidenceDatetimes_with_all_params_recalcZenAzi_calcPol.pkl"
    all_coincidence_events, _ = load_all_coincidence_traces(pkl_path)
    print(f"[INFO] Loaded {len(all_coincidence_events)} coincidence traces.")

    all_2016_backlobes = np.array(all_2016_backlobes)
    all_coincidence_events = np.array(all_coincidence_events)
    print(all_2016_backlobes.shape)
    print(all_coincidence_events.shape)

    if all_2016_backlobes.ndim == 3:
        all_2016_backlobes = all_2016_backlobes[..., np.newaxis]
        print(f'changed to shape {all_2016_backlobes.shape}')
    if all_coincidence_events.ndim == 3:
        all_coincidence_events = all_coincidence_events[..., np.newaxis]
        print(f'changed to shape {all_coincidence_events.shape}')

    for bl in all_2016_backlobes:
      print(np.allclose(bl, all_coincidence_events[149], rtol=1e-01))

    if config['if_dann']:
        prob_backlobe, _ = model.predict(all_2016_backlobes)
        prob_coincidence, _ = model.predict(all_coincidence_events)
    else:
        prob_backlobe = model.predict(all_2016_backlobes)
        prob_coincidence = model.predict(all_coincidence_events)

    prob_backlobe = prob_backlobe.flatten()
    prob_coincidence = prob_coincidence.flatten()

    plot_histogram(prob_backlobe, prob_coincidence, amp, timestamp=model_timestamp, prefix=prefix)

if __name__ == "__main__":
    main()

