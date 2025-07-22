import os
import numpy as np
import re
import time
import pickle
from glob import glob
from datetime import datetime

# --- Configuration ---
def get_config():
    """Returns a dictionary of configuration parameters."""
    return {
        'amp': '200s',
        # 'station_ids_200s': [14, 17, 19, 30],
        # 'station_ids_100s': [13, 15, 18],
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


def load_most_recent_model(model_dir, model_prefix=None):
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})_.*\.h5")

    now = time.time()
    best_file = None
    smallest_diff = float('inf')

    for fname in os.listdir(model_dir):
        if not fname.endswith(".h5"):
            continue
        if model_prefix and model_prefix not in fname:
            continue

        match = pattern.search(fname)
        if match:
            try:
                timestamp = match.group(1)
                model_time = datetime.strptime(timestamp, "%Y-%m-%d_%H-%M").timestamp()
                diff = now - model_time
                if 0 <= diff < smallest_diff:
                    smallest_diff = diff
                    best_file = fname
            except Exception:
                continue

    if best_file:
        return os.path.join(model_dir, best_file)
    else:
        raise FileNotFoundError("No suitable model file found.")


def load_2016_backlobe_templates(file_paths, amp_type='200s'):
    """
    Loads and stacks .npy files into a NumPy array for model prediction,
    and returns metadata (station, chi, snr, trace) for each waveform.

    Args:
        file_paths (list of str): Paths to .npy files.
        amp_type (str): '200s' or '100s' to filter by station group.

    Returns:
        tuple:
            - np.ndarray: Array of shape (N, ...) suitable for model.predict()
            - dict: Metadata for each trace {index: {'station', 'chi', 'snr', 'trace'}}
    """
    station_groups = {
        '200s': [14, 17, 19, 30],
        '100s': [13, 15, 18]
    }

    allowed_stations = station_groups.get(amp_type)

    arrays = []
    metadata = {}

    for path in file_paths:
        match = re.search(r'Stn(\d+)_\d+\.\d+_Chi([\d.]+)_SNR([\d.]+)', path)
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


def load_all_coincidence_traces(pkl_path="/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/6.11.25_CoincidenceDatetimes_with_all_params_recalcZenAzi_calcPol.pkl"):
    """
    Load and concatenate all traces from coincidence pickle file for all events and stations,
    returning:
        - X: concatenated np.ndarray (N, ...)
        - metadata: dict indexed by trace with info (event_id, station_id, SNR, ChiRCR, etc.)

    Args:
        pkl_path (str): path to the coincidence pickle file.

    Returns:
        tuple:
            - np.ndarray: stacked traces
            - dict: metadata per trace
    """
    with open(pkl_path, "rb") as f:
        coinc_dict = pickle.load(f)

    all_traces = []
    metadata = {}
    idx = 0  # global trace index

    for master_id, event_data in coinc_dict.items():
        stations = event_data.get('stations', {})
        for station_id, station_dict in stations.items():
            traces = station_dict.get('Traces')
            if traces is None or len(traces) == 0:
                continue

            all_traces.append(traces)

            n_traces = len(traces)
            for i in range(n_traces):
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

# Example usage
if __name__ == "__main__":
    # amp = "200s"
    # model_dir = f"/pub/tangch3/ARIANNA/DeepLearning/models/{amp}_time/new_chi"
    # path = load_most_recent_model(model_dir, model_prefix="RCR_Backlobe")
    # print("Most recent model:", path)
    
    # template_dir = "/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/templates/confirmed2016Templates"
    # file_paths = glob(f"{template_dir}/Event2016_Stn*.npy")

    # all_2016_backlobes, dict_2016 = load_2016_backlobe_templates(file_paths, amp_type='100s')

    # print("Model input shape:", all_2016_backlobes.shape)
    # print("Metadata for first trace:", dict_2016[0])

    import pickle

    with open("/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/6.11.25_CoincidenceDatetimes_with_all_params_recalcZenAzi_calcPol.pkl", "rb") as f:
        coinc_dict = pickle.load(f)

    print("Total events:", len(coinc_dict))
    print("Example event IDs:", list(coinc_dict.keys())[:5])

    sample_event_id = list(coinc_dict.keys())[0]
    event = coinc_dict[sample_event_id]

    print("Keys under one event:", event.keys())  # should show: numCoincidences, datetime, stations
    print("indices:", event['indices'])
    print("event_ids:", event['event_ids'])
    print("Station IDs:", list(event['stations'].keys()))


