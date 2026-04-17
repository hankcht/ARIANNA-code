# refactor_checks.py

import os
import re
import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from datetime import datetime
import tensorflow
print(tensorflow.__version__)
from tensorflow import keras

from A0_Utilities import load_config
from R01_1D_CNN_train_and_run import load_new_coincidence_data
from R03_Autoencoder_train_and_run import DEFAULT_VALIDATION_SPECIAL_EVENTS


def load_most_recent_model(base_model_path, amp, if_dann, model_prefix=None, specify_model=False):
    """
    Load the most recent model matching the prefix from the specified amp subfolder.
    
    Args:
        base_model_path (str): base path to the model directory.
        amp (str): amplification or timing setting (e.g., '100s', '200s').
        if_dann (bool): to load dann model as custom object.
        model_prefix (str, optional): prefix to match in model filename.
        specify_mdeol (bool): to not load most recent but instead load choice of model 

    Returns:
        tuple: (loaded_model, timestamp_str)
    """
    pattern = re.compile(r"(\d{2}\.\d{2}\.\d{2}_\d{2}-\d{2})_.*\.keras")

    now = time.time()
    best_file = None
    smallest_diff = float('inf')
    best_timestamp = None

    for fname in os.listdir(base_model_path):
        if not fname.endswith(".keras"):
            continue
        if model_prefix and model_prefix not in fname:
            continue

        match = pattern.search(fname)
        if match:
            print(match)
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
        prefix = 'CNN_checks'

    if specify_model:
        # overwrite for specific run
        timestamp =  '12.16.25_14-53' # 11.26.25_13-52
        model_path = f'/dfs8/sbarwick_lab/ariannaproject/tangch3/HGQ2/{timestamp}/models/'
        # print(f"Loading model: {model_path}")
        # model = keras.models.load_model(f'{model_path}12.16.25_14-53_HGQ2_model.keras')
        from tensorflow.keras.layers import InputLayer
        from tensorflow.keras.utils import custom_object_scope
        from hgq.layers import QConv2D, QDense  # wherever your Q-layers are defined

        custom_objects = {
            'InputLayer': InputLayer,
            'QConv2D': QConv2D,
            'QDense': QDense,
            # add gradient_reversal_operation if DANN
        }

        with custom_object_scope(custom_objects):
            model = keras.models.load_model(f'{model_path}{timestamp}_HGQ2_model.keras', compile=False)
        
        prefix = 'hgq'
        return model, timestamp, prefix
    
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
    trace_key : str, optional
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


def plot_histogram(prob_all, prob_passing, prob_special, prob_backlobe, prob_2016, prob_coincidence, prob_coincidence_rcr, amp, timestamp, prefix):
    
    plt.figure(figsize=(8, 6))
    bins = 20
    range_vals = (0, 1)   
    
    hist_pass, bin_edges = np.histogram(prob_passing, bins=bins, range=range_vals)
    hist_back, _ = np.histogram(prob_backlobe, bins=bins, range=range_vals)

    third_pass = hist_pass[2]
    third_back = hist_back[2]

    scale_factor = third_pass / third_back if third_back > 0 else 1

    plt.hist(prob_all, bins=bins, range=range_vals,histtype='step', color='Black', linestyle='solid',
             label=f'{len(prob_all)} All rcr-like data (10.17.25 Cut)')
    # Uncomment to get original plot, currently overwrites
    # plt.hist(prob_passing, bins=bins, range=range_vals,histtype='step', color='Black', linestyle='solid', # weights=np.ones_like(prob_passing)/len(prob_passing),
    #          label=f'Backlobe Coincidence')
    # plt.hist(prob_backlobe, bins=bins, range=range_vals,histtype='step', color='blue', linestyle='solid', weights=np.ones_like(prob_backlobe) * scale_factor,
    #          label=f'Scaled Backlobe-like Data')
    # plt.hist(prob_special, bins=20, range=range_vals,histtype='step', color='green', linestyle='solid', weights=np.ones_like(prob_special)/len(prob_special),
    #          label=f'Special Events {len(prob_special)}')
    # plt.hist(prob_2016, bins=bins, range=range_vals, histtype='step', color='orange', linestyle='solid',
    #          label=f'2016-Backlobes {len(prob_2016)}', density=False)
    # plt.hist(prob_coincidence, bins=bins, range=range_vals, histtype='step', color='black', linestyle='solid',
    #          label=f'Coincidence-Events {len(prob_coincidence)}', density=False)

    plt.xlabel('Network Output', fontsize=18)
    plt.ylabel('Counts', fontsize=18)
    plt.yscale('log')

    # hist_values_2016, _ = np.histogram(prob_2016, bins=20, range=(0, 1))
    # hist_values_coincidence, _ = np.histogram(prob_coincidence, bins=20, range=(0, 1))
    # max_overall_hist = max(np.max(hist_values_2016), np.max(hist_values_coincidence))
    # plt.ylim(7*1e-1, max(10 ** (np.ceil(np.log10(max_overall_hist * 1.1))), 10))

    # plt.text(0.00, 0.85, f'Coincidence RCR network Output is: {prob_coincidence_rcr.item():.2f}',
    #          fontsize=12, verticalalignment='top', transform=plt.gca().transAxes,
    #          bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
    plt.title(f'Passed Events Network Output', fontsize=14)
    plt.legend(loc='upper right', fontsize=12)

    config = load_config()
    filename = config['histogram_filename_template'].format(timestamp=timestamp, amp=amp, prefix=prefix)
    out_path = os.path.join(config['base_plot_path'], 'network_output', filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.savefig(out_path)
    print(f"Saved histogram to {out_path}")
    plt.close()



if __name__ == "__main__":
    config = load_config(config_path="/pub/tangch3/ARIANNA/DeepLearning/code/200s_time/config.yaml")
    amp = config['amp']

    model, model_timestamp, prefix = load_most_recent_model(config['base_model_path'], amp, if_dann=config['if_dann'], model_prefix="CNN", specify_model=True)
    
    template_dir = "/pub/tangch3/ARIANNA/DeepLearning/refactor/confirmed_2016_templates/"
    template_paths = sorted(glob(os.path.join(template_dir, "filtered_Event2016_Stn*.npy"))) # using filtered confirmed BL
    all_2016_backlobes, dict_2016 = load_2016_backlobe_templates(template_paths, amp_type=amp)

    print(f"Loaded {len(all_2016_backlobes)} 2016 backlobe traces.")

    pkl_path = "/pub/tangch3/ARIANNA/DeepLearning/refactor/other/test_bandpass_on_coinc/filtered_coinc.pkl"
    coinc_dict, all_coincidence_events, metadata = load_all_coincidence_traces(pkl_path, "Filtered_Traces") # using filtered coincidence 
    print(f"Loaded {len(all_coincidence_events)} coincidence traces.")

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

    if config['if_dann']:
        print('DANN Evaluation')
        prob_backlobe, _ = model.predict(all_2016_backlobes)
        prob_coincidence, _ = model.predict(all_coincidence_events)
    elif config['if_1D']:
        print('1D CNN Evaluation')
        all_2016_backlobes_transpose = all_2016_backlobes.squeeze(-1).transpose(0, 2, 1)
        all_coincidence_events_transpose = all_coincidence_events.squeeze(-1).transpose(0, 2, 1)
        prob_2016 = model.predict(all_2016_backlobes_transpose)
        prob_coincidence = model.predict(all_coincidence_events_transpose)

        coinc_rcr = all_coincidence_events[1297]
        from A0_Utilities import pT
        pT(coinc_rcr, 'test plot coinc', '/pub/tangch3/ARIANNA/DeepLearning/refactor/other/93_plot_coinc_rcr.png')

        coinc_rcr_transpose = coinc_rcr.squeeze(-1).transpose(1, 0)
        prob_coincidence_rcr = model.predict(np.expand_dims(coinc_rcr_transpose, axis=0))
    else:
        print('2D CNN Evaluation')
        prob_2016 = model.predict(all_2016_backlobes)
        prob_coincidence = model.predict(all_coincidence_events)

        prob_coincidence_rcr = model.predict(np.expand_dims(all_coincidence_events[1297], axis=0))
        print(f'coincidence RCR network output: {prob_coincidence_rcr}')

    prob_2016 = prob_2016.flatten()
    prob_coincidence = prob_coincidence.flatten()
    prob_coincidence_rcr = prob_coincidence_rcr.flatten()

    # --------------------------------------------------
    # Load DEFAULT_VALIDATION_SPECIAL_EVENTS
    # --------------------------------------------------

    validation_pkl_path = "/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/9.24.25_CoincidenceDatetimes_passing_cuts_with_all_params_recalcZenAzi_calcPol.pkl"

    passing_ids = [3047, 3432, 10195, 10231, 10273, 10284, 10444, 10449,
    10466, 10471, 10554, 11197, 11220, 11230, 11236, 11243]  
    special_pairs = DEFAULT_VALIDATION_SPECIAL_EVENTS

    passing_traces, raw_traces, special_traces, passing_meta, raw_meta, special_meta = \
        load_new_coincidence_data(
            validation_pkl_path,
            passing_ids,
            special_pairs
        )

    print(f"Loaded {len(special_traces)} special validation traces.")

    special_traces = np.array(special_traces)

    if special_traces.ndim == 3:
        special_traces = special_traces[..., np.newaxis]

    if config['if_1D']:
        special_traces = special_traces.squeeze(-1).transpose(0, 2, 1)

    prob_special = model.predict(special_traces).flatten()

    print("Special Event Network Outputs:")
    for i, val in enumerate(prob_special):
        meta = special_meta.get(i, {})
        print(f"Event {meta.get('event_id')} Station {meta.get('station_id')} → {val:.4f}")

    passing_traces = np.array(passing_traces)
    print(passing_traces.shape)

    if passing_traces.ndim == 3:
        passing_traces = passing_traces[..., np.newaxis]

    if config['if_1D']:
        passing_traces = passing_traces.squeeze(-1).transpose(0, 2, 1)

    prob_passing = model.predict(passing_traces).flatten()
    from refactor_train_and_run import load_and_prep_data_for_training
    data = load_and_prep_data_for_training(config)
    data_backlobe_traces_rcr_all = data['data_backlobe_tracesRCR']
    data_backlobe_expanded = data_backlobe_traces_rcr_all.transpose(0, 2, 1)
    print(len(data_backlobe_expanded))

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

    combined_pkl_path = f'/dfs8/sbarwick_lab/ariannaproject/tangch3/station_data/above_curve_data/5000evt_10.17.25/above_curve_combined.pkl'    
    snr2016, snrRCR, chi2016, chiRCR, traces2016, tracesRCR, unix2016, unixRCR = load_combined_backlobe_data(combined_pkl_path)

    # Convert to numpy arrays (they should already be arrays from the pickle, but ensure consistency)
    backlobe_traces_2016 = np.array(traces2016)
    backlobe_traces_rcr = np.array(tracesRCR)

    print(f'testing 10.17.25 events, 2016: {len(backlobe_traces_2016)}, rcr: {len(backlobe_traces_rcr)}')
    
    print(f'unix times, 2016: {len(unix2016)}, rcr: {len(unixRCR)}')
    from datetime import timezone

    dt2016 = np.array([datetime.fromtimestamp(t, tz=timezone.utc) for t in unix2016])
    dtRCR  = np.array([datetime.fromtimestamp(t, tz=timezone.utc) for t in unixRCR])

    y2016 = np.random.normal(0, 0.1, len(dt2016))
    yRCR  = np.random.normal(1, 0.1, len(dtRCR))

    plt.figure(figsize=(10, 4))

    plt.scatter(dt2016, y2016, s=10, alpha=0.6)
    plt.scatter(dtRCR,  yRCR,  s=10, alpha=0.6)

    import matplotlib.dates as mdates

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.xticks(rotation=45)
    plt.yticks([0, 1], ['2016', 'RCR'])
    plt.xlabel('Time')
    plt.title('Strip Chart of Unix Times')

    plt.xticks(rotation=45)
    plt.tight_layout()
    print(f'saving as /pub/tangch3/ARIANNA/DeepLearning/plots/miscellaneous/timestrip.png')
    plt.savefig(f'/pub/tangch3/ARIANNA/DeepLearning/plots/miscellaneous/timestrip.png')

    count = np.sum(snrRCR > 20)
    print(f'number of snr > 20 for RCR {count}')
    count = np.sum(snr2016 > 20)
    print(f'number of snr > 20 for 2016 {count}')

    DAY = 86400  # seconds
    days = np.array([int(dt.timestamp() // DAY) for dt in dtRCR])
    unique_days, counts = np.unique(days, return_counts=True)
    day_datetimes = np.array([
    datetime.fromtimestamp(d * 86400, tz=timezone.utc)
    for d in unique_days
    ])
    # ---- Find low-activity days (< 20 events) ----
    low_days = unique_days[counts < 20]
    low_indices = np.where(np.isin(days, low_days))[0]

    print(f"Number of low-activity days (<20 events): {len(low_days)}")
    print(f"Number of events in those days: {len(low_indices)}")

    plt.figure(figsize=(12, 4))

    plt.bar(day_datetimes, counts, width=0.8, alpha=0.7)

    plt.xlabel("Date")
    plt.ylabel("Events per day")
    plt.tight_layout()
    print(f'saving as /pub/tangch3/ARIANNA/DeepLearning/plots/miscellaneous/time_hist.png')
    plt.savefig(f'/pub/tangch3/ARIANNA/DeepLearning/plots/miscellaneous/time_hist.png')

    def find_close_indices(unix_times, label):
        indexed = list(enumerate(unix_times))
        indexed.sort(key=lambda x: x[1])

        close_indices = set()
        pairs = []

        left = 0
        for right in range(len(indexed)):
            while indexed[right][1] - indexed[left][1] > DAY:
                left += 1
            
            for i in range(left, right):
                idx1, t1 = indexed[i]
                idx2, t2 = indexed[right]
                
                # store both index AND timestamp
                pairs.append((idx1, t1, idx2, t2))
                
                close_indices.add(idx1)
                close_indices.add(idx2)

        print(f"{label}: {len(pairs)} pairs")
        print(f"{label}: {len(close_indices)} indices involved")

        return close_indices, pairs

    close2016, pairs2016 = find_close_indices(unix2016, "2016") # 2016: 698029 pairs within 24 hours
    closeRCR, pairsRCR = find_close_indices(unixRCR, "RCR")
    mask = np.ones(len(unixRCR), dtype=bool)
    mask[list(closeRCR)] = False

    tracesRCR_not_close = tracesRCR[mask]
    # for i, trace in enumerate(tracesRCR_not_close):
    #     pT(trace, f'Individual Event', f'/dfs6b/pub/tangch3/ARIANNA/DeepLearning/plots/miscellaneous/tracesRCR_not_close{i}.png')
    #     if i > 5:
    #         break

    data_backlobe_traces_2016_all = data['data_backlobe_traces2016']
    data_backlobe_expanded = data_backlobe_traces_2016_all.transpose(0, 2, 1)
    print(len(data_backlobe_expanded))

    prob_backlobe = model.predict(data_backlobe_expanded)
    prob_backlobe = prob_backlobe.flatten()

    indices_less_25 = np.where(snrRCR < 25)[0]
    backlobe_traces_rcr = backlobe_traces_rcr[low_indices]
    print(f'SNR > 25 removed rcr has size {len(backlobe_traces_rcr)}')
    backlobe_traces_2016_expanded = backlobe_traces_rcr.transpose(0, 2, 1) # all station events cut on 10.17.25, total of 7587
    prob_all = model.predict(backlobe_traces_2016_expanded)
    prob_all = prob_all.flatten()
    
    plot_histogram(prob_all, prob_passing, prob_special, prob_backlobe, prob_2016, prob_coincidence, prob_coincidence_rcr, amp=amp, timestamp=model_timestamp, prefix=prefix)

    backlobe_traces_rcr = backlobe_traces_rcr[prob_all > 0.9]
    for i, trace in enumerate(backlobe_traces_rcr):
        pT(trace, f'Individual Event', f'/dfs6b/pub/tangch3/ARIANNA/DeepLearning/plots/miscellaneous/tracesRCR_event{i}.png')
        if i > 5:
            break
    ############################

    # print(prob_backlobe)

    # indices = [149, 169, 199]
    # for idx in indices:
    #     print(metadata[idx]["master_id"])
    #     print(metadata[idx]["Times"])

    # def load_rcr_events(filepath):
    #     """
    #     Load the saved RCR-passing events dictionary from an .npz file.

    #     Returns a dict with keys:
    #         'station_ids'  : int array (N,) — station ID per event
    #         'event_ids'    : int array (N,) — event ID per event
    #         'times'        : float array (N,) — Unix timestamp per event
    #         'traces'       : float array (N, 4, 256) — waveform traces (4 channels, 256 samples)
    #         'snr'          : float array (N,) — signal-to-noise ratio
    #         'chi_rcr'      : float array (N,) — RCR chi value
    #         'chi_2016'     : float array (N,) — 2016 (backlobe) chi value
    #         'chi_bad'      : float array (N,) — bad-template chi value
    #         'azi'          : float array (N,) — reconstructed azimuth (rad)
    #         'zen'          : float array (N,) — reconstructed zenith (rad)
    #     """
    #     data = np.load(filepath, allow_pickle=False)
    #     return {key: data[key] for key in data.files}


    # def iterate_rcr_events(filepath):
    #     """
    #     Generator that yields one event at a time as a dict.

    #     Each yielded dict contains:
    #         'station_id' : int
    #         'event_id'   : int
    #         'time'       : float (Unix timestamp)
    #         'traces'     : ndarray (4, 256)
    #         'snr'        : float
    #         'chi_rcr'    : float
    #         'chi_2016'   : float
    #         'chi_bad'    : float
    #         'azi'        : float
    #         'zen'        : float

    #     Example:
    #         for evt in iterate_rcr_events('rcr_passing_events.npz'):
    #             print(f"Station {evt['station_id']}, SNR={evt['snr']:.1f}")
    #             traces = evt['traces']  # shape (4, 256)
    #     """
    #     events = load_rcr_events(filepath)
    #     n = len(events['station_ids'])
    #     for i in range(n):
    #         yield {
    #             'station_id': int(events['station_ids'][i]),
    #             'event_id':   int(events['event_ids'][i]),
    #             'time':       float(events['times'][i]),
    #             'traces':     events['traces'][i],
    #             'snr':        float(events['snr'][i]),
    #             'chi_rcr':    float(events['chi_rcr'][i]),
    #             'chi_2016':   float(events['chi_2016'][i]),
    #             'chi_bad':    float(events['chi_bad'][i]),
    #             'azi':        float(events['azi'][i]),
    #             'zen':        float(events['zen'][i]),
    #         }
    # events = load_rcr_events('/pub/tangch3/ARIANNA/DeepLearning/HRAStationDataAnalysis/ErrorAnalysis/output/3.9.26/rcr_passing_events.npz')
    # passing_rcr = []
    # for evt in iterate_rcr_events('/pub/tangch3/ARIANNA/DeepLearning/HRAStationDataAnalysis/ErrorAnalysis/output/3.9.26/rcr_passing_events.npz'):
    #     print(evt['station_id'], evt['snr'], evt['traces'].shape)
    #     passing_rcr.append(evt['traces'])
    # passing_rcr = np.array(passing_rcr)
    
    # print(passing_rcr.shape)
    # passing_rcr = passing_rcr.transpose(0, 2, 1)
    # prob_passing_rcr = model.predict(passing_rcr).flatten()
    # print(np.round(prob_passing_rcr, decimals=2))

    # passing_rcr = passing_rcr.transpose(0, 2, 1) # revert back
    # for i in range(9):
    #     pT(passing_rcr[i], f'plot passing rcr {i}', f'/pub/tangch3/ARIANNA/DeepLearning/refactor/other/passing_rcr_{i}_netout{prob_passing_rcr[i]:.2f}.png')



