import os, re, sys

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from A0_Utilities import pT
from refactor_checks import load_all_coincidence_traces

# to find 2016 events in coincidence
def plot_2016_matches(X, metadata, path_to_2016_events, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)

    files = os.listdir(path_to_2016_events)
    file_times = {}
    for file in files:
        match = re.search(r'Event2016_Stn(\d+)_(\d+\.\d+)_Chi(\d+\.\d+)_SNR(\d+\.\d+)\.npy', file)
        if match:
            unix_timestamp = float(match.group(2))
            file_times[unix_timestamp] = file

    for idx, info in metadata.items():
        coinc_time = float(info['Times'])

        matched_time = next((ts for ts in file_times if abs(coinc_time - ts) < 1e-4), None)

        if matched_time is not None:
            filename = file_times[matched_time]
            save_name = f"BLmatch_idx{idx}_Stn{info['station_id']}_Master{info['master_id']}_Time{coinc_time}.png"
            save_path = os.path.join(plot_dir, save_name)

            print(f"Plotting match: idx={idx}, station={info['station_id']}, master_id={info['master_id']}")
            pT(
                traces=X[idx],
                title=f"BL Match: idx={idx}, station={info['station_id']}, time={coinc_time}",
                saveLoc=save_path,
                sampling_rate=2
            )


if __name__ == '__main__':

    X, metadata = load_all_coincidence_traces("/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/6.11.25_CoincidenceDatetimes_with_all_params_recalcZenAzi_calcPol.pkl")
    path_to_2016_events = "/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/templates/confirmed2016Templates"
    plot_2016_matches(X, metadata, path_to_2016_events, plot_dir="/pub/tangch3/ARIANNA/DeepLearning/refactor/tests/")