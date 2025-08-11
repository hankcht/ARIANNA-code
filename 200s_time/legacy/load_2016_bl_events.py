import numpy as np
from A0_Utilities import load_config
import re, glob, os


def load_2016_backlobe_templates(file_paths, amp_type='200s'):
    """
    Loads the so-called 2016 confirmed Backlobe events

    Returns:
    Array that contains all bl traces, and information about station id, chi, snr, individual traces, and plot filenames     
    """
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

if __name__ == '__main__':
    config = load_config()
    amp = config['amp']

    template_dir = "/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/templates/confirmed2016Templates"
    template_paths = sorted(glob(os.path.join(template_dir, "Event2016_Stn*.npy")))
    all_2016_backlobes, dict_2016 = load_2016_backlobe_templates(template_paths, amp_type=amp)
