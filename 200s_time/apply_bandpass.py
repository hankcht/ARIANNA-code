import os
import numpy as np
import pickle
from NuRadioReco.utilities import fft, units
from scipy import signal
from refactor_checks import load_all_coincidence_traces

def apply_butterworth(spectrum, frequencies, passband, order=8):
    f = np.zeros_like(frequencies, dtype=complex)
    mask = frequencies > 0
    b, a = signal.butter(order, passband, "bandpass", analog=True)
    w, h = signal.freqs(b, a, frequencies[mask])
    f[mask] = h
    return f * spectrum

def butterworth_filter_trace(trace, sampling_frequency, passband, order=8):
    n_samples = len(trace)
    spectrum = fft.time2freq(trace, sampling_frequency)
    frequencies = np.fft.rfftfreq(n_samples, d=1 / sampling_frequency)
    filtered_spectrum = apply_butterworth(spectrum, frequencies, passband, order)
    return fft.freq2time(filtered_spectrum, sampling_frequency)

# Paths
pkl_path = '/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/6.11.25_CoincidenceDatetimes_with_all_params_recalcZenAzi_calcPol.pkl'
output_dir = '/pub/tangch3/ARIANNA/DeepLearning/refactor/other/test_bandpass_on_coinc'
os.makedirs(output_dir, exist_ok=True)

# Load dictionary directly instead of flattened traces
with open(pkl_path, "rb") as f:
    coinc_dict = pickle.load(f)

sampling_rate_hz = 2 * units.MHz
passband = [50 * units.MHz, 1000 * units.MHz]
order = 2

# Process in-place
for master_id, master_data in coinc_dict.items():
    for station_id, station_dict in master_data['stations'].items():
        traces = station_dict.get('Traces')
        if traces is None or len(traces) == 0:
            continue
        
        traces = np.array(traces)
        filtered_traces = [
            butterworth_filter_trace(trace, sampling_rate_hz, passband, order)
            for trace in traces
        ]
        station_dict['Filtered_Traces'] = np.array(filtered_traces)

# Save updated dictionary
new_filename = 'coinc_with_filtered.pkl'
output_path = os.path.join(output_dir, new_filename)
with open(output_path, "wb") as f:
    pickle.dump(coinc_dict, f)

print(f"Saved updated PKL with Filtered_Traces to {output_path}")