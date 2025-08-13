import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units
from scipy import signal

def apply_butterworth(spectrum, frequencies, passband, sampling_frequency, order=8):
    """
    Apply a digital Butterworth bandpass filter to a Fourier spectrum.
    
    Parameters:
    - spectrum: The frequency spectrum (Fourier transform) of the signal to be filtered.
    - frequencies: Array of frequency values corresponding to the input spectrum.
    - passband: Tuple (low_freq, high_freq) defining the passband range to retain.
    - sampling_frequency: The sampling rate of the original signal.
    - order: The order of the Butterworth filter (default is 8).
    
    Returns:
    - The filtered frequency spectrum after applying the Butterworth bandpass filter.
    """
    nyquist = sampling_frequency / 2 # Calculate Nyquist frequency
    norm_passband = [passband[0] / nyquist, passband[1] / nyquist]  # Normalize the passband frequencies by the Nyquist frequency
    b, a = signal.butter(order, norm_passband, btype="bandpass", analog=False) # bandpass filter
    w, h = signal.freqz(b, a, worN=len(frequencies), fs=sampling_frequency) # Calculate the frequency response of the filter
    return spectrum * h

def butterworth_filter_trace(trace, sampling_frequency, passband, order=8):
    """
    Filters a time-domain trace with a digital Butterworth filter.
    """
    n_samples = len(trace)
    spectrum = np.fft.rfft(trace)
    frequencies = np.fft.rfftfreq(n_samples, d=1/sampling_frequency)
    filtered_spectrum = apply_butterworth(spectrum, frequencies, passband, sampling_frequency, order)
    return np.fft.irfft(filtered_spectrum, n_samples)

if __name__ == '__main__':
    pkl_path = '/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/6.11.25_CoincidenceDatetimes_with_all_params_recalcZenAzi_calcPol.pkl'
    output_dir = '/pub/tangch3/ARIANNA/DeepLearning/refactor/other/test_bandpass_on_coinc'
    updated_pkl_path = os.path.join(output_dir, "filtered_coinc.pkl")

    sampling_rate_hz = 2 * units.MHz
    passband = [0.05 * units.MHz, 0.5 * units.MHz]  # 50 kHz – 500 kHz
    order = 2

    print("Loading PKL...")
    from refactor_checks import load_all_coincidence_traces
    coinc_dict, coinc_traces, metadata = load_all_coincidence_traces(pkl_path)
    print(f"Loaded {coinc_traces.shape[0]} traces.")

    filtered_traces = []
    for event_data in coinc_traces:
        filtered_event = []
        for trace_ch in event_data:
            filtered_event.append(butterworth_filter_trace(trace_ch, sampling_rate_hz, passband, order))
        filtered_traces.append(filtered_event)

    filtered_traces = np.array(filtered_traces)
    print(f"Filtering complete. Shape: {filtered_traces.shape}")

    # Add filtered traces back into original dict
    idx = 0
    for master_id, master_data in coinc_dict.items():
        for station_id, station_dict in master_data['stations'].items():
            traces = station_dict.get('Traces')
            if traces is None or len(traces) == 0:
                continue
            n_traces = len(traces)
            station_dict['Filtered_Traces'] = filtered_traces[idx:idx+n_traces]
            idx += n_traces

    # Save updated pickle
    os.makedirs(output_dir, exist_ok=True)
    with open(updated_pkl_path, "wb") as f:
        pickle.dump(coinc_dict, f)
    print(f"Updated PKL saved to {updated_pkl_path}")


    # from A0_Utilities import pT

    # coinc_dict, coinc_traces, metadata = load_all_coincidence_traces(updated_pkl_path, "Filtered_Traces")
    # coinc_traces = np.array(coinc_traces)
    # print(coinc_traces.shape)


