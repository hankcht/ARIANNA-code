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
    from A0_Utilities import load_config, load_data
    # rough draft, clean up later
    config = load_config()

    output_dir = '/pub/tangch3/ARIANNA/DeepLearning/refactor/other/bandpass_on_all_data'

    sampling_rate_hz = 2 * units.MHz
    passband = [0.05 * units.MHz, 0.5 * units.MHz]  # 50 kHz – 500 kHz
    order = 2

    stations = [14] # 13,14,15,18,17,19,30
    for s_id in stations:
        if s_id in [13, 15, 18]:
            amp = '100s'
        elif s_id in [14, 17, 19, 30]:
            amp = '200s'
        else:
            print(f'wrong station {s_id}')

        snr2016, snrRCR, chi2016, chiRCR, traces2016, tracesRCR, unix2016, unixRCR = load_data(config['loading_data_type'], amp_type=amp, station_id=s_id)
        traces2016 = np.array(traces2016)
        tracesRCR = np.array(tracesRCR)
        print(f"Loaded {traces2016.shape[0]} traces.")

        filtered_traces_2016 = []
        for event_data_2016 in traces2016:
            filtered_event_2016 = []
            for trace_ch in event_data_2016:
                filtered_event_2016.append(butterworth_filter_trace(trace_ch, sampling_rate_hz, passband, order))
            filtered_traces_2016.append(filtered_event_2016)

        filtered_traces_rcr = []
        for event_data_rcr in tracesRCR:
            filtered_event_rcr = []
            for trace_ch in event_data_rcr:
                filtered_event_rcr.append(butterworth_filter_trace(trace_ch, sampling_rate_hz, passband, order))
            filtered_traces_rcr.append(filtered_event_rcr)

        filtered_traces_2016 = np.array(filtered_traces_2016)
        print(f"2016 Filtering complete. {s_id} Shape: {filtered_traces_2016.shape}")
        filtered_traces_rcr = np.array(filtered_traces_rcr)
        print(f"RCR Filtering complete. {s_id} Shape: {filtered_traces_rcr.shape}")

        from A0_Utilities import pT
        indices = [202, 510, 648, 763, 879]
        for index in indices:
            pT(filtered_event_2016[index], f'plot filtered 2016 above cuvre stn {s_id}', f'/pub/tangch3/ARIANNA/DeepLearning/refactor/other/test_plot_filtered_data_{s_id}_{index}.png')


    # # Save updated files
    # os.makedirs(output_dir, exist_ok=True)
    # with open(updated_pkl_path, "wb") as f:
    #     pickle.dump(coinc_dict, f)
    # print(f"Updated PKL saved to {updated_pkl_path}")


    
    