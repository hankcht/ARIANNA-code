import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units
from scipy import signal

def apply_butterworth(spectrum, frequencies, passband, sampling_frequency, order=8):
    """
    Apply a digital Butterworth bandpass filter to a Fourier spectrum.
    """
    nyquist = sampling_frequency / 2
    norm_passband = [passband[0] / nyquist, passband[1] / nyquist]
    b, a = signal.butter(order, norm_passband, btype="bandpass", analog=False)
    w, h = signal.freqz(b, a, worN=len(frequencies), fs=sampling_frequency)
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

def plot_and_save_event_traces(traces, output_dir, n_channels=4):
    """
    Plot the first few events for visual inspection.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for event_idx in range(min(4, traces.shape[0])):
        fig, axes = plt.subplots(1, n_channels, figsize=(12, 3), sharex=True)

        for ch_idx in range(n_channels):
            ax = axes[ch_idx] if n_channels > 1 else axes
            ax.plot(traces[event_idx, ch_idx], label=f'Channel {ch_idx+1}')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Event {event_idx+1} - Ch {ch_idx+1}')
            ax.legend(loc='best')

        plt.xlabel('Time (samples)')
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'event_{event_idx}_traces.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot for Event {event_idx} to {plot_path}")

def load_all_coincidence_traces(pkl_path):
    with open(pkl_path, "rb") as f:
        coinc_dict = pickle.load(f)

    all_traces = []
    metadata = {}
    idx = 0
    for master_id, master_data in coinc_dict.items():
        for station_id, station_dict in master_data['stations'].items():
            traces = station_dict.get('Traces')
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

    X = np.stack(all_traces, axis=0)
    return coinc_dict, X, metadata

# ---------------- Main Script ----------------

pkl_path = '/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/6.11.25_CoincidenceDatetimes_with_all_params_recalcZenAzi_calcPol.pkl'
output_dir = '/pub/tangch3/ARIANNA/DeepLearning/refactor/other/test_bandpass_on_coinc'
updated_pkl_path = os.path.join(output_dir, "filtered_coinc.pkl")

sampling_rate_hz = 2 * units.MHz
passband = [0.05 * units.MHz, 0.5 * units.MHz]  # 50 kHz – 500 kHz
order = 2

print("Loading PKL...")
coinc_dict, coinc_traces, metadata = load_all_coincidence_traces(pkl_path)
print(f"Loaded {coinc_traces.shape[0]} traces.")

filtered_traces = []
for event_data in coinc_traces:
    filtered_event = []
    for trace_ch in event_data:
        filtered_event.append(
            butterworth_filter_trace(trace_ch, sampling_rate_hz, passband, order)
        )
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

# Save updated PKL
os.makedirs(output_dir, exist_ok=True)
with open(updated_pkl_path, "wb") as f:
    pickle.dump(coinc_dict, f)
print(f"Updated PKL saved to {updated_pkl_path}")

# Plot first few filtered traces
# plot_and_save_event_traces(filtered_traces, output_dir)

from A0_Utilities import pT
for i in range(4):
    pT(filtered_traces[i], 'plot filtered coinc', f'/pub/tangch3/ARIANNA/DeepLearning/refactor/other/test_bandpass_on_coinc/event_{i}_trace.png')

    pT(filtered_traces[1297], 'plot filtered coinc RCR', f'/pub/tangch3/ARIANNA/DeepLearning/refactor/other/test_bandpass_on_coinc/event_1297_trace.png')
    pT(filtered_traces[1298], 'plot filtered coinc BL', f'/pub/tangch3/ARIANNA/DeepLearning/refactor/other/test_bandpass_on_coinc/event_1298_trace.png')
