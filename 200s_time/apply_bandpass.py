import os
import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.utilities import fft, units
from scipy import signal

def apply_butterworth(spectrum, frequencies, passband, order=8):
    """
    Calculates the response from a Butterworth filter and applies it to the
    input spectrum

    Parameters
    ----------
    spectrum: array of complex
        Fourier spectrum to be filtere
    frequencies: array of floats
        Frequencies of the input spectrum
    passband: (float, float) tuple
        Tuple indicating the cutoff frequencies
    order: integer
        Filter order

    Returns
    -------
    filtered_spectrum: array of complex
        The filtered spectrum
    """

    f = np.zeros_like(frequencies, dtype=complex)
    mask = frequencies > 0
    b, a = signal.butter(order, passband, "bandpass", analog=True)
    w, h = signal.freqs(b, a, frequencies[mask])
    f[mask] = h

    filtered_spectrum = f * spectrum

    return filtered_spectrum

def butterworth_filter_trace(trace, sampling_frequency, passband, order=8):
    """
    Filters a trace using a Butterworth filter.

    Parameters
    ----------
    trace: array of floats
        Trace to be filtered
    sampling_frequency: float
        Sampling frequency
    passband: (float, float) tuple
        Tuple indicating the cutoff frequencies
    order: integer
        Filter order

    Returns
    -------

    filtered_trace: array of floats
        The filtered trace
    """

    n_samples = len(trace)

    spectrum = fft.time2freq(trace, sampling_frequency)
    frequencies = np.fft.rfftfreq(n_samples, d = 1 / sampling_frequency)

    filtered_spectrum = apply_butterworth(spectrum, frequencies, passband, order)
    filtered_trace = fft.freq2time(filtered_spectrum, sampling_frequency)

    return filtered_trace

# Input and output directories
pkl_path = '/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/6.11.25_CoincidenceDatetimes_with_all_params_recalcZenAzi_calcPol.pkl'
output_dir = '/pub/tangch3/ARIANNA/DeepLearning/refactor/other/test_bandpass_on_coinc'

# load input data
from refactor_checks import load_all_coincidence_traces
coinc_traces, metadata = load_all_coincidence_traces(pkl_path)

# Define the sampling rate
sampling_rate_hz = 2 * units.MHz


# Create an empty list to store the filtered traces
filtered_traces = []

# Iterate over each event in the data
for event_data in coinc_traces:
    filtered_event = []
    # Iterate over each channel in the event
    for trace_ch_data_arr in event_data:
        # Define the butterworth filter parameters
        passband = [50 * units.MHz, 1000 * units.MHz] 
        order = 2
        
        # Apply the butterworth filter
        filtered_trace = butterworth_filter_trace(trace_ch_data_arr, sampling_rate_hz, passband, order)
        
        filtered_event.append(filtered_trace)
        
    filtered_traces.append(filtered_event)
    
# Convert the list of filtered traces to a numpy array
filtered_data = np.array(filtered_traces)

# Create the new filename
new_filename = 'filtered_coinc_traces.npy'
output_path = os.path.join(output_dir, new_filename)

# Save the filtered data to the new file
np.save(output_path, filtered_data)
print(f"Saved filtered data to {output_path}")

print(f"All files processed successfully! {filtered_data.shape}")

def plot_and_save_event_traces(traces, output_dir, n_channels=4):
    """
    Plot and save each event individually, displaying `n_channels` traces for each event.
    
    Parameters
    ----------
    traces: np.array
        The filtered traces array (N events, 4 channels, 256 samples)
    output_dir: str
        Directory to save the plots
    n_channels: int
        The number of channels per event (default is 4)
    """
    # Check if output directory exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over each event
    i = 0
    for event_idx in range(traces.shape[0]):
        i += 1
        if i > 4:
            break
        # Create subplots (1 row, n_channels columns)
        fig, axes = plt.subplots(1, n_channels, figsize=(12, 3), sharex=True)
        
        # Loop through each channel and plot
        for ch_idx in range(n_channels):
            ax = axes[ch_idx] if n_channels > 1 else axes  # Handle case with single channel
            ax.plot(traces[event_idx, ch_idx], label=f'Channel {ch_idx+1}')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Event {event_idx+1} - Channel {ch_idx+1}')
            ax.legend(loc='best')
        
        # Set xlabel only once for the whole figure
        plt.xlabel('Time (samples)')
        plt.tight_layout()

        # Save the plot as an image file
        plot_filename = f'event_{event_idx}_traces.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()  # Close the plot to avoid memory issues for large datasets

        print(f"Saved plot for Event {event_idx} to {plot_path}")


plot_and_save_event_traces(filtered_data, output_dir)
print(filtered_data[0])