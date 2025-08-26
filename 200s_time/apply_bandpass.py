import os, re
from glob import glob
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

def butterworth_filter_trace(trace, fs, passband, order=4):
    """
    Filters a time-domain trace with a digital Butterworth filter.
    """
    x = trace - np.mean(trace)
    # Design in SOS for numerical stability; use fs to avoid manual normalization
    sos = signal.butter(order, [passband[0], passband[1]],
                        btype='bandpass', fs=fs, output='sos')

    # Zero-phase filtering with end padding to suppress edge artifacts
    y = signal.sosfiltfilt(sos, x, padtype='even')   # or padtype='odd'
    return y


if __name__ == '__main__':
    from A0_Utilities import load_config, load_data
    # rough draft, clean up later
    config = load_config()

    output_dir = '/pub/tangch3/ARIANNA/DeepLearning/refactor/station_data/above_curve_data/1000evt_8.25.25/'

    sampling_rate_hz = 2 * units.GHz
    passband = [0.05 * units.GHz, 0.99 * units.GHz]  # 50 MHz – 500 MHz
    order = 2


    stations = [13,14,15,18,17,19,30] 
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
        np.save(f'{output_dir}Stn{s_id}_Traces2016_above_filtered.npy', filtered_traces_2016)

        filtered_traces_rcr = np.array(filtered_traces_rcr)
        print(f"RCR Filtering complete. {s_id} Shape: {filtered_traces_rcr.shape}")
        np.save(f'{output_dir}Stn{s_id}_TracesRCR_above_filtered.npy', filtered_traces_rcr)


        testfilteredtraces2016 = np.load(f'{output_dir}Stn{s_id}_Traces2016_above_filtered.npy')
        print(f'LOADING NEW SAVED FROM {output_dir}Stn{s_id}_Traces2016_above_filtered.npy')
        from A0_Utilities import pT
        indices = [49,57,68,70,100]
        for index in indices:
            pT(testfilteredtraces2016[index], f'plot filtered 2016 above cuvre stn {s_id}', f'/pub/tangch3/ARIANNA/DeepLearning/refactor/other/1000826plot_filtered_data_{s_id}_{index}.png')
            


    # template_dir = '/pub/tangch3/ARIANNA/DeepLearning/refactor/confirmed_2016_templates/'
    # from refactor_checks import load_2016_backlobe_templates
    # template_paths = sorted(glob(os.path.join(template_dir, "Event2016_Stn*.npy")))


    # all_2016_backlobes_200s, dict_2016 = load_2016_backlobe_templates(template_paths, amp_type='200s')
    # all_2016_backlobes_200s = np.array(all_2016_backlobes_200s)
    # print(f"Loaded {all_2016_backlobes_200s.shape[0]} traces.")
    # print(template_paths)

    # filtered_traces_2016_200s = []
    # for event_data_2016_200s in all_2016_backlobes_200s:
    #     filtered_event_2016_200s = []
    #     for trace_ch in event_data_2016_200s:
    #         filtered_event_2016_200s.append(butterworth_filter_trace(trace_ch, sampling_rate_hz, passband, order))
    #     filtered_traces_2016_200s.append(filtered_event_2016_200s)

    # filtered_traces_2016_200s = np.array(filtered_traces_2016_200s)
    # print(f"2016 Filtering complete. Shape: {filtered_traces_2016_200s.shape}")

    # all_2016_backlobes_100s, dict_2016 = load_2016_backlobe_templates(template_paths, amp_type='100s')
    # all_2016_backlobes_100s = np.array(all_2016_backlobes_100s)
    # print(f"Loaded {all_2016_backlobes_100s.shape[0]} traces.")
    # print(template_paths)

    # filtered_traces_2016_100s = []
    # for event_data_2016_100s in all_2016_backlobes_100s:
    #     filtered_event_2016_100s = []
    #     for trace_ch in event_data_2016_100s:
    #         filtered_event_2016_100s.append(butterworth_filter_trace(trace_ch, sampling_rate_hz, passband, order))
    #     filtered_traces_2016_100s.append(filtered_event_2016_100s)

    # filtered_traces_2016_100s = np.array(filtered_traces_2016_100s)
    # print(f"2016 Filtering complete. Shape: {filtered_traces_2016_100s.shape}")

    # all_filtered_traces_2016 = np.concatenate([filtered_traces_2016_200s, filtered_traces_2016_100s])
    # print(all_filtered_traces_2016.shape)

    # from A0_Utilities import pT
    # i=0
    # for original_path, filtered_trace in zip(template_paths, all_filtered_traces_2016):
    #     filename = os.path.basename(original_path) 
        
    #     filtered_filename = f"filtered_{filename}"  
        
    #     filtered_path = os.path.join(output_dir, filtered_filename)
        
    #     print(f'saving to {filtered_path}')
    #     np.save(filtered_path, filtered_trace)
    #     pT(filtered_trace, f'plot filtered 2016 confirmed', f'/pub/tangch3/ARIANNA/DeepLearning/refactor/other/826_confirmed_bl_{i}.png')
    #     i += 1
    