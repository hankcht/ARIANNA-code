import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import keras
import templateCrossCorr as txc
from NuRadioReco.utilities import units, fft
from NuRadioReco.utilities.io_utilities import read_pickle


# ---------- CROSS-CORRELATION UTILITIES ---------- #

def get_max_chi(
    traces,
    sampling_rate,
    template_trace,
    template_sampling_rate,
    parallel_channels=None,
    use_average=False,
):
    """
    Compute the maximum absolute cross-correlation with a template.

    Parameters:
        traces (list or array): List/array of trace signals.
        sampling_rate (float): Sampling rate of traces.
        template_trace (array): Template trace for cross-correlation.
        template_sampling_rate (float): Sampling rate of template.
        parallel_channels (list of lists): Groups of channel indices to average.
        use_average (bool): Whether to average over parallel channel groups.

    Returns:
        float: Maximum absolute cross-correlation value.
    """
    if parallel_channels is None:
        parallel_channels = [[0, 2], [1, 3]]

    max_corr = []
    if use_average:
        for group in parallel_channels:
            par_corr = sum(
                np.abs(
                    txc.get_xcorr_for_channel(
                        traces[i], template_trace, sampling_rate, template_sampling_rate
                    )
                )
                for i in group
            ) / len(group)
            max_corr.append(par_corr)
    else:
        max_corr = [
            np.abs(
                txc.get_xcorr_for_channel(
                    trace, template_trace, sampling_rate, template_sampling_rate
                )
            )
            for trace in traces
        ]

    return max(max_corr)


def get_max_all_chi(
    traces,
    sampling_rate,
    template_traces,
    template_sampling_rate,
    parallel_channels=None,
    exclude_match=None,
):
    """
    Compute maximum chi across all templates in a dictionary.

    Parameters:
        traces (list or array): Trace signals.
        sampling_rate (float): Sampling rate of traces.
        template_traces (dict): Dictionary of templates keyed by string.
        template_sampling_rate (float): Sampling rate of templates.
        parallel_channels (list of lists): Channel groups for averaging.
        exclude_match (str or None): Key to exclude from templates.

    Returns:
        float: Maximum chi value.
    """
    if parallel_channels is None:
        parallel_channels = [[0, 2], [1, 3]]

    return max(
        get_max_chi(traces, sampling_rate, trace, template_sampling_rate, parallel_channels)
        for key, trace in template_traces.items()
        if key != str(exclude_match)
    )


def load_single_template(series):
    """
    Load single template pickle for given series (e.g., 100 or 200).

    Returns:
        np.ndarray: Template array.
    """
    path = f"/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/templates/reflectedCR_template_{series}eries.pkl"
    templates = read_pickle(path)
    return next(iter(templates.values()))


def load_multiple_templates(series, date='3.29.25', add_single=False):
    """
    Load multiple templates for a given series and date.

    Parameters:
        series (int): Series number (100 or 200).
        date (str): Date string or '2016' for special case.
        add_single (bool): Whether to append single template to loaded templates.

    Returns:
        dict: Dictionary of templates.
    """
    templates = {}

    if date != '2016':
        template_dir = f"/pub/tangch3/ARIANNA/DeepLearning/RCR_templates/{date}/"
        for i, filename in enumerate(os.listdir(template_dir)):
            if filename.startswith(f'{series}'):
                path = os.path.join(template_dir, filename)
                templates[i] = np.load(path)
    else:
        path = "/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/templates/confirmed2016Templates/"
        for filename in os.listdir(path):
            template = np.load(os.path.join(path, filename))
            max_temp = max(template, key=lambda t: np.max(np.abs(t)))
            key = filename.split('_')[1]
            templates[key] = max_temp

    if add_single:
        templates[len(templates)] = load_single_template(series)

    return templates


def get_max_snr(traces, noise_rms):
    """
    Calculate the maximum SNR for a set of traces.

    Parameters:
        traces (list or array): Trace signals.
        noise_rms (float): Noise RMS value.

    Returns:
        float: Maximum SNR value.
    """
    snrs = [((np.max(t) + np.abs(np.min(t))) * units.V) / (2 * noise_rms) for t in traces]
    max_snr = max(snrs)

    if max_snr == 0:
        print("Warning: Max SNR is zero. Rechecking traces:")
        for i, t in enumerate(traces):
            print(f"Trace {i}: {t}")

    return max_snr


# ---------- DATA LOADING ---------- #


def load_npy_files(directory, station_id):
    """
    Load all numpy files starting with 'Stn{station_id}' from directory and concatenate them.

    Parameters:
        directory (str): Directory path.
        station_id (int or str): Station ID number.

    Returns:
        list: Concatenated list of all traces loaded.
    """
    matched = [
        np.load(os.path.join(directory, f), allow_pickle=True)
        for f in os.listdir(directory)
        if f.startswith(f"Stn{station_id}")
    ]
    data = []
    for file in matched:
        data.extend(file)
    return data


def load_data(data_type, amp_type, station_id):
    """
    Load data arrays depending on data type and station.

    Parameters:
        data_type (str): Data category, e.g., 'All_data', 'AboveCurve_data', 'Circled_data', 'new_chi_above_curve'.
        amp_type (str): Amplifier type, e.g., '200s'.
        station_id (int): Station number.

    Returns:
        tuple: Numpy arrays with data.
    """
    folder = f"/pub/tangch3/ARIANNA/DeepLearning/{data_type}"

    if data_type == 'new_chi_above_curve':
        folder = f"{folder}/new_chi/5000evt"
        return (
            np.load(f"{folder}/Stn{station_id}_SNR_above.npy"),
            np.load(f"{folder}/Stn{station_id}_Chi2016_above.npy"),
            np.load(f"{folder}/Stn{station_id}_ChiRCR_above.npy"),
            np.load(f"{folder}/Stn{station_id}_Traces2016_above.npy"),
            np.load(f"{folder}/Stn{station_id}_TracesRCR_above.npy"),
            np.load(f"{folder}/Stn{station_id}_UNIX_above.npy")
        )

    snr = np.load(f"{folder}/Station_SNR/{amp_type}/Stn{station_id}_SNR.npy")
    chi = np.load(f"{folder}/Station_Chi/{amp_type}/Stn{station_id}_Chi.npy")
    traces = load_npy_files(f"{folder}/Station_Traces/{amp_type}/", station_id)
    unix = np.load(f"{folder}/Station_UNIX/{amp_type}/Stn{station_id}_UNIX.npy")
    return snr, chi, traces, unix


# ---------- SIMULATION DATA ---------- #


def load_sim(path, rcr_path, backlobe_path, amp):
    """
    Load simulation data for RCR and Backlobe.

    Parameters:
        path (str): Base directory path.
        rcr_path (str): Subdirectory for RCR files.
        backlobe_path (str): Subdirectory for Backlobe files.
        amp (str): Amplifier type.

    Returns:
        tuple: Arrays of RCR and Backlobe simulation traces.
    """
    rcr_files = [
        os.path.join(path, rcr_path, f)
        for f in os.listdir(os.path.join(path, rcr_path))
        if f.startswith(f"FilteredSimRCR_{amp}_")
    ]

    rcr = np.empty((0, 4, 256))
    for file in rcr_files:
        rcr_data = np.load(file)[:, 0:4]
        rcr = np.concatenate((rcr, rcr_data))

    backlobe_files = [
        os.path.join(path, backlobe_path, f)
        for f in os.listdir(os.path.join(path, backlobe_path))
        if f.startswith(f"Backlobe_{amp}_")
    ]

    backlobe = np.empty((0, 4, 256))
    for file in backlobe_files:
        backlobe_data = np.load(file)[:, 0:4]
        backlobe = np.concatenate((backlobe, backlobe_data))

    return rcr, backlobe


def load_sim_rcr(sim_path, noise_enabled, filter_enabled, amp):
    """
    Load simulation RCR data with specific noise and filter flags.

    Parameters:
        sim_path (str): Path to simulation files.
        noise_enabled (bool): Noise enabled flag.
        filter_enabled (bool): Filter enabled flag.
        amp (str): Amplifier type.

    Returns:
        np.ndarray or None: Loaded simulation traces or None if not found.
    """
    noise_str = "NoiseTrue" if noise_enabled else "NoiseFalse"
    filter_str = "FilterTrue" if filter_enabled else "FilterFalse"

    base_prefix = f"SimRCR_{amp}_{noise_str}_forcedFalse_"
    base_suffix = f"events_{filter_str}_part0.npy"

    found_file = None
    for filename in os.listdir(sim_path):
        if filename.startswith(base_prefix) and filename.endswith(base_suffix):
            found_file = filename
            break

    if found_file:
        full_filepath = os.path.join(sim_path, found_file)
        sim_traces = np.load(full_filepath)
        return sim_traces
    else:
        print(f"No matching file found in '{sim_path}' for Noise={noise_enabled}, Filter={filter_enabled}.")
        return None


def siminfo_for_plotting(type_, amp, simulation_date, templates_2016, templates_RCR, noise_rms):
    """
    Load and process simulation data for plotting.

    Parameters:
        type_ (str): 'RCR' or 'Backlobe'
        amp (str): Amplifier type.
        simulation_date (str): Date string.
        templates_2016 (dict): Templates from 2016.
        templates_RCR (dict): RCR templates.
        noise_rms (float): Noise RMS.

    Returns:
        tuple: Sorted simulation arrays and metadata.
    """
    base_path = "/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/"
    if type_ == 'RCR':
        path = os.path.join(base_path, amp, simulation_date)
        if amp == '200s':
            sim = np.load(f"{path}/SimRCR_200s_NoiseFalse_forcedFalse_4344events_FilterTrue_part0.npy")
            weights = np.load(f"{path}/SimWeights_SimRCR_200s_NoiseFalse_forcedFalse_4344events_part0.npy")
        elif amp == '100s':
            sim = np.load(f"{path}/SimRCR_100s_NoiseFalse_forcedFalse_4826events_FilterTrue_part0.npy")
            weights = np.load(f"{path}/SimWeights_SimRCR_100s_NoiseFalse_forcedFalse_4826events_part0.npy")
    else:
        # Backlobe not implemented
        return

    sim_snr = []
    sim_chi2016 = []
    sim_chi_rcr = []
    sim_weights = []

    for i, event in enumerate(sim):
        traces = [trace * units.V for trace in event]
        sim_snr.append(get_max_snr(traces, noise_rms=noise_rms))
        sim_chi2016.append(get_max_all_chi(traces, 2 * units.GHz, templates_2016, 2 * units.GHz))
        sim_chi_rcr.append(get_max_all_chi(traces, 2 * units.GHz, templates_RCR, 2 * units.GHz))
        sim_weights.append(weights[i])

    sim_snr = np.array(sim_snr)
    sim_chi2016 = np.array(sim_chi2016)
    sim_chi_rcr = np.array(sim_chi_rcr)
    sim_weights = np.array(sim_weights)

    sort_order = sim_weights.argsort()
    sim = sim[sort_order]
    sim_snr = sim_snr[sort_order]
    sim_chi_rcr = sim_chi_rcr[sort_order]
    sim_chi2016 = sim_chi2016[sort_order]
    sim_weights = sim_weights[sort_order]

    return sim, sim_chi2016, sim_chi_rcr, sim_snr, sim_weights, simulation_date


# ---------- PLOTTING ---------- #


def pT(
    traces,
    title,
    save_loc,
    sampling_rate=2,
    show=False,
    average_fft_per_channel=None,
):
    """
    Plot time-domain and frequency-domain traces.

    Parameters:
        traces (list or array): List of channel traces.
        title (str): Title of the plot.
        save_loc (str): File path to save the plot.
        sampling_rate (float): Sampling rate in GHz.
        show (bool): Whether to display the plot.
        average_fft_per_channel (list or None): Optional average FFT per channel.
    """
    if average_fft_per_channel is None:
        average_fft_per_channel = []

    x = np.linspace(1, int(256 / sampling_rate), num=256)
    x_freq = np.fft.rfftfreq(len(x), d=(1 / sampling_rate * units.GHz)) / units.MHz

    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(6, 5), sharex=False)

    fmax = 0
    vmax = 0

    for ch_id, trace in enumerate(traces):
        trace = trace.reshape(len(trace))
        freqtrace = np.abs(fft.time2freq(trace, sampling_rate * units.GHz))

        axs[ch_id][0].plot(x, trace)

        if average_fft_per_channel:
            axs[ch_id][1].plot(x_freq, average_fft_per_channel[ch_id], color='gray', linestyle='--')
        axs[ch_id][1].plot(x_freq, freqtrace)

        fmax = max(fmax, max(freqtrace))
        vmax = max(vmax, max(trace))

        axs[ch_id][0].grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
        axs[ch_id][1].grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)

    axs[3][0].set_xlabel('time [ns]', fontsize=12)
    axs[3][1].set_xlabel('Frequency [MHz]', fontsize=12)

    for ch_id, trace in enumerate(traces):
        axs[ch_id][0].set_ylabel(f'ch{ch_id}', labelpad=10, rotation=0, fontsize=10)
        axs[ch_id][0].set_xlim(-3, 260 / sampling_rate)
        axs[ch_id][1].set_xlim(-3, 1000)
        axs[ch_id][0].tick_params(labelsize=10)
        axs[ch_id][1].tick_params(labelsize=10)
        axs[ch_id][0].set_ylim(-vmax * 1.1, vmax * 1.1)
        axs[ch_id][1].set_ylim(-0.05, fmax * 1.1)

    fig.text(0.05, 0.5, 'Voltage [V]', ha='right', va='center', rotation='vertical', fontsize=12)
    plt.suptitle(title)
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(left=0.175)

    if show:
        plt.show()
    else:
        plt.savefig(save_loc, format='png')

    plt.clf()
    plt.close(fig)


def combine_plots(plot_path, station_id, data_type):
    """
    Combine multiple PNG plot parts horizontally.

    Parameters:
        plot_path (str): Path to plots.
        station_id (int): Station ID.
        data_type (str): Data type string for filenames.
    """
    plot_part_1 = Image.open(f'{plot_path}/All_part_1_{data_type}NO.png')
    plot_part_2 = Image.open(f'{plot_path}/All_part_2_{data_type}NO.png')

    combined_width = plot_part_1.width + plot_part_2.width
    combined_height = max(plot_part_1.height, plot_part_2.height)

    if station_id == 17:
        plot_part_3 = Image.open(f'{plot_path}/All_part_3_{data_type}NO.png')
        combined_width += plot_part_3.width
        combined_height = max(combined_height, plot_part_3.height)

    combined = Image.new('RGB', (combined_width, combined_height))
    combined.paste(plot_part_1, (0, 0))
    combined.paste(plot_part_2, (plot_part_1.width, 0))

    if station_id == 17:
        combined.paste(plot_part_3, (plot_part_1.width + plot_part_2.width, 0))

    combined.save(f'{plot_path}/All_{data_type}NO_plot.png')


# ---------- MODEL RUNNING ---------- #


def run_trained_model(events, model_path):
    """
    Run a Keras model on event data.

    Parameters:
        events (np.ndarray): Event data array.
        model_path (str): Path to model file.

    Returns:
        np.ndarray: Model prediction probabilities.
    """
    model = keras.models.load_model(model_path)
    prob_events = model.predict(events)
    return prob_events


