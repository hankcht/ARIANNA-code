import os
import re
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Flatten, Dense
from pathlib import Path
import pickle
import templateCrossCorr as txc
import NuRadioReco
from NuRadioReco.utilities import units, fft
from NuRadioReco.utilities.io_utilities import read_pickle
import yaml


def load_config(config_path="/pub/tangch3/ARIANNA/DeepLearning/code/200s_time/config.yaml", station_id=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    amp = config['amp'] 

    config['noise_rms'] = config['noise_rms_200s'] * units.mV if amp == '200s' else config['noise_rms_100s'] * units.mV
    config['station_ids'] = config['station_ids_200s'] if amp == '200s' else config['station_ids_100s']

    return config
    
def getMaxChi(traces, sampling_rate, template_trace, template_sampling_rate, parallelChannels=[[0, 2], [1, 3]], use_average=False):
    #Parallel channels should be index corresponding to the channel in traces

    maxCorr = []
    if use_average:
        for parChans in parallelChannels:
            parCorr = 0
            for chan in parChans:
                xCorr = txc.get_xcorr_for_channel(traces[chan], template_trace, sampling_rate, template_sampling_rate)
                parCorr += np.abs(xCorr)
            maxCorr.append(parCorr / len(parChans))
    else:
        for trace in traces:
            xCorr = txc.get_xcorr_for_channel(trace, template_trace, sampling_rate, template_sampling_rate)
            maxCorr.append(np.abs(xCorr))

    return max(maxCorr)

def getMaxAllChi(traces, sampling_rate, template_traces, template_sampling_rate, parallelChannels=[[0, 2], [1, 3]], exclude_match=None):
    # get Maximum Chi across all RCR templates

    maxCorr = []
    for key in template_traces:
        # ic(key, exclude_match, key == str(exclude_match))
        if key == str(exclude_match):
            continue
        trace = template_traces[key]
        maxCorr.append(getMaxChi(traces, sampling_rate, trace, template_sampling_rate, parallelChannels=parallelChannels))

    return max(maxCorr)

def loadSingleTemplate(series):
    # Series should be 200 or 100
    # Loads the first version of a template made for an average energy/zenith
    templates_RCR = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/templates/reflectedCR_template_{series}eries.pkl' 
    templates_RCR = read_pickle(templates_RCR)
    for key in templates_RCR:
        temp = templates_RCR[key]
    templates_RCR = temp

    return templates_RCR

def loadMultipleTemplates(series, date='3.29.25', addSingle=False, bad=False):
    # Dates - 9.16.24 (noise included), 10.1.24 (no noise)
    #       - 2016 : found backlobe events from 2016
    #       - 3.29.25 : noiseless, 100s and 200s, pruned by-hand for 'good' templates

    # 10.1.24 has issues with the templates, so use 9.16.24
    # Series should be 200 or 100
    # Loads all the templates made for an average energy/zenith
    template_series_RCR = {}
    if not date == '2016':
        template_series_RCR_location = f'/pub/tangch3/ARIANNA/DeepLearning/RCR_templates/{date}/' 
        i = 0
        for filename in os.listdir(template_series_RCR_location):
            if filename.startswith(f'{series}'):
                temp = np.load(os.path.join(template_series_RCR_location, filename))
                template_series_RCR[i] = temp
                i += 1
    else:
        templates_2016_location = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/templates/confirmed2016Templates/'
        for filename in os.listdir(templates_2016_location):
            temp = np.load(os.path.join(templates_2016_location, filename))
            # Only use the channel of highest amplitude
            max_temp = [0]
            for t in temp:
                if max(np.abs(t)) > max(max_temp):
                    max_temp = t
            key = filename.split('_')[1]
            template_series_RCR[key] = max_temp

    if addSingle:
        template_series_RCR.append(loadSingleTemplate(series))

    return template_series_RCR

def getMaxSNR(traces, noiseRMS):

    SNRs = []
    for trace in traces:
        p2p = (np.max(trace) + np.abs(np.min(trace))) * units.V
        SNRs.append(p2p / (2*noiseRMS))
    
    if max(SNRs)==0:
        print(f'zero error')
        SNRs = []
        for trace in traces:
            print(f'trace {trace}')
            p2p = (np.max(trace) + np.abs(np.min(trace))) * units.V
            SNRs.append(p2p / (2*noiseRMS))

    return max(SNRs)

def load_data(type, amp_type, station_id):

    data_folder = f'/pub/tangch3/ARIANNA/DeepLearning/{type}'

    if type == 'All_data':
        print(f'using {type} files')
        All_data_SNR = np.load(f'{data_folder}/Station_SNR/{amp_type}/Stn{station_id}_SNR.npy')
        All_data_Chi = np.load(f'{data_folder}/Station_Chi/{amp_type}/Stn{station_id}_Chi.npy')

        temporary_Traces = []
        for filename in os.listdir(f'{data_folder}/Station_Traces/{amp_type}/'):
            if filename.startswith(f'Stn{station_id}'):
                print(filename)
                temporary_Traces.append(np.load(f'{data_folder}/Station_Traces/{amp_type}/{filename}'))
        
        All_data_Traces = []
        for file in temporary_Traces:
            All_data_Traces.extend(file)
        
        All_data_UNIX = np.load(f'{data_folder}/Station_UNIX/{amp_type}/Stn{station_id}_UNIX.npy')

        return All_data_SNR, All_data_Chi, All_data_Traces, All_data_UNIX
    

    if type == 'AboveCurve_data':
        print(f'using {type}')        
        above_curve_data_SNR = np.load(f'{data_folder}/Station_SNR/{amp_type}/Stn{station_id}_SNR.npy')
        above_curve_data_Chi = np.load(f'{data_folder}/Station_Chi/{amp_type}/Stn{station_id}_Chi.npy')

        temporary_Traces = []
        for filename in os.listdir(f'{data_folder}/Station_Traces/{amp_type}/'):
            if filename.startswith(f'Stn{station_id}'):
                print(filename)
                temporary_Traces.append(np.load(f'{data_folder}/Station_Traces/{amp_type}/{filename}'))
        
        above_curve_data_Traces = []
        for file in temporary_Traces:
            above_curve_data_Traces.extend(file)

        above_curve_data_UNIX = np.load(f'{data_folder}/Station_UNIX/{amp_type}/Stn{station_id}_UNIX.npy')

        return above_curve_data_SNR, above_curve_data_Chi, above_curve_data_Traces, above_curve_data_UNIX
        

    if type == 'Circled_data':
        print(f'using {type}')
        amp_type = '200s'
        Circled_data_SNR = np.load(f'{data_folder}/Station_SNR/{amp_type}/Stn{station_id}_SNR.npy')
        Circled_data_Chi = np.load(f'{data_folder}/Station_Chi/{amp_type}/Stn{station_id}_Chi.npy')

        temporary_Traces = []
        for filename in os.listdir(f'{data_folder}/Station_Traces/{amp_type}/'):
            if filename.startswith(f'Stn{station_id}'):
                print(filename)
                temporary_Traces.append(np.load(f'{data_folder}/Station_Traces/{amp_type}/{filename}'))
        
        Circled_data_Traces = []
        for file in temporary_Traces:
            Circled_data_Traces.extend(file)

        Circled_data_UNIX = np.load(f'{data_folder}/Station_UNIX/{amp_type}/Stn{station_id}_UNIX.npy')

        return Circled_data_SNR, Circled_data_Chi, Circled_data_Traces, Circled_data_UNIX
    
    if type == 'new_chi_above_curve':
        print(f'loading {type} data')        
        Above_curve_data_folder =  f'/pub/tangch3/ARIANNA/DeepLearning/AboveCurve_data/new_chi/5000evt'
        Above_curve_data_SNR = np.load(f'{Above_curve_data_folder}/Stn{station_id}_SNR_above.npy')
        Above_curve_data_Chi2016 = np.load(f'{Above_curve_data_folder}/Stn{station_id}_Chi2016_above.npy')
        Above_curve_data_ChiRCR = np.load(f'{Above_curve_data_folder}/Stn{station_id}_ChiRCR_above.npy')
        Above_curve_data_Traces2016 = np.load(f'{Above_curve_data_folder}/Stn{station_id}_Traces2016_above.npy')
        Above_curve_data_TracesRCR = np.load(f'{Above_curve_data_folder}/Stn{station_id}_TracesRCR_above.npy')
        Above_curve_data_UNIX = np.load(f'{Above_curve_data_folder}/Stn{station_id}_UNIX_above.npy')

        return Above_curve_data_SNR, Above_curve_data_Chi2016, Above_curve_data_ChiRCR, Above_curve_data_Traces2016, Above_curve_data_TracesRCR, Above_curve_data_UNIX

def load_sim(path, RCR_path, backlobe_path, amp):
    RCR_files = []
    print(f'path {path + RCR_path}')
    for filename in os.listdir(path + RCR_path):
        if filename.startswith(f'FilteredSimRCR_{amp}_'):
            RCR_files.append(path + RCR_path +  filename)
    rcr = np.empty((0, 4, 256))
    for file in RCR_files:
        print(f'RCR file {file}')
        RCR_data = np.load(file)[0:, 0:4]
        print(f'RCR data shape {RCR_data.shape} and RCR shape {rcr.shape}')
        rcr = np.concatenate((rcr, RCR_data))
    
    Backlobes_files = []
    for filename in os.listdir(path + backlobe_path):
        if filename.startswith(f'Backlobe_{amp}_'):
            Backlobes_files.append(path + backlobe_path + filename)
    Backlobe = np.empty((0, 4, 256))
    for file in Backlobes_files:
        print(f'Backlobe file {file}')
        Backlobe_data = np.load(file)[0:, 0:4]
        Backlobe = np.concatenate((Backlobe, Backlobe_data))
    
    # # prints out every byte in this RCR file, was printing only zeros
    # with open('/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/simulatedRCRs/200s_10.30.24/SimRCR_200s_forcedTrue_5214events_part0.npy', mode="rb") as f:
    #     data = f.read()
    #     for c in data:
    #         print(c, end = " ")

    print(f'loaded sim RCR {len(rcr)} Backlobe {len(Backlobe)}')

    return rcr, Backlobe

def load_sim_rcr(sim_path: str, noise_enabled: bool, filter_enabled: bool, amp) -> np.ndarray:
    """
    The expected file format is:
    'SimRCR_200s_Noise{True/False}_forcedFalse_{events}events_Filter{True/False}_part0.npy'
    """
    noise_str = "NoiseTrue" if noise_enabled else "NoiseFalse"
    filter_str = "FilterTrue" if filter_enabled else "FilterFalse"

    base_prefix = f"SimRCR_{amp}_{noise_str}_forcedFalse_"
    base_suffix = f"events_{filter_str}_part0.npy"

    found_file = None
    print(f"Searching for file in: '{sim_path}' matching pattern '{base_prefix}*{base_suffix}'")

    for filename in os.listdir(sim_path):
        if filename.startswith(base_prefix) and filename.endswith(base_suffix):
            if f"events_{filter_str}" in filename:
                found_file = filename
                break # Found the first matching file, assuming only one per combination

    if found_file:
        full_filepath = os.path.join(sim_path, found_file)
        print(f"Found and loading: '{full_filepath}'")
        sim_Traces = np.load(full_filepath)
        print(f"Successfully loaded data with shape: {sim_Traces.shape}")
        return sim_Traces
    else:
        print(f"No matching file found in '{sim_path}' for Noise='{noise_enabled}', Filter='{filter_enabled}'.")
        return None

def siminfo_forplotting(type, amp, simulation_date, templates_2016, templates_RCR, noiseRMS):
    # alter path to load desired sim 
    path = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/'
    
    files = []
    if type == 'RCR':
        path += f'{amp}/{simulation_date}'
        if amp == '200s':
            sim = np.load(f'{path}/SimRCR_200s_NoiseFalse_forcedFalse_4344events_FilterTrue_part0.npy')
            weights = np.load(f'{path}/SimWeights_SimRCR_200s_NoiseFalse_forcedFalse_4344events_part0.npy')
        elif amp == '100s':
            sim = np.load(f'{path}/SimRCR_100s_NoiseFalse_forcedFalse_4826events_FilterTrue_part0.npy')
            weights = np.load(f'{path}/SimWeights_SimRCR_100s_NoiseFalse_forcedFalse_4826events_part0.npy')

    elif type == 'Backlobe':
        # not yet developed 
        return 

    sim_SNRs = []
    sim_Chi2016 = []
    sim_ChiRCR = []
    sim_weights = []
    for iR, event in enumerate(sim):
            
        traces = []
        for trace in event:
            traces.append(trace * units.V)
        sim_SNRs.append(getMaxSNR(traces, noiseRMS=noiseRMS))
        sim_Chi2016.append(getMaxAllChi(traces, 2*units.GHz, templates_2016, 2*units.GHz))
        sim_ChiRCR.append(getMaxAllChi(traces, 2*units.GHz, templates_RCR, 2*units.GHz))
        sim_weights.append(weights[iR])
        
    SNRbins = np.logspace(0.477, 2, num=80)
    maxCorrBins = np.arange(0, 1.0001, 0.01)

    sim_weights = np.array(sim_weights)
    sim_SNRs = np.array(sim_SNRs)
    sim_Chi2016 = np.array(sim_Chi2016)
    sim_ChiRCR = np.array(sim_ChiRCR)

    sort_order = sim_weights.argsort()
    sim = sim[sort_order]
    sim_SNRs = sim_SNRs[sort_order]
    sim_ChiRCR = sim_ChiRCR[sort_order]
    sim_Chi2016 = sim_Chi2016[sort_order]
    sim_weights = sim_weights[sort_order]

    if type == 'RCR':
        cmap = 'seismic'
    else:
        cmap = 'PiYG'

    # plt.scatter(sim_SNRs, sim_Chi2016, c=sim_weights, cmap=cmap, alpha=0.9, norm=matplotlib.colors.LogNorm())

    return sim, sim_Chi2016, sim_ChiRCR, sim_SNRs, sim_weights, simulation_date

def pT(traces, title, saveLoc, sampling_rate=2, show=False, average_fft_per_channel=[]):
    # Sampling rate should be in GHz
    print(f'printing')
    # Important Clarification: In our actual experiment, we receive one data point per 0.5ns, so our duration of 128ns gives 256 data points
    # it is different from here where I organize one data point to one ns and make the total time 256ns (these two are mathematically identical)
    x = np.linspace(1, int(256 / sampling_rate), num=256)
    x_freq = np.fft.rfftfreq(len(x), d=(1 / sampling_rate * units.GHz)) / units.MHz

    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(6, 5), sharex=False)
    fmax = 0
    vmax = 0

    for chID, trace in enumerate(traces):
        trace = trace.reshape(len(trace)) 
        freqtrace = np.abs(fft.time2freq(trace, sampling_rate * units.GHz))

        # Plot time-domain trace
        axs[chID][0].plot(x, trace)
        
        # Plot frequency-domain trace and average FFT if provided
        if len(average_fft_per_channel) > 0:
            axs[chID][1].plot(x_freq, average_fft_per_channel[chID], color='gray', linestyle='--')
        axs[chID][1].plot(x_freq, freqtrace)

        # Update fmax and vmax for axis limits
        fmax = max(fmax, max(freqtrace))
        vmax = max(vmax, max(trace))

        # Add grid to each subplot
        axs[chID][0].grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)  # Time-domain grid
        axs[chID][1].grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)  # Frequency-domain grid

    # Set axis labels
    axs[3][0].set_xlabel('time [ns]', fontsize=12)
    axs[3][1].set_xlabel('Frequency [MHz]', fontsize=12)

    for chID, trace in enumerate(traces):
        axs[chID][0].set_ylabel(f'ch{chID}', labelpad=10, rotation=0, fontsize=10)
        axs[chID][0].set_xlim(-3, 260 / sampling_rate)
        axs[chID][1].set_xlim(-3, 1000)
        axs[chID][0].tick_params(labelsize=10)
        axs[chID][1].tick_params(labelsize=10)

        # Set y-axis limits
        axs[chID][0].set_ylim(-vmax * 1.1, vmax * 1.1)
        axs[chID][1].set_ylim(-0.05, fmax * 1.1)

    axs[0][0].tick_params(labelsize=10)
    axs[0][1].tick_params(labelsize=10)
    axs[0][0].set_ylabel(f'ch{0}', labelpad=10, rotation=0, fontsize=10)

    # Final x and y axis limits
    axs[chID][0].set_xlim(-3, 260 / sampling_rate)
    axs[chID][1].set_xlim(-3, 1000)

    # Add a common y-axis label for the entire figure
    fig.text(0.05, 0.5, 'Voltage [V]', ha='right', va='center', rotation='vertical', fontsize=12)

    plt.suptitle(title)

    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(left=0.175) 

    if show:
        plt.show()
    else:
        print(f'saving to {saveLoc}')
        plt.savefig(saveLoc, format='png')
    
    plt.clf()
    plt.close(fig)

    return

def combine_plots(plot_path, station_id, type):
    
    plot_part_1 = Image.open(f'{plot_path}/All_part_1_{type}NO.png')
    plot_part_2 = Image.open(f'{plot_path}/All_part_2_{type}NO.png')

    # We might take different partitions, so the general form applies
    combined_width = plot_part_1.width + plot_part_2.width
    combined_height = max(plot_part_1.height, plot_part_2.height)


    if station_id == 17:
        plot_part_3 = Image.open(f'{plot_path}/All_part_3_{type}NO.png')
        combined_width += plot_part_3.width
        combined_height = max(plot_part_1.height, plot_part_2.height, plot_part_3.height)       


    combined = Image.new('RGB', (combined_width, combined_height))

    combined.paste(plot_part_1, (0, 0))
    combined.paste(plot_part_2, (plot_part_1.width, 0))

    if station_id == 17:
        combined.paste(plot_part_3, (plot_part_1.width + plot_part_2.width, 0))

    combined.save(f'{plot_path}/All_{type}NO_plot.png')

def RunTrainedModel(events, model_path):

    # window size 10 on first layer
    # FIRST TRY data_data_2025-01-10_22-31_RCR_Backlobe_model_2Layer.h5
    # BEST      data_data_2025-01-28_22-52_RCR_Backlobe_model_2Layer.h5
    # CURRENT /pub/tangch3/ARIANNA/DeepLearning/models/200s_time/data_data_2025-01-30_16-38_RCR_Backlobe_model_2Layer.h5
    
    # window size 30 on first layer
    # 
    # CURRENT DeepLearning/models/200s_time/data_data_2025-01-30_21-38_RCR_Backlobe_model_2Layer.h5

    model = keras.models.load_model(f'{model_path}200s_time/data_data_2025-01-30_16-38_RCR_Backlobe_model_2Layer.h5')
    prob_events = model.predict(events)
    

    return prob_events

def profiling():
    start = time.time()

    end = time.time()
    print(f"task: {end - start}s")
    start = end

def deleting():
    # --- to delete files ---
    directory = '/pub/tangch3/ARIANNA/DeepLearning/logs'
    for i in range(154):
        files_to_delete = glob.glob(os.path.join(directory, f'Stn17_{i}.out'))

        for file in files_to_delete:
            os.remove(file)
            print(f'Deleted :{file}')

def load_520_data(station_id, param, data_folder, date_filter="5.20.25", single_load = True):
    '''
    quick load function for 5/20 after nosie cut data with very specific filenames
    from: '/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/5.20.25/'
    param: Azi, Chi2016, ChiRCR, ChiBad, EventIDs, MaxAmplitude, SNR, Times, Traces, Zen
    '''
    if single_load:
        pattern = os.path.join(data_folder, f"{date_filter}_Station{station_id}_{param}*")
        matched_files = sorted(glob.glob(pattern))

        data_list = [np.load(f, allow_pickle=True) for f in matched_files]

        if not data_list:
            print(f"No files found for Station {station_id}, Parameter '{param}'")
            return None

        data = np.concatenate(data_list, axis=0).squeeze()
        return data
    else:
        SNR_files = sorted(glob.glob(os.path.join(data_folder, f"{date_filter}_Station{station_id}_SNR*")))
        Chi2016_files = sorted(glob.glob(os.path.join(data_folder, f"{date_filter}_Station{station_id}_Chi2016*")))
        ChiRCR_files = sorted(glob.glob(os.path.join(data_folder, f"{date_filter}_Station{station_id}_ChiRCR*")))
        Times_files = sorted(glob.glob(os.path.join(data_folder, f"{date_filter}_Station{station_id}_Times*")))
        Traces_files = sorted(glob.glob(os.path.join(data_folder, f"{date_filter}_Station{station_id}_Traces*")))

        SNR_list = [np.load(f, allow_pickle=True) for f in SNR_files]
        Chi2016_list = [np.load(f, allow_pickle=True) for f in Chi2016_files]
        ChiRCR_list = [np.load(f, allow_pickle=True) for f in ChiRCR_files]
        Times_list = [np.load(f, allow_pickle=True) for f in Times_files]
        Traces_list = [np.load(f, allow_pickle=True) for f in Traces_files]

        SNRs = np.concatenate(SNR_list, axis=0).squeeze()
        Chi2016 = np.concatenate(Chi2016_list, axis=0).squeeze()
        ChiRCR = np.concatenate(ChiRCR_list, axis=0).squeeze()
        Times = np.concatenate(Times_list, axis=0).squeeze()
        Traces = np.concatenate(Traces_list, axis=0).squeeze()

        return {'SNR': SNRs, 'Chi2016': Chi2016, 'ChiRCR': ChiRCR, 'Times': Times, 'Traces': Traces}
    


def load_coincidence_pkl(master_id, argument, station_id,
    pkl_path="/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/6.11.25_CoincidenceDatetimes_with_all_params_recalcZenAzi_calcPol.pkl"
):
    """
    Load data from coincidence pickle file for a master event and argument.

    Args:
        master_id (int): master event ID.
        argument (str): key under the event.
        station_id (str): station ID key or empty string.
        pkl_path (str): path to the pickle file.

    Returns:
        value corresponding to keys or empty dict if not found.
    """
    with open(pkl_path, "rb") as f:
        coinc_dict = pickle.load(f)

    value = coinc_dict.get(master_id, {})
    if not value:
        return {}

    sub_value = value.get(argument, {})
    if not sub_value:
        return {}

    if station_id == "":
        return sub_value

    return sub_value.get(station_id, {})


if __name__ == "__main__":

    sim_rcr_730 = np.load('/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/7.30.25/200s/all_traces_200s_RCR_part0_50000events.npy', allow_pickle=True)
    print(f'number of traces is {len(sim_rcr_730)}')
    print(sim_rcr_730[0])
    print(type(sim_rcr_730[0]))
    print(sim_rcr_730.shape)
    print([np.shape(ch) for ch in sim_rcr_730[0]])

    print(sim_rcr_730.shape) 

    # for i in range(5):
    #     saveLoc = f'/pub/tangch3/ARIANNA/DeepLearning/refactor/other/test_plot_730_sim_rcr_{i}.png'
    #     pT(sim_rcr_730[i], f'7/30 sim RCR event, index: {i}', saveLoc)





















