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

    station_id = [14, 17, 19, 30, 13, 15, 18]
    parameters = ['ChiRCR', 'Chi2016']
    plot_folder = f'/pub/tangch3/ARIANNA/DeepLearning'
    extraname = 'quicktest'
    if_sim = ''
    station_data_folder = '/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/5.20.25/'
    date_filter = '5.20.25'

    file_path = "/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/6.11.25_CoincidenceDatetimes_with_all_params_recalcZenAzi_calcPol.pkl"

    amp = '200s'

    '''run all station data'''
    # data_folder = '/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/5.20.25/'
    # data = load_520_data(14, '', data_folder, single_load=False)
    # All_Traces = data['Traces']

    # model_path = f'/pub/tangch3/ARIANNA/DeepLearning/models/200s_time/new_chi'
    # model_filename = '2025-07-21_12-00_RCR_Backlobe_model_2Layer.h5'
    # model = keras.models.load_model(os.path.join(model_path, model_filename))

    # All_Traces_expanded = np.expand_dims(All_Traces, axis=-1)

    # no = model.predict(All_Traces_expanded)

    # no = no.flatten()
    # original_size = len(no)
    # no = no[np.isfinite(no)]
    # filtered_size = len(no)

    # if original_size != filtered_size:
    #     print(f"Warning: Removed {original_size - filtered_size} non-finite values (NaN/Inf) from 'no' before plotting.")

    # plt.figure(figsize=(10, 6))
    # plt.hist(no, bins='auto', edgecolor='black')
    # plt.ylim(bottom=0)
    # plt.title('Histogram of CNN Model Output')
    # plt.xlabel('Output Value (e.g., Probability, Score)')
    # plt.ylabel('Number of Events (Frequency)')
    # plt.grid(axis='y', alpha=0.75)

    # plot_path = '/pub/tangch3/ARIANNA/DeepLearning/refactor/other/'
    # os.makedirs(plot_path, exist_ok=True)
    # plt.savefig(f'{plot_path}testplotall_netoutput__RCR_Backlobe_model_2Layer.png')

    # print(f"Histogram saved to: {plot_path}testplotall_netoutput__RCR_Backlobe_model_2Layer.png")
    '''run on event 578'''
    # data =[]
    # for id in station_id:
    #     snr, chi2016, chiRCR, traces2016, tracesRCR, unix = load_data('new_chi_above_curve', amp_type = amp, station_id=id)
    #     data.extend(tracesRCR)

    # data = np.array(data)
    # master_id = 578
    # argument = 'stations'
    
    # coinc_data = load_coincidence_pkl(master_id, argument, 13)
    # event = coinc_data['Traces']
    # print(isinstance(event, list))  
    # event = np.array(event)
    # print(event.shape)

    # model_path = f'/pub/tangch3/ARIANNA/DeepLearning/models/200s_time/new_chi'
    # model = keras.models.load_model(f'{model_path}/2025-07-21_12-00_RCR_Backlobe_model_2Layer.h5')
    # no = model.predict(event)
    # print(no)
    # event_id = coinc_data['event_ids']
    # print(f'event id is {event_id}')
    # prob_Backlobe = model.predict(data)



    # network_output_plot_path = f'/pub/tangch3/ARIANNA/DeepLearning/plots/A1_Training/Network_Output'

    # dense_val = False
    # plt.figure(figsize=(8, 6))
    # plt.hist(prob_Backlobe, bins=20, range=(0,1), histtype='step', color='blue', linestyle='solid', label=f'Backlobe {len(prob_Backlobe)}', density=dense_val)

    # plt.xlabel('Network Output', fontsize=18)
    # plt.ylabel('Number of Events', fontsize=18)
    # plt.yscale('log')
    # plt.title(f'{amp}_time RCR-Backlobe network output')
    # plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
    # plt.yticks(fontsize=18)

    # hist_values, bin_edges, _ = plt.hist(prob_Backlobe, bins=20, range=(0,1), histtype='step')
    # plt.ylim(0, max(10 ** (np.ceil(np.log10(hist_values)))))
    # plt.tick_params(axis='both', which='major', labelsize=12)
    # plt.legend(loc='upper left', fontsize=8)
    # # plt.text(0.375, 0.75, f'RCR efficiency: {RCR_efficiency}%', fontsize=12, transform=plt.gcf().transFigure)
    # # plt.text(0.375, 0.7, f'Backlobe efficiency: {Backlobe_efficiency}%', fontsize=12, transform=plt.gcf().transFigure)
    # # plt.text(0.375, 0.65, f'TrainCut: {TrainCut}', fontsize=12, transform=plt.gcf().transFigure)
    # # plt.axvline(x=output_cut_value, color='y', label='cut')
    # plt.text(0.05, -0.12, 'BL', verticalalignment='center', horizontalalignment='center', fontsize=12, transform=plt.gca().transAxes, color='blue')
    # plt.text(0.96, -0.12, 'RCR', verticalalignment='center', horizontalalignment='center', fontsize=12, transform=plt.gca().transAxes, color='red')
    # plt.subplots_adjust(left=0.2, right=0.85, bottom=0.2, top=0.8)

    # print(f'saving /pub/tangch3/ARIANNA/DeepLearning/plots/Simulation/network_output/{amp}_time/new_chi/test_histogram.png')
    # plt.savefig(f'{network_output_plot_path}/test_{amp}_histogram.png')



    '''find event 578'''
    # eventid = load_520_data(13, 'EventIDs', station_data_folder)
    # for evtid in eventid:
    #     if evtid == 17121:
    #         print('found event 578')


    # index = np.where(eventid == 17121)[0]
    # idx = index[0] 
    
    # snrs = load_520_data(13, 'SNR', station_data_folder)
    # chircrs = load_520_data(13, 'ChiRCR', station_data_folder)
    # chi2016s = load_520_data(13, 'Chi2016', station_data_folder)
    # times = load_520_data(13, 'Times', station_data_folder)

    # snr = snrs[idx]
    # chircr = chircrs[idx]
    # chi2016 = chi2016s[idx]
    # thetime = times[idx]

    # from datetime import datetime, timezone
    # utc_time = datetime.fromtimestamp(thetime, tz=timezone.utc)

    # print("UTC time:", utc_time)
    # print(snr, chircr, chi2016, thetime)

    '''test load coincidence pickle'''
    # test = np.load(f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/Station14_SNR_Chi.npy', allow_pickle=True)
    # print('loaded test')

    # for id in station_id:
    #     for param in parameters:
    #         chi = load_coincidence_pkl(id, param) 
    #         print(len(chi))

    '''load and plot SNR-ChiRCR/2016'''
    # for id in station_id:
    #     snr = load_520_data(id, 'SNR', station_data_folder)
    #     for param in parameters:
    #         chi = load_520_data(id, param, station_data_folder)

    #         SNRbins = np.logspace(0.477, 2, num=80)
    #         maxCorrBins = np.arange(0, 1.0001, 0.01)
    #         plt.hist2d(snr, chi, bins=[SNRbins, maxCorrBins], norm=matplotlib.colors.LogNorm())
    #         plt.colorbar()
    #         plt.xlim((3, 100))
    #         plt.ylim((0, 1))
    #         plt.xlabel('SNR')
    #         plt.ylabel('Avg Chi Highest Parallel Channels')
    #         # plt.legend()
    #         plt.xscale('log')
    #         plt.tick_params(axis='x', which='minor', bottom=True)
    #         plt.grid(visible=True, which='both', axis='both') 
    #         plt.title(f'Station {id} - SNR vs. Chi (Events: {num:,})')
    #         print(f'Saving {plot_folder}/{extraname}Stn{id}_SNR-Chi{param}_All{if_sim}.png')
    #         # plt.scatter(sim_SNRs, sim_Chi, c=sim_weights, cmap=cmap, alpha=0.9, norm=matplotlib.colors.LogNorm())
    #         plt.savefig(f'{plot_folder}/{extraname}Stn{id}_SNR-Chi{param}_All{if_sim}.png')
    #         plt.close()

    '''test model on different events'''
    # station_id = [14,17,19,30]
    # amp_type = '200s'
    # for id in station_id:
    #     '''load old rcr'''
    #     # path = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/'                                                                                                     #Set which amplifier to run on
    #     # RCR_path = f'simulatedRCRs/{amp_type}_2.9.24/'
    #     # backlobe_path = f'simulatedBacklobes/{amp_type}_2.9.24/'
    #     # rcr, sim_Backlobe = load_sim(path, RCR_path, backlobe_path, amp_type)
    #     '''load new rcr'''
    #     sim_folder = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/{amp_type}/5.28.25/'
    #     sim_rcr = load_sim_rcr(sim_folder, noise_enabled=True, filter_enabled=True, amp=amp_type)

    #     '''load old above curve'''
    #     # All_data_SNR, All_data_Chi, All_Traces, All_data_UNIX = load_data('AboveCurve_data', amp_type, id)
    #     '''load new above curve'''
    #     Above_curve_data_SNR, Above_curve_data_Chi2016, Above_curve_data_ChiRCR, all_Traces2016, all_TracesRCR, Above_curve_data_UNIX = load_data('new_chi_above_curve', amp_type, id)
        
    #     '''select 200 events for each station'''
    #     All_Traces = np.array(all_TracesRCR)
    #     # num_total_events = All_Traces.shape[0]
    #     # selected_indices = np.random.choice(num_total_events, 200, replace=False)
    #     # random_200_events = All_Traces[selected_indices]

    #     '''predict with old model'''
    #     # network_output = RunTrainedModel(random_200_events, '/pub/tangch3/ARIANNA/DeepLearning/models/')
    #     # prob_RCR = RunTrainedModel(sim_rcr, '/pub/tangch3/ARIANNA/DeepLearning/models/')
    #     '''predict with new model'''
    #     model = keras.models.load_model(f'/pub/tangch3/ARIANNA/DeepLearning/models/200s_time/new_chi_2025-06-20_16-06_RCR_Backlobe_model_2Layer.h5')
    #     network_output = model.predict(All_Traces)
    #     prob_RCR = model.predict(sim_rcr)

    #     plt.figure(figsize=(10, 6))
    #     plt.hist(network_output, bins=50, range=(0, 1), edgecolor='black', alpha=0.7)
    #     plt.hist(prob_RCR, bins=20, range=(0,1), histtype='step', color='red', linestyle='solid', label=f'RCR {len(prob_RCR)}', density=False)
    #     plt.title(f'Distribution of Network Output for Station {id} {len(network_output)} events')
    #     plt.xlabel('Network Output (Probability)')
    #     plt.ylabel('Number of Events')
    #     plt.yscale('log')
    #     plt.grid(axis='y', alpha=0.75)
        # plt.axvline(x=0.9, color='red', linestyle='--', label='Threshold = 0.9')
        # plt.legend()
        # plt.show() 
        # plot_output_dir = '/pub/tangch3/ARIANNA/DeepLearning/'
        # os.makedirs(plot_output_dir, exist_ok=True)
        # plt.savefig(os.path.join(plot_output_dir, f'new_mod_on_RCRtemplate_network_output_distribution_stn{id}.png'))
        # plt.clf() # Clear the current figure

        # '''plot RCR like events'''
        # threshold = 0.9
        # high_output_indices = np.where(network_output > threshold)[0]
        # events_above_threshold_traces = All_Traces[high_output_indices]

    #     print(f"\nTotal events: {len(network_output)}")
    #     print(f"Events with network output > {threshold}: {len(high_output_indices)}")
    #     print(f"Shape of traces for events above threshold: {events_above_threshold_traces.shape}")

    #     plot_output_directory = '/pub/tangch3/ARIANNA/DeepLearning/potential_RCR_plots/'
    #     os.makedirs(plot_output_directory, exist_ok=True)

    #     for event_data, original_index in zip(events_above_threshold_traces, high_output_indices):
    #         plot_filename = os.path.join(plot_output_directory, f'7.2_potential_RCR_event_original_idx_{original_index}.png')
    #         pT(event_data, f'Potential RCR (Original Event Index: {original_index})', plot_filename)
    #         print(f"Plotting and saving event with original index {original_index} to {plot_filename}")

    ''''''

    '''train new CNN with 5000 evt'''
    # from A1_TrainAndRunCNN import Train_CNN
    # amp_type = '200s'
    # if amp_type == '200s':
    #     noiseRMS = 22.53 * units.mV
    #     station_id = [14,17,19,30]
    # model_path = f'/pub/tangch3/ARIANNA/DeepLearning/models/{amp_type}_time/new_chi' 
    # loss_accuracy_plot_path = f'/pub/tangch3/ARIANNA/DeepLearning/plots/A1_Training/Loss_Accuracy' 
    # network_output_plot_path = f'/pub/tangch3/ARIANNA/DeepLearning/plots/A1_Training/Network_Output'   
    # from datetime import datetime
    # current_datetime = datetime.now() # Get the current date and time
    # timestamp = current_datetime.strftime("%Y-%m-%d_%H-%M") # Format the datetime object as a string with seconds

    # '''load sim'''
    # sim_folder = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/{amp_type}/5.28.25/'
    # # sim_rcr = load_sim_rcr(sim_folder, noise_enabled=True, filter_enabled=True, amp=amp_type)

    # templates_2016 = loadMultipleTemplates(200, date='2016')
    # templates_RCR = loadMultipleTemplates(200) 
    # noiseRMS = 22.53 * units.mV
    # sim, sim_Chi2016, sim_ChiRCR, sim_SNRs, sim_weights, simulation_date = siminfo_forplotting('RCR', '200s', '5.28.25', templates_2016, templates_RCR, noiseRMS)

    # no_red_indices = np.where(sim_ChiRCR > 0.55)[0]
    # no_red_sim = sim[no_red_indices]

    # '''load above curve'''
    # data_Backlobe = []
    # data_Backlobe_chi2016 = []
    # data_Backlobe_UNIX = [] 
    # for id in station_id:
    #     snr, chi2016, chiRCR, traces2016, tracesRCR, unix = load_data('new_chi_above_curve', amp_type, station_id=id)
    #     data_Backlobe.extend(traces2016)
    #     data_Backlobe_chi2016.extend(chi2016)
    #     data_Backlobe_UNIX.extend(unix)

    # no_red_sim = np.array(no_red_sim) 
    # data_Backlobe = np.array(data_Backlobe)
    # data_Backlobe_chi2016 = np.array(data_Backlobe_chi2016)
    # data_Backlobe_UNIX = np.array(data_Backlobe_UNIX)


    # RCR_training_indices = np.random.choice(no_red_sim.shape[0], size=len(no_red_sim), replace=False)
    # BL_training_indices = np.random.choice(data_Backlobe.shape[0], size=len(no_red_sim), replace=False)
    # training_RCR = no_red_sim[RCR_training_indices, :]
    # training_Backlobe = data_Backlobe[BL_training_indices, :]

    # model = Train_CNN(training_RCR, training_Backlobe, model_path, loss_accuracy_plot_path, timestamp, amp_type)  

    # model.save(f'{model_path}/{timestamp}_RCR_Backlobe_model_2Layer.h5') # currently saving in h5
    # print('------> Training is Done!')

    # # timestamp = '2025-07-01_11-08'
    # model = keras.models.load_model(f'{model_path}/{timestamp}_RCR_Backlobe_model_2Layer.h5')

    # prob_RCR = model.predict(training_RCR)
    # prob_Backlobe = model.predict(data_Backlobe)

    # # Finding not weighted RCR efficiency (percentage of RCR events that would pass the cut) 
    # sim_RCR_output = prob_RCR
    # RCR_efficiency = (sim_RCR_output > 0.9).sum() / len(sim_RCR_output)
    # RCR_efficiency = (100*RCR_efficiency).round(decimals=2)
    # print(f'RCR efficiency: {RCR_efficiency}')

    # # Finding Backlobe efficiency (percentage of backlobe that would remain after our cut)
    # sim_Backlobe_output = prob_Backlobe
    # Backlobe_efficiency = (sim_Backlobe_output > 0.9).sum() / len(sim_Backlobe_output)
    # Backlobe_efficiency = (100*Backlobe_efficiency).round(decimals=4)
    # print(f'Backlobe efficiency: {Backlobe_efficiency}')

    # print(f'lengths {len(prob_RCR)} and {len(prob_Backlobe)}')


    # # Set up for Network Output histogram
    # dense_val = False
    # plt.figure(figsize=(8, 6))
    # plt.hist(prob_Backlobe, bins=20, range=(0,1), histtype='step', color='blue', linestyle='solid', label=f'Backlobe {len(prob_Backlobe)}', density=dense_val)
    # plt.hist(prob_RCR, bins=20, range=(0,1), histtype='step', color='red', linestyle='solid', label=f'RCR {len(prob_RCR)}', density=dense_val)

    # plt.xlabel('Network Output', fontsize=18)
    # plt.ylabel('Number of Events', fontsize=18)
    # plt.yscale('log')
    # plt.title(f'{amp_type}_time RCR-Backlobe network output')
    # plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
    # plt.yticks(fontsize=18)

    # hist_values, bin_edges, _ = plt.hist(prob_Backlobe, bins=20, range=(0,1), histtype='step')
    # plt.ylim(0, max(10 ** (np.ceil(np.log10(hist_values)))))
    # plt.tick_params(axis='both', which='major', labelsize=12)
    # plt.legend(loc='upper left', fontsize=8)
    # plt.text(0.375, 0.75, f'RCR efficiency: {RCR_efficiency}%', fontsize=12, transform=plt.gcf().transFigure)
    # plt.text(0.375, 0.7, f'Backlobe efficiency: {Backlobe_efficiency}%', fontsize=12, transform=plt.gcf().transFigure)
    # plt.text(0.375, 0.65, f'TrainCut: {len(training_RCR)}', fontsize=12, transform=plt.gcf().transFigure)
    # plt.axvline(x=0.9, color='y', label='cut')
    # plt.text(0.05, -0.12, 'BL', verticalalignment='center', horizontalalignment='center', fontsize=12, transform=plt.gca().transAxes, color='blue')
    # plt.text(0.96, -0.12, 'RCR', verticalalignment='center', horizontalalignment='center', fontsize=12, transform=plt.gca().transAxes, color='red')
    # plt.subplots_adjust(left=0.2, right=0.85, bottom=0.2, top=0.8)

    # print(f'saving to {network_output_plot_path}/{timestamp}_{amp_type}_histogram.png')
    # plt.savefig(f'{network_output_plot_path}/{timestamp}_{amp_type}_histogram.png')
    # print(f'------> {amp_type} Done!')
    # ''''''

    '''plot SNR values for 5.28.25 simulated RCRs'''
    # templates_2016 = loadMultipleTemplates(200, date='2016')
    # templates_RCR = loadMultipleTemplates(200) 
    # noiseRMS = 22.53 * units.mV
    # sim, sim_Chi2016, sim_ChiRCR, sim_SNRs, sim_weights, simulation_date = siminfo_forplotting('RCR', '200s', '5.28.25', templates_2016, templates_RCR, noiseRMS)
    
    # plt.figure(figsize=(10, 6))
    # plt.hist(sim_SNRs, bins=50, edgecolor='black', alpha=0.7)
    # plt.title(f'Distribution of {len(sim_SNRs)} Simulated SNRs')
    # plt.xlabel('SNR Value')
    # plt.ylabel('Number of Events')

    # plt.grid(axis='y', alpha=0.75)

    # plot_output_dir = '/pub/tangch3/ARIANNA/DeepLearning/' # You can change this to your desired output directory
    # os.makedirs(plot_output_dir, exist_ok=True)
    # plt.savefig(os.path.join(plot_output_dir, 'sim_SNRs_histogram.png'))
    # plt.clf()
    # print(f"Histogram of sim_SNRs saved to {os.path.join(plot_output_dir, 'sim_SNRs_histogram.png')}")

    '''plot sim high weight events '''
    # amp='200s'
    # sim_folder = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/{amp}/5.28.25/'
    # indices = np.where((sim_Chi2016 > 0.4) & (sim_Chi2016 < 0.42) & 
    #                (sim_SNRs > 4) & (sim_SNRs < 5))[0]
    # # plot_sim = sim[indices]

    # sim = np.array(sim)
    # for i, index in enumerate(indices):
    #     print(f'number of sim is{len(sim)}')
    #     pT(sim[index], 'test plot new sim', f'/pub/tangch3/ARIANNA/DeepLearning/plot_new_sim_{amp}_{index}.png')

    #     if i > 10:
    #         break



    #     sim_RCR = load_sim_rcr(sim_folder, noise_enabled=False, filter_enabled=True, amp=amp)
    #     print(f'number of sim is{len(sim_RCR)}')
    #     pT(sim_RCR[index], 'test plot new sim', f'/pub/tangch3/ARIANNA/DeepLearning/test_new_sim_{amp}_noisefalse_{index}.png')


    #     sim_RCR = load_sim_rcr(sim_folder, noise_enabled=True, filter_enabled=False, amp=amp)
    #     print(f'number of sim is{len(sim_RCR)}')
    #     pT(sim_RCR[index], 'test plot new sim', f'/pub/tangch3/ARIANNA/DeepLearning/test_new_sim_{amp}_filterfalse_{index}.png')



    '''check specific Station 13, Feb 16, 2017 at 19:09:51 UTC evt'''
    # amp = '200s'
    # station_id = [13, 17]
    # all_Backlobe = []
    # all_Backlobe_UNIX = [] 
    # all_Backlobe_SNR = []
    # all_Backlobe_Chi2016 = []
    # for id in station_id:
    #     # snr, chi, trace, unix = load_data('All_data', amp_type = amp, station_id=id)
    #     unix = np.load(f'/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/4.4.25/Station{id}/station{id}_all_Times.npy')
    #     trace = np.load(f'/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/4.4.25/Station{id}/station{id}_all_Traces.npy')
    #     snr = np.load(f'/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/4.4.25/Station{id}/station{id}_all_SNR.npy')
    #     chi2016 = np.load(f'/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/4.4.25/Station{id}/station{id}_all_Chi2016.npy')
    #     all_Backlobe.extend(trace)
    #     all_Backlobe_UNIX.extend(unix)
    #     all_Backlobe_SNR.extend(snr)
    #     all_Backlobe_Chi2016.extend(chi2016)
    # print(len(all_Backlobe_UNIX))
    

    # from datetime import datetime, timezone
    # target_unix = 1487300991

    # if target_unix in all_Backlobe_UNIX:
    #     idx = all_Backlobe_UNIX.index(target_unix)
    #     trace = all_Backlobe[idx]
    #     snr = all_Backlobe_SNR[idx]
    #     chi = all_Backlobe_Chi2016[idx]
    #     thetime = all_Backlobe_UNIX[idx]
    #     utc_time = datetime.fromtimestamp(thetime, tz=timezone.utc)

    #     print(f"\n Event found at index {idx}")
    #     print(f"  UTC Time:    {utc_time}")
    #     print(f"  UNIX Time:   {thetime}")
    #     print(f"  SNR:         {snr}")
    #     print(f"  Chi2016:     {chi}")
    #     print(f"  Trace shape: {trace.shape}")

    

    # for id in station_id:
    #     unix, count = load_520_data(id, 'Times', station_data_folder)
    #     trace, num = load_520_data(id, 'Traces', station_data_folder)
    #     all_Backlobe_UNIX.extend(unix.tolist())
    #     all_Backlobe.extend(trace.tolist())
    # all_Backlobe_UNIX = np.array(all_Backlobe_UNIX)
    # all_Backlobe = np.array(all_Backlobe)

    # from datetime import timezone, datetime

    # target_unix_time = 1487272191 # Feb 16, 2017 at 19:09:51 UTC
    # similarity_window_seconds = 5

    # exact_match_indices = []
    # exact_match_count = 0

    # print(f"\nSearching for EXACT matches to UNIX time: {target_unix_time}")
    # print(f"Total events in all_Backlobe_UNIX: {all_Backlobe_UNIX.shape}")

    # # Use all_Backlobe_UNIX consistently
    # for idx, unix_time in enumerate(all_Backlobe_UNIX):
    #     if unix_time == target_unix_time:
    #         exact_match_count += 1
    #         # Use timezone.utc for precise conversion if Unix time is UTC
    #         std_time = datetime.fromtimestamp(unix_time, tz=timezone.utc)
    #         print(f"  Exact match found: Event {exact_match_count} at index {idx} with time {std_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    #         exact_match_indices.append(idx)

    # print(f"\n--- Summary of Exact Matches ---")
    # print(f"Total exact matches found: {exact_match_count}")
    # print(f"Indices of exact matching events: {exact_match_indices}")

    # for index in exact_match_indices:
    #     pT(all_Backlobe[index], 'test plot confirmed RCR', f'/pub/tangch3/ARIANNA/DeepLearning/78_test_plot_confirmed_RCR_{amp}_{index}.png')

    # # --- Search for SIMILAR matches ---
    # similar_match_indices = []
    # similar_match_count = 0

    # print(f"\nSearching for SIMILAR matches (within +/- {similarity_window_seconds} seconds) to UNIX time: {target_unix_time}")

    # # Using all_Backlobe_UNIX for the search
    # for idx, unix_time in enumerate(all_Backlobe_UNIX):
    #     # Check if the UNIX time falls within the defined window
    #     if target_unix_time - similarity_window_seconds <= unix_time <= target_unix_time + similarity_window_seconds:
    #         similar_match_count += 1
    #         # Use timezone.utc for precise conversion if Unix time is UTC
    #         std_time = datetime.fromtimestamp(unix_time, tz=timezone.utc)
    #         print(f"  Similar match found: Event {similar_match_count} at index {idx} with time {std_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    #         similar_match_indices.append(idx)

    # print(f"\n--- Summary of Similar Matches ---")
    # print(f"Total similar matches found: {similar_match_count}")
    # print(f"Indices of similar matching events: {similar_match_indices}")


    # amp_type = '200s'

    # sim_folder = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/{amp_type}/5.28.25/'
    # sim_rcr = load_sim_rcr(sim_folder, noise_enabled=True, filter_enabled=True, amp=amp_type)


        
