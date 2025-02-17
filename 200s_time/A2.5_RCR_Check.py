from NuRadioReco.utilities.io_utilities import read_pickle
import os
import numpy as np
from NuRadioReco.utilities import units
import datetime
import matplotlib.pyplot as plt
from A0_Utilities import RunTrainedModel, load_data, pT, load_sim
from A2_RealRunCNN import plot_data_histogram
from A1_TrainAndRunCNN import output_cut_value

def loadSingleTemplate(series):
    # Series should be 200 or 100
    # Loads the first version of a template made for an average energy/zenith
    templates_RCR = f'../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/templates/reflectedCR_template_{series}series.pkl'
    templates_RCR = read_pickle(templates_RCR)
    for key in templates_RCR:
        temp = templates_RCR[key]
    templates_RCR = temp

    return templates_RCR

def loadMultipleTemplates(series, date='9.16.24', addSingle=True):
    # Dates - 9.16.24 (noise included), 10.1.24 (no noise)
    #       - 2016 : found backlobe events from 2016

    # 10.1.24 has issues with the templates, so use 9.16.24
    # Series should be 200 or 100
    # Loads all the templates made for an average energy/zenith
    if not date == 'confirmed2016Templates':
        template_series_RCR_location = f'../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/templates/RCR/{date}/' 
        template_series_RCR = []
        for filename in os.listdir(template_series_RCR_location):
            if filename.startswith(f'{series}s'):
                temp = np.load(os.path.join(template_series_RCR_location, filename))
                template_series_RCR.append(temp)
    else:
        templates_2016_location = f'../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/templates/confirmed2016Templates/'
        template_series_RCR = []
        for filename in os.listdir(templates_2016_location):
            temp = np.load(os.path.join(templates_2016_location, filename))
            template_series_RCR.append(temp)

    if addSingle:
        template_series_RCR.append(loadSingleTemplate(series))

    return template_series_RCR

if __name__ == '__main__':

    model_path = f'/pub/tangch3/ARIANNA/DeepLearning/models/' 
   
#    # first test confirmed BL
#     confirmed_BL_path = '../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/templates/confirmed2016Templates'
 
#     confirmed_Traces = []
#     for filename in os.listdir(confirmed_BL_path):
#         trace = np.load(f'{confirmed_BL_path}/{filename}')
#         confirmed_Traces.append(trace)

#     confiremd_Traces = loadMultipleTemplates('200', date='confirmed2016Templates', addSingle=False)

#     confirmed_Traces = np.array(confirmed_Traces)
#     print(confirmed_Traces.shape)

#     prob_template = RunTrainedModel(confirmed_Traces, model_path)

#     fig, ax = plt.subplots(figsize=(8, 6))
#     hist_values, bin_edges, _ = ax.hist(prob_template, bins=20, range=(0, 1), histtype='step', color='blue', linestyle='solid', label=f'template count {len(prob_template)}', density=False)
#     ax.set_xlabel('Network Output', fontsize=18)
#     ax.set_ylabel('Number of Events', fontsize=18)
#     # ax.set_yscale('log') # Set logarithmic scale for y-axis, if needed
#     # ax.set_ylim(1, max(10 ** (np.ceil(np.log10(hist_values))))) # uncomment if needed for semi-log plot
#     ax.set_title(f'confirmed BL network output')
#     ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
#     ax.tick_params(axis='both', which='both', labelsize=12, length=5)
#     ax.text(0.05, -0.12, 'BL', verticalalignment='center', horizontalalignment='center', fontsize=12, transform=ax.transAxes, color='blue')
#     ax.text(0.96, -0.12, 'RCR', verticalalignment='center', horizontalalignment='center', fontsize=12, transform=ax.transAxes, color='red')
#     plt.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)
#     ax.axvline(x=output_cut_value, color='y', label='cut')
#     ax.legend(loc='upper right', fontsize=8, bbox_to_anchor=(0.85, 0.99))
#     print(f'/pub/tangch3/ARIANNA/DeepLearning/plots/Histograms/confirmed_histogram.png')
#     plt.savefig(f'/pub/tangch3/ARIANNA/DeepLearning/plots/Histograms/confirmed_histogram.png')
#     print('Done!')


    # now check data Backlobe
    amp = '200s'
    window_size = 10

    if amp == '200s':
        noiseRMS = 22.53 * units.mV
        station_id = [14,17,19,30]
    elif amp == '100s':
        noiseRMS = 20 * units.mV
        station_id = [13,15,18]

    path = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/'    
    RCR_path = f'simulatedRCRs/{amp}_2.9.24/'
    backlobe_path = f'simulatedBacklobes/{amp}_2.9.24/'
    rcr, sim_Backlobe = load_sim(path, RCR_path, backlobe_path, amp)

    data_Backlobe = []
    data_Backlobe_UNIX = [] 
    for id in station_id:
        snr, chi, trace, unix = load_data('AboveCurve_data', amp_type = amp, station_id=id)
        data_Backlobe.extend(trace)
        data_Backlobe_UNIX.extend(unix)


    data_Backlobe_UNIX = np.array(data_Backlobe_UNIX)
    data_Backlobe = np.array(data_Backlobe)
    prob_Backlobe = RunTrainedModel(data_Backlobe, model_path)
    prob_RCR = RunTrainedModel(rcr, model_path)

    # Finding Backlobe efficiency (percentage of backlobe that would remain after our cut)
    sim_Backlobe_output = prob_Backlobe
    Backlobe_efficiency = (sim_Backlobe_output > output_cut_value).sum() / len(sim_Backlobe_output)
    Backlobe_efficiency = (100*Backlobe_efficiency).round(decimals=4)
    print(f'Backlobe efficiency: {Backlobe_efficiency}')

    # Set up for Network Output histogram
    dense_val = False
    fig, ax = plt.subplots(figsize=(8, 6))  
    hist_values, bin_edges, _ = ax.hist(prob_Backlobe, bins=20, range=(0,1), histtype='step', color='blue', linestyle='solid', label=f'Backlobe {len(prob_Backlobe)}', density=dense_val)
    ax.hist(prob_RCR, bins=20, range=(0,1), histtype='step', color='red', linestyle='solid', label=f'RCR {len(prob_RCR)}', density=dense_val)
    
    # for i in range(len(hist_values)):
    #     plt.text(bin_edges[i] + 0.1, hist_values[i] + 10, str(hist_values[i]), color='black', ha='center')

    ax.set_xlabel('Network Output', fontsize=18)
    ax.set_ylabel('Number of Events', fontsize=18)
    ax.set_yscale('log')
    ax.set_title(f'{amp}_time RCR-Backlobe network output')
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylim(0, max(10 ** (np.ceil(np.log10(hist_values)))))
    ax.tick_params(axis='both', which='major', labelsize=12)
    fig.subplots_adjust(left=0.2, right=0.85, bottom=0.2, top=0.8)
    ax.legend(loc='upper left', fontsize=8)
    fig.text(0.375, 0.7, f'Backlobe efficiency: {Backlobe_efficiency}%', fontsize=12)
    ax.axvline(x=output_cut_value, color='y', label='cut')
    ax.text(0.05, -0.12, 'BL', verticalalignment='center', horizontalalignment='center', fontsize=12, transform=ax.transAxes, color='blue')
    ax.text(0.96, -0.12, 'RCR', verticalalignment='center', horizontalalignment='center', fontsize=12, transform=ax.transAxes, color='red')
    print(f'/pub/tangch3/ARIANNA/DeepLearning/plots/Histograms/All_Stations/Size_{window_size}_histogram.png')
    plt.savefig(f'/pub/tangch3/ARIANNA/DeepLearning/plots/Histograms/All_Stations/Size_{window_size}_histogram.png')
    print(f'------> {amp} Done!')

    
    # find potential RCR events
    potential_RCR_index = []
    for i, network_output in enumerate(prob_Backlobe):
        if network_output > 0.95:
            potential_RCR_index.append(i)
    potential_RCR_index.extend([2668, 2767, 4967])

    print(potential_RCR_index)
    potential_RCR = data_Backlobe[potential_RCR_index]
    potential_RCR_UNIX = data_Backlobe_UNIX[potential_RCR_index]

    potential_RCR_calendar = []
    for ts in potential_RCR_UNIX:
        formatted_time = datetime.datetime.fromtimestamp(ts).strftime("%m-%d-%Y %H:%M:%S")
        potential_RCR_calendar.append(formatted_time)

    print(potential_RCR_calendar)
    get_station_number = []
    for index in potential_RCR_index: # maybe I will collapse the ugly sums later when numbers become more familiar 
        if index <= 318 - 1:
            get_station_number.append(14)
        elif 318 <= index <= 318 + 2326 - 1:
            get_station_number.append(17)
        elif 318 + 2326 <= index <= 318 + 2326 + 317 - 1:
            get_station_number.append(19)
        elif 318 + 2326 + 317 <= index <= 318 + 2326 + 317 + 5618 - 1:
            get_station_number.append(30)
    print(get_station_number)

    for station_number, date, trace in zip(get_station_number, potential_RCR_calendar, potential_RCR):
        pT(trace, f'potential RCR', f'/pub/tangch3/ARIANNA/DeepLearning/stn{station_number} {date}.png')

    # 87 94 3031 5872 6106 2668 4967

        



















    # # old, checking network output on templates
    # template_series_RCR = loadMultipleTemplates('200')

    # complete_template = []
    # # since templates are defined on one channel, I test if the network picks up parallel features by creating parallel for half and all chan for the other half
    # for index, template in enumerate(template_series_RCR):
         
    #     if index % 2 == 1:
    #         all_four_channels_temporary_template = []
    #         for i in range(4):
    #             all_four_channels_temporary_template.append(template)
    #         complete_template.append(all_four_channels_temporary_template)
    #     elif index % 2 == 0:
    #         parallel_channels_temporary_template = np.zeros((4, 256))
    #         parallel_channels_temporary_template[0,:] = template
    #         parallel_channels_temporary_template[2,:] = template
    #         complete_template.append(parallel_channels_temporary_template)
     
    # complete_template = np.array(complete_template)
    # print(complete_template.shape)

    # model_path = f'/pub/tangch3/ARIANNA/DeepLearning/models/' 

    # prob_template = RunTrainedModel(complete_template, model_path)

    # RCR_like_template_index = []
    # BL_like_template_index = []
    # for i, network_output in enumerate(prob_template):
    #     if network_output > 0.8:
    #         RCR_like_template_index.append(i)
    #     if network_output < 0.2:
    #         BL_like_template_index.append(i)

    # print(RCR_like_template_index, BL_like_template_index)

    # date='9.16.24'
    # series = '200'
    # template_RCR_location = f'../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/templates/RCR/{date}/' 
    # filenames = []
    # for filename in os.listdir(template_RCR_location):
    #     if filename.startswith(f'{series}s'):
    #         filenames.append(filename)

    # # this show which template files are recognized as RCR/BL
    # for file_index, filename in enumerate(filenames):
        
    #     if  file_index in RCR_like_template_index:
    #         print(f'RCR {filename}')
    #     elif file_index in BL_like_template_index:
    #         print(f'BL {filename}')


    # fig, ax = plt.subplots(figsize=(8, 6))
    # hist_values, bin_edges, _ = ax.hist(prob_template, bins=20, range=(0, 1), histtype='step', color='red', linestyle='solid', label=f'template count {len(prob_template)}', density=False)
    # ax.set_xlabel('Network Output', fontsize=18)
    # ax.set_ylabel('Number of Events', fontsize=18)
    # # ax.set_yscale('log') # Set logarithmic scale for y-axis, if needed
    # # ax.set_ylim(1, max(10 ** (np.ceil(np.log10(hist_values))))) # uncomment if needed for semi-log plot
    # ax.set_title(f'RCR template network output')
    # ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ax.tick_params(axis='both', which='both', labelsize=12, length=5)
    # ax.text(0.05, -0.12, 'BL', verticalalignment='center', horizontalalignment='center', fontsize=12, transform=ax.transAxes, color='blue')
    # ax.text(0.96, -0.12, 'RCR', verticalalignment='center', horizontalalignment='center', fontsize=12, transform=ax.transAxes, color='red')
    # plt.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)
    # ax.axvline(x=output_cut_value, color='y', label='cut')
    # ax.legend(loc='upper right', fontsize=8, bbox_to_anchor=(0.85, 0.99))
    # print(f'/pub/tangch3/ARIANNA/DeepLearning/plots/Histograms/template_check_histogram.png')
    # plt.savefig(f'/pub/tangch3/ARIANNA/DeepLearning/plots/Histograms/template_check_histogram_all.png')
    # print('Done!')

