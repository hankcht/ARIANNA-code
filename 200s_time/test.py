import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import tracemalloc
from A0_Utilities import load_data, pT, getMaxChi, getMaxSNR
import re
import datetime
import B1_BLcurve 
from NuRadioReco.utilities import units

# arg = 'time'

# station_path = f'/pub/tangch3/ARIANNA/DeepLearning/all_data/indv_{arg}'
# output_file = '/pub/tangch3/ARIANNA/DeepLearning/all_data/stn17_traces.npy'  # Final output file
# batch_size = 50  # Number of files to process at once

# amp_type = '200s'
# if amp_type == '200s':
#     noiseRMS = 22.53 * units.mV

# elif amp_type == '100s':
#     noiseRMS = 20 * units.mV

# station_id = 17 # [14,17,19,30]

# all_snr = []
# all_chi = []
# all_trace = []
# all_unix = []

# sampling_rate = 2
# RCR_freq = np.abs(np.fft.rfft(training_RCR, axis=-1)/ sampling_rate * 2 ** 0.5) # this is the same way freq is calculated from NuRadioMC
# Backlobe_freq = np.abs(np.fft.rfft(training_Backlobe, axis=-1)/ sampling_rate * 2 ** 0.5)
# print(RCR_freq.shape, Backlobe_freq.shape)


from tensorflow import keras
model = keras.models.load_model('/pub/tangch3/ARIANNA/DeepLearning/models/200s_time/data_data_2025-06-02_09-45_RCR_BL_model_2Layer_two_ws_stdy.h5')

rcr, sim_Backlobe = load_sim(path, RCR_path, backlobe_path, amp)
data_Backlobe = []
data_Backlobe_UNIX = [] 
data_chi = []
for id in station_id: # since we load traces depending on stn, we need to make data_Backlobe a full list
    snr, chi, trace, unix = load_data('AboveCurve_data', amp_type = amp, station_id=id)
    data_Backlobe.extend(trace)
    data_Backlobe_UNIX.extend(unix)
    data_chi.extend(chi)
rcr = np.array(rcr)
prob_RCR = model.predict(rcr)
prob_Backlobe = model.predict(Backlobe)


import sys
sys.exit()

algorithm = 'RS'
best_result = np.load(f'/pub/tangch3/ARIANNA/DeepLearning/sherpa_output/{algorithm}secnd_wdw.npy')
print(best_result)
from collections import Counter
count = Counter(best_result)
most_common_element, most_common_count = count.most_common(1)[0]
print(most_common_element)
print(len(best_result))
bins = np.linspace(1,60,60)
plt.hist(best_result, bins)
plt.xlabel('Window size')
plt.ylabel('count')
plt.text(10, 10, f'total # of studies:{len(best_result)}')
plt.text(most_common_element, most_common_count, f'Best: {most_common_element}', ha='center', va='bottom', fontsize=10, color='red')
print(f'saving fig for {algorithm}')
plt.savefig(f'/pub/tangch3/ARIANNA/DeepLearning/sherpa_output/{algorithm}secnd_wdw.png')

# for id in station_id:
#         snr, chi, trace, unix = load_data('All_data', amp_type=amp_type, station_id=id)
#         all_snr.extend(snr)
#         all_chi.extend(chi)
#         all_trace.extend(trace)
#         all_unix.extend(unix)
# all_snr ,all_chi, all_trace, all_unix = load_data('All_data', amp_type, station_id)

# # all_snr = [round(snr, 2) for snr in all_snr]
# # all_chi = [round(chi, 2) for chi in all_chi]
# # index = []
# # for i, unix in enumerate(all_unix):
# #     if unix == 1449861609:
# #         index.append(i)

# # snr_index = []
# # for i, (snr,chi) in enumerate(zip(all_snr, all_chi)):
# #     if 20.32 <= snr <= 20.33 and 0.67 <= chi <= 0.68:
# #         snr_index.append(i)
# #         print(snr, chi)

# # print(index, snr_index)

# # data = np.load(f'../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/Station{station_id}_SNR_Chi.npy', allow_pickle=True)

# # All_SNRs = data[0]
# # All_RCR_Chi = data[1]
# # All_Azi = data[2]
# # All_Zen = data[3]

# a = 0
# confirmed_BL_unix_200s = {1454540191,1455263868,1449861609,1450734371,1457453131,1455205950,1455513662,1458294171}
# # 1450734371,1457453131 not in all?
# confirmed_BL_unix_100s = {1450734371,1449861609,1450268467,1455205950,1455513662,1458294171}

# ii = 0
# type = 'data'
# plot_folder = '/pub/tangch3/ARIANNA/DeepLearning/'
# for unix, chi, snr, trace in zip(all_unix, all_chi, all_snr, all_trace):
#         if unix in confirmed_BL_unix_200s:
#                 print('found in saved', unix, chi, snr)
#                 # pT(trace, f'data', f'{plot_folder}{type}_SNR_{snr}_Chi{chi}.png')
#                 ii += 1
# print(ii)

# confirmed_trace = []
# for filename in os.listdir('../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/templates/confirmed2016Templates/'):
#     if filename.startswith(f'Event2016_Stn{station_id}'):
#             trace = np.load(f'../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/templates/confirmed2016Templates/{filename}')
#             print(trace.shape)
#             confirmed_trace.append(trace)
#             # pT(confirmed_trace, f'confirmed trace', f'/pub/tangch3/ARIANNA/DeepLearning/{filename}.png')

# confirmed_snr = []
# confirmed_chi = []
# templates_RCR = B1_BLcurve.loadTemplate(type='RCR', amp=amp_type)
# for event in confirmed_trace:
        
#     traces = []
#     for trace in event:
#         traces.append(trace * units.V)
#     confirmed_snr.append(getMaxSNR(traces, noiseRMS=noiseRMS))
#     confirmed_chi.append(getMaxChi(traces, 2*units.GHz, templates_RCR, 2*units.GHz))

# print('confirmed')
# for snr, chi in zip(confirmed_snr, confirmed_chi):
#       chi = round(chi, 2) 
#       snr = round(snr,2)
#       print(chi, snr)
      


# calendar_time = []
# for unix in confirmed_BL_unix_200s:
#     formatted_time = datetime.datetime.fromtimestamp(unix).strftime("%m-%d-%Y, %H:%M:%S")
#     calendar_time.append(formatted_time)
#     print(unix, formatted_time)


# i = 0
# n = 0
# for filename in os.listdir(station_path):
#     # print(filename)
#     i += 1
#     time = np.load(f'{station_path}/{filename}')
#     # print(len(time))
#     n += len(time)

# print(n)
# i = 0
# traces1 = []
# traces2 = []
# traces3 = []
# traces4 = []

# # Assuming station_path contains the list of filenames
# for filename in os.listdir(station_path):
#     filepath = os.path.join(station_path, filename)
    
#     if i < 50:
#         # Use np.memmap for large files to avoid loading everything into memory at once
#         traces1.append(np.load(filepath, mmap_mode='r'))
#         i += 1
#     elif 100 > i >= 50:
#         traces2.append(np.load(filepath, mmap_mode='r'))
#         i += 1
#     elif i >= 100:
#         traces3.append(np.load(filepath, mmap_mode='r'))
#         i += 1

# tracemalloc.start()
# trace1 = np.concatenate(traces1, axis=0) if traces1 else None
# print(f"Memory tr1: {tracemalloc.get_traced_memory()[1]} bytes")
# print(trace1.shape)
# np.save('/pub/tangch3/ARIANNA/DeepLearning/all_data/Station_Traces/200s/Stn17_Traces_part0.npy', trace1)
# trace2 = np.concatenate(traces2, axis=0) if traces2 else None
# print(f"Memory tr2: {tracemalloc.get_traced_memory()[1]} bytes")
# print(trace2.shape)
# np.save('/pub/tangch3/ARIANNA/DeepLearning/all_data/Station_Traces/200s/Stn17_Traces_part1.npy', trace2)
# trace3 = np.concatenate(traces3, axis=0) if traces3 else None
# print(f"Memory tr3: {tracemalloc.get_traced_memory()[1]} bytes")
# print(trace3.shape)
# np.save('/pub/tangch3/ARIANNA/DeepLearning/all_data/Station_Traces/200s/Stn17_Traces_part2.npy', trace3)

# trace = np.concatenate([trace1,trace2],axis=0)
# print(f"Memory tr3: {tracemalloc.get_traced_memory()[1]} bytes")
# trace = np.load('/pub/tangch3/ARIANNA/DeepLearning/all_data/indv_traces/stn17_Data_station_17_run_00000_Traces.npy') 
# np.save('/pub/tangch3/ARIANNA/DeepLearning/all_data/Station_Traces/200s/Stn17_Traces.py')
# print(len(trace),trace.shape)
# print(i)   

# n = 0
# i = 0
# time = []
# for filename in os.listdir(station_path):
#     # print(filename)
#     yo = np.load(f'/pub/tangch3/ARIANNA/DeepLearning/all_data/indv_{arg}/{filename}')
#     time.append(yo)
    
#     # print(lent)
#     # shape = yo.shape
#     # first_dim = shape[0]
#     # print(first_dim)
#     lent = len(yo)
#     # print(lent, filename)
#     n += lent
#     i += 1
# print(n,i) 
# times = np.concatenate(time)
# print(times[0:5])
# print('saving')
# np.save('/pub/tangch3/ARIANNA/DeepLearning/all_data/Station_UNIX/200s/Stn17_UNIX.npy',times)   

# ii = 0
# station_path2 = f'/dfs8/sbarwick_lab/ariannaproject/station_nur/station_17/'
# for filename2 in os.listdir(station_path2):
#     if filename2.endswith('_statDatPak.root.nur'):
#         continue
#     else:
#         if os.path.getsize(os.path.join(station_path, filename2)) == 0:
#                     print(f'File {filename2} is empty, skipping')
#                     continue
#         ii += 1 
# print(ii)



# station_id = [14,19,30]
# for i in station_id:
#     if i in [13,15,18]:
#         amp = '100s'
#     else:
#         amp = '200s'
#     unix = np.load(f'/pub/tangch3/ARIANNA/DeepLearning/circled_data/Station_UNIX/{amp}/Stn{i}_UNIX.npy')
#     print(unix[10:15])
#     print(len(unix))
#     snr = np.load(f'/pub/tangch3/ARIANNA/DeepLearning/all_data/Station_SNR/{amp}/Stn{i}_SNR.npy')
#     chi = np.load(f'/pub/tangch3/ARIANNA/DeepLearning/all_data/Station_Chi/{amp}/Stn{i}_Chi.npy')
    # print(len(snr), len(chi))
    # print(snr[10:15])
    # print(chi[10:15])
    # traces = np.load(f'/pub/tangch3/ARIANNA/DeepLearning/all_data/Station_Traces/{amp}/Stn{i}_Traces.npy')
    # print(len(traces),traces.shape)
    # print(traces[1:5])
    # data = np.load(f'../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/Station{i}_SNR_Chi.npy', allow_pickle=True)
    # print(len(data[4])==len(snr), len(data[5])==len(chi))




# # np.save('/pub/tangch3/ARIANNA/DeepLearning/all_data/Station_Traces/200s/Stn17_Traces.npy', stn17)
# data_path = '../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/events_prepared_for_model_training/'
# for filename in os.listdir(data_path):
#     if filename.startswith(f'FilteredStation17'):
#         print(filename)

# station_id = 17

# All_SNRs = data[0]
# All_RCR_Chi = data[1]
# All_Azi = data[2]
# All_Zen = data[3]
# PassingCut_SNRs = data[4]
# PassingCut_RCR_Chi = data[5]
# PassingCut_Azi = data[6]
# PassingCut_Zen = data[7]

# print(len(All_SNRs),len(All_RCR_Chi))

