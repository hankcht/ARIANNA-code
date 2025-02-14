import matplotlib.pyplot as plt
import numpy as np
import os
import keras
from tensorflow.keras.utils import to_categorical
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import random
import datetime
import pandas as pd
from glob import glob
import matplotlib

# Set parameters
path = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/eventsPassingNoiseCuts/'    #Edit path to properly point to folder
model_path = f'/data/homezvol3/tangch3/ARIANNA/DeepLearning/models/'  #Path to save models
amp = '200s' 
station_number = 14                                                  

def load_data():
    station_files = []
    print(f'path {path}')
    for filename in os.listdir(path):
        print(f'filename {filename}')
        if filename.startswith(f'Station{station_number}'):
            print(f'appending')
            station_files.append(path +  filename)
    station = np.empty((0, 4, 256))
    print(f'station files {station_files}')
    for file in station_files:
        print(f'station file {file}')
        station_data = np.load(file)[0:, 0:4]
        print(f'station data shape {station_data.shape} and station shape {station.shape}')
        station = np.concatenate((station, station_data))

    return (station, station_data.shape) 

station, total_number_events = load_data()

#station contained all passed events

#IMPORTANT:
#In the following block of code, I will choose the events that are in the top red (good) region
#To do this, I had a line of code in Chi-SNRgraph.py that gave the indices of passed events that had for example, Chi-value >= 0.5
#The following arrays contain the indices of chosen passed events for each station 

selected_events_dict = {
'stn14selectedevents' : [3, 6, 13, 15, 18, 19, 21, 22, 24, 28, 35, 39, 40, 49],
'stn17selectedevents' : [1, 3, 18, 21, 22, 26, 28, 29, 31, 33, 44, 45, 47, 48, 49, 50],
'stn19selectedevents' : [1, 2, 4, 5, 7, 15, 17, 18, 19, 25, 27, 31, 34], #Chosen passed events with Chi >= 0.5 6/20/2024
'stn30selectedevents' : [4, 6, 7, 8, 9, 11, 12, 13, 16, 17, 20, 22, 25, 26, 29, 30, 32, 33, 34, 36, 38, 43, 49, 50, 55, 60, 61, 62]
}

selected_station = []
selected_station = station[selected_events_dict[f'stn{station_number}selectedevents']] #change stn number accordingly (or come up a way to use station_number)

print('selected passed events', len(selected_station))
print(selected_station.shape)

# Load the model, copy from /models
model = keras.models.load_model(f'/data/homezvol3/tangch3/ARIANNA/DeepLearning/models/200s_freq/2024-04-10_16-09-09_RCR_Backlobe_model_2Layer_0.h5')

#turn station time-series to frequency domain
sampling_rate = 2
station_freq = np.fft.rfft(selected_station, axis=-1)
print(station_freq.shape)

prob_station_freq = model.predict(station_freq)

output_cut_value = 0.05 #Change this depending on chosen cut
print(f'output cut value: {output_cut_value}')

#Finding efficiency (percentage of events that has passed the cut)
station_output = prob_station_freq
passed_events = (station_output < output_cut_value).sum() / len(station_output)
passed_events = passed_events.round(decimals=4)*100
print(f'number of passed events: {(station_output < output_cut_value).sum()}')
print(f'total events: {len(station_output)}')

total_number_events = len(selected_station) #used to be actual total passed events, with selected_station, it is now selected passed events
number_passed_events = (station_output < output_cut_value).sum()
# number_passed_events = number_passed_events.round(decimals=2)

dense_val = False
hist_values, bin_edges, _ = plt.hist(prob_station_freq, bins=20, range=(0, 1), histtype='step', color='green', linestyle='solid', label='output', density=dense_val)

# Set logarithmic scale for y-axis, if needed
# plt.yscale('log')

# Set labels and title
plt.xlabel('Network Output', fontsize = 18)
plt.ylabel('Number of Events', fontsize = 18)
plt.title(f'Station {station_number} network output (200s_freq)')
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],size=18)
plt.yticks(size=18)
# plt.ylim(1, max(10 ** (np.ceil(np.log10(hist_values))))) # uncomment if needed for semi-log plot
plt.tick_params(axis='both', which='major', labelsize=12)
plt.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)
# handles, labels = plt.get_legend_handles_labels()
# new_handles = [Line2D([], [], c=h.get_edgecolor(), linestyle=h.get_linestyle()) for h in handles]
plt.legend(loc='upper right', fontsize=8)
plt.figtext(0.3,0.75, f'Total Passed Events: {number_passed_events}')
plt.figtext(0.3,0.7, f'Total Selected Events: {total_number_events}')
plt.axvline(x = 0.05, color = 'y', label = 'cut')
# Save the plot to a file (in PNG format)
plt.savefig(f'/data/homezvol3/tangch3/ARIANNA/DeepLearning/plots/Candidates/Station {station_number}/freqStn{station_number}histogram.png')

#parameters to change: window size
#might want to have less than window size 10 for frequency (maybe 8) window size 10 means 10 samples, so over 5ns
#try with highest accuracy
