import matplotlib.pyplot as plt
import numpy as np
import os
import keras
import time
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
#from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import random
import datetime
import pandas as pd
from glob import glob
import matplotlib
import argparse

#Set parameters
path = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/'    #Edit path to properly point to folder
amp = '200s'                                                                  #Set which amplifier to run on
RCR_path = f'simulatedRCRs/{amp}_2.9.24/'
backlobe_path = f'simulatedBacklobes/{amp}_2.9.24/'

# load data
# first we load RCR data

RCR_files = []
print(f'path {path + RCR_path}')
for filename in os.listdir(path + RCR_path):
    print(f'filename {filename}')
    if filename.startswith(f'FilteredSimRCR_{amp}_'):
        print(f'appending')
        RCR_files.append(path + RCR_path +  filename)
RCR = np.empty((0, 4, 256))
print(f'rcr files {RCR_files}')
for file in RCR_files:
    print(f'RCR file {file}')
    RCR_data = np.load(file)[0:, 0:4]
    print(f'RCR data shape {RCR_data.shape} and RCR shape {RCR.shape}')
    RCR = np.concatenate((RCR, RCR_data))

# Now, we either load [1] sim BL, or [2] "BL data events"

parser = argparse.ArgumentParser(description='Determine To Use sim BL or "BL data events"')
parser.add_argument('BLsimOrdata', type=str, default='sim', help='Use sim BL or "BL data events')
args = parser.parse_args()
if_sim = args.BLsimOrdata

if if_sim == 'sim':
    #[1] Simulated Backlobe 
    Backlobes_files = []
    for filename in os.listdir(path + backlobe_path):
        if filename.startswith(f'Backlobe_{amp}_'):
            Backlobes_files.append(path + backlobe_path + filename)
    Backlobe = np.empty((0, 4, 256))
    for file in Backlobes_files:
        print(f'Backlobe file {file}')
        Backlobe_data = np.load(file)[0:, 0:4]
        Backlobe = np.concatenate((Backlobe, Backlobe_data))

elif if_sim == 'data':  
    #[2] BL Curve Cut on Station Data, comment this section out if using [1]
    data_path = '/pub/tangch3/ARIANNA/DeepLearning/AboveCurve_npfiles/'
    all_data_files = [] 

    for filename in os.listdir(data_path):
        print(f'appending {filename}')
        all_data_files.append(os.path.join(data_path, filename))

    print(f'size of data: {len(all_data_files)}') 

    shapes = [np.load(file).shape for file in all_data_files]
    total_rows = sum(shape[0] for shape in shapes)
    first_shape = shapes[0][1:]
    print(f'first shape is: {first_shape}')
    all_station_data = np.empty((total_rows, *first_shape), dtype=np.float32)

    start_idx = 0

    for i, file in enumerate(all_data_files):
        data = np.load(file)
        num_rows = data.shape[0]

        all_station_data[start_idx:start_idx + num_rows] = data

        start_idx += num_rows 

    print(all_station_data.shape)

    Backlobe = all_station_data

RCR, Backlobe = RCR[5000:], Backlobe[5000:] #I used the first 5000 simulation for training, so I run the trained model on the remaining

# Load the model, copy from /models
model = keras.models.load_model(f'/pub/tangch3/ARIANNA/DeepLearning/models/200s_time/2024-10-12_11-52-29_RCR_Backlobe_model_2Layer_0.h5')

##################################
# Here I am saving a list of good models by the time they were created
# 2024-10-12_11-52-29 (Used Sim BL)

##################################

prob_RCR = model.predict(RCR)
prob_Backlobe = model.predict(Backlobe)

output_cut_value = 0.95 #Change this depending on chosen cut, we get our passed events from this 
print(f'output cut value: {output_cut_value}')

#Finding RCR efficiency (percentage of RCR events that would pass the cut)
sim_RCR_output = prob_RCR
RCR_efficiency = (sim_RCR_output > output_cut_value).sum() / len(sim_RCR_output)
RCR_efficiency = (100*RCR_efficiency).round(decimals=2)
print(f'{if_sim} RCR efficiency: {RCR_efficiency}')

#Finding Backlobe efficiency (percentage of backlobe that would remain after our cut)
sim_Backlobe_output = prob_Backlobe
Backlobe_efficiency = (sim_Backlobe_output > output_cut_value).sum() / len(sim_Backlobe_output)
Backlobe_efficiency = (100*Backlobe_efficiency).round(decimals=4)
print(f'{if_sim} Backlobe efficiency: {Backlobe_efficiency}')

dense_val = False
hist_values, bin_edges, _ = plt.hist(prob_Backlobe, bins=20, range=(0, 1), histtype='step', color='red', linestyle='solid', label='Backlobe', density=dense_val)
plt.hist(prob_RCR, bins=20, range=(0, 1), histtype='step',color='blue', linestyle='solid',label='RCR',density = dense_val)

# Set logarithmic scale for y-axis
plt.yscale('log')

# Set labels and title
plt.xlabel('Network Output', fontsize = 18)
plt.ylabel('Number of Events', fontsize = 18)
plt.title(f'{if_sim} RCR vs Backlobe network output (200s_time)')
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],size=18)
plt.yticks(size=18)
plt.ylim(1, max(10 ** (np.ceil(np.log10(hist_values)))))
plt.tick_params(axis='both', which='major', labelsize=12)
plt.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)
# handles, labels = plt.get_legend_handles_labels()
# new_handles = [Line2D([], [], c=h.get_edgecolor(), linestyle=h.get_linestyle()) for h in handles]
plt.legend(loc='upper left', fontsize=8)
plt.figtext(0.375,0.75, f'{if_sim} RCR efficiency: {RCR_efficiency}%')
plt.figtext(0.375,0.7, f'{if_sim} Backlobe efficiency: {Backlobe_efficiency}%')
plt.axvline(x = 0.95, color = 'y', label = 'cut')
# plt.text(0, 1.1, 'BL', 
#     verticalalignment='center', 
#     horizontalalignment='center',
#     fontsize=12, 
#     color='black')
# plt.text(0, -0.1, 'RCR', 
#     verticalalignment='center', 
#     horizontalalignment='center',
#     fontsize=12, 
#     color='black')
# Save the plot to a file (in PNG format)
plt.savefig(f'/pub/tangch3/ARIANNA/DeepLearning/plots/Simulation/network_output/200s_time/{if_sim} histogram.png')
print('Done!')

#parameters to change: window size
#might want to have less than window size 10 for frequency (maybe 8) window size 10 means 10 samples, so over 5ns
#try with highest accuracy
