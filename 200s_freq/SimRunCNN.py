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

#Set parameters
path = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/'                            #Edit path to properly point to source folder
model_path = f'/data/homezvol3/tangch3/ARIANNA/DeepLearning/models/200s_freq/'                #Path to save models
plots_path_accuracy = '/data/homezvol3/tangch3/ARIANNA/DeepLearning/plots/accuracy/200s_freq/'#Path to save plots
plots_path_loss = '/data/homezvol3/tangch3/ARIANNA/DeepLearning/plots/loss/200s_freq/'        #Path to save plots
amp = '200s'                                                                                  #Set which amplifier to run on
RCR_path = f'simulatedRCRs/{amp}_2.9.24/'
backlobe_path = f'simulatedBacklobes/{amp}_2.9.24/'
TrainCut = 5000                                                                     #Number of events to use for training

def load_data():
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
    
    Backlobes_files = []
    for filename in os.listdir(path + backlobe_path):
        if filename.startswith(f'Backlobe_{amp}_'):
            Backlobes_files.append(path + backlobe_path + filename)
    Backlobe = np.empty((0, 4, 256))
    for file in Backlobes_files:
        print(f'Backlobe file {file}')
        Backlobe_data = np.load(file)[0:, 0:4]
        Backlobe = np.concatenate((Backlobe, Backlobe_data))
    
    return (RCR, Backlobe) 

RCR, Backlobe = load_data()
RCR, Backlobe = RCR[5000:], Backlobe[5000:]

# turn RCR time-series to frequency domain
sampling_rate = 2
RCR_freq = np.fft.rfft(RCR, axis=-1)
print(RCR_freq.shape)

#turn Backlobes time-series to frequency domain
sampling_rate = 2
Backlobe_freq = np.fft.rfft(Backlobe, axis=-1)

# Load the model, copy from /models
model = keras.models.load_model(f'/data/homezvol3/tangch3/ARIANNA/DeepLearning/models/200s_freq/2024-04-10_16-09-09_RCR_Backlobe_model_2Layer_0.h5')

prob_RCR_freq = model.predict(RCR_freq)

prob_backlobe_freq = model.predict(Backlobe_freq)

output_cut_value = 0.05 #Change this depending on chosen cut
print(f'output cut value: {output_cut_value}')

#Finding RCR efficiency (percentage of RCR events that would pass the cut)
sim_RCR_output = prob_RCR_freq
RCR_efficiency = (sim_RCR_output < output_cut_value).sum() / len(sim_RCR_output)
RCR_efficiency = RCR_efficiency.round(decimals=4)*100
print(f'RCR efficiency: {RCR_efficiency}')

#Finding Backlobe efficiency (percentage of backlobe that would remain after our cut)
sim_Backlobe_output = prob_backlobe_freq
Backlobe_efficiency = (sim_Backlobe_output < output_cut_value).sum() / len(sim_Backlobe_output)
Backlobe_efficiency = Backlobe_efficiency.round(decimals=4)*100
print(f'Backlobe efficiency: {Backlobe_efficiency}')


dense_val = False
hist_values, bin_edges, _ = plt.hist(prob_backlobe_freq, bins=20, range=(0, 1), histtype='step', color='red', linestyle='solid', label='Backlobe', density=dense_val)
plt.hist(prob_RCR_freq, bins=20, range=(0, 1), histtype='step',color='blue', linestyle='solid',label='RCR',density = dense_val)

# Set logarithmic scale for y-axis
plt.yscale('log')

# Set labels and title
plt.xlabel('Network Output', fontsize = 18)
plt.ylabel('Number of Events', fontsize = 18)
plt.title('RCR vs Backlobe network output (200s_freq)')
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],size=18)
plt.yticks(size=18)
plt.ylim(1, max(10 ** (np.ceil(np.log10(hist_values)))))
plt.tick_params(axis='both', which='major', labelsize=13)
plt.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)
# handles, labels = plt.get_legend_handles_labels()
# new_handles = [Line2D([], [], c=h.get_edgecolor(), linestyle=h.get_linestyle()) for h in handles]
plt.legend(loc='upper left', fontsize=8)
plt.figtext(0.375,0.75, f'RCR efficiency: {RCR_efficiency}%')
plt.figtext(0.375,0.7, f'Backlobe efficiency: {Backlobe_efficiency}%')
plt.axvline(x = 0.05, color = 'y', label = 'cut')
# Save the plot to a file (in PNG format)
plt.savefig('/data/homezvol3/tangch3/ARIANNA/DeepLearning/plots/Simulation/network_output/200s_freq/histogram.png')

#include RCR efficiency RCR files/total files
#now train on frequency
#parameters to change: window size
#might want to have less than window size 10 for frequency (maybe 8) window size 10 means 10 samples, so over 5ns