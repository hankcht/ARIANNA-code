import matplotlib.pyplot as plt
import numpy as np
import os
import time
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import random
import datetime
import pandas as pd
from glob import glob
import matplotlib

path = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/'    #Edit path to properly point to folder
amp = '100s'                                                                  #Set which amplifier to run on
RCR_path = f'simulatedRCRs/{amp}_2.9.24/'
backlobe_path = f'simulatedBacklobes/{amp}_2.9.24/'

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

#Now setup for the plot
RCR_simulation_number = 408 #this represents which simulation event we want to plot, check total event number to decide how much this goes to
Backlobe_simulation_number = 247  

RCR, Backlobe = RCR[RCR_simulation_number - 1], Backlobe[Backlobe_simulation_number - 1]

# for c in RCR[0]:
#     print(c, end = " ")

#Important Clarification: In our actual experiment, we receive one data point per 0.5ns, so our duration of 128ns gives 256 data points
#it is different from here where I organize one data point to one ns and make the total time 256ns (these two are mathematically identical)
time_length = 256
time = np.arange(0, time_length, 1) 
#print(time)

def RCRPlotfunction(channel_label): 
    plt.plot(time, RCR[channel_label,:time_length])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'{amp} RCR Time Trace #{RCR_simulation_number} Channel {channel_label}')
    plt.grid(True)
    plt.show()
    plt.savefig(f'/data/homezvol3/tangch3/ARIANNA/DeepLearning/plots/traces/{amp}_time/RCR/#{RCR_simulation_number}Channel{channel_label}timetrace.png')
    plt.clf()

#only allowed values are from 0, 1, 2, 3, corresponding to our physical Channels 0, 1, 2, and 3
for channel_label in np.arange(0, 4, 1):
    RCRPlotfunction(channel_label)

def BacklobePlotfunction(channel_label): 
    plt.plot(time, Backlobe[channel_label,:time_length])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'{amp} Backlobe Time Trace #{Backlobe_simulation_number} Channel {channel_label}')
    plt.grid(True)
    plt.show()
    plt.savefig(f'/data/homezvol3/tangch3/ARIANNA/DeepLearning/plots/traces/{amp}_time/Backlobe/#{Backlobe_simulation_number}Channel{channel_label}timetrace.png')
    plt.clf()

#only allowed values are from 0, 1, 2, 3, corresponding to our physical Channels 0, 1, 2, and 3
for channel_label in np.arange(0, 4, 1):
    BacklobePlotfunction(channel_label)