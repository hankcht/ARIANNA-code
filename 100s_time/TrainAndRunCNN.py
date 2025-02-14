import os
import pickle
import argparse
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
#from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
import random
import math
from datetime import datetime
from icecream import ic
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import BLcurve
from NuRadioReco.utilities import units
import templateCrossCorr as txc

import matplotlib
matplotlib.use('Agg')

#                   + sim BL              (1)                
# Train: sim RCR <
#                   + "BL data events"    (2)
# 
# ----------------------------------------------
# 
#                   + sim BL              (1)                
# Run:   sim RCR <
#                   + "BL data events"    (2)

# example: 1.2 is sim_data, 2.2 is data_data 

#Set parameters
path = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/'                              #Edit path to properly point to source folder
model_path = f'/pub/tangch3/ARIANNA/DeepLearning/models/100s_time/'                             #Path to save models
plots_path_accuracy = '/pub/tangch3/ARIANNA/DeepLearning/plots/Simulation/accuracy/100s_time/'  #Path to save plots
plots_path_loss = '/pub/tangch3/ARIANNA/DeepLearning/plots/Simulation/loss/100s_time/'          #Path to save plots
amp = '100s'                                                                                    #Set which amplifier to run on
RCR_path = f'simulatedRCRs/{amp}_10.30.24/'
backlobe_path = f'simulatedBacklobes/{amp}_10.25.24/'
station_data_path = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/eventsPassingNoiseCuts/'
TrainCut = 2000                                                                                 #Number of events to use for training

def getMaxChi(traces, sampling_rate, template_trace, template_sampling_rate, parallelChannels=[[0, 2], [1, 3]]):
    #Parallel channels should be index corresponding to the channel in traces

    maxCorr = []
    for parChans in parallelChannels:
        parCorr = 0
        for chan in parChans:
            xCorr = txc.get_xcorr_for_channel(traces[chan], template_trace, sampling_rate, template_sampling_rate)
            parCorr += np.abs(xCorr)
        maxCorr.append(parCorr / len(parChans))

    return max(maxCorr)

def getMaxSNR(traces, noiseRMS=22.53 * units.mV):

    SNRs = []
    for trace in traces:
        p2p = (np.max(trace) + np.abs(np.min(trace))) * units.V
        SNRs.append(p2p / (2*noiseRMS))

    return max(SNRs)

#First load RCR 
RCR_files = []
print(f'RCR path: {path + RCR_path}')
for filename in os.listdir(path + RCR_path):
    print(f'filename {filename}')
    if filename.startswith(f'SimRCR_{amp}_'):
        print(f'appending')
        RCR_files.append(path + RCR_path +  filename)
RCR = np.empty((0, 4, 256))
print(f'RCR files array: {RCR_files}')
for file in RCR_files:
    print(f'RCR file {file}')
    RCR_data = np.load(file)[0:, 0:4]
    print(f'RCR data shape {RCR_data.shape} and RCR shape {RCR.shape}')
    RCR = np.concatenate((RCR, RCR_data))

# #prints out every byte in this RCR file, was printing only zeros
# with open('../../../../../dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/simulatedRCRs/100s_10.30.24/SimRCR_100s_forcedTrue_3115events_part0.npy', mode="rb") as f:
#     data = f.read()
#     for c in data:
#         print(c, end = " ")

# # Now load BL
# #[1] Simulated Backlobe 
# Backlobes_files = []
# for filename in os.listdir(path + backlobe_path): 
#     if filename.startswith(f'Backlobe_{amp}_'):
#         Backlobes_files.append(path + backlobe_path + filename)
# sim_Backlobe = np.empty((0, 4, 256))
# for file in Backlobes_files:
#     print(f'Backlobe file {file}')
#     Backlobe_data = np.load(file)[0:, 0:4]
#     sim_Backlobe = np.concatenate((sim_Backlobe, Backlobe_data))



#[2] BL Curve Cut on Station Data
data_path = '/pub/tangch3/ARIANNA/DeepLearning/AboveCurve_npfiles/100s/'
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

data_Backlobe = all_station_data

# Now, we use either [1] sim BL, or [2] "BL data events"
parser = argparse.ArgumentParser(description='Determine To Use sim BL or "BL data events"')
parser.add_argument('BLsimOrdata', type=str, help='Use sim BL or "BL data events')
args = parser.parse_args()
if_sim = args.BLsimOrdata

if if_sim == 'sim_data' or if_sim == 'sim_sim':
    Backlobe = sim_Backlobe
    print(f'{if_sim}')
elif if_sim == 'data_sim' or if_sim == 'data_data':
    Backlobe = data_Backlobe
    print('BL data')

print(f'Backlobe shape is: {if_sim} {Backlobe.shape}')

#Make a cut on data, and then can run model on the uncut data after training to see effectiveness (In addition to validation)
#Ie train on 5k, test on 3k if total is 8k

#take a random selection because events are ordered based off CR simulated, so avoids overrepresenting particular Cosmic Rays
RCR_training_indices = np.random.choice(RCR.shape[0], size=TrainCut, replace=False)
BL_training_indices = np.random.choice(Backlobe.shape[0], size=TrainCut, replace=False)
training_RCR = RCR[RCR_training_indices, :]
training_Backlobe = Backlobe[BL_training_indices, :]

print(f'Shape RCR {training_RCR.shape} Shape Backlobe {training_Backlobe.shape} TrainCut {TrainCut}')
print(f'Entering {if_sim}')

#Before I continue, I want to save the indices of non_trained_events, to use them for our test later
RCR_non_training_indices = np.setdiff1d(np.arange(RCR.shape[0]), RCR_training_indices)
print(len(RCR_non_training_indices))
BL_non_training_indices = np.setdiff1d(np.arange(Backlobe.shape[0]), BL_training_indices)
print(len(BL_non_training_indices))

#Now we can train
x = np.vstack((training_RCR, training_Backlobe))
n_samples = x.shape[2]
n_channels = x.shape[1]
x = np.expand_dims(x, axis=-1)
#Zeros are BL, Ones for RCR

#y is output array
y = np.vstack((np.ones((training_RCR.shape[0], 1)), np.zeros((training_Backlobe.shape[0], 1))))
s = np.arange(x.shape[0])
np.random.shuffle(s)
x = x[s]
y = y[s]
print(x.shape)

BATCH_SIZE = 32
EPOCHS = 100 

#callback automatically saves when loss increases over a number of patience cycles
callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]
model = Sequential()
model.add(Conv2D(20, (4, 10), activation='relu', input_shape=(n_channels, n_samples, 1), groups = 1))
model.add(Conv2D(10, (1, 10), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='Adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

history = model.fit(x, y, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks_list)
#validation_split is the fraction of training data used as validation data

print(f'Model path: {model_path}')

#Get the current date and time
current_datetime = datetime.now()

#Format the datetime object as a string with seconds
timestamp_with_seconds = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

#Save the history as a pickle file
with open(f'{model_path}{timestamp_with_seconds}_RCR_Backlobe_model_2Layer_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

#defining simulation multiplier, should depend on how many training cycles done
simulation_multiplier = 1

# Plot the training and validation loss
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title(f'Model 1: Simulation File Used {simulation_multiplier} Times')
plt.savefig(f'{plots_path_loss}{if_sim}_loss_plot_{timestamp_with_seconds}_RCR_Backlobe_model_2Layer.png')
plt.clf()

#Plot the training and validation accuracy
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title(f'Model 1: Simulation File Used {simulation_multiplier} Times')
plt.savefig(f'{plots_path_accuracy}{if_sim}_accuracy_plot_{timestamp_with_seconds}_RCR_Backlobe_model_2Layer.png')
plt.clf()

model.summary()

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(x[-int(0.2 * len(x)):], y[-int(0.2 * len(y)):], verbose=0)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_acc}')

#input the path and file you'd like to save the model as (in h5 format)
if if_sim == 'sim_sim' or if_sim == 'sim_data':
    model.save(f'{model_path}{if_sim}_{timestamp_with_seconds}_RCR_Backlobe_model_2Layer.h5')
    sim_model_path = f'{model_path}{if_sim}_{timestamp_with_seconds}_RCR_Backlobe_model_2Layer.h5'
elif if_sim == 'data_sim' or if_sim == 'data_data':
    model.save(f'{model_path}{if_sim}_{timestamp_with_seconds}_RCR_Backlobe_model_2Layer.h5')
    data_model_path = f'{model_path}{if_sim}_{timestamp_with_seconds}_RCR_Backlobe_model_2Layer.h5'

print('Training is Done!')

# Load the model that we just trained (Extra/Unnecessary Step, but I want to show it clearly what model we are using) 
if if_sim == 'sim_sim' or if_sim == 'sim_data':
    model_path = sim_model_path
elif if_sim == 'data_sim' or if_sim == 'data_data':
    model_path = data_model_path

# model_path = '/pub/tangch3/ARIANNA/DeepLearning/models/200s_time/data_data_2024-10-12_18-12-21_RCR_Backlobe_model_2Layer.h5'

model = keras.models.load_model(model_path)

# #Or can load another previous model, get from models folder
# model = keras.models.load_model(f'/pub/tangch3/ARIANNA/DeepLearning/models/200s_time/2024-10-12_11-52-29_RCR_Backlobe_model_2Layer_0.h5')

##################################
# Here I am saving a list of good models by the time they were created
# 2024-10-12_11-52-29 (Used Sim BL)

##################################

#Now we can test run our trained model on the non trained events
#We can either run on sim BL or "BL data events"
if if_sim == 'sim_sim' or if_sim == 'data_sim':
    non_trained_RCR, non_trained_Backlobe = RCR[RCR_non_training_indices,:], sim_Backlobe[BL_non_training_indices,:] 
elif if_sim == 'sim_data' or if_sim == 'data_data':
    non_trained_RCR, non_trained_Backlobe = RCR[RCR_non_training_indices,:], data_Backlobe[BL_non_training_indices,:] 

prob_RCR = model.predict(non_trained_RCR)
prob_Backlobe = model.predict(non_trained_Backlobe)
print(len(prob_Backlobe))

##########################################################
##########################################################
##########################################################

#Here we can get quickly get the network output, Chi, and SNR of certain events that we want to look at
BL_identify = []
BL_identify_index = []
for i, BL in enumerate(prob_Backlobe):
    if BL > 0.15 and BL < 0.4:
        BL_identify.append(BL)
        BL_identify_index.append(i)

    
print(f'Number of BL_identify is {len(BL_identify)}')
print(f'indices of BL_identify are {BL_identify_index}')

RCR_identify = []
RCR_identify_index = []
for i, RCR in enumerate(prob_Backlobe): # We want to see what data events the network identified as RCR
    if RCR > 0.7:
        RCR_identify.append(RCR)
        RCR_identify_index.append(i)


print(f'Number of RCR_identify is {len(RCR_identify)}')
print(f'indices of RCR_identify are {RCR_identify_index}')

#then we get the Chi and SNR values of these events
noiseRMS = 22.53 * units.mV
BL_data_SNRs = []
BL_data_Chi = []
templates_RCR = BLcurve.loadTemplate(type='RCR', amp=amp)
for iR, RCR in enumerate(data_Backlobe):
        
    traces = []
    for trace in RCR:
        traces.append(trace * units.V)
    BL_data_SNRs.append(getMaxSNR(traces, noiseRMS=noiseRMS))
    BL_data_Chi.append(getMaxChi(traces, 2*units.GHz, templates_RCR, 2*units.GHz))

print(f'Sizes of BL data Chi {len(BL_data_Chi)} & SNR {len(BL_data_SNRs)}')

BL_identify_Chi = [BL_data_Chi[i] for i in BL_identify_index]
BL_identify_SNR = [BL_data_SNRs[i] for i in BL_identify_index]

print(f'BL Chis are {BL_identify_Chi}')

print(f'Sizes of BL identify Chi {len(BL_identify_Chi)} & SNR {len(BL_identify_SNR)}')

RCR_identify_Chi = [BL_data_Chi[i] for i in RCR_identify_index]
RCR_identify_SNR = [BL_data_SNRs[i] for i in RCR_identify_index]

print(f'Sizes of RCR identify Chi {len(RCR_identify_Chi)} & SNR {len(RCR_identify_SNR)}')

# # We first plot the traces
# # We want to use the indices of identified data to find what station they belong to
# data_Backlobe_copy = data_Backlobe.copy()
# data_Backlobe_copy = data_Backlobe_copy.tolist()
# BL_identify_events = [data_Backlobe_copy.index(non_trained_Backlobe[i]) for i in BL_identify_index]
# print(len(BL_identify_events))
# print(BL_identify_events)

# RCR_identify_events = [data_Backlobe_copy.index(non_trained_Backlobe[i]) for i in RCR_identify_index]
# print(len(RCR_identify_events))
# print(RCR_identify_events)


##########################################################
##########################################################
##########################################################

output_cut_value = 0.95 #Change this depending on chosen cut, we get our passed events from this 
print(f'output cut value: {output_cut_value}')

#Finding RCR efficiency (percentage of RCR events that would pass the cut)
sim_RCR_output = prob_RCR
RCR_efficiency = (sim_RCR_output > output_cut_value).sum() / len(sim_RCR_output)
RCR_efficiency = (100*RCR_efficiency).round(decimals=2)
print(f'{if_sim} RCR efficiency: {RCR_efficiency}')

# Finding Backlobe efficiency (percentage of backlobe that would remain after our cut)
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
plt.title(f'RCR-Backlobe network output (100s_time)')
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],size=18)
plt.yticks(size=18)
plt.ylim(0, max(10 ** (np.ceil(np.log10(hist_values)))))
plt.tick_params(axis='both', which='major', labelsize=12)
plt.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)
# handles, labels = plt.get_legend_handles_labels()
# new_handles = [Line2D([], [], c=h.get_edgecolor(), linestyle=h.get_linestyle()) for h in handles]
plt.legend(loc='upper left', fontsize=8)
plt.figtext(0.375,0.75, f'RCR efficiency: {RCR_efficiency}%')
plt.figtext(0.375,0.7, f'Backlobe efficiency: {Backlobe_efficiency}%')
plt.axvline(x = output_cut_value, color = 'y', label = 'cut')
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
print(f'saving /pub/tangch3/ARIANNA/DeepLearning/plots/Simulation/network_output/100s_time/{if_sim}_{timestamp_with_seconds}_histogram.png')
plt.savefig(f'/pub/tangch3/ARIANNA/DeepLearning/plots/Simulation/network_output/100s_time/{if_sim}_{timestamp_with_seconds}_histogram.png')
print('Done!')

# Now we also want to get certain events according to Network Output value
# We want to be able to find their unix time stamp and then plot them

# np.save above_curve_events

