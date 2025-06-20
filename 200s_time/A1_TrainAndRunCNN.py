import os
import sherpa
import pickle
import argparse
import numpy as np
from tensorflow import keras
from keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D
from keras.metrics import Recall
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
import random
import math
from datetime import datetime
from icecream import ic
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import B1_BLcurve 
from NuRadioReco.utilities import units
import templateCrossCorr as txc
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')
from A0_Utilities import getMaxChi, getMaxSNR, load_sim_rcr, load_data

def save_best_result(best_result, algorithm=''):
    """
    Input type of algorithm for saving
    1) Saves result into numpy array
    2) Creates histogram and shows best setting
    
    """
    hparam = best_result['window_size']
    hparam_arr = np.array([hparam])
    
    try:
        best_hparam = np.load(f'/pub/tangch3/ARIANNA/DeepLearning/sherpa_output/{algorithm}best_windowsize.npy')
    except FileNotFoundError as e:
        print(e)
        best_hparam = np.array([])  

    best_hparam = np.concatenate((best_hparam, hparam_arr))

    print('Saving best hyperparameter setting')
    np.save(f'/pub/tangch3/ARIANNA/DeepLearning/sherpa_output/{algorithm}best_windowsize.npy', best_hparam)

    from collections import Counter

    count = Counter(best_result)
    most_common_element, most_common_count = count.most_common(1)[0]
    print(f'best setting: {most_common_element}')
    bins = np.linspace(1,60,60)
    plt.hist(best_result, bins)
    plt.xlabel('Window size')
    plt.ylabel('count')
    plt.text(most_common_element, most_common_count, f'Best: {most_common_element}', ha='center', va='bottom', fontsize=10, color='red')
    print(f'saving fig for {algorithm}')
    plt.savefig(f'/pub/tangch3/ARIANNA/DeepLearning/sherpa_output/{algorithm}best_windowsize.png')
    plt.clf()

    print(best_hparam)
    print(len(best_hparam))

def Train_CNN():
    x = np.vstack((training_RCR, training_Backlobe))
    n_samples = x.shape[2]
    n_channels = x.shape[1]
    x = np.expand_dims(x, axis=-1)

    # y is output array (Zeros are BL, Ones for RCR)    
    y = np.vstack((np.ones((training_RCR.shape[0], 1)), np.zeros((training_Backlobe.shape[0], 1)))) 
    s = np.arange(x.shape[0])
    np.random.shuffle(s)
    x = x[s]
    y = y[s]
    print(x.shape, y.shape)
    
    BATCH_SIZE = 32
    EPOCHS = 20 

    # callback automatically saves when loss increases over a number of patience cycles
    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]

    # # SHERPA###################################################
    # parameters = [sherpa.Discrete('window_size', [1,30])] # range of window sizes to test
    # alg = sherpa.algorithms.RandomSearch(max_num_trials=100) # 100 trials

    # study = sherpa.Study(parameters=parameters,
    #                     algorithm=alg,
    #                     lower_is_better=True,
    #                     disable_dashboard=True,
    #                     output_dir='/pub/tangch3/ARIANNA/DeepLearning/sherpa_output')


    # for trial in study:
    #     model = Sequential()
    #     # change window size to capture the 100 Mhz difference in Backlobe (shadowing effect, we have integer frequencies amplified)
    #     # window size default is 10, on 256 floats 
    #     # model.add(Conv2D(20, (4, 10), activation='relu', input_shape=(n_channels, n_samples, 1), groups = 1))
    #     model.add(Conv2D(20, (4, trial.parameters['window_size']), activation='relu', input_shape=(n_channels, n_samples, 1), groups = 1))
    #     model.add(Conv2D(10, (1, trial.parameters['window_size']), activation='relu')) # (1,20)
    #     model.add(Dropout(0.5))
    #     model.add(Flatten())
    #     # model.add(Dense(64, activation='relu'))
    #     model.add(Dense(1, activation='sigmoid'))
    #     model.compile(optimizer='Adam',
    #                     loss='binary_crossentropy',
    #                     metrics=['accuracy'])

    #     # validation_split is the fraction of training data used as validation data
    #     # history = model.fit(x, y, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks_list)
    #     history = model.fit(x, y, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=[study.keras_callback(trial, objective_name='val_loss')])
    #     study.finalize(trial)

    # best_result = study.get_best_result()
    # print(best_result)

    # algorithm = 'RS'
    # save_best_result(best_result, algorithm)

    # exit()
    ###############################################
    model = Sequential()
    # change window size to capture the 100 Mhz difference in Backlobe (shadowing effect, we have integer frequencies amplified)
    # window size default is 10, on 256 floats 
    model.add(Conv2D(20, (4, 10), activation='relu', input_shape=(n_channels, n_samples, 1), groups = 1))
    model.add(Conv2D(10, (1, 10), activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='Adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    # validation_split is the fraction of training data used as validation data
    history = model.fit(x, y, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks_list)

    print(f'Model path: {model_path}')

    # Save the history as a pickle file
    with open(f'{model_path}{timestamp}_RCR_Backlobe_model_2Layer_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    # Plot the training and validation loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'sim RCR vs data BL loss plot')
    plt.savefig(f'{loss_plot_path}_loss_plot_{timestamp}_RCR_Backlobe_model_2Layer.png')
    plt.clf()

    # Plot the training and validation accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'sim RCR vs data BL accuracy plot')
    plt.savefig(f'{accuracy_plot_path}_accuracy_plot_{timestamp}_RCR_Backlobe_model_2Layer.png')
    plt.clf()

    model.summary()

    # Evaluate the model on the validation set
    val_loss, val_acc = model.evaluate(x[-int(0.2 * len(x)):], y[-int(0.2 * len(y)):], verbose=0)
    print(f'Validation Loss: {val_loss}')
    print(f'Validation Accuracy: {val_acc}')

    return model

# Set parameters
amp = '200s' 
output_cut_value = 0.6 # Change this depending on chosen cut, we get our passed events from this  # Originally 0.95
TrainCut = 800 # Number of events to use for training, change accordingly if we do not have enough events


if amp == '200s':
    noiseRMS = 22.53 * units.mV
    station_id = [14,17,19,30]
elif amp == '100s':
    noiseRMS = 20 * units.mV
    station_id = [13,15,18]

# path = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/'                                                                                                     #Set which amplifier to run on
sim_folder = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/{amp}/5.28.25/'

model_path = f'/pub/tangch3/ARIANNA/DeepLearning/models/{amp}_time/new_chi'                                  
accuracy_plot_path = f'/pub/tangch3/ARIANNA/DeepLearning/plots/Simulation/accuracy/{amp}_time/new_chi' 
loss_plot_path = f'/pub/tangch3/ARIANNA/DeepLearning/plots/Simulation/loss/{amp}_time/new_chi'         

current_datetime = datetime.now() # Get the current date and time
timestamp = current_datetime.strftime("%Y-%m-%d_%H-%M") # Format the datetime object as a string with seconds

if __name__ == "__main__":  

    sim_RCR = load_sim_rcr(sim_folder, noise_enabled=True, filter_enabled=True, amp=amp)
    print(f'number of sim is{len(sim_RCR)}')
    # since we load traces depending on stn, we need to make data_Backlobe a full list
    data_Backlobe = []
    data_Backlobe_UNIX = [] 
    for id in station_id:
        snr, chi2016, chiRCR, traces, unix = load_data('new_chi_above_curve', amp_type = amp, station_id=id)
        data_Backlobe.extend(traces)
        data_Backlobe_UNIX.extend(unix)


    data_Backlobe = np.array(data_Backlobe)
    sim_RCR = np.array(sim_RCR)

    data_Backlobe_UNIX = np.array(data_Backlobe_UNIX)
    print(f'RCR shape: {sim_RCR.shape} Backlobe shape: {data_Backlobe.shape}')

    # take a random selection because events are ordered based off CR simulated, so avoids overrepresenting particular Cosmic Rays
    RCR_training_indices = np.random.choice(sim_RCR.shape[0], size=TrainCut, replace=False)
    BL_training_indices = np.random.choice(data_Backlobe.shape[0], size=TrainCut, replace=False)
    training_RCR = sim_RCR[RCR_training_indices, :]
    training_Backlobe = data_Backlobe[BL_training_indices, :]
    # I also want to save the indices of non_trained_events, to use them for our test later
    RCR_non_training_indices = np.setdiff1d(np.arange(sim_RCR.shape[0]), RCR_training_indices)
    BL_non_training_indices = np.setdiff1d(np.arange(data_Backlobe.shape[0]), BL_training_indices)
    print(f'Training shape RCR {training_RCR.shape} Training Shape Backlobe {training_Backlobe.shape} TrainCut {TrainCut}')
    print(f'Non-training RCR count {len(RCR_non_training_indices)} Non-training Backlobe count {len(BL_non_training_indices)}')

    # Now Train
    model = Train_CNN()  

    model.save(f'{model_path}_{timestamp}_RCR_Backlobe_model_2Layer.h5') # currently saving in h5
    model_path = f'{model_path}_{timestamp}_RCR_Backlobe_model_2Layer.h5'
    model = keras.models.load_model(model_path)


    print('------> Training is Done!')




    # Now we run our trained model on the remaining (non-trained) events

    ##################################
    # Here I can make a list of good models by the title/time they were created

    ##################################

    # Now we can test run our trained model on the non trained events (or just all events)
    non_trained_RCR = sim_RCR[RCR_non_training_indices,:]
    non_trained_Backlobe =  data_Backlobe[BL_non_training_indices,:]
    non_trained_Backlobe_UNIX = data_Backlobe_UNIX[BL_non_training_indices] 

    # prob_RCR = model.predict(non_trained_RCR) # Network output of RCR
    # prob_Backlobe = model.predict(non_trained_Backlobe) # Network output of Backlobe
    # print(len(prob_Backlobe))

    print(f'output cut value: {output_cut_value}')

    sim_RCR = np.array(sim_RCR)
    prob_RCR = model.predict(sim_RCR)
    prob_Backlobe = model.predict(data_Backlobe)

    # Finding not weighted RCR efficiency (percentage of RCR events that would pass the cut) 
    sim_RCR_output = prob_RCR
    RCR_efficiency = (sim_RCR_output > output_cut_value).sum() / len(sim_RCR_output)
    RCR_efficiency = (100*RCR_efficiency).round(decimals=2)
    print(f'RCR efficiency: {RCR_efficiency}')

    # Finding Backlobe efficiency (percentage of backlobe that would remain after our cut)
    sim_Backlobe_output = prob_Backlobe
    Backlobe_efficiency = (sim_Backlobe_output > output_cut_value).sum() / len(sim_Backlobe_output)
    Backlobe_efficiency = (100*Backlobe_efficiency).round(decimals=4)
    print(f'Backlobe efficiency: {Backlobe_efficiency}')

    print(f'lengths {len(prob_RCR)} and {len(prob_Backlobe)}')

    # Set up for Network Output histogram
    dense_val = False
    fig, ax = plt.subplots(figsize=(8, 6))  
    hist_values, bin_edges, _ = ax.hist(prob_Backlobe, bins=20, range=(0,1), histtype='step', color='blue', linestyle='solid', label=f'Backlobe {len(prob_Backlobe)}', density=dense_val)
    ax.hist(prob_RCR, bins=20, range=(0,1), histtype='step', color='red', linestyle='solid', label=f'RCR {len(prob_RCR)}', density=dense_val)

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
    fig.text(0.375, 0.75, f'RCR efficiency: {RCR_efficiency}%', fontsize=12)
    fig.text(0.375, 0.7, f'Backlobe efficiency: {Backlobe_efficiency}%', fontsize=12)
    fig.text(0.375, 0.65, f'TrainCut: {TrainCut}', fontsize=12)
    ax.axvline(x=output_cut_value, color='y', label='cut')
    ax.text(0.05, -0.12, 'BL', verticalalignment='center', horizontalalignment='center', fontsize=12, transform=ax.transAxes, color='blue')
    ax.text(0.96, -0.12, 'RCR', verticalalignment='center', horizontalalignment='center', fontsize=12, transform=ax.transAxes, color='red')
    print(f'saving /pub/tangch3/ARIANNA/DeepLearning/plots/Simulation/network_output/{amp}_time/new_chi/{timestamp}_histogram.png')
    plt.savefig(f'/pub/tangch3/ARIANNA/DeepLearning/plots/Simulation/network_output/{amp}_time/new_chi/{timestamp}_histogram.png')
    print(f'------> {amp} Done!')



    # # We can get get the network output, Chi, and SNR of certain events that we want to look at
    
    # BL_identify = []
    # BL_identify_index = []
    # RCR_identify = []
    # RCR_identify_index = []

    # for i, network_output in enumerate(prob_Backlobe):
    #     if network_output >= 0.15 and network_output <= 0.4:
    #         BL_identify.append(network_output)
    #         BL_identify_index.append(i)
    #     if network_output > 0.7:
    #         RCR_identify.append(RCR)
    #         RCR_identify_index.append(i)

    # print(f'Number of BL_identify is {len(BL_identify)}')
    # # print(f'indices of BL_identify are {BL_identify_index}')
    # print(f'Number of RCR_identify is {len(RCR_identify)}')
    # # print(f'indices of RCR_identify are {RCR_identify_index}')

    # # then we get the Chi and SNR values of these events?
    # BL_data_SNRs = []
    # BL_data_Chi = []
    # templates_RCR = B1_BLcurve.loadTemplate(type='RCR', amp=amp)

    # for iR, RCR in enumerate(non_trained_Backlobe):
            
    #     traces = []
    #     for trace in RCR:
    #         traces.append(trace * units.V)
    #     BL_data_SNRs.append(getMaxSNR(traces, noiseRMS=noiseRMS))
    #     if getMaxSNR(traces, noiseRMS=noiseRMS) == float('nan'):
    #         print(getMaxSNR(traces, noiseRMS=noiseRMS))
    #         print('oh no')
    #     BL_data_Chi.append(getMaxChi(traces, 2*units.GHz, templates_RCR, 2*units.GHz))

    # BL_data_SNRs = [round(snr, 2) for snr in BL_data_SNRs]
    # BL_data_Chi = [round(chi, 2) for chi in BL_data_Chi]

    # BL_identify_Chi = [BL_data_Chi[i] for i in BL_identify_index]
    # BL_identify_SNR = [BL_data_SNRs[i] for i in BL_identify_index]
    # BL_identify_UNIX = [non_trained_Backlobe_UNIX[i] for i in BL_identify_index]

    # RCR_identify_Chi = [BL_data_Chi[i] for i in RCR_identify_index]
    # RCR_identify_SNR = [BL_data_SNRs[i] for i in RCR_identify_index]
    # RCR_identify_UNIX = [non_trained_Backlobe_UNIX[i] for i in RCR_identify_index]
    
    # print(f'Sizes of BL data Chi {len(BL_data_Chi)} & SNR {len(BL_data_SNRs)}')
    # print(f'Sizes of RCR identify Chi {len(RCR_identify_Chi)} & SNR {len(RCR_identify_SNR)}')
    # print(f'Sizes of BL identify Chi {len(BL_identify_Chi)} & SNR {len(BL_identify_SNR)}')

    # confirmed_BL_unix = {1449861609, 1455513662, 1455205950, 1458294171, 1449861609, 1450268467, 1450734371, 1455205950, 1455513662, 1458294171, 1450734371, 1449861609, 1450734371, 1457453131, 1454540191, 1455263868}

    # for unix, chi, snr in zip(BL_identify_UNIX, BL_identify_Chi, BL_identify_SNR):
    #     if unix in confirmed_BL_unix:

    #         print('found one')
    #         print(unix, chi, snr)


    # print(f'histogram data: {len(non_trained_Backlobe_UNIX)} and {len(BL_data_SNRs)}')
    # for unix, chi, snr, network_output in zip(non_trained_Backlobe_UNIX, BL_data_SNRs, BL_data_Chi, prob_Backlobe):
    #     if unix in confirmed_BL_unix:
    #         print('found, just not identified')
    #         print(unix, chi, snr, network_output)

    

    # TODO: Plot the traces of identified events

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


