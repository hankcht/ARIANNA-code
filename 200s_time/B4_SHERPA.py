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
from A0_Utilities import getMaxChi, getMaxSNR, load_sim_rcr, load_data, pT

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

    # SHERPA###################################################
    parameters = [sherpa.Discrete('window_size', [1,30])] # range of window sizes to test
    alg = sherpa.algorithms.RandomSearch(max_num_trials=100) # 100 trials

    study = sherpa.Study(parameters=parameters,
                        algorithm=alg,
                        lower_is_better=True,
                        disable_dashboard=True,
                        output_dir='/pub/tangch3/ARIANNA/DeepLearning/sherpa_output')


    for trial in study:
        model = Sequential()
        # change window size to capture the 100 Mhz difference in Backlobe (shadowing effect, we have integer frequencies amplified)
        # window size default is 10, on 256 floats 
        # model.add(Conv2D(20, (4, 10), activation='relu', input_shape=(n_channels, n_samples, 1), groups = 1))
        model.add(Conv2D(20, (4, trial.parameters['window_size']), activation='relu', input_shape=(n_channels, n_samples, 1), groups = 1))
        model.add(Conv2D(10, (1, trial.parameters['window_size']), activation='relu')) # (1,20)
        model.add(Dropout(0.5))
        model.add(Flatten())
        # model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='Adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

        # validation_split is the fraction of training data used as validation data
        # history = model.fit(x, y, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks_list)
        history = model.fit(x, y, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=[study.keras_callback(trial, objective_name='val_loss')])
        study.finalize(trial)

    best_result = study.get_best_result()
    print(best_result)

    algorithm = 'RS'
    save_best_result(best_result, algorithm)

    exit()

if __name__ == "__main__":

    # Set parameters
    amp = '200s' 
    output_cut_value = 0.95 # Change this depending on chosen cut, we get our passed events from this  # Originally 0.95
    TrainCut = 50 # Number of events to use for training, change accordingly if we do not have enough events


    if amp == '200s':
        noiseRMS = 22.53 * units.mV
        station_id = [14,17,19,30]
    elif amp == '100s':
        noiseRMS = 20 * units.mV
        station_id = [13,15,18]

    # path = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/'                                                                                                     #Set which amplifier to run on
    sim_folder = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/{amp}/5.28.25/'

    model_path = f'/pub/tangch3/ARIANNA/DeepLearning/models/{amp}_time/new_chi'                                  
    accuracy_plot_path = f'/pub/tangch3/ARIANNA/DeepLearning/plots/Simulation/accuracy/{amp}_time/new_chi/' 
    loss_plot_path = f'/pub/tangch3/ARIANNA/DeepLearning/plots/Simulation/loss/{amp}_time/new_chi/'         

    current_datetime = datetime.now() # Get the current date and time
    timestamp = current_datetime.strftime("%Y-%m-%d_%H-%M") # Format the datetime object as a string with seconds

    sim_RCR = load_sim_rcr(sim_folder, noise_enabled=True, filter_enabled=True, amp=amp)
    print(f'number of sim is{len(sim_RCR)}')
    # since we load traces depending on stn, we need to make data_Backlobe a full list
    data_Backlobe = []
    data_Backlobe_chi2016 = []
    data_Backlobe_UNIX = [] 
    for id in station_id:
        snr, chi2016, chiRCR, traces, unix = load_data('new_chi_above_curve', amp_type = amp, station_id=id)
        data_Backlobe.extend(traces)
        data_Backlobe_chi2016.extend(chi2016)
        data_Backlobe_UNIX.extend(unix)

    # data_Backlobe_chi2016 = np.array(data_Backlobe_chi2016)
    # indices = np.where(data_Backlobe_chi2016 > 0.8)[0]
    # print(indices)

    # for index in indices: 
    #     pT(data_Backlobe[index], 'test plot data BL', f'/pub/tangch3/ARIANNA/DeepLearning/test_plot_data_BL/test_new_data_BL_{amp}_{index}.png')

    # exit()


    data_Backlobe = np.array(data_Backlobe)
    sim_RCR = np.array(sim_RCR) #

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