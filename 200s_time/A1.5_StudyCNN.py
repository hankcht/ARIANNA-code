import os # test
import pickle
import argparse
import numpy as np
from tensorflow import keras
import keras_tuner 
from keras_tuner.tuners import RandomSearch
import sherpa
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
from A0_Utilities import getMaxChi, getMaxSNR, load_sim, load_data, pT

def saves_best_result(best_result, algorithm=''):
    """
    Input type of algorithm for saving
    1) Saves result into numpy array
    2) Creates histogram and shows best setting
    # options: best_windowsize (first window), secnd_wdw
    """
    hparam = best_result['window_size2']
    hparam_arr = np.array([hparam])
    
    try:
        best_hparam = np.load(f'/pub/tangch3/ARIANNA/DeepLearning/sherpa_output/{algorithm}secnd_wdw.npy')
    except FileNotFoundError as e:
        print(e)
        best_hparam = np.array([])  

    best_hparam = np.concatenate((best_hparam, hparam_arr))

    print('Saving best hyperparameter setting')
    np.save(f'/pub/tangch3/ARIANNA/DeepLearning/sherpa_output/{algorithm}secnd_wdw.npy', best_hparam)

    from collections import Counter

    count = Counter(best_result)
    most_common_element, most_common_count = count.most_common(1)[0]
    print(f'best setting: {most_common_element}')
    bins = np.linspace(1,60,60)
    plt.hist(best_result, bins)
    plt.xlabel('Window size2')
    plt.ylabel('count')
    plt.text(most_common_element, most_common_count, f'Best: {most_common_element}', ha='center', va='bottom', fontsize=10, color='red')
    print(f'saving fig for {algorithm}')
    plt.savefig(f'/pub/tangch3/ARIANNA/DeepLearning/sherpa_output/{algorithm}secnd_wdw.png')
    plt.clf()

    print(best_hparam)
    print(len(best_hparam))


def save_best_result(best_result, algorithm=''):
    """
    Tracks and visualizes the best 2D hyperparameter settings:
    window_size1 and window_size2 from the two Conv2D layers.
    """
    
    ws1 = best_result['window_size1']
    ws2 = best_result['window_size2']
    hparam_pair = (ws1, ws2)

    save_path = f'/pub/tangch3/ARIANNA/DeepLearning/sherpa_output/{algorithm}window_pair_log.npy'

    # Load existing data or initialize new
    try:
        best_hparam = np.load(save_path, allow_pickle=True)
    except FileNotFoundError:
        print("No existing file found. Creating new array.")
        best_hparam = np.empty((0, 2), dtype=int)

    best_hparam = np.vstack((best_hparam, hparam_pair))
    np.save(save_path, best_hparam)
    from collections import Counter
    pair_list = [tuple(row) for row in best_hparam]
    count = Counter(pair_list)
    most_common_pair, freq = count.most_common(1)[0]
    print(f'Best setting so far: {most_common_pair} occurred {freq} times')

    ws1_vals = [pair[0] for pair in pair_list]
    ws2_vals = [pair[1] for pair in pair_list]
    ws1_bins = np.arange(min(ws1_vals), max(ws1_vals) + 2)
    ws2_bins = np.arange(min(ws2_vals), max(ws2_vals) + 2)

    heatmap, xedges, yedges = np.histogram2d(ws1_vals, ws2_vals, bins=[ws1_bins, ws2_bins])

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap.T, origin='lower', aspect='auto',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               cmap='viridis')
    plt.colorbar(label='Frequency')
    plt.xlabel('Window Size 1 (Conv2D-1 width)')
    plt.ylabel('Window Size 2 (Conv2D-2 width)')
    plt.title(f'2D Hyperparameter Map - {algorithm}')

    # Mark the most frequent point
    plt.text(most_common_pair[0], most_common_pair[1],
             f'{most_common_pair}\nFreq: {freq}', ha='center', va='center',
             color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    # Save 
    fig_path = f'/pub/tangch3/ARIANNA/DeepLearning/sherpa_output/{algorithm}window_pair_heatmap.png'
    plt.savefig(fig_path)
    plt.clf()

    print(f'Saved heatmap to: {fig_path}')
    print(best_hparam)
    print(f'Total entries: {len(best_hparam)}')
    

def prep_training_data():
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

    return x, y

class HyperModel(keras_tuner.HyperModel): # using a hypermodel to flexibly train/tune
    def __init__(self, x, y, fixed_config=None):
        self.x = x
        self.y = y
        self.input_shape = x.shape[1:]
        self.fixed_config = fixed_config
    
    def build(self, hp):
        def param(name, default, type_fn):
            """Use fixed parameters if provided, else use tuning"""
            if self.fixed_config:
                return self.fixed_config.get(name, default) # name of hyperparameter, default value for training
            return type_fn(name)
        
        model = Sequential()

        # change window size to capture the 100 Mhz difference in Backlobe (shadowing effect, we have integer frequencies amplified)
        # window size default is 10, on 256 floats 
        # model.add(Conv2D(20, (4, 10), activation='relu', input_shape=(n_channels, n_samples, 1), groups = 1))
        model.add(Conv2D(
            filters=param("conv1_filters", 20, lambda n: hp.Choice(n, [5, 10, 15, 20])),
            kernel_size=(4, param("kernel_width_1", 10, lambda n: hp.Int(n, 10, 40, step=10))), activation='relu', input_shape=self.input_shape))

        model.add(Conv2D(
            filters=param("conv2_filters", 10, lambda n: hp.Choice(n, [5, 10, 15, 20])),
            kernel_size=(1, param("kernel_width_2", 10, lambda n: hp.Int(n, 10, 40, step=10))), activation='relu'))
        
        model.add(Dropout(param("dropout_rate", 0.5, lambda n: hp.Float(n, 0.3, 0.7, step=0.1))))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.compile(
            optimizer=optimizers.Adam(learning_rate=param("learning_rate", 1e-3, lambda n: hp.Float(n, 1e-4, 1e-2, sampling="log"))),
            loss='binary_crossentropy',
            metrics=['accuracy'])

        return model
    
    def fit(self, hp, model, *args, **kwargs):
        def param(name, default, type_fn):
            """Use fixed parameters if provided, else use tuning"""
            if self.fixed_config:
                return self.fixed_config.get(name, default) # name of hyperparameter, default value for training
            return type_fn(name)
        return model.fit(
            *args,
            validation_split=0.2,
            epochs=param("epochs", 100, lambda n: hp.Int(n, 50, 150, step=25)),
            batch_size=param("batch_size", 32, lambda n: hp.Choice(n, [16, 32, 64])),
            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)],
            verbose=1,
            **kwargs
        )

def Sherpa_Train_CNN():
    x, y = prep_training_data()
    n_samples = x.shape[2]
    n_channels = x.shape[1]

    BATCH_SIZE = 32
    EPOCHS = 100 

    # callback automatically saves when loss increases over a number of patience cycles
    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]

    parameters = [
                  sherpa.Discrete('window_size2', [1,30])] # range of window sizes to test # sherpa.Discrete('window_size1', [1,30]),
    alg = sherpa.algorithms.RandomSearch(max_num_trials=60) # 100 trials

    study = sherpa.Study(parameters=parameters,
                        algorithm=alg,
                        lower_is_better=True,
                        disable_dashboard=False,
                        output_dir='/pub/tangch3/ARIANNA/DeepLearning/sherpa_output')


    for trial in study:
        model = Sequential()
        #
        # change window size to capture the 100 Mhz difference in Backlobe (shadowing effect, we have integer frequencies amplified)
        # window size default is 10, on 256 floats 
        # model.add(Conv2D(20, (4, 10), activation='relu', input_shape=(n_channels, n_samples, 1), groups = 1)) # trial.parameters['window_size1']
        model.add(Conv2D(20, (4, 10), activation='relu', input_shape=(n_channels, n_samples, 1), groups = 1))
        model.add(Conv2D(10, (1, trial.parameters['window_size2']), activation='relu')) # (1,20)
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
    saves_best_result(best_result, algorithm)

    exit()
    ##############################################
    

    print(f'Model path: {model_path}')

    # Save the history as a pickle file
    with open(f'{model_path}{timestamp}_RCR_BL_model_2Layer_two_ws_stdy_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)


    # defining simulation multiplier, should depend on how many training cycles done
    simulation_multiplier = 1 # Not required anymore

    # Plot the training and validation loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Model 1: Simulation File Used {simulation_multiplier} Times')
    plt.savefig(f'{loss_plot_path}{if_sim}_loss_plot_{timestamp}_RCR_BL_model_2Layer_two_ws_stdy.png')
    plt.clf()

    # Plot the training and validation accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Model 1: Simulation File Used {simulation_multiplier} Times')
    plt.savefig(f'{accuracy_plot_path}{if_sim}_accuracy_plot_{timestamp}_RCR_BL_model_2Layer_two_ws_stdy.png')
    plt.clf()

    model.summary()

    # Evaluate the model on the validation set
    val_loss, val_acc = model.evaluate(x[-int(0.2 * len(x)):], y[-int(0.2 * len(y)):], verbose=0)
    print(f'Validation Loss: {val_loss}')
    print(f'Validation Accuracy: {val_acc}')

    return model

def Keras_Study_CNN():
    x, y = prep_training_data()

    # A bunch of different Tuners RS, GS being the most common, I want to try the BO tuner
    tuner = RandomSearch(
        HyperModel(x, y),
        objective='val_loss',
        max_trials=10,  # Number of hyperparameter combinations to try
        directory='my_dir',
        project_name='cnn_rcr_bl_tuning'
    )

    # Perform the search (this will tune the hyperparameters)
    tuner.search(x, y)

    # Get the best model
    best_model = tuner.get_best_models(num_models=1)[0]

    # Evaluate the best model
    val_loss, val_acc = best_model.evaluate(x[-int(0.2 * len(x)):], y[-int(0.2 * len(y)):], verbose=0)
    print(f'Best Model Validation Loss: {val_loss}')
    print(f'Best Model Validation Accuracy: {val_acc}')
    return

def Train_CNN():
    # first define training hyperparameters
    fixed_config = {
        "conv1_filters": 20,
        "conv2_filters": 10,
        "kernel_width_1": 2, #(2,10) (1,5) (10,10)
        "kernel_width_2": 10,
        "dropout_rate": 0.5,
        "learning_rate": 1e-3,
        "epochs": 100,
        "batch_size": 32
    }
    x, y = prep_training_data()
    hypermodel = HyperModel(x, y, fixed_config=fixed_config)
    model = hypermodel.build(hp=None)
    history = model.fit(x, y, validation_split=0.2, epochs=100, batch_size=32, callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)])
    print(f'Model path: {model_path}')
    with open(f'{model_path}{timestamp}_RCR_Backlobe_model_2Layer_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    # Plot the training and validation loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Model Training Loss')
    plt.savefig(f'{loss_plot_path}{if_sim}_loss_plot_{timestamp}_RCR_Backlobe_model_2Layer.png')
    plt.clf()

    # Plot the training and validation accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Model Training Accuracy')
    plt.savefig(f'{accuracy_plot_path}{if_sim}_accuracy_plot_{timestamp}_RCR_Backlobe_model_2Layer.png')
    plt.clf()
    print('loss and accuracy plots done!')

    model.summary()
    val_loss, val_acc = model.evaluate(x[-int(0.2 * len(x)):], y[-int(0.2 * len(y)):], verbose=0)
    print(f'Validation Loss: {val_loss}')
    print(f'Validation Accuracy: {val_acc}')
    
    return model


# Set parameters 
amp = '200s' 
output_cut_value = 0.6 # Change this depending on chosen cut, we get our passed events from this  # Originally 0.95
TrainCut = 5000 # Number of events to use for training, change accordingly if we do not have enough events

if amp == '200s':
    noiseRMS = 22.53 * units.mV
    station_id = [14,17,19,30]
elif amp == '100s':
    noiseRMS = 20 * units.mV
    station_id = [13,15,18]

path = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/'                                                                                                     #Set which amplifier to run on
RCR_path = f'simulatedRCRs/{amp}_2.9.24/'
backlobe_path = f'simulatedBacklobes/{amp}_2.9.24/'

model_path = f'/pub/tangch3/ARIANNA/DeepLearning/models/{amp}_time/'                                  
accuracy_plot_path = f'/pub/tangch3/ARIANNA/DeepLearning/plots/Simulation/accuracy/{amp}_time/' 
loss_plot_path = f'/pub/tangch3/ARIANNA/DeepLearning/plots/Simulation/loss/{amp}_time/'         

current_datetime = datetime.now() # Get the current date and time
timestamp = current_datetime.strftime("%Y-%m-%d_%H-%M") # Format the datetime object as a string with seconds

if __name__ == "__main__":  

    rcr, sim_Backlobe = load_sim(path, RCR_path, backlobe_path, amp)
    data_Backlobe = []
    data_Backlobe_UNIX = [] 
    data_chi = []
    for id in station_id: # since we load traces depending on stn, we need to make data_Backlobe a full list
        snr, chi, trace, unix = load_data('AboveCurve_data', amp_type = amp, station_id=id)
        data_Backlobe.extend(trace)
        data_Backlobe_UNIX.extend(unix)
        data_chi.extend(chi)


    model = keras.models.load_model('/pub/tangch3/ARIANNA/DeepLearning/models/200s_time/data_data_2025-06-02_09-45_RCR_BL_model_2Layer_two_ws_stdy.h5')
    rcr = np.array(rcr)
    prob_RCR = model.predict(rcr)
    prob_Backlobe = model.predict(data_Backlobe)

    RCR_like_indices = np.where(prob_Backlobe > 0.2)[0]
    print(f'RCR-like events indices {RCR_like_indices}') # use this later to get station
    RCR_like_BL = len(RCR_like_indices)
    assert RCR_like_BL == 5


    print(f'Plotting {RCR_like_BL} misidentified Backlobe event traces using pT function.')

    traces_to_plot = data_Backlobe[RCR_like_indices]
    unix_times_to_plot = data_Backlobe_UNIX[RCR_like_indices]
    RCR_like_network_output = prob_Backlobe[RCR_like_indices] 
    RCR_like_chi = data_chi[RCR_like_indices]

    save_dir = f'/pub/tangch3/ARIANNA/DeepLearning/plots/RCR_like_BL/{amp}_time/RCR_like_Traces/'
    os.makedirs(save_dir, exist_ok=True) 

    for i, trace in enumerate(traces_to_plot):
        unix_timestamp = unix_times_to_plot[i]
        RCR_like_no = RCR_like_network_output[i].item()
        RCR_like_no = round(RCR_like_no, 2)
        RCR_like_chi_val = RCR_like_chi[i]
        
        dt_object = datetime.fromtimestamp(unix_timestamp)
        formatted_time_for_filename = dt_object.strftime('%Y%m%d_%H%M%S')

        original_event_index = RCR_like_indices[i]
        plot_filename = os.path.join(save_dir, f'data_data_{timestamp}_pot_RCR_event_{original_event_index}_{formatted_time_for_filename}_netout{RCR_like_no}_chi{RCR_like_chi_val}.png')
        
        plot_title = f'pot_RCR Trace (Event {original_event_index})\nTime: {dt_object.strftime("%Y-%m-%d %H:%M:%S")}'

        pT(traces=trace, title=plot_title, saveLoc=plot_filename)
        
        print(f'------> Saved pot_RCR trace for event {original_event_index} to {plot_filename}')
    
    import sys
    sys.exit()


    # With argparse, we can either use [1] sim BL, or [2] "BL data events" 
    parser = argparse.ArgumentParser(description='Determine To Use sim BL or "BL data events"') 
    parser.add_argument('BLsimOrdata', type=str, default='data_data', help='Use sim BL or "BL data events')
    args = parser.parse_args()
    if_sim = args.BLsimOrdata

    if if_sim == 'sim_data' or if_sim == 'sim_sim':
        Backlobe = sim_Backlobe # Using [1]
        print(f'using sim Backlobe for training')
    elif if_sim == 'data_sim' or if_sim == 'data_data':
        Backlobe = data_Backlobe # Using [2]
        print('using data Backlobe for training')

    Backlobe = np.array(Backlobe)
    data_Backlobe_UNIX = np.array(data_Backlobe_UNIX)
    print(f'RCR shape: {rcr.shape} Backlobe shape: {if_sim} {Backlobe.shape}')

    # take a random selection because events are ordered based off CR simulated, so avoids overrepresenting particular Cosmic Rays
    RCR_training_indices = np.random.choice(rcr.shape[0], size=TrainCut, replace=False)
    BL_training_indices = np.random.choice(Backlobe.shape[0], size=TrainCut, replace=False)

    training_RCR = rcr[:5000, :] # training_RCR = rcr[RCR_training_indices, :]
    training_Backlobe = Backlobe[:5000, :] # training_Backlobe = Backlobe[BL_training_indices, :]

    # I also want to save the indices of non_trained_events, to use them for our test later
    RCR_non_training_indices = np.setdiff1d(np.arange(rcr.shape[0]), RCR_training_indices)
    BL_non_training_indices = np.setdiff1d(np.arange(Backlobe.shape[0]), BL_training_indices)
    print(f'Training shape RCR {training_RCR.shape} Training Shape Backlobe {training_Backlobe.shape} TrainCut {TrainCut}')
    print(f'Non-training RCR count {len(RCR_non_training_indices)} Non-training Backlobe count {len(BL_non_training_indices)}')

  


    # Now Train
    model = Train_CNN()




    # input the path and file you'd like to save the model as (in h5 format)
    if if_sim == 'sim_sim' or if_sim == 'sim_data':
        model.save(f'{model_path}{if_sim}_{timestamp}_RCR_BL_model_2Layer_two_ws_stdy.h5')
        sim_model_path = f'{model_path}{if_sim}_{timestamp}_RCR_BL_model_2Layer_two_ws_stdy.h5'
        print(f'model saved at {sim_model_path}')
    elif if_sim == 'data_sim' or if_sim == 'data_data':
        model.save(f'{model_path}{if_sim}_{timestamp}_RCR_BL_model_2Layer_two_ws_stdy.h5')
        data_model_path = f'{model_path}{if_sim}_{timestamp}_RCR_BL_model_2Layer_two_ws_stdy.h5'
        print(f'model saved at {data_model_path}')

    print('------> Training is Done!')



################################################################################################


    # Now we run our trained model on the remaining (non-trained) events

    # Load the model that we just trained (Extra/Unnecessary Step, but I want to clearly show what model we are using) 
    if if_sim == 'sim_sim' or if_sim == 'sim_data':
        model_path = sim_model_path
    elif if_sim == 'data_sim' or if_sim == 'data_data':
        model_path = data_model_path

    # model_path = '/pub/tangch3/ARIANNA/DeepLearning/models/200s_time/data_data_2024-10-12_18-12-21_RCR_Backlobe_model_2Layer.h5'

    model = keras.models.load_model(model_path)


    ##################################
    # Here I can make a list of good models by the title/time they were created
    # 2024-10-12_11-52-29 (Used Sim BL)

    ##################################

    # # Now we can test run our trained model on the non trained events
    # non_trained_RCR = rcr[RCR_non_training_indices,:]
    # We can either run on sim BL or "BL data events"
    # if if_sim == 'sim_sim' or if_sim == 'data_sim':
    #     non_trained_Backlobe =  Backlobe[BL_non_training_indices,:] 
    # elif if_sim == 'sim_data' or if_sim == 'data_data':
    #     non_trained_Backlobe =  Backlobe[BL_non_training_indices,:]
    #     non_trained_Backlobe_UNIX = data_Backlobe_UNIX[BL_non_training_indices] 

    # prob_RCR = model.predict(non_trained_RCR) # Network output of RCR
    # prob_Backlobe = model.predict(non_trained_Backlobe) # Network output of Backlobe
    # print(len(prob_Backlobe))

    print(f'output cut value: {output_cut_value}')

    rcr = np.array(rcr)
    prob_RCR = model.predict(rcr)
    prob_Backlobe = model.predict(Backlobe)

    # Finding not weighted RCR efficiency (percentage of RCR events that would pass the cut) 
    sim_RCR_output = prob_RCR
    RCR_efficiency = (sim_RCR_output > output_cut_value).sum() / len(sim_RCR_output)
    RCR_efficiency = (100*RCR_efficiency).round(decimals=2)
    print(f'{if_sim} RCR efficiency: {RCR_efficiency}')

    # Finding Backlobe efficiency (percentage of backlobe that would remain after our cut)
    sim_Backlobe_output = prob_Backlobe
    Backlobe_efficiency = (sim_Backlobe_output > output_cut_value).sum() / len(sim_Backlobe_output)
    Backlobe_efficiency = (100*Backlobe_efficiency).round(decimals=4)
    print(f'{if_sim} Backlobe efficiency: {Backlobe_efficiency}')

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
    print(f'saving /pub/tangch3/ARIANNA/DeepLearning/plots/Simulation/network_output/{amp}_time/{if_sim}_{timestamp}_histogram.png')
    plt.savefig(f'/pub/tangch3/ARIANNA/DeepLearning/plots/Simulation/network_output/{amp}_time/{if_sim}_{timestamp}_histogram.png')
    print(f'------> network output plot for {amp} {if_sim} is done!')


    # Here we plot data events that are RCR-like
    threshold = 25
    RCR_like_indices = np.where(prob_Backlobe > output_cut_value)[0]
    print(f'RCR-like events indices {RCR_like_indices}') # use this later to get station
    RCR_like_BL = len(RCR_like_indices)

    print(f'\nChecking for Backlobe events misidentified as RCR (above {output_cut_value} threshold)...')
    print(f'Found {RCR_like_BL} such events.')

    if 0 < RCR_like_BL <= threshold:
        print(f'Plotting {RCR_like_BL} misidentified Backlobe event traces using pT function.')

        traces_to_plot = Backlobe[RCR_like_indices]
        unix_times_to_plot = data_Backlobe_UNIX[RCR_like_indices]
        RCR_like_network_output = prob_Backlobe[RCR_like_indices] 

        save_dir = f'/pub/tangch3/ARIANNA/DeepLearning/plots/RCR_like_BL/{amp}_time/RCR_like_Traces/'
        os.makedirs(save_dir, exist_ok=True) 

        for i, trace in enumerate(traces_to_plot):
            unix_timestamp = unix_times_to_plot[i]
            RCR_like_no = RCR_like_network_output[i].item()
            RCR_like_no = round(RCR_like_no, 2)
            
            dt_object = datetime.fromtimestamp(unix_timestamp)
            formatted_time_for_filename = dt_object.strftime('%Y%m%d_%H%M%S')

            original_event_index = RCR_like_indices[i]
            plot_filename = os.path.join(save_dir, f'{if_sim}_{timestamp}_pot_RCR_event_{original_event_index}_{formatted_time_for_filename}_netout{RCR_like_no}.png')
            
            plot_title = f'pot_RCR Trace (Event {original_event_index})\nTime: {dt_object.strftime("%Y-%m-%d %H:%M:%S")}'

            pT(traces=trace, title=plot_title, saveLoc=plot_filename)
            
            print(f'------> Saved pot_RCR trace for event {original_event_index} to {plot_filename}')

    elif RCR_like_BL > threshold:
        print(f'Skipping plotting individual traces: {RCR_like_BL} events found, which is more than the limit of{threshold}.')
    else:
        print('No Backlobe events were potentially RCR. No individual trace plots needed.')

