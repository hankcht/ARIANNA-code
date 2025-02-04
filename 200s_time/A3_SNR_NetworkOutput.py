import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow
import keras
import time
from tensorflow import keras
from keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
#from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import random
import datetime
import pandas as pd
from glob import glob
from PIL import Image
import pickle
import argparse
from pathlib import Path
from NuRadioReco.utilities import units
from NuRadioReco.utilities.io_utilities import read_pickle
from matplotlib.markers import MarkerStyle
from A0_Utilities import getMaxSNR, load_sim, load_data, RunTrainedModel

#this file creates SNR-NetworkOutput plots of:
#   [1] simualted RCR(blue) 
#   [2] simulated backlobe(red) 
#   [3] all station data 
#   [4] Above curve (data) Backlobe
#   [5] TODO: selected station data 
#   [6] TODO: station data that passed Ryan's noise cut (only 200 series)

def generalplotSNRNetworkOutput(SNRs, NO, labeling, facecolor, plot_folder, plot_name, type='general', part=0):

    plt.figure(figsize=(8, 6))
    plt.scatter(SNRs, NO, label=f'{len(NO)} {labeling}', facecolor=facecolor, edgecolor='none')
    plt.xlim((3, 400))
    plt.xscale('log') # can also do linear
    plt.ylim((-0.05, 1.05)) 
    plt.xlabel('SNR')
    plt.ylabel('Network Output')
    plt.legend()
    plt.tick_params(axis='x', which='minor', bottom=True)
    plt.grid(visible=True, which='both', axis='both')
    plt.text(3, 1.1, 'RCR', verticalalignment='center', horizontalalignment='center', fontsize=12, color='black')
    plt.text(3, -0.1, 'BL', verticalalignment='center', horizontalalignment='center', fontsize=12, color='black')
 
    if type == 'general':
        plt.title(f'{plot_name} SNR-NetworkOutput')
        print(f'Saving {plot_folder}/{plot_name}_SNRNO.png')
        plt.savefig(f'{plot_folder}/{plot_name}_SNRNO.png')
    if type == 'All_data':
        plt.title(f'{plot_name} part_{part} SNR-NetworkOutput')
        print(f'Saving {plot_folder}/{plot_name}_part_{part}_SNRNO.png')
        plt.savefig(f'{plot_folder}/{plot_name}_part_{part}_SNRNO.png')

    plt.clf()


def plotSimSNRNetworkOutput(noiseRMS, type='combined'):

    events_RCR = RCR
    prob_RCR = RunTrainedModel(RCR, model_path)
    facecolor_RCR = 'blue'
    labeling_RCR = 'RCR Simulations'
    plot_name_RCR = 'simRCR'

    events_Backlobe = sim_Backlobe
    prob_Backlobe = RunTrainedModel(sim_Backlobe, model_path)
    facecolor_Backlobe = 'red'
    labeling_Backlobe = 'Backlobe Simulations'
    plot_name_Backlobe = 'simBacklobe'

    sim_SNRs_RCR = []
    for i, event in enumerate(events_RCR):
        traces = [trace * units.V for trace in event]
        sim_SNRs_RCR.append(getMaxSNR(traces, noiseRMS=noiseRMS))
    
    sim_SNRs_Backlobe = []
    for i, event in enumerate(events_Backlobe):
        traces = [trace * units.V for trace in event]
        sim_SNRs_Backlobe.append(getMaxSNR(traces, noiseRMS=noiseRMS))

    print(f'len sim snr RCR: {len(sim_SNRs_RCR)}')
    print(f'len sim snr Backlobe: {len(sim_SNRs_Backlobe)}')

    sim_plot_folder = '/pub/tangch3/ARIANNA/DeepLearning/plots/SNR_NetworkOutput/Simulation'

    if type == 'individual':
        generalplotSNRNetworkOutput(sim_SNRs_RCR, prob_RCR, labeling_RCR, facecolor_RCR, sim_plot_folder, plot_name_RCR)
        generalplotSNRNetworkOutput(sim_SNRs_Backlobe, prob_Backlobe, labeling_Backlobe, facecolor_Backlobe, sim_plot_folder, plot_name_Backlobe)
    elif type == 'combined':
        plt.figure(figsize=(8, 6))
        plt.scatter(sim_SNRs_Backlobe, prob_Backlobe, label=f'{len(prob_Backlobe)} {labeling_Backlobe}', facecolor=facecolor_Backlobe, edgecolor='none')
        plt.scatter(sim_SNRs_RCR, prob_RCR, label=f'{len(prob_RCR)} {labeling_RCR}', facecolor=facecolor_RCR, edgecolor='none') 
        plt.xlim((3, 400))
        plt.xscale('log')
        plt.ylim((-0.05, 1.05))
        plt.xlabel('SNR')
        plt.ylabel('Network Output')
        plt.legend()
        plt.tick_params(axis='x', which='minor', bottom=True)
        plt.grid(visible=True, which='both', axis='both')
        plt.title('Simulation SNR-NetworkOutput')
        plt.text(3, 1.1, 'RCR', verticalalignment='center', horizontalalignment='center', fontsize=12, color='black')
        plt.text(3, -0.1, 'BL', verticalalignment='center', horizontalalignment='center', fontsize=12, color='black')

        print(f'Saving {sim_plot_folder}/sim_SNRNO.png')
        plt.savefig(f'{sim_plot_folder}/sim_SNRNO.png')
        plt.clf()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Determine to plot sim or data') 
    parser.add_argument('simOrdata', type=str,  choices=['sim', 'data'], help='plot sim or data')
    parser.add_argument('--station', type=int, default=14, help='Station to run on') # station only makes sense for data
    args = parser.parse_args()
    simOrdata = args.simOrdata
    station_id = args.station

    model_path = f'/pub/tangch3/ARIANNA/DeepLearning/models/'  

    if simOrdata == 'sim':

        #Set parameters
        ampforsim = '200s'  
        sim_RCR_path = f'simulatedRCRs/{ampforsim}_2.9.24/'
        sim_Backlobe_path = f'simulatedBacklobes/{ampforsim}_10.25.24/' 
        path = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/'                                                     
   
        if ampforsim == '200s':
            noiseRMS = 22.53 * units.mV
        elif ampforsim == '100s':
            noiseRMS = 20 * units.mV

        RCR, sim_Backlobe = load_sim(path, sim_RCR_path, sim_Backlobe_path, ampforsim) 

        plotSimSNRNetworkOutput(noiseRMS, type='combined') # type = individual or combined

        print('Simulation done!')

    elif simOrdata == 'data':

        facecolor = 'black'
        plot_folder = f'/pub/tangch3/ARIANNA/DeepLearning/plots/SNR_NetworkOutput/Station_{station_id}'

        if station_id in [14, 17, 19, 30]:
            amp_type = '200s'
            noiseRMS = 22.53 * units.mV # keeping this in case I need to calculate snr for verification
        elif station_id in [13, 15, 18]:
            amp_type = '100s'
            noiseRMS = 20 * units.mV
        
        All_data_SNR, All_data_Chi, All_data_Traces, All_data_UNIX = load_data('All_data', amp_type, station_id)

        partition = 3
        chunk = 600000

        if partition == 1:
            All_data_Traces = All_data_Traces[:chunk]
            All_data_SNR = All_data_SNR[:chunk]
        elif partition == 2:
            if station_id == 17:
                All_data_Traces = All_data_Traces[chunk:2*chunk]
                All_data_SNR = All_data_SNR[chunk:2*chunk]
            else:
                All_data_Traces = All_data_Traces[chunk:]
                All_data_SNR = All_data_SNR[chunk:]
        elif partition == 3:
            All_data_Traces = All_data_Traces[2*chunk:]
            All_data_SNR = All_data_SNR[2*chunk:]

        All_data_Traces = np.array(All_data_Traces)

        prob_All = RunTrainedModel(All_data_Traces, model_path)

        generalplotSNRNetworkOutput(All_data_SNR, prob_All, labeling='All Station Data', facecolor=facecolor, 
                                    plot_folder=plot_folder, plot_name='All', type='All_data', part=partition)

        print('All station data done!')
        
        above_curve_data_SNR, above_curve_data_Chi, above_curve_data_Traces, above_curve_data_UNIX = load_data('AboveCurve_data', amp_type, station_id)
        above_curve_data_Traces = np.array(above_curve_data_Traces)

        prob_Above_curve = RunTrainedModel(above_curve_data_Traces, model_path)

        generalplotSNRNetworkOutput(above_curve_data_SNR, prob_Above_curve, labeling='Above Curve Station Data', facecolor=facecolor, plot_folder=plot_folder, plot_name='AboveCurve')

        print('Above Curve data done!')

    else:
        print('option does not exist')



