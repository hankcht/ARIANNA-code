import matplotlib.pyplot as plt
import numpy as np
import os
import keras
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import random
import datetime
import pandas as pd
from glob import glob
import matplotlib
import argparse
from A0_Utilities import load_sim
from A0_Utilities import getMaxSNR, load_sim, load_data, RunTrainedModel
from A1_TrainAndRunCNN import output_cut_value # We import the network output cut variable from A1 
from NuRadioReco.utilities import units

# here I define this "out-of-place" function because I am learning how to apply the high cohesion, low coupling programming philosophy
def add_confirmed_histogram(ax, prob_confirmed):
    # does not include whole plotting, use with plot_data_histogram()
    hist_values, bin_edges, _ = ax.hist(prob_confirmed, bins=20, range=(0, 1), histtype='step', color='green', linestyle='solid', label=f'confirmed BL {len(prob_confirmed)}', density=False)

def plot_data_histogram(plot_folder, prob_station, number_passed_events, prob_confirmed=[], if_confirmed=False):
    fig, ax = plt.subplots(figsize=(8, 6))
    hist_values, bin_edges, _ = ax.hist(prob_station, bins=20, range=(0, 1), histtype='step', color='black', linestyle='solid', label=f'All data {len(prob_station)}', density=False)
    if if_confirmed:
        print('adding confirmed BL')
        add_confirmed_histogram(ax, prob_confirmed)
    ax.set_xlabel('Network Output', fontsize=18)
    ax.set_ylabel('Number of Events', fontsize=18)
    ax.set_yscale('log')  # Set logarithmic scale for y-axis, if needed
    # ax.set_ylim(1, max(10 ** (np.ceil(np.log10(hist_values))))) # uncomment if needed for semi-log plot
    ax.set_title(f'Station {station_id} network output (200s_time)')
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='both', which='both', labelsize=12, length=5)
    ax.text(0.05, 0.93, f'Total Passed Events: {number_passed_events}', transform=ax.transAxes, fontsize=12)
    ax.text(0.05, -0.12, 'BL', verticalalignment='center', horizontalalignment='center', fontsize=12, transform=ax.transAxes, color='blue')
    ax.text(0.96, -0.12, 'RCR', verticalalignment='center', horizontalalignment='center', fontsize=12, transform=ax.transAxes, color='red')
    plt.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)
    ax.axvline(x=output_cut_value, color='y', label='cut')
    ax.legend(loc='upper right', fontsize=8, bbox_to_anchor=(0.85, 0.99))
    print(f'{plot_folder}/Station {station_id}/Stn{station_id}_histogram.png')
    plt.savefig(f'{plot_folder}/Station {station_id}/Stn{station_id}_histogram.png')
    print('Done!')

def plot_sim_histogram(prob_RCR, prob_Backlobe, ampforsim):

    # Finding RCR efficiency (percentage of RCR events that would pass the cut)
    sim_RCR_output = prob_RCR
    RCR_efficiency = (sim_RCR_output > output_cut_value).sum() / len(sim_RCR_output)
    RCR_efficiency = (100*RCR_efficiency).round(decimals=2)
    print(f'RCR efficiency: {RCR_efficiency}')

    # Finding Backlobe efficiency (percentage of backlobe that would remain after our cut)
    sim_Backlobe_output = prob_Backlobe
    Backlobe_efficiency = (sim_Backlobe_output > output_cut_value).sum() / len(sim_Backlobe_output)
    Backlobe_efficiency = (100*Backlobe_efficiency).round(decimals=4)
    print(f'Backlobe efficiency: {Backlobe_efficiency}')

    dense_val = False
    fig, ax = plt.subplots(figsize=(8, 6))  
    hist_values, bin_edges, _ = ax.hist(prob_Backlobe, bins=20, range=(0, 1), histtype='step', color='blue', linestyle='solid', label=f'Backlobe {len(prob_Backlobe)}', density=dense_val)
    ax.hist(prob_RCR, bins=20, range=(0, 1), histtype='step', color='red', linestyle='solid', label=f'RCR {len(prob_RCR)}', density=dense_val)

    ax.set_xlabel('Network Output', fontsize=18)
    ax.set_ylabel('Number of Events', fontsize=18)
    ax.set_yscale('log')
    ax.set_title(f'{ampforsim}_time RCR-Backlobe network output')
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylim(0, max(10 ** (np.ceil(np.log10(hist_values)))))
    ax.tick_params(axis='both', which='major', labelsize=12)
    fig.subplots_adjust(left=0.2, right=0.85, bottom=0.2, top=0.8)
    ax.legend(loc='upper left', fontsize=8)
    fig.text(0.38, 0.75, f'RCR efficiency: {RCR_efficiency}%', fontsize=12)
    fig.text(0.38, 0.7, f'Backlobe efficiency: {Backlobe_efficiency}%', fontsize=12)
    ax.axvline(x=output_cut_value, color='y', label='cut')
    ax.text(0.05, -0.12, 'BL', verticalalignment='center', horizontalalignment='center', fontsize=12, transform=ax.transAxes, color='blue')
    ax.text(0.96, -0.12, 'RCR', verticalalignment='center', horizontalalignment='center', fontsize=12, transform=ax.transAxes, color='red')
    print(f'saving /pub/tangch3/ARIANNA/DeepLearning/plots/Histograms/Simulation/{ampforsim}_time/sim_histogram.png')
    plt.savefig(f'/pub/tangch3/ARIANNA/DeepLearning/plots/Histograms/Simulation/{ampforsim}_time/sim_histogram.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Determine to plot sim or data') 
    parser.add_argument('pickRundata', type=str,  choices=['sim','confirmed_BL', 'data'], help='plot sim, confirmed BL, or data')
    parser.add_argument('--station', type=int, default=14, help='Station to run on') # station only makes sense for data
    args = parser.parse_args()
    pickRundata = args.pickRundata
    station_id = args.station

    model_path = f'/pub/tangch3/ARIANNA/DeepLearning/models/' 

    if pickRundata == 'sim':

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

        prob_RCR = RunTrainedModel(RCR, model_path)
        prob_Backlobe = RunTrainedModel(sim_Backlobe, model_path)

        plot_sim_histogram(prob_RCR, prob_Backlobe, ampforsim)

    elif pickRundata == 'data':
        # above curve (data Backlobe) for now:
        # for all_data, need to fix memory issue

        if station_id in [14, 17, 19, 30]:
            amp_type = '200s'
            noiseRMS = 22.53 * units.mV
        elif station_id in [13, 15, 18]:
            amp_type = '100s'
            noiseRMS = 20 * units.mV

        plot_folder = '/pub/tangch3/ARIANNA/DeepLearning/plots/Histograms'

        above_curve_data_SNR, above_curve_data_Chi, above_curve_data_Traces, above_curve_data_UNIX = load_data('AboveCurve_data', amp_type, station_id)

        above_curve_data_Traces = np.array(above_curve_data_Traces)
        prob_station = RunTrainedModel(above_curve_data_Traces, model_path)

        print(len(prob_station))

        print(f'output cut value: {output_cut_value}')

        # =================================================================================================================================
        # Here we get useful information about how our model did with out network output cut
        # note: depending on the input into the model, our interpretation below is different. We could just have passed events for example.
        # (11/29/2024) We used selected_station, so the passed events are actually selected-passed events
        # =================================================================================================================================

        number_passed_events = (prob_station > output_cut_value).sum() # number of events passing output cut
        passed_events_efficiency = number_passed_events / len(prob_station)
        passed_events_efficiency = passed_events_efficiency.round(decimals=4)*100 # efficiency

        print(f'number of passed events: {number_passed_events}')
        print(f'Output cut efficiency: {passed_events_efficiency}')

        plot_data_histogram(plot_folder, prob_station, number_passed_events)









# Old, Unused stuff
selected_events_dict = {
'stn14selectedevents' : [3, 6, 13, 15, 18, 19, 21, 22, 24, 28, 35, 39, 40, 49],
'stn17selectedevents' : [1, 3, 18, 21, 22, 26, 28, 29, 31, 33, 44, 45, 47, 48, 49, 50],
'stn19selectedevents' : [1, 2, 4, 5, 7, 15, 17, 18, 19, 25, 27, 31, 34], 
'stn30selectedevents' : [4, 6, 7, 8, 9, 11, 12, 13, 16, 17, 20, 22, 25, 26, 29, 30, 32, 33, 34, 36, 38, 43, 49, 50, 55, 60, 61, 62]
}