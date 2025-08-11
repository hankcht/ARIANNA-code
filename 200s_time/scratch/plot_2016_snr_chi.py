import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import os, sys

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from A0_Utilities import load_520_data

station_id = []
station_data_folder = '/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/5.20.25/'
parameters = ['ChiRCR', 'Chi2016']
plot_folder = f'/pub/tangch3/ARIANNA/DeepLearning'
extraname = 'quicktest'
if_sim = ''


for id in station_id:
    snr = load_520_data(id, 'SNR', station_data_folder)
    for param in parameters:
        chi = load_520_data(id, param, station_data_folder)

        SNRbins = np.logspace(0.477, 2, num=80)
        maxCorrBins = np.arange(0, 1.0001, 0.01)
        plt.hist2d(snr, chi, bins=[SNRbins, maxCorrBins], norm=matplotlib.colors.LogNorm())
        plt.colorbar()
        plt.xlim((3, 100))
        plt.ylim((0, 1))
        plt.xlabel('SNR')
        plt.ylabel('Avg Chi Highest Parallel Channels')
        # plt.legend()
        plt.xscale('log')
        plt.tick_params(axis='x', which='minor', bottom=True)
        plt.grid(visible=True, which='both', axis='both') 
        plt.title(f'Station {id} - SNR vs. Chi')
        print(f'Saving {plot_folder}/{extraname}Stn{id}_SNR-Chi{param}_All{if_sim}.png')
        # plt.scatter(sim_SNRs, sim_Chi, c=sim_weights, cmap=cmap, alpha=0.9, norm=matplotlib.colors.LogNorm())
        plt.savefig(f'{plot_folder}/{extraname}Stn{id}_SNR-Chi{param}_All{if_sim}.png')
        plt.close()