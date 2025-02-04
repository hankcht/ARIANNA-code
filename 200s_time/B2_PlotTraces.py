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
import NuRadioReco
from NuRadioReco.utilities import units
from NuRadioReco.utilities.io_utilities import read_pickle
from A0_Utilities import load_sim, pT, getMaxChi, getMaxSNR
from B1_BLcurve import loadTemplate, plotSimSNRChi

amp = '200s'  
type = 'Backlobe'

if amp == '200s':
    noiseRMS = 22.53 * units.mV        
elif amp == '100s':
    noiseRMS = 20 * units.mV       

path = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/'                                                                                                     #Set which amplifier to run on
RCR_path = f'simulatedRCRs/{amp}_10.30.24/'
backlobe_path = f'simulatedBacklobes/{amp}_10.25.24/'

#SO1_stationdata...
#Line 186
# find the pT function
# show both unix and datetime on traces plot
# Also Save the events that pass the neural net  


templates_RCR = loadTemplate(type='RCR', amp=amp)
sim_traces, sim_Chi, sim_SNRs, sim_weights, simulation_date = plotSimSNRChi(templates_RCR, noiseRMS, amp, type=type)

chi_indices = []
plot_chi = []
for i, chi in enumerate(sim_Chi):
    if chi > 0.74:
        plot_chi.append(chi)
        chi_indices.append(i)

print(len(plot_chi))
plot_traces = sim_traces[chi_indices]
plot_SNR = sim_SNRs[chi_indices]
plot_SNR = [round(snr, 2) for snr in plot_SNR]
plot_chi = [round(chi, 2) for chi in plot_chi]

plot_folder = '/pub/tangch3/ARIANNA/DeepLearning/Traces/'
for trace, snr, chi in zip(plot_traces, plot_SNR, plot_chi):
    pT(trace, f'trace', f'{plot_folder}/{type}_SNR_{snr}_Chi{chi}.png')

# template = []
# for i in range(4):
#     template.append(templates_RCR)

# pT(template, f'template', f'{plot_folder}/ReflectedCR_template.png')

# plt.savefig('/pub/tangch3/ARIANNA/DeepLearning/yo.png')










