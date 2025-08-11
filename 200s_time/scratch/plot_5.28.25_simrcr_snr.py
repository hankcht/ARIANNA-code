import os, sys, keras
import numpy as np
import matplotlib.pyplot as plt
from NuRadioReco.utilities import units

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from A0_Utilities import loadMultipleTemplates, siminfo_forplotting, load_data


templates_2016 = loadMultipleTemplates(200, date='2016')
templates_RCR = loadMultipleTemplates(200) 
noiseRMS = 22.53 * units.mV
sim, sim_Chi2016, sim_ChiRCR, sim_SNRs, sim_weights, simulation_date = siminfo_forplotting('RCR', '200s', '5.28.25', templates_2016, templates_RCR, noiseRMS)

plt.figure(figsize=(10, 6))
plt.hist(sim_SNRs, bins=50, edgecolor='black', alpha=0.7)
plt.title(f'Distribution of {len(sim_SNRs)} Simulated SNRs')
plt.xlabel('SNR Value')
plt.ylabel('Number of Events')

plt.grid(axis='y', alpha=0.75)

plot_output_dir = '/pub/tangch3/ARIANNA/DeepLearning/' # You can change this to your desired output directory
os.makedirs(plot_output_dir, exist_ok=True)
plt.savefig(os.path.join(plot_output_dir, 'sim_SNRs_histogram.png'))
plt.clf()
print(f"Histogram of sim_SNRs saved to {os.path.join(plot_output_dir, 'sim_SNRs_histogram.png')}")