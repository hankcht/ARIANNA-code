import os, sys, keras
import numpy as np
import matplotlib.pyplot as plt

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from A0_Utilities import loadMultipleTemplates, siminfo_forplotting, load_data
from NuRadioReco.utilities import units
from A1_TrainAndRunCNN import Train_CNN


amp_type = '200s'
if amp_type == '200s':
    noiseRMS = 22.53 * units.mV
    station_id = [14,17,19,30]
model_path = f'/pub/tangch3/ARIANNA/DeepLearning/models/{amp_type}_time/new_chi' 
loss_accuracy_plot_path = f'/pub/tangch3/ARIANNA/DeepLearning/plots/A1_Training/Loss_Accuracy' 
network_output_plot_path = f'/pub/tangch3/ARIANNA/DeepLearning/plots/A1_Training/Network_Output'   
from datetime import datetime
current_datetime = datetime.now() # Get the current date and time
timestamp = current_datetime.strftime("%Y-%m-%d_%H-%M") # Format the datetime object as a string with seconds

'''load sim'''
sim_folder = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/{amp_type}/5.28.25/'
# sim_rcr = load_sim_rcr(sim_folder, noise_enabled=True, filter_enabled=True, amp=amp_type)

templates_2016 = loadMultipleTemplates(200, date='2016')
templates_RCR = loadMultipleTemplates(200) 
noiseRMS = 22.53 * units.mV
sim, sim_Chi2016, sim_ChiRCR, sim_SNRs, sim_weights, simulation_date = siminfo_forplotting('RCR', '200s', '5.28.25', templates_2016, templates_RCR, noiseRMS)

no_red_indices = np.where(sim_ChiRCR > 0.55)[0]
no_red_sim = sim[no_red_indices]

'''load above curve'''
data_Backlobe = []
data_Backlobe_chi2016 = []
data_Backlobe_UNIX = [] 
for id in station_id:
    snr, chi2016, chiRCR, traces2016, tracesRCR, unix = load_data('new_chi_above_curve', amp_type, station_id=id)
    data_Backlobe.extend(traces2016)
    data_Backlobe_chi2016.extend(chi2016)
    data_Backlobe_UNIX.extend(unix)

no_red_sim = np.array(no_red_sim) 
data_Backlobe = np.array(data_Backlobe)
data_Backlobe_chi2016 = np.array(data_Backlobe_chi2016)
data_Backlobe_UNIX = np.array(data_Backlobe_UNIX)


RCR_training_indices = np.random.choice(no_red_sim.shape[0], size=len(no_red_sim), replace=False)
BL_training_indices = np.random.choice(data_Backlobe.shape[0], size=len(no_red_sim), replace=False)
training_RCR = no_red_sim[RCR_training_indices, :]
training_Backlobe = data_Backlobe[BL_training_indices, :]

model = Train_CNN(training_RCR, training_Backlobe, model_path, loss_accuracy_plot_path, timestamp, amp_type)  

model.save(f'{model_path}/{timestamp}_RCR_Backlobe_model_2Layer.h5') # currently saving in h5
print('------> Training is Done!')

# timestamp = '2025-07-01_11-08'
model = keras.models.load_model(f'{model_path}/{timestamp}_RCR_Backlobe_model_2Layer.h5')

prob_RCR = model.predict(training_RCR)
prob_Backlobe = model.predict(data_Backlobe)

# Finding not weighted RCR efficiency (percentage of RCR events that would pass the cut) 
sim_RCR_output = prob_RCR
RCR_efficiency = (sim_RCR_output > 0.9).sum() / len(sim_RCR_output)
RCR_efficiency = (100*RCR_efficiency).round(decimals=2)
print(f'RCR efficiency: {RCR_efficiency}')

# Finding Backlobe efficiency (percentage of backlobe that would remain after our cut)
sim_Backlobe_output = prob_Backlobe
Backlobe_efficiency = (sim_Backlobe_output > 0.9).sum() / len(sim_Backlobe_output)
Backlobe_efficiency = (100*Backlobe_efficiency).round(decimals=4)
print(f'Backlobe efficiency: {Backlobe_efficiency}')

print(f'lengths {len(prob_RCR)} and {len(prob_Backlobe)}')


# Set up for Network Output histogram
dense_val = False
plt.figure(figsize=(8, 6))
plt.hist(prob_Backlobe, bins=20, range=(0,1), histtype='step', color='blue', linestyle='solid', label=f'Backlobe {len(prob_Backlobe)}', density=dense_val)
plt.hist(prob_RCR, bins=20, range=(0,1), histtype='step', color='red', linestyle='solid', label=f'RCR {len(prob_RCR)}', density=dense_val)

plt.xlabel('Network Output', fontsize=18)
plt.ylabel('Number of Events', fontsize=18)
plt.yscale('log')
plt.title(f'{amp_type}_time RCR-Backlobe network output')
plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
plt.yticks(fontsize=18)

hist_values, bin_edges, _ = plt.hist(prob_Backlobe, bins=20, range=(0,1), histtype='step')
plt.ylim(0, max(10 ** (np.ceil(np.log10(hist_values)))))
plt.tick_params(axis='both', which='major', labelsize=12)
plt.legend(loc='upper left', fontsize=8)
plt.text(0.375, 0.75, f'RCR efficiency: {RCR_efficiency}%', fontsize=12, transform=plt.gcf().transFigure)
plt.text(0.375, 0.7, f'Backlobe efficiency: {Backlobe_efficiency}%', fontsize=12, transform=plt.gcf().transFigure)
plt.text(0.375, 0.65, f'TrainCut: {len(training_RCR)}', fontsize=12, transform=plt.gcf().transFigure)
plt.axvline(x=0.9, color='y', label='cut')
plt.text(0.05, -0.12, 'BL', verticalalignment='center', horizontalalignment='center', fontsize=12, transform=plt.gca().transAxes, color='blue')
plt.text(0.96, -0.12, 'RCR', verticalalignment='center', horizontalalignment='center', fontsize=12, transform=plt.gca().transAxes, color='red')
plt.subplots_adjust(left=0.2, right=0.85, bottom=0.2, top=0.8)

print(f'saving to {network_output_plot_path}/{timestamp}_{amp_type}_histogram.png')
plt.savefig(f'{network_output_plot_path}/{timestamp}_{amp_type}_histogram.png')
print(f'------> {amp_type} Done!')