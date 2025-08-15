import numpy as np
import os, sys

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from A0_Utilities import pT, load_config

sim_rcr_730 = np.load(f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/8.14.25/200s/all_traces_200s_RCR_part0_4473events.npy', allow_pickle=True) 
sim_bl_730 = np.load(f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedBacklobe/8.14.25/200s/all_traces_200s_part0_11239events.npy', allow_pickle=True)

sim_rcr_730_100s = np.load(f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/8.14.25/100s/all_traces_100s_RCR_part0_4200events.npy', allow_pickle=True) 
sim_bl_730_100s = np.load(f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedBacklobe/8.14.25/100s/all_traces_100s_part0_9511events.npy', allow_pickle=True)


config = load_config()
from refactor_train_and_run import load_and_prep_data_for_training
data = load_and_prep_data_for_training(config)
sim_rcr_all = data['sim_rcr_all']

# Usage
indices = [2917, 2926, 4075, 4343] 
for i in range(10,15):
    pT(sim_rcr_730_100s[i], 'test plot 8/14 sim RCR', f"/pub/tangch3/ARIANNA/DeepLearning/refactor/other/test_plot_814_sim_rcr_{i}_100s.png")
    pT(sim_bl_730_100s[i], 'test plot 8/14 sim BL', f"/pub/tangch3/ARIANNA/DeepLearning/refactor/other/test_plot_814_sim_bl_{i}_100s.png")
    # pT(sim_rcr_all[i], 'test plot 5/28 sim rcr', f"/pub/tangch3/ARIANNA/DeepLearning/refactor/other/test_plot_528_sim_rcr_{i}_noiseTrue.png")

