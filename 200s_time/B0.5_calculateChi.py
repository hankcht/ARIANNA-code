import os
import configparser
import templateCrossCorr as txc
import numpy as np
from NuRadioReco.utilities import units
from NuRadioReco.utilities.io_utilities import read_pickle
from icecream import ic
from A0_Utilities import getMaxChi, getMaxAllChi





if __name__ == "__main__":

    # load_path = f'/pub/tangch3/ARIANNA/DeepLearning/new_chi_data/4.4.25/Station14/' 

    # stations_100s = [13, 15, 18]
    # stations_200s = [14, 17, 19, 30]
    # stations = {100: stations_100s, 200: stations_200s}

    ### -- Calculate Chi for Simulation events --- ###
    series = 200
     
    sim_RCR_path = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/{series}s/5.28.25/'
    sim_RCR_events = np.load(os.path.join(sim_RCR_path,'SimRCR_200s_NoiseTrue_forcedFalse_4363events_FilterTrue_part0.npy'))

    chi_2016 = np.zeros((len(sim_RCR_events)))
    chi_RCR = np.zeros((len(sim_RCR_events)))

    for iT, traces in enumerate(sim_RCR_events):

        chi_2016[iT] = getMaxAllChi(traces, 2*units.GHz, templates_2016, 2*units.GHz)
        chi_RCR[iT] = getMaxAllChi(traces, 2*units.GHz, templates_series, 2*units.GHz)

    print(chi_2016[0:20])
    print(chi_RCR[0:20])

    # save_folder = f'/pub/tangch3/ARIANNA/DeepLearning/simRCR_chi/3.29.25'
    # np.save(f'{save_folder}/3.29.25_simRCR_chi2016.npy', chi_2016)
    # np.save(f'{save_folder}/3.29.25_simRCR_chiRCR.npy', chi_RCR)
