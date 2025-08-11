import numpy as np
import sys, os

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from A0_Utilities import load_coincidence_pkl

station_id = [14, 17, 19, 30, 13, 15, 18]
parameters = ['ChiRCR', 'Chi2016']

test = np.load(f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/Station14_SNR_Chi.npy', allow_pickle=True)
print('loaded test')

for id in station_id:
    for param in parameters:
        chi = load_coincidence_pkl(id, param) 
        print(len(chi))