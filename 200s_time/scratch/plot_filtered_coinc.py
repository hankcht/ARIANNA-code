import numpy as np
import sys, os

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from A0_Utilities import pT


# plot filtered coincidence events
conic_traces = np.load('/pub/tangch3/ARIANNA/DeepLearning/refactor/other/test_bandpass_on_coinc/filtered_coinc_traces.npy')
print(f'number of traces is {len(conic_traces)}')
conic_traces = np.array(conic_traces)

for i in range(5):
    saveLoc = f'/pub/tangch3/ARIANNA/DeepLearning/refactor/other/test_bandpass_on_coinc/ryans_filtered_data_event_{i}.png'
    pT(conic_traces[i], f'filtered coincidence event, index: {i}', saveLoc)