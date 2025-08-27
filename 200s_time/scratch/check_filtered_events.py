import os, sys
import numpy as np
from glob import glob
from A0_Utilities import load_config, pT, load_data

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

config = load_config()
amp = config['amp']
num = config["num_above_curve_events"]
station_id = 19
Above_curve_data_SNR2016, Above_curve_data_SNRRCR, Above_curve_data_Chi2016, Above_curve_data_ChiRCR,\
           Above_curve_data_Traces2016, Above_curve_data_TracesRCR, Above_curve_data_UNIX2016, Above_curve_data_UNIXRCR = load_data(config, amp_type=amp, station_id=station_id)

save_path = "/pub/tangch3/ARIANNA/DeepLearning/refactor/other"
for i in range(4):
    pT(Above_curve_data_Traces2016[i], f'check filter index {i}', os.path.join(save_path, f"827_{num}evt_{i}.png"))

trace_type = 'Filtered_Traces'
pkl_path = '/pub/tangch3/ARIANNA/DeepLearning/refactor/coincidence_events/filtered_coinc.pkl'

from refactor_checks import load_all_coincidence_traces, load_2016_backlobe_templates
from glob import glob
coinc_dict, coinc_traces, metadata = load_all_coincidence_traces(pkl_path, trace_key=trace_type) 
coinc_traces = np.array(coinc_traces)
print(coinc_traces.shape)
for i in np.arange(15,30):
    pT(coinc_traces[i], f"test plot old coinc index {i}", f'/pub/tangch3/ARIANNA/DeepLearning/refactor/other/827_plot_coinc_{i}.png')

config = load_config()
amp = config['amp']
template_dir = "/pub/tangch3/ARIANNA/DeepLearning/refactor/confirmed_2016_templates/"
template_paths = sorted(glob(os.path.join(template_dir, "filtered_Event2016_Stn*.npy"))) # using filtered confirmed BL
all_2016_backlobes, dict_2016 = load_2016_backlobe_templates(template_paths, amp_type=amp)

for i in np.arange(5):
    pT(all_2016_backlobes[i], f"test plot 2016 BL index {i}", f'/pub/tangch3/ARIANNA/DeepLearning/refactor/other/827_plot_2016BL_{i}.png')