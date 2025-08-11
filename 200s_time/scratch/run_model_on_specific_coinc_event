import numpy as np
import keras, sys, os

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from A0_Utilities import load_data, load_coincidence_pkl

station_id = []
amp = '200s'

data =[]
for id in station_id:
    snr, chi2016, chiRCR, traces2016, tracesRCR, unix = load_data('new_chi_above_curve', amp_type = amp, station_id=id)
    data.extend(tracesRCR)

data = np.array(data)
master_id = 578 # run on master event 578
argument = 'stations'

coinc_data = load_coincidence_pkl(master_id, argument, 13)
event = coinc_data['Traces']
print(isinstance(event, list))  
event = np.array(event)
print(event.shape)

model_path = f'/pub/tangch3/ARIANNA/DeepLearning/models/200s_time/new_chi'
model = keras.models.load_model(f'{model_path}/2025-07-21_12-00_RCR_Backlobe_model_2Layer.h5')
no = model.predict(event)
print(no)
event_id = coinc_data['event_ids']
print(f'event id is {event_id}')
prob_Backlobe = model.predict(data)
