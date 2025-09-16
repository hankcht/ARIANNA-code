import os
import pickle
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Dense, BatchNormalization, ReLU, GlobalAveragePooling1D 
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from NuRadioReco.utilities import units
from A0_Utilities import load_sim_rcr, load_data, pT, load_config



config = load_config()
amp = config['amp']



from refactor_train_and_run import load_and_prep_data_for_training, evaluate_model_performance, plot_network_output_histogram, save_and_plot_training_history
data = load_and_prep_data_for_training(config)
training_rcr = data['sim_rcr_all']
training_backlobe = data['data_backlobe_traces2016']



x = np.vstack((training_rcr, training_backlobe))
y = np.vstack((np.ones((training_rcr.shape[0], 1)), np.zeros((training_backlobe.shape[0], 1)))) # 1s for RCR (signal)
s = np.arange(x.shape[0])
np.random.shuffle(s)
x = x[s]
y = y[s]

# If data is already in variables:
# x: (n_events, 4, 256)
# y: (n_events,)
# Keras Conv layers expect (length, channels), so transpose last two dims:
x = x.transpose(0, 2, 1)   # now shape = (n_events, 256, 4)
# maybe reverse the transpose

model = Sequential()

# Multi-scale idea approximated with stacked Conv1D layers
model.add(Conv1D(32, kernel_size=5, padding="same", activation="relu", input_shape=(256, 4)))
model.add(Conv1D(32, kernel_size=15, padding="same", activation="relu"))
model.add(Conv1D(32, kernel_size=31, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(ReLU())

# Downstream feature extractor
model.add(Conv1D(64, kernel_size=7, padding="same", activation="relu"))

# Collapse across time
model.add(GlobalAveragePooling1D())

# Dense classification head
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# Compile
model.compile(
     optimizer='Adam',
     loss="binary_crossentropy",
     metrics=["accuracy"]
)

model.summary()

# Train
history = model.fit(
     x, y,
     validation_split=0.2,
     epochs=config['keras_epochs'],
     batch_size=config['keras_batch_size']
)

timestamp = datetime.now().strftime('%m.%d.%y_%H-%M')
print(f"Starting CNN training at {timestamp} for {amp} amplifier.")

sim_rcr_all = data['sim_rcr_all']
data_backlobe_traces_rcr_all = data['data_backlobe_tracesRCR']
prefix = config['prefix']

sim_rcr_expanded = sim_rcr_all.transpose(0, 2, 1)
data_backlobe_expanded = data_backlobe_traces_rcr_all.transpose(0, 2, 1)

model_save_path = os.path.join(config['base_model_path'], config['model_filename_template'].format(timestamp=timestamp, amp=amp, prefix=prefix))
model.save(model_save_path)
print(f'Model saved to: {model_save_path}')
save_and_plot_training_history(history, config['base_model_path'], config['base_plot_path'], timestamp, amp, config)

prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency = \
    evaluate_model_performance(model, sim_rcr_expanded, data_backlobe_expanded, config['output_cut_value'], config)

plot_network_output_histogram(prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency, config, timestamp)

indices = np.where(prob_backlobe.flatten() > config['output_cut_value'])[0]
for index in indices:
     plot_traces_save_path = os.path.join(config['base_plot_path'], 'traces', f'{timestamp}_plot_pot_rcr_{amp}_{index}.png')
     pT(data['data_backlobe_tracesRCR'][index], f'Backlobe Trace {index} (Output > {config["output_cut_value"]:.2f})', plot_traces_save_path)
     print(f"Saved trace plot for Backlobe event {index} to {plot_traces_save_path}")
     
print("Script finished successfully.")