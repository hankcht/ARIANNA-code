# HGQ_1D_CNN.py
import os
import pickle
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Keras / TF
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, ReLU, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# HGQ2 imports
# install with: pip install HGQ2
from hgq.layers import QConv1D, QDense     # quantized layer replacements
from hgq.config import QuantizerConfigScope, LayerConfigScope, QuantizerConfig
from hgq.utils.sugar import BetaScheduler   # optional: schedule beta during training

# Your project imports (unchanged)
from NuRadioReco.utilities import units
from A0_Utilities import load_sim_rcr, load_data, pT, load_config
from refactor_train_and_run import (
    load_and_prep_data_for_training,
    evaluate_model_performance,
    plot_network_output_histogram,
    save_and_plot_training_history
)

# load config & data
config = load_config()
amp = config['amp']
data = load_and_prep_data_for_training(config)
training_rcr = data['sim_rcr_all']
training_backlobe = data['data_backlobe_traces2016']

x = np.vstack((training_rcr, training_backlobe))
y = np.vstack((np.ones((training_rcr.shape[0], 1)), np.zeros((training_backlobe.shape[0], 1))))
s = np.arange(x.shape[0])
np.random.shuffle(s)
x = x[s]
y = y[s]

# transpose to (n_events, length, channels) expected by Conv1D/QConv1D
x = x.transpose(0, 2, 1)   # -> (n_events, 256, 4)

# === HGQ2 configuration scope ===
# Note: tune these settings. beta0 controls resource regularization strength.
# resource_reg (below) is a simple name I've used here to indicate a small weighting
# for EBOPs; the exact kwarg name may differ in your installed HGQ2 version — check the docs.
beta0 = 1e-5            # starting tradeoff between loss and resource (increase to push more pruning)
resource_reg = 1e-8     # small weight for EBOP/resource regularization (tune)

# Example quantizer config for datalane (activations) if you want a different quantizer type
# fr = fractional bits, ir = integer bits initial guesses (optional)
fr_init = 4
ir_init = 1
oq_conf = QuantizerConfig('kif', 'datalane', fr=fr_init, ir=ir_init)

# Optional: BetaScheduler to ramp beta during training (recommended)
beta_scheduler = BetaScheduler(initial_beta=beta0, final_beta=1e-3, ramp_epochs=10)

# Create the quantized model inside the HGQ2 config scopes
with (
    QuantizerConfigScope(place='all', default_q_type='kbi', overflow_mode='SAT_SYM'),
    QuantizerConfigScope(place='datalane', default_q_type='kif', overflow_mode='WRAP'),
    LayerConfigScope(enable_ebops=True, beta0=beta0, resource_reg=resource_reg),
):
    # Build the quantized sequential model (use QConv1D / QDense)
    model = Sequential()
    model.add(QConv1D(32, kernel_size=5, padding="same", activation='relu', input_shape=(256, 4)))
    model.add(QConv1D(32, kernel_size=15, padding="same", activation='relu'))
    model.add(QConv1D(32, kernel_size=31, padding="same", activation='relu'))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(QConv1D(64, kernel_size=7, padding="same", activation='relu'))

    model.add(GlobalAveragePooling1D())
    model.add(QDense(32, activation="relu"))
    # final head - QDense, but the output is still a float sigmoided probability.
    model.add(QDense(1, activation="sigmoid"))

# Compile
# You can attach the beta scheduler callback so HGQ's beta is changed over epochs if desired
model.compile(
     optimizer=Adam(),
     loss="binary_crossentropy",
     metrics=["accuracy"]
)

model.summary()

# Train (HGQ2 uses fake-quantization internally; training as usual)
callbacks = []
# attach BetaScheduler if it exists in your installed version
try:
    callbacks.append(beta_scheduler.to_keras_callback())
except Exception:
    # older/newer APIs may differ; consult hgq.utils.sugar docs if callback creation fails
    pass

history = model.fit(
     x, y,
     validation_split=0.2,
     epochs=config['keras_epochs'],
     batch_size=config['keras_batch_size'],
     callbacks=callbacks
)

timestamp = datetime.now().strftime('%m.%d.%y_%H-%M')
print(f"Starting HGQ2 CNN training at {timestamp} for {amp} amplifier.")

# Save original training artifacts same as your script
sim_rcr_all = data['sim_rcr_all']
data_backlobe_traces_rcr_all = data['data_backlobe_tracesRCR']
prefix = config['prefix']

sim_rcr_expanded = sim_rcr_all.transpose(0, 2, 1)
data_backlobe_expanded = data_backlobe_traces_rcr_all.transpose(0, 2, 1)

model_save_path = os.path.join(config['base_model_path'], config['model_filename_template'].format(timestamp=timestamp, amp=amp, prefix=prefix))
# When saving a model wrapped with HGQ2 quantizers, use standard keras save unless the package
# recommends a specific exporter. Confirm in docs for bit-exact exporting if you plan to deploy to FPGA.
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