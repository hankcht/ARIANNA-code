import os
import pickle
import time
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, Flatten, BatchNormalization, ReLU, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import pandas as pd

# --- HGQ2 imports ---
from hgq.layers import QDense, QConv1D
from hgq.config.layer import LayerConfigScope
from hgq.config.quantizer import QuantizerConfigScope, QuantizerConfig # scope allows consistent controll of quantization in all layers 
from hgq.utils.sugar.beta_scheduler import BetaScheduler
from hgq.utils.minmax_trace import trace_minmax
from hgq.utils.sugar import FreeEBOPs, PBar
ebops = FreeEBOPs()

# Your project imports
from A0_Utilities import load_sim_rcr, load_data, pT, load_config
from refactor_train_and_run import (
    load_and_prep_data_for_training,
    evaluate_model_performance,
    plot_network_output_histogram,
    save_and_plot_training_history
)

# --- Helper Functions ---
def build_fp32_model(input_shape):
    # this is an equivalent 1D model as the 2D model I originally gave Albert
    model = Sequential()
    model.add(Input(shape=input_shape)) # adding Input layer to avoid passing input_shape as an argument
    model.add(Conv1D(20, kernel_size=10, activation='relu'))
    model.add(Conv1D(10, kernel_size=10, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='Adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    
    return model

def build_hgq_model(input_shape, beta0=1e-12, beta_final=1e-3, ramp_epochs=20):
    """
    Build HGQ2 model inside HGQ2 config scopes.

    Notes about HGQ2 config scopes used here:
    - QuantizerConfigScope(place='all', default_q_type='kbi', overflow_mode='SAT_SYM'):
        Sets quantizer defaults for all places (weights/activations). 'kbi' is a typical
        quantizer type in HGQ2 for kernel/weight integer-like quantization; 'SAT_SYM' sets
        symmetric saturation overflow behavior.
    - QuantizerConfigScope(place='datalane', default_q_type='kif', overflow_mode='WRAP'):
        Sets quantizer defaults for data lanes; 'WRAP' overflow simulates modular arithmetic.
    - LayerConfigScope(enable_ebops=True, beta0=beta0, resource_reg=1e-8):
        Enables EBOP tracing (EBOPs are measured when performing trace_minmax),
        sets initial beta for EBOP-related regularization, and a tiny resource reg to
        influence quantizer/resource optimization.

    These are HGQ2-specific constructs; adjust parameters to taste for your hardware/targets.
    """
    # Define BetaScheduler (linear ramp)
    def linear_beta_fn(epoch):
        if epoch >= ramp_epochs:
            return beta_final
        return beta0 + (beta_final - beta0) * (epoch / ramp_epochs)
    beta_scheduler = BetaScheduler(beta_fn=linear_beta_fn)

    # Create the model inside the HGQ2 configuration scopes so quantizers/EBOPs are configured.
    with (
        QuantizerConfigScope(
            q_type='kbi', place='weight',
            overflow_mode='SAT_SYM', round_mode='RND',
            b0=12, i0=12
        ),
        QuantizerConfigScope(
            q_type='kif', place='datalane',
            overflow_mode='SAT_SYM', round_mode='RND',
            i0=12, f0=6            
        ),
        LayerConfigScope(enable_ebops=True, beta0=beta0)
    ):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(QConv1D(20, kernel_size=10, activation='relu'))
        model.add(QConv1D(10, kernel_size=10, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(QDense(1, activation='sigmoid'))

        model.compile(optimizer='Adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    return model, beta_scheduler

# --- Main Script ---
def main():
    config = load_config(config_path="/dfs8/sbarwick_lab/ariannaproject/tangch3/ARIANNA-code/200s_time/config.yaml")
    amp = config['amp']
    prefix = config.get('prefix', '')
    timestamp = datetime.now().strftime('%m.%d.%y_%H-%M')
    print(f"Starting training at {timestamp} for {amp} amplifier.")

    # Load data
    data = load_and_prep_data_for_training(config)
    training_rcr = data['training_rcr']
    training_backlobe = data['training_backlobe']

    x = np.vstack((training_rcr, training_backlobe))
    y = np.vstack((np.ones((training_rcr.shape[0], 1)), np.zeros((training_backlobe.shape[0], 1))))
    s = np.arange(x.shape[0])
    np.random.shuffle(s)
    # Expecting shape after stacking: (n_events, channels, length) maybe — user original code transposed to (n_events, length, channels)
    x = x[s].transpose(0, 2, 1)  # ensure (n_events, length, channels)
    y = y[s]

    input_shape = x.shape[1:]  # (length, channels)
    print(f"Input Shape: {input_shape}. For 1D CNN, should be (256, 4)")

    # --- Train Baseline FP32 Model ---
    baseline_model = build_fp32_model(input_shape)
    baseline_history = baseline_model.fit(x, y,
                                          validation_split=0.2,
                                          epochs=config['keras_epochs'],
                                          batch_size=config['keras_batch_size'],
                                          verbose=config['verbose_fit'])
    baseline_acc = baseline_history.history.get('val_accuracy', baseline_history.history.get('val_acc'))[-1]
    baseline_model_path = os.path.join(config['base_model_path'], f"{timestamp}_baseline_model.h5")
    baseline_model.save(baseline_model_path)

    # --- Train HGQ2 Model ---
    hgq_model, beta_scheduler = build_hgq_model(input_shape)

    nan_terminate = keras.callbacks.TerminateOnNaN()
    pbar = PBar('loss: {loss:.3f}/{val_loss:.3f} - acc: {accuracy:.3f}/{val_accuracy:.3f}')

    hgq_history = hgq_model.fit(x, y,
                                validation_split=0.2,
                                epochs=config['keras_epochs'],
                                batch_size=config['keras_batch_size'],
                                callbacks=[ebops, pbar, nan_terminate], # It is recommended to use the FreeEBOPs callback to monitor EBOPs during training
                                verbose=config['verbose_fit'])
    hgq_acc = hgq_history.history.get('val_accuracy', hgq_history.history.get('val_acc'))[-1]
    hgq_model_path = os.path.join(config['base_model_path'], f"{timestamp}_HGQ2_model.h5")
    hgq_model.save(hgq_model_path)

    baseline_model.summary()
    hgq_model.summary()

    hgq_train_acc = hgq_history.history.get('accuracy', hgq_history.history.get('acc'))
    hgq_val_acc = hgq_history.history.get('val_accuracy', hgq_history.history.get('val_acc'))
    hgq_ebops = hgq_history.history.get('ebops')

    # FP32 baseline EBOPs: if you have a way to measure FP32 EBOPs, replace this.
    baseline_ebops = float('nan')  # FP32 baseline EBOPs not computed here


    # --- Plots: Accuracy / Loss ---
    plot_dir = os.path.join(config['base_plot_path'], 'HGQ2', timestamp)
    os.makedirs(os.path.join(plot_dir, 'loss'), exist_ok=True)
    os.makedirs(os.path.join(plot_dir, 'accuracy'), exist_ok=True)
    os.makedirs(os.path.join(plot_dir, 'hgq2_results'), exist_ok=True)

    # Train Accuracy
    plt.figure(figsize=(6,4))
    plt.plot(baseline_history.history.get('accuracy', baseline_history.history.get('acc')),
             label='Baseline train_acc')
    plt.plot(hgq_history.history.get('accuracy', hgq_history.history.get('acc')),
             label='HGQ2 train_acc')
    plt.xlabel('Epoch'); plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy Comparison')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'accuracy', 'train_accuracy_comparison.png'))
    plt.close()

    # train Loss
    plt.figure(figsize=(6,4))
    plt.plot(baseline_history.history['loss'], label='Baseline train_loss')
    plt.plot(hgq_history.history['loss'], label='HGQ2 train_loss')
    plt.xlabel('Epoch'); plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'loss', 'train_loss_comparison.png'))
    plt.close()

    # Val Accuracy
    plt.figure(figsize=(6,4))
    plt.plot(baseline_history.history.get('val_accuracy', baseline_history.history.get('val_acc')), label='Baseline val_acc')
    plt.plot(hgq_history.history.get('val_accuracy', hgq_history.history.get('val_acc')), label='HGQ2 val_acc')
    plt.xlabel('Epoch'); plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'accuracy', 'val_accuracy_comparison.png'))
    plt.close()

    # Val Loss
    plt.figure(figsize=(6,4))
    plt.plot(baseline_history.history.get('val_loss'), label='Baseline val_loss')
    plt.plot(hgq_history.history.get('val_loss'), label='HGQ2 val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'loss', 'val_loss_comparison.png'))
    plt.close()

    # EBOPs vs Epoch
    plt.figure(figsize=(6,4))
    plt.plot(hgq_ebops, '.')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('EBOPs')
    plt.title('HGQ2 EBOPs per Epoch')
    plt.savefig(os.path.join(plot_dir, 'hgq2_results', 'ebops_per_epoch.png'))
    plt.close()

    # 
    plt.figure(figsize=(6,4))
    plt.plot(hgq_ebops, hgq_val_acc, '.')
    plt.xscale('log')
    plt.xlabel('EBOPs')
    plt.ylabel('Validation Accuracy')
    plt.title('HGQ2 EBOPs vs Validation Accuracy')
    plt.savefig(os.path.join(plot_dir, 'hgq2_results', 'ebops_vs_val_accuracy.png'))
    plt.close()

    # --- Summary Table and Reduction Calculation ---
    # Explanation of metrics (comments):
    #  - Accuracy: validation accuracy measured after training. Higher is better.
    #  - EBOPs: Energy-Bit-Operations (estimated) per inference — an HGQ2-traced proxy for power/compute cost.
    #           Smaller EBOPs indicate fewer bit-ops, usually implying lower power/energy consumption on quantized hardware.
    #  - Latency (s): measured average per-inference time (single example) in seconds.
    #  - Model Size (MB): file size of the saved model on disk in megabytes.
    #
    # Reduction % logic:
    #  For a metric M, Reduction % = 100 * (Baseline_M - HGQ2_M) / Baseline_M
    #  - It's the percent decrease from baseline to HGQ2.
    #  - If Baseline_M is zero or NaN (for example EBOPs not computed for FP32 baseline), the reduction is undefined -> set to NaN.
    #  - Positive Reduction % means HGQ2 reduced the metric compared to baseline (good for EBOPs/Latency/Size).
    #  - For Accuracy, we typically report absolute values or relative change depending on preference; here we compute accuracy delta (HGQ2 - Baseline).
    #
    def compute_reduction_percent(baseline_val, new_val):
        try:
            if baseline_val is None or np.isnan(baseline_val):
                return float('nan')
            if baseline_val == 0:
                return float('nan')  # undefined
            return 100.0 * (baseline_val - new_val) / baseline_val
        except Exception:
            return float('nan')
    reduction_ebops = compute_reduction_percent(baseline_ebops, hgq_ebops)

    # For accuracy, show absolute difference (HGQ2 - Baseline). Positive means HGQ2 higher accuracy.
    accuracy_delta = hgq_acc - baseline_acc

    df = pd.DataFrame({
        'Metric': ['Validation Accuracy (delta)', 'EBOPs (estimated)', 'Latency (s)', 'Model Size (MB)'],
        'Baseline': [baseline_acc, baseline_ebops],
        'HGQ2': [hgq_acc, hgq_ebops[-1]],
        'Reduction %': [accuracy_delta, reduction_ebops]
    })

    summary_file = os.path.join(plot_dir, 'hgq2_results', 'summary_table.csv')
    df.to_csv(summary_file, index=False)
    pd.set_option('display.max_columns', None)
    print(df)
    print(f"Saved summary table to {summary_file}")

if __name__ == "__main__":
    main()
