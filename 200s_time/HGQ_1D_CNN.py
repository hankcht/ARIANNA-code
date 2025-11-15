import os
import pickle
import time
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, BatchNormalization, ReLU, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import pandas as pd

# --- HGQ2 imports (robust) ---
from hgq.layers import QConv1D, QDense
from hgq.config import QuantizerConfigScope, LayerConfigScope, QuantizerConfig
from hgq.utils.sugar.beta_scheduler import BetaScheduler
from hgq.utils.minmax_trace import trace_minmax

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
    model = Sequential()
    model.add(Conv1D(32, kernel_size=5, padding="same", activation='relu', input_shape=input_shape))
    model.add(Conv1D(32, kernel_size=15, padding="same", activation='relu'))
    model.add(Conv1D(32, kernel_size=31, padding="same", activation='relu'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv1D(64, kernel_size=7, padding="same", activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_hgq_model(input_shape, beta0=1e-5, beta_final=1e-3, ramp_epochs=10):
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
        QuantizerConfigScope(place='all', default_q_type='kbi', overflow_mode='SAT_SYM'),
        QuantizerConfigScope(place='datalane', default_q_type='kif', overflow_mode='WRAP'),
        LayerConfigScope(enable_ebops=True, beta0=beta0, resource_reg=1e-8)
    ):
        model = Sequential()
        model.add(QConv1D(32, kernel_size=5, padding="same", activation='relu', input_shape=input_shape))
        model.add(QConv1D(32, kernel_size=15, padding="same", activation='relu'))
        model.add(QConv1D(32, kernel_size=31, padding="same", activation='relu'))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(QConv1D(64, kernel_size=7, padding="same", activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(QDense(32, activation="relu"))
        model.add(QDense(1, activation="sigmoid"))
        model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    return model, beta_scheduler

def measure_latency(model, x, n_runs=50):
    """
    Measures per-inference latency (in seconds) by running model.predict on a single example
    n_runs times and averaging. Warm-up call first to avoid cold-start effects.
    """
    model.predict(x[:1])  # warm-up
    start = time.time()
    for _ in range(n_runs):
        model.predict(x[:1])
    end = time.time()
    return (end - start) / n_runs

# --- Main Script ---
def main():
    config = load_config()
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
    callbacks = [beta_scheduler]
    hgq_history = hgq_model.fit(x, y,
                                validation_split=0.2,
                                epochs=config['keras_epochs'],
                                batch_size=config['keras_batch_size'],
                                callbacks=callbacks,
                                verbose=config['verbose_fit'])
    hgq_acc = hgq_history.history.get('val_accuracy', hgq_history.history.get('val_acc'))[-1]
    hgq_model_path = os.path.join(config['base_model_path'], f"{timestamp}_HGQ2_model.h5")
    hgq_model.save(hgq_model_path)

    # --- EBOPs (Energy / bit operations) ---
    # trace_minmax is an HGQ2 utility that runs through the model and reports min/max ranges and,
    # when EBOP tracing is enabled, estimated EBOPs or a related power metric per inference.
    # We request return_results=True to get structured output. Different hgq2 versions may return
    # different structures (scalar, dict, etc.) so we handle them robustly below.
    try:
        # Request results (safer to ask for return_results and parse)
        ebop_results = trace_minmax(hgq_model, x, batch_size=1024, verbose=1, return_results=True)
        hgq_ebops = None

        # Possible return types: scalar (float), dict containing 'ebops' or 'power' keys, or custom object.
        if isinstance(ebop_results, (float, int, np.floating, np.integer)):
            hgq_ebops = float(ebop_results)
        elif isinstance(ebop_results, dict):
            # check common keys
            for k in ['ebops', 'EBOPs', 'power', 'estimated_ebops', 'estimated_power']:
                if k in ebop_results:
                    hgq_ebops = float(ebop_results[k])
                    break
            # If there is a nested per-layer summary, we can sum if present
            if hgq_ebops is None:
                # e.g. {'layer_summaries': [{'ebops': ...}, ...]}
                if 'layer_summaries' in ebop_results and isinstance(ebop_results['layer_summaries'], (list, tuple)):
                    try:
                        hgq_ebops = float(sum(ls.get('ebops', 0.0) for ls in ebop_results['layer_summaries']))
                    except Exception:
                        hgq_ebops = float('nan')
        else:
            # If it's an object with attribute 'ebops' or 'power', try to extract.
            hgq_ebops = float(getattr(ebop_results, 'ebops', getattr(ebop_results, 'power', float('nan'))))
    except Exception as e:
        # If trace_minmax fails or does not produce EBOPs for your version, record NaN and warn.
        print("Warning: trace_minmax failed or returned unexpected format. EBOPs set to NaN. Error:", e)
        hgq_ebops = float('nan')

    # FP32 baseline EBOPs: if you have a way to measure FP32 EBOPs, replace this.
    baseline_ebops = float('nan')  # FP32 baseline EBOPs not computed here

    # --- Latency (per single inference) ---
    baseline_latency = measure_latency(baseline_model, x)
    hgq_latency = measure_latency(hgq_model, x)

    # --- Model Sizes (MB) ---
    baseline_size = os.path.getsize(baseline_model_path) / 1024**2
    hgq_size = os.path.getsize(hgq_model_path) / 1024**2

    # --- Plots: Accuracy / Loss ---
    plot_dir = os.path.join(config['base_plot_path'], 'HGQ2', timestamp)
    os.makedirs(os.path.join(plot_dir, 'loss'), exist_ok=True)
    os.makedirs(os.path.join(plot_dir, 'accuracy'), exist_ok=True)
    os.makedirs(os.path.join(plot_dir, 'hgq2_results'), exist_ok=True)

    # Accuracy
    plt.figure(figsize=(6,4))
    plt.plot(baseline_history.history.get('val_accuracy', baseline_history.history.get('val_acc')), label='Baseline val_acc')
    plt.plot(hgq_history.history.get('val_accuracy', hgq_history.history.get('val_acc')), label='HGQ2 val_acc')
    plt.xlabel('Epoch'); plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'accuracy', 'val_accuracy_comparison.png'))
    plt.close()

    # Loss
    plt.figure(figsize=(6,4))
    plt.plot(baseline_history.history.get('val_loss'), label='Baseline val_loss')
    plt.plot(hgq_history.history.get('val_loss'), label='HGQ2 val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'loss', 'val_loss_comparison.png'))
    plt.close()

    # EBOPs Bar
    plt.figure(figsize=(4,4))
    # If baseline_ebops is NaN, show 0 for baseline to keep plotting simple (but CSV will have NaN).
    baseline_ebops_for_plot = 0 if np.isnan(baseline_ebops) else baseline_ebops
    hgq_ebops_for_plot = 0 if np.isnan(hgq_ebops) else hgq_ebops
    plt.bar(['Baseline', 'HGQ2'], [baseline_ebops_for_plot, hgq_ebops_for_plot])
    plt.ylabel('EBOPs (estimated)')
    plt.title('Power/EBOPs Comparison')
    plt.savefig(os.path.join(plot_dir, 'hgq2_results', 'ebops_comparison.png'))
    plt.close()

    # Latency Bar
    plt.figure(figsize=(4,4))
    plt.bar(['Baseline', 'HGQ2'], [baseline_latency, hgq_latency])
    plt.ylabel('Latency (s)')
    plt.title('Latency Comparison')
    plt.savefig(os.path.join(plot_dir, 'hgq2_results', 'latency_comparison.png'))
    plt.close()

    # Model Size Bar
    plt.figure(figsize=(4,4))
    plt.bar(['Baseline', 'HGQ2'], [baseline_size, hgq_size])
    plt.ylabel('Model Size (MB)')
    plt.title('Model Footprint Comparison')
    plt.savefig(os.path.join(plot_dir, 'hgq2_results', 'model_size_comparison.png'))
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

    reduction_latency = compute_reduction_percent(baseline_latency, hgq_latency)
    reduction_size = compute_reduction_percent(baseline_size, hgq_size)
    reduction_ebops = compute_reduction_percent(baseline_ebops, hgq_ebops)

    # For accuracy, show absolute difference (HGQ2 - Baseline). Positive means HGQ2 higher accuracy.
    accuracy_delta = hgq_acc - baseline_acc

    df = pd.DataFrame({
        'Metric': ['Validation Accuracy (delta)', 'EBOPs (estimated)', 'Latency (s)', 'Model Size (MB)'],
        'Baseline': [baseline_acc, baseline_ebops, baseline_latency, baseline_size],
        'HGQ2': [hgq_acc, hgq_ebops, hgq_latency, hgq_size],
        'Reduction %': [accuracy_delta, reduction_ebops, reduction_latency, reduction_size]
    })

    summary_file = os.path.join(plot_dir, 'hgq2_results', 'summary_table.csv')
    df.to_csv(summary_file, index=False)
    print(df)
    print(f"Saved summary table to {summary_file}")

if __name__ == "__main__":
    main()
