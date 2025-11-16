import os
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
import pandas as pd

# --- HGQ2 imports ---
from hgq.layers import QDense, QConv1D
from hgq.config.layer import LayerConfigScope
from hgq.config.quantizer import QuantizerConfigScope
from hgq.utils.sugar.beta_scheduler import BetaScheduler
from hgq.utils.sugar import FreeEBOPs, PBar

# Project imports
from A0_Utilities import load_config
from refactor_train_and_run import load_and_prep_data_for_training

# --- Helper Functions ---
def build_fp32_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(20, kernel_size=10, activation='relu'),
        Conv1D(10, kernel_size=10, activation='relu'),
        Dropout(0.5),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def build_hgq_model(input_shape, beta0=1e-12, beta_final=1e-3, ramp_epochs=20):
    def linear_beta_fn(epoch):
        if epoch >= ramp_epochs:
            return beta_final
        return beta0 + (beta_final - beta0) * (epoch / ramp_epochs)
    
    beta_scheduler = BetaScheduler(beta_fn=linear_beta_fn)
    ebops = FreeEBOPs()

    with (
        QuantizerConfigScope(q_type='kbi', place='weight', overflow_mode='SAT_SYM', round_mode='RND', b0=12, i0=12),
        QuantizerConfigScope(q_type='kif', place='datalane', overflow_mode='SAT_SYM', round_mode='RND', i0=12, f0=6),
        LayerConfigScope(enable_ebops=True, beta0=beta0)
    ):
        model = Sequential([
            Input(shape=input_shape),
            QConv1D(20, kernel_size=10, activation='relu'),
            QConv1D(10, kernel_size=10, activation='relu'),
            Dropout(0.5),
            Flatten(),
            QDense(1, activation='sigmoid')
        ])
        model.compile(optimizer='Adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    return model, beta_scheduler, ebops

def save_plot(history_dict, metric_name, plot_dir, title=None, validation=False):
    plt.figure(figsize=(6, 4))
    key = metric_name
    val_key = f'val_{metric_name}'
    if metric_name not in history_dict and 'acc' in metric_name:
        key = 'acc'
        val_key = 'val_acc'
    if key in history_dict:
        plt.plot(history_dict[key], label=f'Train {metric_name}')
    if validation and val_key in history_dict:
        plt.plot(history_dict[val_key], label=f'Val {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.replace('_',' ').title())
    plt.title(title or f'{metric_name.title()} over Epochs')
    plt.legend()
    filename = f"{'val_' if validation else 'train_'}{metric_name}_comparison.png"
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

def compute_reduction_percent(baseline_val, new_val):
    if baseline_val is None or np.isnan(baseline_val) or baseline_val == 0:
        return float('nan')
    return 100.0 * (baseline_val - new_val) / baseline_val

# --- Main Function ---
def main():
    config = load_config(config_path="/dfs8/sbarwick_lab/ariannaproject/tangch3/ARIANNA-code/200s_time/config.yaml")
    amp = config['amp']
    timestamp = datetime.now().strftime('%m.%d.%y_%H-%M')

    # Setup directories
    RUN_DIR = os.path.join("/dfs8/sbarwick_lab/ariannaproject/tangch3/HGQ2", timestamp)
    MODEL_DIR = os.path.join(RUN_DIR, "models")
    PLOT_DIR = os.path.join(RUN_DIR, "plots")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(os.path.join(PLOT_DIR, 'loss'), exist_ok=True)
    os.makedirs(os.path.join(PLOT_DIR, 'accuracy'), exist_ok=True)
    os.makedirs(os.path.join(PLOT_DIR, 'hgq2_results'), exist_ok=True)

    print(f"Starting training at {timestamp} for {amp} amplifier.")

    # Load and prepare data
    data = load_and_prep_data_for_training(config)
    training_rcr, training_backlobe = data['training_rcr'], data['training_backlobe']

    x = np.vstack((training_rcr, training_backlobe))
    y = np.vstack((np.ones((training_rcr.shape[0], 1)), np.zeros((training_backlobe.shape[0], 1))))
    s = np.arange(x.shape[0])
    np.random.shuffle(s)
    x = x[s].transpose(0, 2, 1)
    y = y[s]
    input_shape = x.shape[1:]
    print(f"Input Shape: {input_shape}")

    # Train FP32 model
    baseline_model = build_fp32_model(input_shape)
    baseline_history = baseline_model.fit(
        x, y,
        validation_split=0.2,
        epochs=config['keras_epochs'],
        batch_size=config['keras_batch_size'],
        verbose=config['verbose_fit']
    )
    baseline_acc = baseline_history.history.get('val_accuracy', baseline_history.history.get('val_acc'))[-1]
    baseline_model_path = os.path.join(MODEL_DIR, f"{timestamp}_baseline_model.h5")
    baseline_model.save(baseline_model_path)

    # Train HGQ2 model
    hgq_model, beta_scheduler, ebops_cb = build_hgq_model(input_shape)
    nan_terminate = keras.callbacks.TerminateOnNaN()
    pbar = PBar('loss: {loss:.3f}/{val_loss:.3f} - acc: {accuracy:.3f}/{val_accuracy:.3f}')

    hgq_history = hgq_model.fit(
        x, y,
        validation_split=0.2,
        epochs=config['keras_epochs'],
        batch_size=config['keras_batch_size'],
        callbacks=[ebops_cb, pbar, nan_terminate],
        verbose=config['verbose_fit']
    )
    hgq_acc = hgq_history.history.get('val_accuracy', hgq_history.history.get('val_acc'))[-1]
    hgq_model_path = os.path.join(MODEL_DIR, f"{timestamp}_HGQ2_model.h5")
    hgq_model.save(hgq_model_path)

    # Plot Accuracy and Loss
    save_plot(baseline_history.history, 'accuracy', os.path.join(PLOT_DIR, 'accuracy'), title='Training Accuracy Comparison', validation=True)
    save_plot(baseline_history.history, 'loss', os.path.join(PLOT_DIR, 'loss'), title='Training Loss Comparison', validation=True)
    save_plot(hgq_history.history, 'accuracy', os.path.join(PLOT_DIR, 'accuracy'), title='HGQ2 Accuracy Comparison', validation=True)
    save_plot(hgq_history.history, 'loss', os.path.join(PLOT_DIR, 'loss'), title='HGQ2 Loss Comparison', validation=True)

    # EBOP plots
    hgq_ebops = hgq_history.history.get('ebops', [])
    hgq_val_acc = hgq_history.history.get('val_accuracy', hgq_history.history.get('val_acc', []))
    if hgq_ebops:
        plt.figure(figsize=(6,4))
        plt.plot(hgq_ebops, '.')
        plt.yscale('log')
        plt.xlabel('Epoch'); plt.ylabel('EBOPs')
        plt.title('HGQ2 EBOPs per Epoch')
        plt.savefig(os.path.join(PLOT_DIR, 'hgq2_results', 'ebops_per_epoch.png'))
        plt.close()

        plt.figure(figsize=(6,4))
        plt.plot(hgq_ebops, hgq_val_acc, '.')
        plt.xscale('log')
        plt.xlabel('EBOPs'); plt.ylabel('Validation Accuracy')
        plt.title('HGQ2 EBOPs vs Validation Accuracy')
        plt.savefig(os.path.join(PLOT_DIR, 'hgq2_results', 'ebops_vs_val_accuracy.png'))
        plt.close()

    # Summary Table
    baseline_ebops = float('nan')  # FP32 EBOPs not computed
    reduction_ebops = compute_reduction_percent(baseline_ebops, hgq_ebops[-1] if hgq_ebops else float('nan'))
    accuracy_delta = hgq_acc - baseline_acc

    df = pd.DataFrame({
        'Metric': ['Validation Accuracy (delta)', 'EBOPs (estimated)'],
        'Baseline': [baseline_acc, baseline_ebops],
        'HGQ2': [hgq_acc, hgq_ebops[-1] if hgq_ebops else float('nan')],
        'Reduction %': [accuracy_delta, reduction_ebops]
    })

    summary_file = os.path.join(PLOT_DIR, 'hgq2_results', 'summary_table.csv')
    df.to_csv(summary_file, index=False)
    print(df)
    print(f"Saved summary table to {summary_file}")

if __name__ == "__main__":
    main()
