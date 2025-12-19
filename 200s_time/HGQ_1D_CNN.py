import os
import pickle
import time
import argparse
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tensorflow import keras
print(keras.__version__)
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, Flatten, BatchNormalization, ReLU, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import pandas as pd
import random
import tensorflow as tf

seed = 67

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# --- HGQ2 imports ---
from hgq.layers import QDense, QConv1D
from hgq.config.layer import LayerConfigScope
from hgq.config.quantizer import QuantizerConfigScope, QuantizerConfig # scope allows consistent controll of quantization in all layers 
from hgq.utils.sugar.beta_scheduler import BetaScheduler
from hgq.utils.minmax_trace import trace_minmax
from hgq.utils.sugar import FreeEBOPs, PBar
from hgq.regularizers import MonoL1
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
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='Adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    
    return model

def build_hgq_model(input_shape, ramp_epochs, beta0=1e-5, beta_final=1e-4):

    # Define BetaScheduler (linear ramp)
    def linear_beta_fn(epoch):
        if epoch >= ramp_epochs:
            return beta_final
        return beta0 + (beta_final - beta0) * (epoch / ramp_epochs)
    beta_scheduler = BetaScheduler(beta_fn=linear_beta_fn)

    # with QuantizerConfigScope(q_type='kbi', place='weight', overflow_mode='SAT_SYM', round_mode='RND_CONV'):
    #     # For activations, use different configa
    #     with QuantizerConfigScope(q_type='kif', place='datalane', overflow_mode='SAT_SYM', round_mode='RND_CONV'):
    #         with LayerConfigScope(enable_ebops=True, beta0=beta0):
    #             # Create model with quantized layers
    #             model = keras.Sequential([
    #                 keras.layers.Input(shape=input_shape),
    #                 QConv1D(20, kernel_size=10, activation='relu'),
    #                 keras.layers.Conv1D(10, kernel_size=10, activation='relu'),
    #                 # keras.layers.Dropout(0.5),
    #                 keras.layers.Flatten(),
    #                 keras.layers.Dense(1, activation='sigmoid')
    #             ])


    # Define Config Scopes 
    scope0 = QuantizerConfigScope(place='all', b0=3, i0=0, default_q_type='kbi', overflow_mode='SAT_SYM') #  k0=1,
    scope1 = QuantizerConfigScope(place='datalane', f0=3, i0=3, default_q_type='kif', overflow_mode='SAT_SYM') # 
    with scope0, scope1: 
        iq_conf = QuantizerConfig(place='datalane') # input quantizer
        oq_conf = QuantizerConfig(place='datalane', fr=MonoL1(1e-3)) # output quantizer   
        model = keras.Sequential([
                    keras.layers.Input(shape=input_shape),
                    QConv1D(20, kernel_size=10, beta0=beta0, iq_conf=iq_conf, activation='relu', name='conv1d_0'),
                    # keras.layers.Conv1D(20, kernel_size=10, activation='relu', name='conv1d_0'),
                    QConv1D(10, kernel_size=10, beta0=beta0, iq_conf=iq_conf, activation='relu', name='conv1d_1'),
                    # keras.layers.Dropout(0.5),
                    keras.layers.Flatten(),
                    QDense(1, beta0=beta0, activation='sigmoid', name='dense_0', oq_conf=oq_conf)
                ])

    # Compile model as usual
    model.compile(optimizer=keras.optimizers.Adam(), #learning_rate=5e-3
                loss='binary_crossentropy',
                metrics=['accuracy'])
        
    return model, beta_scheduler

# --- Main Script ---
def main():
    parser = argparse.ArgumentParser(description="Train baseline and HGQ2 models.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (overrides config file)")
    args = parser.parse_args()

    config = load_config(config_path="/pub/tangch3/ARIANNA/DeepLearning/code/200s_time/config.yaml")

    epochs = args.epochs if args.epochs is not None else config['keras_epochs']

    amp = config['amp']
    timestamp = datetime.now().strftime('%m.%d.%y_%H-%M')

    # setup saving paths
    hgq2_root = config['base_plot_path']

    # Timestamp folder inside HGQ2
    run_dir = os.path.join(hgq2_root, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # Subfolders
    model_dir = os.path.join(run_dir, "models")
    plot_dir = os.path.join(run_dir, "plots")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Inside plots keep your existing structure
    os.makedirs(os.path.join(plot_dir, 'loss'), exist_ok=True)
    os.makedirs(os.path.join(plot_dir, 'accuracy'), exist_ok=True)
    os.makedirs(os.path.join(plot_dir, 'hgq2_results'), exist_ok=True)


    print(f"Starting training at {timestamp} for {amp} amplifier.")


    # Load data
    data = load_and_prep_data_for_training(config)
    training_rcr = data['training_rcr']
    training_backlobe = data['training_backlobe']

    x = np.vstack((training_rcr, training_backlobe))
    y = np.vstack((np.ones((training_rcr.shape[0], 1)), np.zeros((training_backlobe.shape[0], 1))))
    s = np.arange(x.shape[0])
    np.random.seed(42)
    np.random.shuffle(s)
    x = x[s].transpose(0, 2, 1)  # ensure (n_events, length, channels)
    y = y[s]

    x = x.astype('float32') 
    y = y.astype('float32') 

    input_shape = x.shape[1:]  # (length, channels)
    print(f"Input Shape: {input_shape}. For 1D CNN, should be (256, 4)")

    # --- Train Baseline FP32 Model ---
    baseline_model = build_fp32_model(input_shape)
    print(f'------- training Baseline model -------')
    baseline_history = baseline_model.fit(x, y,
                                          validation_split=0.2,
                                          epochs=epochs,
                                          batch_size=config['keras_batch_size'],
                                          verbose=config['verbose_fit'])
    
    baseline_model_path = os.path.join(model_dir, f"{timestamp}_baseline_model.keras")
    baseline_model.save(baseline_model_path)

    # --- Train HGQ2 Model ---
    hgq_model, beta_scheduler = build_hgq_model(input_shape, ramp_epochs=epochs)

    nan_terminate = keras.callbacks.TerminateOnNaN()
    pbar = PBar('loss: {loss:.3f}/{val_loss:.3f} - acc: {accuracy:.3f}/{val_accuracy:.3f}')
    print(f'------- training HGQ2 model -------')
    hgq_history = hgq_model.fit(x, y,
                                validation_split=0.2,
                                epochs=epochs,
                                batch_size=config['keras_batch_size'],
                                callbacks=[ebops, pbar, nan_terminate], # It is recommended to use the FreeEBOPs callback to monitor EBOPs during training
                                verbose=config['verbose_fit'])
    
    hgq_model_path = os.path.join(model_dir, f"{timestamp}_HGQ2_model.keras")
    hgq_model.save(hgq_model_path)

    baseline_model.summary()
    hgq_model.summary()

    # --- Fit results: accuracy, loss, ebop ---
    baseline_train_acc = baseline_history.history.get('accuracy', baseline_history.history.get('acc'))
    hgq_train_acc = hgq_history.history.get('accuracy', hgq_history.history.get('acc'))
    
    baseline_train_loss = baseline_history.history['loss']
    hgq_train_loss = hgq_history.history['loss']

    baseline_val_acc = baseline_history.history.get('val_accuracy', baseline_history.history.get('val_acc'))
    hgq_val_acc = hgq_history.history.get('val_accuracy', hgq_history.history.get('val_acc'))

    baseline_val_loss = baseline_history.history.get('val_loss')
    hgq_val_loss = hgq_history.history.get('val_loss')

    baseline_ebops = float('nan')  # FP32 baseline EBOPs not computed here
    hgq_ebops = hgq_history.history.get('ebops')

    # Train Accuracy
    plt.figure(figsize=(8,6)    )
    plt.plot(baseline_train_acc, label='Baseline train_acc')
    plt.plot(hgq_train_acc, label='HGQ2 train_acc')
    plt.xlabel('Epoch'); plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy Comparison')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'accuracy', 'train_accuracy_comparison.png'))
    plt.close()

    # train Loss
    plt.figure(figsize=(8,6))
    plt.plot(baseline_train_loss, label='Baseline train_loss')
    plt.plot(hgq_train_loss, label='HGQ2 train_loss')
    plt.xlabel('Epoch'); plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'loss', 'train_loss_comparison.png'))
    plt.close()

    # Val Accuracy
    plt.figure(figsize=(8,6))
    plt.plot(baseline_val_acc, label='Baseline val_acc')
    plt.plot(hgq_val_acc, label='HGQ2 val_acc')
    plt.xlabel('Epoch'); plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'accuracy', 'val_accuracy_comparison.png'))
    plt.close()

    # Val Loss
    plt.figure(figsize=(8,6))
    plt.plot(baseline_val_loss, label='Baseline val_loss')
    plt.plot(hgq_val_loss, label='HGQ2 val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'loss', 'val_loss_comparison.png'))
    plt.close()

    # EBOPs vs Epoch
    plt.figure(figsize=(8,6))
    plt.plot(hgq_ebops, '.')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('EBOPs')
    plt.title('HGQ2 EBOPs per Epoch')
    plt.savefig(os.path.join(plot_dir, 'hgq2_results', 'ebops_per_epoch.png'))
    plt.close()

    # 
    plt.figure(figsize=(8,6))
    plt.plot(hgq_ebops, hgq_val_acc, '.')
    plt.xscale('log')
    plt.xlabel('EBOPs')
    plt.ylabel('Validation Accuracy')
    plt.title('HGQ2 EBOPs vs Validation Accuracy')
    plt.savefig(os.path.join(plot_dir, 'hgq2_results', 'ebops_vs_val_accuracy.png'))
    plt.close()


    sim_rcr_all = data['sim_rcr_all']
    data_backlobe_traces_rcr_all = data['data_backlobe_tracesRCR']
    sim_rcr_expanded = sim_rcr_all.transpose(0, 2, 1)
    data_backlobe_expanded = data_backlobe_traces_rcr_all.transpose(0, 2, 1)

    # Evaluate & plot network output histogram ON RCR-like TRACES!
    baseline_prob_rcr, baseline_prob_backlobe, baseline_rcr_efficiency, baseline_bl_efficiency = \
        evaluate_model_performance(baseline_model, sim_rcr_expanded, data_backlobe_expanded, config['output_cut_value'], config)
    
    hgq_prob_rcr, hgq_prob_backlobe, hgq_rcr_efficiency, hgq_bl_efficiency = \
        evaluate_model_performance(hgq_model, sim_rcr_expanded, data_backlobe_expanded, config['output_cut_value'], config)

    plot_network_output_histogram(baseline_prob_rcr, baseline_prob_backlobe, baseline_rcr_efficiency, baseline_bl_efficiency,
                                    config, timestamp, model_tag="baseline")

    plot_network_output_histogram(hgq_prob_rcr, hgq_prob_backlobe, hgq_rcr_efficiency, hgq_bl_efficiency, 
                                    config, timestamp, model_tag="hgq2")

    # For accuracy, show absolute difference (HGQ2 - Baseline). Positive means HGQ2 higher accuracy.

    train_accuracy_delta = baseline_train_acc[-1] - hgq_train_acc[-1]
    val_accuracy_delta = baseline_val_acc[-1] - hgq_val_acc[-1]

    train_loss_delta = baseline_train_loss[-1] - hgq_train_loss[-1]
    val_loss_delta = baseline_val_loss[-1] - hgq_val_loss[-1]

    df = pd.DataFrame({
        'Metric': [
            'Training Accuracy',
            'Validation Accuracy',
            'Training Loss',
            'Validation Loss',
            'EBOPs (estimated)'
        ],
        'Baseline': [
            baseline_train_acc[-1],
            baseline_val_acc[-1],
            baseline_train_loss[-1],
            baseline_val_loss[-1],
            baseline_ebops
        ],
        'HGQ2': [
            hgq_train_acc[-1],
            hgq_val_acc[-1],
            hgq_train_loss[-1],
            hgq_val_loss[-1],
            hgq_ebops[-1]
        ],
        'Difference': [
            train_accuracy_delta,
            val_accuracy_delta,
            train_loss_delta,
            val_loss_delta,
            float('nan')
        ]
    })

    df[['Baseline', 'HGQ2', 'Difference']] = df[['Baseline', 'HGQ2', 'Difference']].round(6)

    for col in ['Baseline', 'HGQ2', 'Difference']:
        df[col] = df[col].apply(lambda x: f"{x:.6f}" if pd.notnull(x) else x)

    summary_file = os.path.join(plot_dir, 'hgq2_results', 'summary_table.csv')
    df.to_csv(summary_file, index=False)
    pd.set_option('display.max_columns', None)
    print(df)
    print(f"Saved summary table to {summary_file}")

    

if __name__ == "__main__":
    main()

   
    
    
