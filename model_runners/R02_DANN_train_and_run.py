"""
Trains a Domain-Adversarial Neural Network (DANN)
to separate Signal (Sim RCR) from Background (Data Backlobe)
while penalizing the ability to distinguish Sim vs. Data.
"""

import os
import sys
import re
import pickle
import argparse
from datetime import datetime
from glob import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling1D
from NuRadioReco.utilities import units
from pathlib import Path

# --- Local Imports from your project structure ---
sys.path.append(str(Path(__file__).resolve().parents[1] / '200s_time'))
from A0_Utilities import load_sim_rcr, load_data, pT, load_config

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_builder import (
    build_1d_model,
    build_1d_model_freq
)
from model_builder_DANN import build_dann_model, GradientReversalLayer  # Import DANN builder + custom layer
from data_channel_cycling import cycle_channels
# Import original data loading functions (modified slightly)
from R01_1D_CNN_train_and_run import (
    load_and_prep_data_for_training, 
    _compute_frequency_magnitude, 
    _apply_frequency_edge_filter,
    save_and_plot_training_history,
    plot_network_output_histogram,
    load_2016_backlobe_templates,
    load_new_coincidence_data,
    plot_check_histogram,
    plot_layer_activations
)

# --- Model Selection (Feature Extractors) ---
# We select the *base* feature extractor here. The DANN head will be added.
MODEL_BUILDERS = {
    '1d_cnn': build_1d_model,
    '1d_cnn_freq': build_1d_model_freq
}

# Register custom layers so saved models can be reloaded without manual custom_objects.
keras.utils.get_custom_objects()['GradientReversalLayer'] = GradientReversalLayer

def get_base_model_builder(model_type):
    if model_type not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(MODEL_BUILDERS.keys())}")
    
    # This function returns the *uncompiled* base model (feature extractor)
    def builder_func(input_shape, learning_rate):
        # We call the original builder but don't care about the learning_rate
        # as it will be compiled in the DANN model.
        # We also need to strip the original head.
        
        orig_model, requires_transpose = MODEL_BUILDERS[model_type](input_shape=input_shape, learning_rate=learning_rate)
        
        # Find the last layer before GlobalAveragePooling or Flatten
        layer_to_pop = None
        for layer in reversed(orig_model.layers):
            if isinstance(layer, (GlobalAveragePooling1D, keras.layers.Flatten)):
                layer_to_pop = layer
                break
        
        if layer_to_pop is None:
            raise ValueError("Could not find a pooling/flatten layer to strip the head.")

        # Create a new model that stops before the head
        base_extractor = Model(inputs=orig_model.input, 
                               outputs=layer_to_pop.input, # Get the *input* to the pooling layer
                               name=f"base_extractor_{orig_model.name}")
        
        print(f"--- Built base feature extractor: {base_extractor.name} ---")
        base_extractor.summary()
        
        return base_extractor, requires_transpose

    return builder_func


def train_dann_model(training_rcr, training_backlobe, config, learning_rate, model_type, lambda_weight):
    """
    Trains the DANN model.
    """
    # 1. Get the base model builder
    builder_func = get_base_model_builder(model_type)

    # 2. Determine data format (transpose or not)
    _, requires_transpose = MODEL_BUILDERS[model_type]()
    if not requires_transpose:
        raise ValueError("Selected model type is not supported by the DANN pipeline (expects 1D models).")

    # Determine expected input shape directly from the training arrays to mirror R01 handling
    if training_rcr.ndim != 3:
        raise ValueError(f"Unexpected training_rcr shape {training_rcr.shape}; expected (N, channels, samples).")
    sample_axis = -1  # Trace length dimension
    channel_axis = -2  # Channel dimension before transpose
    input_shape = (training_rcr.shape[sample_axis], training_rcr.shape[channel_axis])

    # 3. Prepare data
    x_rcr = training_rcr
    x_bl = training_backlobe
    
    x_train = np.vstack((x_rcr, x_bl))
    
    # --- Create TWO sets of labels ---
    # 1. Classifier labels: Signal (RCR) = 1, Background (Backlobe) = 0
    y_classifier = np.vstack((np.ones((x_rcr.shape[0], 1)), 
                              np.zeros((x_bl.shape[0], 1))))
    
    # 2. Domain labels: Sim (RCR) = 1, Data (Backlobe) = 0
    y_domain = np.vstack((np.ones((x_rcr.shape[0], 1)), 
                          np.zeros((x_bl.shape[0], 1))))

    # Prepare data shape for model
    if requires_transpose:
        x_train = x_train.transpose(0, 2, 1)
    else:
        if x_train.ndim == 3:
            x_train = np.expand_dims(x_train, axis=-1)

    # Shuffle all data consistently
    s = np.arange(x_train.shape[0])
    np.random.shuffle(s)
    x_train = x_train[s]
    y_classifier = y_classifier[s]
    y_domain = y_domain[s]
    
    # y_train must match the model's two outputs
    y_train = [y_classifier, y_domain]

    print(f"Training data shape: {x_train.shape}, requires_transpose: {requires_transpose}")
    print(f"Classifier label shape: {y_train[0].shape}, Domain label shape: {y_train[1].shape}")

    # 4. Build the full DANN model
    base_extractor, _ = builder_func(input_shape=input_shape, learning_rate=learning_rate)
    
    dann_model = build_dann_model(base_extractor, 
                                  input_shape=input_shape, 
                                  learning_rate=learning_rate, 
                                  lambda_weight=lambda_weight)
    
    dann_model.summary()

    # 5. Train the model
    callbacks_list = [keras.callbacks.EarlyStopping(
        monitor='val_classifier_output_loss', # Monitor the validation loss of the *main task*
        patience=config['early_stopping_patience']
    )]

    history = dann_model.fit(x_train, y_train,
                             validation_split=0.25,
                             epochs=config['keras_epochs'],
                             batch_size=config['keras_batch_size'],
                             verbose=config['verbose_fit'],
                             callbacks=callbacks_list)

    return dann_model, history, requires_transpose


def evaluate_model_performance_dann(model, sim_rcr_all, data_backlobe_traces_rcr_all, output_cut_value, config, model_type, requires_transpose):
    """
    Evaluates the DANN model.
    The model.predict() will return [classifier_output, domain_output].
    We only care about the classifier_output for performance.
    """
    # Prepare data based on model type
    if requires_transpose:
        sim_rcr_expanded = sim_rcr_all.transpose(0, 2, 1)
        data_backlobe_expanded = data_backlobe_traces_rcr_all.transpose(0, 2, 1)
    else:
        sim_rcr_expanded = sim_rcr_all if sim_rcr_all.ndim == 4 else sim_rcr_all[..., np.newaxis]
        data_backlobe_expanded = data_backlobe_traces_rcr_all if data_backlobe_traces_rcr_all.ndim == 4 else data_backlobe_traces_rcr_all[..., np.newaxis]

    # model.predict() returns a list of [classifier_probs, domain_probs]
    pred_rcr_list = model.predict(sim_rcr_expanded, batch_size=config['keras_batch_size'])
    pred_backlobe_list = model.predict(data_backlobe_expanded, batch_size=config['keras_batch_size'])
    
    # Get the classifier output (the first item in the list)
    prob_rcr = pred_rcr_list[0]
    prob_backlobe = pred_backlobe_list[0]
    
    # Get the domain output (the second item) to see what it learned
    domain_prob_rcr = pred_rcr_list[1]
    domain_prob_backlobe = pred_backlobe_list[1]
    
    print(f'--- Classifier Performance ---')
    rcr_efficiency = (np.sum(prob_rcr > output_cut_value) / len(prob_rcr)) * 100
    backlobe_efficiency = (np.sum(prob_backlobe > output_cut_value) / len(prob_backlobe)) * 100
    print(f'RCR efficiency (Signal): {rcr_efficiency:.2f}%')
    print(f'Backlobe efficiency (BG): {backlobe_efficiency:.4f}%')
    
    print(f'--- Domain Discriminator Performance ---')
    print(f'Mean domain output for Sim (RCR): {np.mean(domain_prob_rcr):.3f} (closer to 1 is bad)')
    print(f'Mean domain output for Data (BL): {np.mean(domain_prob_backlobe):.3f} (closer to 0 is bad)')
    # Calculate domain accuracy
    domain_acc_rcr = (np.sum(domain_prob_rcr > 0.5) / len(domain_prob_rcr)) * 100
    domain_acc_bl = (np.sum(domain_prob_backlobe < 0.5) / len(domain_prob_backlobe)) * 100
    total_domain_acc = (domain_acc_rcr * len(domain_prob_rcr) + domain_acc_bl * len(domain_prob_backlobe)) / (len(domain_prob_rcr) + len(domain_prob_backlobe))
    print(f'Domain accuracy: {total_domain_acc:.2f}% (Goal is ~50%)')


    return prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency


def main():
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train DANN model.')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate (e.g., 0.0001)')
    parser.add_argument('--model_type', type=str, required=True, choices=list(MODEL_BUILDERS.keys()),
                        help=f'Type of base model to train. Choose from: {list(MODEL_BUILDERS.keys())}')
    parser.add_argument('--lambda_weight', type=float, default=0.5,
                        help='Weight for the domain discriminator loss (lambda).')

    args = parser.parse_args()
    
    learning_rate = args.learning_rate
    model_type = args.model_type
    lambda_weight = args.lambda_weight
    
    config = load_config()
    amp = config['amp']
    prefix = config['prefix']
    
    # --- Setup Paths ---
    is_freq_model = model_type.endswith('_freq')
    config['is_freq_model'] = is_freq_model
    config['domain_label'] = 'freq' if is_freq_model else 'time'
    config['domain_suffix'] = '_freq' if is_freq_model else ''
    
    base_model_root = config['base_model_path']
    base_plot_root = config['base_plot_path']

    lr_folder = f'lr_{learning_rate:.0e}'.replace('-', '')
    lambda_folder = f'lambda_{lambda_weight:.1e}'.replace('-', '')
    model_folder = f"{model_type}_DANN" # Add DANN suffix
    timestamp = datetime.now().strftime('%m.%d.%y_%H-%M')
    
    config['base_model_path'] = os.path.join(base_model_root, model_folder, lr_folder, lambda_folder)
    config['base_plot_path'] = os.path.join(base_plot_root, f"{timestamp}", model_folder, lr_folder, lambda_folder)
    
    print(f"--- Starting DANN training at {timestamp} ---")
    print(f"Model type: {model_type}, LR: {learning_rate}, Lambda: {lambda_weight}")
    
    # Data load & prep
    data = load_and_prep_data_for_training(config)
    training_rcr = data['training_rcr']
    training_backlobe = data['training_backlobe']
    sim_rcr_all = data['sim_rcr_all']
    data_backlobe_traces_rcr_all = data['data_backlobe_tracesRCR']

    # Apply channel cycling (augmentation)
    print("Applying channel cycling augmentation...")
    training_rcr_aug = cycle_channels(training_rcr.copy(), channel_axis=1)
    training_backlobe_aug = cycle_channels(training_backlobe.copy(), channel_axis=1)
    
    # Train model
    model, history, requires_transpose = train_dann_model(
        training_rcr_aug, training_backlobe_aug, config, learning_rate, model_type, lambda_weight
    )
    print('------> DANN Training is Done!')

    # Save model
    domain_suffix = config['domain_suffix']
    lr_str = f"lr{learning_rate:.0e}".replace('-', '')
    lambda_str = f"lambda{lambda_weight:.1e}".replace('-', '')
    model_filename = f'{timestamp}_{amp}_{model_type}_DANN_model_{prefix}_{lr_str}_{lambda_str}{domain_suffix}.h5'
    model_save_path = os.path.join(config['base_model_path'], model_filename)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f'Model saved to: {model_save_path}')

    # Save training history & plots
    save_and_plot_training_history(history, config['base_model_path'], config['base_plot_path'], 
                                   timestamp, amp, config, learning_rate, model_type)

    # Evaluate & plot network output histogram
    prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency = \
        evaluate_model_performance_dann(model, sim_rcr_all, data_backlobe_traces_rcr_all, 
                                        config['output_cut_value'], config, model_type, requires_transpose)

    plot_network_output_histogram(prob_rcr, prob_backlobe, rcr_efficiency, backlobe_efficiency, 
                                  config, timestamp, learning_rate, model_type)

    # --- Run additional checks (using imported functions) ---
    print("\n--- Running additional checks on 2016 backlobes and coincidence events ---")
    
    # (This section is copied from R01 and should work if the helper functions are correct)
    template_dir = "/pub/tangch3/ARIANNA/DeepLearning/refactor/confirmed_2016_templates/"
    template_paths = sorted(glob(os.path.join(template_dir, "filtered_Event2016_Stn*.npy")))
    
    if os.path.exists(template_dir) and template_paths:
        all_2016_backlobes, dict_2016 = load_2016_backlobe_templates(template_paths, amp_type=amp)
        
        pkl_path = "/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/9.24.25_CoincidenceDatetimes_passing_cuts_with_all_params_recalcZenAzi_calcPol.pkl"
        passing_event_ids = [3047, 3432, 10195, 10231, 10273, 10284, 10444, 10449, 10466, 10471, 10554, 11197, 11220, 11230, 11236, 11243]
        special_event_id = 11230
        special_station_id = 13
        
        if os.path.exists(pkl_path):
            passing_traces, raw_traces, special_traces, _, _, _ = \
                load_new_coincidence_data(pkl_path, passing_event_ids, special_event_id, special_station_id)
            
            if len(passing_traces) > 0:
                all_2016_backlobes = np.array(all_2016_backlobes)
                passing_traces = np.array(passing_traces)
                raw_traces = np.array(raw_traces)
                special_traces = np.array(special_traces) if len(special_traces) > 0 else special_traces

                sampling_rate = float(config.get('frequency_sampling_rate', 2.0))
                if is_freq_model:
                    all_2016_backlobes = _compute_frequency_magnitude(all_2016_backlobes, sampling_rate)
                    passing_traces = _compute_frequency_magnitude(passing_traces, sampling_rate)
                    raw_traces = _compute_frequency_magnitude(raw_traces, sampling_rate)
                    if len(special_traces) > 0:
                        special_traces = _compute_frequency_magnitude(special_traces, sampling_rate)

                # Prepare data for model
                if requires_transpose:
                    all_2016_backlobes_prepped = all_2016_backlobes.transpose(0, 2, 1)
                    passing_traces_prepped = passing_traces.transpose(0, 2, 1)
                    raw_traces_prepped = raw_traces.transpose(0, 2, 1)
                    special_traces_prepped = special_traces.transpose(0, 2, 1) if len(special_traces) > 0 else special_traces
                else:
                    all_2016_backlobes_prepped = all_2016_backlobes[..., np.newaxis] if all_2016_backlobes.ndim == 3 else all_2016_backlobes
                    passing_traces_prepped = passing_traces[..., np.newaxis] if passing_traces.ndim == 3 else passing_traces
                    raw_traces_prepped = raw_traces[..., np.newaxis] if raw_traces.ndim == 3 else raw_traces
                    special_traces_prepped = special_traces[..., np.newaxis] if len(special_traces) > 0 and special_traces.ndim == 3 else special_traces

                # Get classifier output only (index [0])
                prob_2016_backlobe = model.predict(all_2016_backlobes_prepped)[0].flatten()
                prob_passing = model.predict(passing_traces_prepped)[0].flatten()
                prob_raw = model.predict(raw_traces_prepped)[0].flatten()
                prob_special = model.predict(special_traces_prepped)[0].flatten() if len(special_traces) > 0 else np.array([])
                
                plot_check_histogram(prob_2016_backlobe, prob_passing, prob_raw, prob_special,
                                     amp, timestamp, prefix, learning_rate, model_type, config)
            else:
                print("Warning: No traces loaded from coincidence data. Skipping check histogram.")
        else:
            print(f"Warning: Coincidence PKL file not found at {pkl_path}. Skipping check histogram.")
    else:
        print(f"Warning: 2016 backlobe template directory not found at {template_dir}. Skipping check histogram.")
         
    print(f"Script finished successfully for DANN model {model_type}, lr {learning_rate}, lambda {lambda_weight}")

if __name__ == "__main__":
    main()
