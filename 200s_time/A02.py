import os
import pickle
import argparse
import numpy as np
from tensorflow import keras
import keras_tuner
from keras_tuner.tuners import RandomSearch
import sherpa
from keras import optimizers
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')
from datetime import datetime

# Import necessary functions from your A01.py file
from A01 import get_config, load_and_prep_data_for_training 
from abc import ABC, abstractmethod


# --- Abstract Base Class for Tuners ---
class BaseHyperparameterOptimizer(ABC):
    def __init__(self, hypermodel_builder, x_data, y_data, project_name, output_directory):
        self.hypermodel_builder = hypermodel_builder
        self.x_data = x_data
        self.y_data = y_data
        self.project_name = project_name
        self.output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True) # Ensure output directory exists

    @abstractmethod
    def execute_search(self):
        """Executes the hyperparameter search process."""
        pass

    @abstractmethod
    def get_optimized_model(self):
        """Retrieves and returns the best model found during the search."""
        pass

    @abstractmethod
    def get_best_hyperparameter_config(self):
        """Returns the best hyperparameter configuration found."""
        pass

# --- Keras Tuner Implementation ---
class KerasRandomSearchOptimizer(BaseHyperparameterOptimizer):
    def __init__(self, hypermodel_builder, x_data, y_data, project_name, output_directory, max_trials=10):
        super().__init__(hypermodel_builder, x_data, y_data, project_name, output_directory)
        self.max_trials = max_trials
        self.hypermodel_instance = self.hypermodel_builder(self.x_data, self.y_data)
        self.tuner = RandomSearch(
            self.hypermodel_instance,
            objective='val_loss',
            max_trials=self.max_trials,
            directory=self.output_directory,
            project_name=self.project_name,
            overwrite=True
        )

    def execute_search(self):
        print(f"Starting Keras Tuner Random Search for project: {self.project_name} in {self.output_directory}")
        self.tuner.search(self.x_data, self.y_data)

    def get_optimized_model(self):
        return self.tuner.get_best_models(num_models=1)[0]

    def get_best_hyperparameter_config(self):
        best_trial = self.tuner.oracle.get_best_trials(num_trials=1)[0]
        return best_trial.hyperparameters.values

# --- Sherpa Tuner Implementation ---
class SherpaRandomSearchOptimizer(BaseHyperparameterOptimizer):
    def __init__(self, hypermodel_builder, x_data, y_data, project_name, output_directory, max_trials=100):
        super().__init__(hypermodel_builder, x_data, y_data, project_name, output_directory)
        self.max_trials = max_trials
        self.study = None
        self.best_result_sherpa = None

    def execute_search(self):
        print(f"Starting Sherpa Random Search for project: {self.project_name} in {self.output_directory}")
        parameters = [
            sherpa.Discrete('kernel_width_1', [10, 20, 30, 40]),
            sherpa.Discrete('kernel_width_2', [10, 20, 30, 40]),
            sherpa.Discrete('conv1_filters', [5, 10, 15, 20]),
            sherpa.Discrete('conv2_filters', [5, 10, 15, 20]),
            sherpa.Choice('dropout_rate', [0.3, 0.4, 0.5, 0.6, 0.7]),
            sherpa.Continuous('learning_rate', 1e-4, 1e-2, scale='log'),
            sherpa.Discrete('epochs', [50, 75, 100, 125, 150]),
            sherpa.Discrete('batch_size', [16, 32, 64])
        ]
        alg = sherpa.algorithms.RandomSearch(max_num_trials=self.max_trials)

        self.study = sherpa.Study(parameters=parameters,
                                  algorithm=alg,
                                  lower_is_better=True,
                                  disable_dashboard=False,
                                  output_dir=self.output_directory,
                                  file_name=f'{self.project_name}_sherpa_trials.json')

        for trial in self.study:
            hypermodel_instance = self.hypermodel_builder(self.x_data, self.y_data, fixed_config=trial.parameters)
            model = hypermodel_instance.build(hp=None)

            history = model.fit(
                self.x_data, self.y_data,
                validation_split=0.2,
                epochs=trial.parameters['epochs'],
                batch_size=trial.parameters['batch_size'],
                verbose=0,
                callbacks=[self.study.keras_callback(trial, objective_name='val_loss')]
            )
            self.study.finalize(trial)
        self.best_result_sherpa = self.study.get_best_result()

    def get_optimized_model(self):
        if self.best_result_sherpa is None:
            raise ValueError("Sherpa search not performed yet. Call execute_search() first.")

        best_params = self.best_result_sherpa.parameters
        hypermodel_instance = self.hypermodel_builder(self.x_data, self.y_data, fixed_config=best_params)
        model = hypermodel_instance.build(hp=None)
        return model

    def get_best_hyperparameter_config(self):
        if self.best_result_sherpa is None:
            raise ValueError("Sherpa search not performed yet. Call execute_search() first.")
        return self.best_result_sherpa.parameters


# --- Visualization Functions (renamed for clarity) ---
def plot_single_hyperparameter_distribution_history(best_result_dict, algorithm_name=''):
    """
    Plots the distribution of a single hyperparameter (kernel_width_2).
    Expects best_result_dict to contain 'kernel_width_2'.
    """
    hparam = best_result_dict.get('kernel_width_2')
    if hparam is None:
        print("Warning: 'kernel_width_2' not found in best_result_dict. Cannot plot single hyperparameter distribution.")
        return

    hparam_arr = np.array([hparam])

    save_dir = '/pub/tangch3/ARIANNA/DeepLearning/hyperparameter_optimization_logs/single_param_plots/'
    os.makedirs(save_dir, exist_ok=True)
    npy_path = os.path.join(save_dir, f'{algorithm_name}_kernel_width_2_history.npy')

    try:
        current_hparams = np.load(npy_path)
    except FileNotFoundError:
        print(f"No existing history file found at {npy_path}. Initializing new array.")
        current_hparams = np.empty(0, dtype=int)

    updated_hparams = np.concatenate((current_hparams, hparam_arr))
    np.save(npy_path, updated_hparams)
    print(f'Saved updated hyperparameter history to {npy_path}')

    from collections import Counter
    hparam_list = updated_hparams.tolist()
    if not hparam_list: # Check if list is empty after adding current hparam
        print("Not enough data to plot single hyperparameter distribution yet.")
        return
    
    count = Counter(hparam_list)
    most_common_element, most_common_count = count.most_common(1)[0]

    print(f'Most common kernel_width_2 setting: {most_common_element}')

    # Ensure bins cover the range, even if only one value exists
    if len(set(hparam_list)) == 1:
        bins = np.array([min(hparam_list) - 0.5, min(hparam_list) + 0.5]) # Single bin
    else:
        bins = np.arange(min(hparam_list), max(hparam_list) + 2)
    
    plt.figure(figsize=(8, 6))
    plt.hist(hparam_list, bins=bins, edgecolor='black')
    plt.xlabel('Kernel Width 2')
    plt.ylabel('Count')
    plt.text(most_common_element, most_common_count + 0.5,
             f'Most Common: {most_common_element}',
             ha='center', va='bottom', fontsize=10, color='red')
    
    fig_path = os.path.join(save_dir, f'{algorithm_name}_kernel_width_2_dist.png')
    plt.savefig(fig_path)
    plt.clf()
    print(f'Saved single hyperparameter distribution plot to: {fig_path}')
    print(f"Current recorded kernel_width_2 values: {hparam_list}")
    print(f"Total entries: {len(hparam_list)}")


def plot_2d_hyperparameter_heatmap_history(best_result_dict, algorithm_name=''):
    """
    Plots a 2D heatmap of hyperparameter pairs (kernel_width_1, kernel_width_2).
    Expects best_result_dict to contain 'kernel_width_1' and 'kernel_width_2'.
    """
    ws1 = best_result_dict.get('kernel_width_1')
    ws2 = best_result_dict.get('kernel_width_2')
    if ws1 is None or ws2 is None:
        print("Warning: 'kernel_width_1' or 'kernel_width_2' not found in best_result_dict. Cannot plot 2D heatmap.")
        return

    hparam_pair = (ws1, ws2)

    save_dir = '/pub/tangch3/ARIANNA/DeepLearning/hyperparameter_optimization_logs/2d_param_plots/'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{algorithm_name}_kernel_pair_history.npy')

    try:
        best_hparam_pairs = np.load(save_path, allow_pickle=True)
    except FileNotFoundError:
        print("No existing 2D history file found. Creating new array.")
        best_hparam_pairs = np.empty((0, 2), dtype=object) # Use object dtype for tuples

    # Convert to list of tuples for concatenation and Counter
    current_pairs_list = [tuple(row) for row in best_hparam_pairs.tolist()]
    current_pairs_list.append(hparam_pair)
    updated_hparam_pairs = np.array(current_pairs_list, dtype=object)

    np.save(save_path, updated_hparam_pairs)
    print(f'Saved updated 2D hyperparameter history to {save_path}')

    from collections import Counter
    pair_counts = Counter(current_pairs_list)
    most_common_pair, freq = pair_counts.most_common(1)[0]
    print(f'Most common 2D setting so far: {most_common_pair} occurred {freq} times')

    if not current_pairs_list:
        print("No 2D hyperparameter pairs recorded to plot.")
        return

    ws1_vals = [pair[0] for pair in current_pairs_list]
    ws2_vals = [pair[1] for pair in current_pairs_list]

    # Ensure bins cover the full range of observed values
    ws1_min, ws1_max = min(ws1_vals), max(ws1_vals)
    ws2_min, ws2_max = min(ws2_vals), max(ws2_vals)

    ws1_bins = np.arange(ws1_min, ws1_max + 2)
    ws2_bins = np.arange(ws2_min, ws2_max + 2)

    heatmap, xedges, yedges = np.histogram2d(ws1_vals, ws2_vals, bins=[ws1_bins, ws2_bins])

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap.T, origin='lower', aspect='auto',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               cmap='viridis')
    plt.colorbar(label='Frequency')
    plt.xlabel('Kernel Width 1 (Conv2D-1 width)')
    plt.ylabel('Kernel Width 2 (Conv2D-2 width)')
    plt.title(f'2D Hyperparameter Map - {algorithm_name}')

    plt.text(most_common_pair[0], most_common_pair[1],
             f'{most_common_pair}\nFreq: {freq}', ha='center', va='center',
             color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    fig_path = os.path.join(save_dir, f'{algorithm_name}_kernel_pair_heatmap.png')
    plt.savefig(fig_path)
    plt.clf()

    print(f'Saved 2D heatmap to: {fig_path}')
    print(f"Current recorded 2D values: {updated_hparam_pairs.tolist()}")
    print(f'Total 2D entries: {len(updated_hparam_pairs)}')


# --- Keras HyperModel Definition for Tuners ---
class CnnHyperModel(keras_tuner.HyperModel):
    def __init__(self, x_data, y_data, fixed_config=None):
        self.x_data = x_data
        self.y_data = y_data
        self.input_shape = x_data.shape[1:]
        self.fixed_config = fixed_config # Allows passing fixed params for Sherpa or single-run

    def _get_param_value(self, hp_obj, name, default_value, hp_function):
        """Helper to get parameter value from fixed_config or HyperParameter tuner."""
        if self.fixed_config and name in self.fixed_config:
            return self.fixed_config.get(name)
        # For Keras Tuner, 'hp_obj' is the hp object. For Sherpa (fixed_config), hp_obj is None.
        if hp_obj:
            return hp_function(name)
        return default_value # Fallback if no hp_obj and not in fixed_config

    def build(self, hp):
        model = keras.Sequential() # Use keras.Sequential instead of global Sequential due to import change

        model.add(Conv2D(
            filters=self._get_param_value(hp, "conv1_filters", 20, lambda n: hp.Choice(n, [5, 10, 15, 20])),
            kernel_size=(4, self._get_param_value(hp, "kernel_width_1", 10, lambda n: hp.Int(n, 10, 40, step=10))),
            activation='relu',
            input_shape=self.input_shape
        ))

        model.add(Conv2D(
            filters=self._get_param_value(hp, "conv2_filters", 10, lambda n: hp.Choice(n, [5, 10, 15, 20])),
            kernel_size=(1, self._get_param_value(hp, "kernel_width_2", 10, lambda n: hp.Int(n, 10, 40, step=10))),
            activation='relu'
        ))

        # Dropout layer needs to be created dynamically based on hp.Float
        dropout_rate = self._get_param_value(hp, "dropout_rate", 0.5, lambda n: hp.Float(n, 0.3, 0.7, step=0.1))
        model.add(Dropout(dropout_rate))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        
        learning_rate = self._get_param_value(hp, "learning_rate", 1e-3, lambda n: hp.Float(n, 1e-4, 1e-2, sampling="log"))
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        epochs = self._get_param_value(hp, "epochs", 100, lambda n: hp.Int(n, 50, 150, step=25))
        batch_size = self._get_param_value(hp, "batch_size", 32, lambda n: hp.Choice(n, [16, 32, 64]))
        
        return model.fit(
            *args,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
            verbose=0, # Set to 0 for less verbose output during hyperparam search trials
            **kwargs
        )


# --- Main Orchestration Function ---
def run_hyperparameter_optimization_pipeline(tuner_type_str, max_optimization_trials, project_name_suffix):
    """
    Orchestrates the entire hyperparameter optimization process.
    """
    config = get_config()
    current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Adjust config for data loading
    # The load_and_prep_data_for_training in A01.py expects 'station_ids' and 'noise_rms'
    # to be set based on 'amp' in the config. We set them here before passing config.
    amp = config['amp']
    if amp == '200s':
        config['noise_rms'] = config['noise_rms_200s']
        config['station_ids'] = config['station_ids_200s']
    elif amp == '100s':
        config['noise_rms'] = config['noise_rms_100s']
        config['station_ids'] = config['station_ids_100s']

    # Load and prepare data using the imported A01 function
    # Note: load_and_prep_data_for_training returns a dict, extract what's needed for HyperModel
    data_dict = load_and_prep_data_for_training(config)
    x_train = np.vstack((data_dict['training_rcr'], data_dict['training_backlobe']))
    x_train = np.expand_dims(x_train, axis=-1)
    y_train = np.vstack((np.ones((data_dict['training_rcr'].shape[0], 1)), np.zeros((data_dict['training_backlobe'].shape[0], 1))))
    # Shuffle after stacking and expanding
    s = np.arange(x_train.shape[0])
    np.random.shuffle(s)
    x_train = x_train[s]
    y_train = y_train[s]

    # Determine project name and output directory based on tuner type
    if tuner_type_str == 'keras_tuner':
        project_name = f"{config['keras_tuner_project_name_prefix']}_{project_name_suffix}"
        tuner_output_dir = os.path.join(config['hyperparam_log_base_path'], 'keras_tuner_runs')
        optimizer_instance = KerasRandomSearchOptimizer(CnnHyperModel, x_train, y_train,
                                                        project_name=project_name,
                                                        output_directory=tuner_output_dir,
                                                        max_trials=max_optimization_trials)
    elif tuner_type_str == 'sherpa':
        project_name = f"{config['sherpa_project_name_prefix']}_{project_name_suffix}"
        tuner_output_dir = os.path.join(config['hyperparam_log_base_path'], 'sherpa_runs')
        optimizer_instance = SherpaRandomSearchOptimizer(CnnHyperModel, x_train, y_train,
                                                         project_name=project_name,
                                                         output_directory=tuner_output_dir,
                                                         max_trials=max_optimization_trials)
    else:
        raise ValueError(f"Invalid tuner_type: {tuner_type_str}. Must be 'keras_tuner' or 'sherpa'.")

    print(f"\n--- Starting {tuner_type_str.upper()} Hyperparameter Optimization ---")
    optimizer_instance.execute_search() # Run the search

    best_model = optimizer_instance.get_optimized_model()
    best_hyperparameters = optimizer_instance.get_best_hyperparameter_config()

    print("\n--- Hyperparameter Optimization Results ---")
    print(f"Best Hyperparameters Found: {best_hyperparameters}")

    # Evaluate the best model on the training data's validation split
    # For a final evaluation, you'd typically train this best model on the full training set
    # and evaluate on a separate test set. Here, we re-evaluate on the validation split for confirmation.
    val_loss, val_acc = best_model.evaluate(x_train[-int(0.2 * len(x_train)):], y_train[-int(0.2 * len(y_train)):], verbose=0)
    print(f'Best Model Validation Loss (from optimization phase): {val_loss}')
    print(f'Best Model Validation Accuracy (from optimization phase): {val_acc}')

    # Save the best model
    os.makedirs(config['best_model_save_path'], exist_ok=True)
    model_save_path = os.path.join(config['best_model_save_path'], f'optimized_cnn_model_{tuner_type_str}_{current_timestamp}.h5')
    best_model.save(model_save_path)
    print(f'Optimized model saved to: {model_save_path}')

    # Plotting hyperparameter history
    plot_single_hyperparameter_distribution_history(best_hyperparameters, algorithm_name=tuner_type_str)
    plot_2d_hyperparameter_heatmap_history(best_hyperparameters, algorithm_name=tuner_type_str)

    print(f"--- {tuner_type_str.upper()} Hyperparameter Optimization Complete! ---")


# --- Main Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CNN hyperparameter optimization.')
    parser.add_argument('--tuner_type', type=str, default='keras_tuner', choices=['keras_tuner', 'sherpa'],
                        help='Specify the hyperparameter optimization algorithm: "keras_tuner" or "sherpa".')
    parser.add_argument('--max_trials', type=int, default=10,
                        help='Maximum number of hyperparameter combinations to test. Default: 10 (Keras Tuner), 100 (Sherpa recommended).')
    parser.add_argument('--project_suffix', type=str, default='run_1',
                        help='An optional suffix for the project name to help identify runs (e.g., "initial_test", "final_run").')
    
    args = parser.parse_args()

    # Adjust default max_trials for Sherpa if user didn't specify and it's the default 10
    if args.tuner_type == 'sherpa' and args.max_trials == 10:
        print("Note: For Sherpa, 100 trials is often a more typical default. Adjusting max_trials to 100.")
        args.max_trials = 100 

    run_hyperparameter_optimization_pipeline(
        tuner_type_str=args.tuner_type,
        max_optimization_trials=args.max_trials,
        project_name_suffix=args.project_suffix
    )