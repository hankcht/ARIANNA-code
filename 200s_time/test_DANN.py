import os
import numpy as np
import tensorflow as tf
import pickle
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from NuRadioReco.utilities import units

from A0_Utilities import load_sim_rcr, load_data
from refactor_train_and_run import load_and_prep_data_for_training

# --- Configuration ---
def get_config():
    amp = '200s'
    config = {
        'amp': amp,
        'output_cut_value': 0.6,
        'train_cut': 4000,
        'noise_rms_200s': 22.53 * units.mV,
        'noise_rms_100s': 20 * units.mV,
        'station_ids_200s': [14, 17, 19, 30],
        'station_ids_100s': [13, 15, 18],
        'sim_date': '5.28.25',
        'base_sim_folder': '/dfs8/sbarwick_lab/ariannaproject/rricesmi/simulatedRCRs/',
        'base_model_path': '/pub/tangch3/ARIANNA/DeepLearning/refactor/models/',
        'base_plot_path': '/pub/tangch3/ARIANNA/DeepLearning/refactor/plots/',
        'loading_data_type': 'new_chi_above_curve',
        'model_filename_template': '{timestamp}_{amp}_RCR_Backlobe_model_2Layer.h5',
        'history_filename_template': '{timestamp}_{amp}_RCR_Backlobe_model_2Layer_history.pkl',
        'loss_plot_filename_template': '{timestamp}_{amp}_{prefix}_loss_plot_RCR_Backlobe_model_2Layer.png',
        'accuracy_plot_filename_template': '{timestamp}_{amp}_{prefix}_accuracy_plot_RCR_Backlobe_model_2Layer.png',
        'tsne_plot_filename_template': '{timestamp}_{amp}_tsne_feature_space_RCR_Backlobe_model_2Layer.png',
        'histogram_filename_template': '{timestamp}_{amp}_train_and_run_histogram.png',
        'early_stopping_patience': 5,
        'keras_epochs': 50,
        'keras_batch_size': 64,
        'verbose_fit': 1,
        'lambda_adversary': 0.01,
        'input_shape': (4, 256, 1),
    }
    config['noise_rms'] = config['noise_rms_200s'] if amp == '200s' else config['noise_rms_100s']
    config['station_ids'] = config['station_ids_200s'] if amp == '200s' else config['station_ids_100s']
    return config

# --- Gradient Reversal Layer ---
@tf.custom_gradient
def gradient_reversal_operation(x, lambda_):
    def grad(dy):
        return -lambda_ * dy, tf.zeros_like(lambda_)
    return x, grad

def GradientReversalLayer(lambda_):
    return Lambda(lambda x: gradient_reversal_operation(x, lambda_), name='gradient_reversal_operation')

# --- DANN Model ---
def build_dann_model(input_shape, lambda_):
    inputs = Input(shape=input_shape)
    x = Conv2D(16, (4, 10), activation='relu')(inputs)
    x = Conv2D(8, (1, 10), activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Flatten(name='flatten')(x)
    label_output = Dense(1, activation='sigmoid', name='label_output')(x)
    reversed_x = GradientReversalLayer(lambda_)(x)
    domain_output = Dense(1, activation='sigmoid', name='domain_output')(reversed_x)
    model = Model(inputs=inputs, outputs=[label_output, domain_output])
    model.compile(
        optimizer=Adam(learning_rate=1e-4, clipvalue=1.0),  # gradient clipping added to limit how much a single batch can affect weights.
        loss={'label_output': 'binary_crossentropy', 'domain_output': 'binary_crossentropy'},
        loss_weights={'label_output': 1.0, 'domain_output': 1.0},
        metrics={'label_output': 'accuracy', 'domain_output': 'accuracy'}
    )
    return model

# --- Prepare Data ---
def prep_dann_data(config):
    data = load_and_prep_data_for_training(config)
    x_rcr = data['training_rcr']
    x_bl = data['training_backlobe']
    y_rcr = np.ones(len(x_rcr))
    y_bl = np.zeros(len(x_bl))
    x_source = np.concatenate([x_rcr, x_bl])
    y_source = np.concatenate([y_rcr, y_bl])
    x_target = data['data_backlobe_tracesRCR_all']
    x_domain = np.concatenate([x_source, x_target])
    y_domain = np.concatenate([
        np.zeros(len(x_source)),
        np.ones(len(x_target))
    ])
    x_source = x_source[..., np.newaxis]
    x_domain = x_domain[..., np.newaxis]
    return x_source, y_source, x_domain, y_domain

# --- t-SNE Visualization ---
from sklearn.manifold import TSNE

def plot_tsne_features(model, x_source, x_target, config):
    feature_model = Model(inputs=model.input, outputs=model.get_layer('flatten').output)
    features_src = feature_model.predict(x_source, batch_size=128)
    features_tgt = feature_model.predict(x_target, batch_size=128)
    features = np.concatenate([features_src, features_tgt])
    labels = np.array([0] * len(features_src) + [1] * len(features_tgt))
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    plt.figure(figsize=(8, 6))
    plt.scatter(features_2d[labels == 0, 0], features_2d[labels == 0, 1], label='Source (Sim+BL)', alpha=0.6)
    plt.scatter(features_2d[labels == 1, 0], features_2d[labels == 1, 1], label='Target (All Station Data)', alpha=0.6)
    plt.title('t-SNE of Feature Representations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('/pub/tangch3/ARIANNA/DeepLearning/refactor/tests/', config['tsne_plot_filename_template'].format(timestamp=timestamp, amp=config['amp'])))
    plt.close()

# --- Training History Plotting ---
def plot_dann_training_history(history, config, amp, timestamp, output_dir='.'):
    hist = history.history
    def plot_metric(metric, val_metric, title, ylabel, filename):
        plt.figure(figsize=(8, 6))
        plt.plot(hist[metric], label='Train')
        plt.plot(hist[val_metric], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
    plot_metric('label_output_loss', 'val_label_output_loss', 'Label Loss (RCR task)', 'Loss',
                config['loss_plot_filename_template'].format(timestamp=timestamp, amp=amp, prefix='label'))
    plot_metric('label_output_accuracy', 'val_label_output_accuracy', 'Label Accuracy (RCR task)', 'Accuracy',
                config['accuracy_plot_filename_template'].format(timestamp=timestamp, amp=amp, prefix='label'))
    plot_metric('domain_output_loss', 'val_domain_output_loss', 'Domain Loss (Domain Discriminator)', 'Loss',
                config['loss_plot_filename_template'].format(timestamp=timestamp, amp=amp, prefix='domain'))
    plot_metric('domain_output_accuracy', 'val_domain_output_accuracy', 'Domain Accuracy (should → 50%)', 'Accuracy',
                config['accuracy_plot_filename_template'].format(timestamp=timestamp, amp=amp, prefix='domain'))

# --- Main ---
if __name__ == '__main__':
    cfg = get_config()
    x_source, y_source, x_domain, y_domain = prep_dann_data(cfg)
    n_target = len(x_domain) - len(x_source)
    y_target_dummy = np.zeros(n_target)
    y_combined = np.concatenate([y_source, y_target_dummy])
    y_domain = y_domain.reshape(-1, 1)
    y_combined = y_combined.reshape(-1, 1)

    model = build_dann_model(input_shape=cfg['input_shape'], lambda_=cfg['lambda_adversary'])
    model.summary()

    callbacks = [
        EarlyStopping(patience=cfg['early_stopping_patience'], restore_best_weights=True)
    ]

    history = model.fit(
        x=x_domain,
        y={'label_output': y_combined, 'domain_output': y_domain},
        epochs=cfg['keras_epochs'],
        batch_size=cfg['keras_batch_size'],
        validation_split=0.2,
        callbacks=callbacks,
        verbose=cfg['verbose_fit'],
        shuffle=True
    )

    timestamp = datetime.now().strftime('%m.%d.%y_%H-%M')
    model_filename = cfg['model_filename_template'].format(timestamp=timestamp, amp=cfg['amp'])
    model_path = os.path.join(cfg['base_model_path'], model_filename)
    model.save(model_path)
    print(f'Model saved to: {model_path}')

    x_target = x_domain[len(x_source):]
    plot_tsne_features(model, x_source, x_target, config=cfg)

    plot_dann_training_history(history, config=cfg, amp=cfg['amp'], timestamp=timestamp,
                               output_dir='/pub/tangch3/ARIANNA/DeepLearning/refactor/tests/')
