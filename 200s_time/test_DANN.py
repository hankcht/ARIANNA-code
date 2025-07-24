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
    """Returns a dictionary of configuration parameters with derived fields set."""
    amp = '200s'  

    # Base config
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
        'loss_plot_filename_template': '{timestamp}_{amp}_loss_plot_RCR_Backlobe_model_2Layer.png',
        'accuracy_plot_filename_template': '{timestamp}_{amp}_accuracy_plot_RCR_Backlobe_model_2Layer.png',
        'histogram_filename_template': '{timestamp}_{amp}_train_and_run_histogram.png',
        'early_stopping_patience': 5,
        'keras_epochs': 50,
        'keras_batch_size': 64,
        'verbose_fit': 1,
        'lambda_adversary': 1.0,
        'input_shape': (4, 256, 1),
    }

    if amp == '200s':
        config['noise_rms'] = config['noise_rms_200s']
        config['station_ids'] = config['station_ids_200s']
    elif amp == '100s':
        config['noise_rms'] = config['noise_rms_100s']
        config['station_ids'] = config['station_ids_100s']

    return config

# --- Gradient Reversal Layer ---
@tf.custom_gradient
def gradient_reversal(x, lambda_):
    def grad(dy):
        return -lambda_ * dy, tf.zeros_like(lambda_)
    return x, grad

def GradientReversal(lambda_):
    return Lambda(lambda x: gradient_reversal(x, lambda_), name='gradient_reversal')

# --- DANN Model ---
def build_dann_model(input_shape, lambda_):
    inputs = Input(shape=input_shape)

    # Feature extractor
    x = Conv2D(20, (4, 10), activation='relu', groups=1)(inputs)
    x = Conv2D(10, (1, 10), activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)

    # Task classifier
    label_output = Dense(1, activation='sigmoid', name='label_output')(x)

    # Domain classifier
    reversed_x = GradientReversal(lambda_)(x)
    domain_output = Dense(1, activation='sigmoid', name='domain_output')(reversed_x)

    model = Model(inputs=inputs, outputs=[label_output, domain_output])

    model.compile(
        optimizer=Adam(),
        loss={'label_output': 'binary_crossentropy', 'domain_output': 'binary_crossentropy'},
        loss_weights={'label_output': 1.0, 'domain_output': 1.0},
        metrics={'label_output': 'accuracy', 'domain_output': 'accuracy'}
    )
    return model

# --- Prepare Data ---
def prep_dann_data(config):
    data = load_and_prep_data_for_training(config)

    # Source domain: labeled
    x_rcr = data['training_rcr']
    x_bl = data['training_backlobe']
    y_rcr = np.ones(len(x_rcr))
    y_bl = np.zeros(len(x_bl))

    x_src = np.concatenate([x_rcr, x_bl])
    y_src = np.concatenate([y_rcr, y_bl])

    # Target domain: unlabeled
    x_tgt = data['data_backlobe_tracesRCR_all']

    # Domain labels: 0 for source, 1 for target
    x_dom = np.concatenate([x_src, x_tgt])
    y_dom = np.concatenate([
        np.zeros(len(x_src)),
        np.ones(len(x_tgt))
    ])

    # Reshape
    x_src = x_src[..., np.newaxis]
    x_dom = x_dom[..., np.newaxis]

    return x_src, y_src, x_dom, y_dom

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_tsne_features(model, x_src, x_tgt):
    # Extract the feature extractor part (before classification heads)
    feature_model = Model(inputs=model.input, outputs=model.get_layer('flatten').output)

    features_src = feature_model.predict(x_src, batch_size=128)
    features_tgt = feature_model.predict(x_tgt, batch_size=128)

    features = np.concatenate([features_src, features_tgt])
    labels = np.array([0] * len(features_src) + [1] * len(features_tgt))  # 0=source, 1=target

    # Run t-SNE to reduce to 2D
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(features_2d[labels == 0, 0], features_2d[labels == 0, 1], label='Source (Sim+BL)', alpha=0.6)
    plt.scatter(features_2d[labels == 1, 0], features_2d[labels == 1, 1], label='Target (All Station Data)', alpha=0.6)
    plt.title('t-SNE of Feature Representations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/pub/tangch3/ARIANNA/DeepLearning/refactor/tests/tsne_feature_space.png')
    plt.close()

def plot_dann_training_history(history, output_dir='.', prefix='dann'):
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
        plt.savefig(os.path.join(output_dir, f'{prefix}_{filename}.png'))
        plt.close()

    # Label task loss & accuracy
    plot_metric('label_output_loss', 'val_label_output_loss', 'Label Loss (RCR task)', 'Loss', 'label_loss')
    plot_metric('label_output_accuracy', 'val_label_output_accuracy', 'Label Accuracy (RCR task)', 'Accuracy', 'label_accuracy')

    # Domain loss & accuracy
    plot_metric('domain_output_loss', 'val_domain_output_loss', 'Domain Loss (Domain Discriminator)', 'Loss', 'domain_loss')
    plot_metric('domain_output_accuracy', 'val_domain_output_accuracy', 'Domain Accuracy (should → 50%)', 'Accuracy', 'domain_accuracy')

# --- Main ---
if __name__ == '__main__':
    cfg = get_config()

    # Load data
    x_src, y_src, x_dom, y_dom = prep_dann_data(cfg)

    n_target = len(x_dom) - len(x_src)
    y_target_dummy = np.zeros(n_target)
    y_combined = np.concatenate([y_src, y_target_dummy])

    # Build model
    model = build_dann_model(input_shape=cfg['input_shape'], lambda_=cfg['lambda_adversary'])
    model.summary()

    callbacks = [
        EarlyStopping(patience=cfg['early_stopping_patience'], restore_best_weights=True)
    ]

    # Train model
    history = model.fit(
        x=x_dom,
        y={'label_output': y_combined, 'domain_output': y_dom},
        epochs=cfg['keras_epochs'],
        batch_size=cfg['keras_batch_size'],
        validation_split=0.2,
        callbacks=callbacks,
        verbose=cfg['verbose_fit'],
        shuffle=True
    )

    # Save
    timestamp = datetime.now().strftime('%m.%d.%y_%H-%M')
    model_filename = cfg['model_filename_template'].format(timestamp=timestamp, amp=cfg['amp'])
    model_path = os.path.join(cfg['base_model_path'], model_filename)
    model.save(model_path)
    print(f'Model saved to: {model_path}')

    x_tgt = x_dom[len(x_src):]  # From your prep_dann_data()
    plot_tsne_features(model, x_src, x_tgt)

    plot_dann_training_history(history, output_dir='/pub/tangch3/ARIANNA/DeepLearning/refactor/tests/', prefix=cfg['amp'])
