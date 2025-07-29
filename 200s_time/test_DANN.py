import os
import numpy as np
import tensorflow as tf
import pickle
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Dropout, Lambda, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from NuRadioReco.utilities import units

from A0_Utilities import load_sim_rcr, load_data, load_config
from refactor_train_and_run import load_and_prep_data_for_training


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
    features_src = feature_model.predict(x_source, batch_size=config['keras_batch_size'])
    features_tgt = feature_model.predict(x_target, batch_size=config['keras_batch_size'])
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
        print(f'Training plots saved to: {output_dir}')
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
    config = load_config("config.yaml")
    x_source, y_source, x_domain, y_domain = prep_dann_data(config)
    n_target = len(x_domain) - len(x_source)
    y_target_dummy = np.zeros(n_target)
    y_combined = np.concatenate([y_source, y_target_dummy])
    y_domain = y_domain.reshape(-1, 1)
    y_combined = y_combined.reshape(-1, 1)

    model = build_dann_model(input_shape=config['input_shape'], lambda_=config['lambda_adversary'])
    model.summary()

    callbacks = [
        EarlyStopping(patience=config['early_stopping_patience'], restore_best_weights=True)
    ]

    history = model.fit(
        x=x_domain,
        y={'label_output': y_combined, 'domain_output': y_domain},
        epochs=config['keras_epochs'],
        batch_size=config['keras_batch_size'],
        validation_split=0.2,
        callbacks=callbacks,
        verbose=config['verbose_fit'],
        shuffle=True
    )

    timestamp = datetime.now().strftime('%m.%d.%y_%H-%M')
    model_filename = config['model_filename_template'].format(timestamp=timestamp, amp=config['amp'])
    model_path = os.path.join(config['base_model_path'], model_filename)
    model.save(model_path)
    print(f'Model saved to: {model_path}')

    x_target = x_domain[len(x_source):]
    plot_tsne_features(model, x_source, x_target, config=config)

    plot_dann_training_history(history, config=config, amp=config['amp'], timestamp=timestamp,
                               output_dir='/pub/tangch3/ARIANNA/DeepLearning/refactor/tests/')
