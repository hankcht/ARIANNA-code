import os
import random

# JAX runs the fastest for hgq in general based on our experience
# If you don't have jax, or if you want to use another backend, you can change this to 'tensorflow' or 'torch'
# os.environ['KERAS_BACKEND'] = 'jax'
# tested for tensorflow, jax, torch. Openvino support is not tested yet.
# For the best performance, we recommend using jax, or tensorflow with XLA enabled.
# Jit compilation for torch (torch dynamo) is not supported yet.

import keras
import numpy as np
from matplotlib import pyplot as plt

from hgq.config import QuantizerConfig, QuantizerConfigScope
from hgq.layers import QDense, QSoftmax
from hgq.utils.sugar import FreeEBOPs, PBar

import pickle as pkl
from pathlib import Path

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_data(data_path: Path, seed=42):
    try:
        import zstd
    except ImportError:
        zstd = None
    if not os.path.exists(data_path):
        print('Downloading data...')
        data = fetch_openml('hls4ml_lhc_jets_hlf')
        buf = pkl.dumps(data)
        with open(data_path, 'wb') as f:
            if zstd is not None:
                buf = zstd.compress(buf)
            f.write(buf)
    else:
        os.makedirs(data_path.parent, exist_ok=True)
        with open(data_path, 'rb') as f:
            buf = f.read()
            if zstd is not None:
                buf = zstd.decompress(buf)
            data = pkl.loads(buf)

    X, y = data['data'], data['target']
    codecs = {'g': 0, 'q': 1, 't': 4, 'w': 2, 'z': 3}
    y = np.array([codecs[i] for i in y])

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    X_train_val, X_test, y_train_val, y_test = X_train_val.astype(np.float32), X_test.astype(np.float32), y_train_val, y_test

    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(X_train_val)
    X_test = scaler.transform(X_test)

    X_train_val = X_train_val.astype(np.float32)
    y_train_val = y_train_val.astype(np.float32)

    return X_train_val, X_test, y_train_val, y_test


X_train, X_test, y_train, y_test = get_data(Path('/tmp/inp_data.zst'))

from hgq.utils.sugar import Dataset

_y_train = keras.utils.to_categorical(y_train, 5)
_y_test = keras.utils.to_categorical(y_test, 5)

dataset_train = Dataset(X_train, _y_train, batch_size=33200, device='gpu:0')
dataset_test = Dataset(X_test, _y_test, batch_size=33200, device='gpu:0')

np.random.seed(42)
random.seed(42)

from hgq.regularizers import MonoL1

# Skipping these should also work.
# Usually, the default configs are good enough for most cases, but the initial number of bits, `[bif]0`
# may need to be increased. If you see that the model is not converging, you can try increasing these values.
scope0 = QuantizerConfigScope(place='all', k0=1, b0=3, i0=0, default_q_type='kbi', overflow_mode='sat_sym')
scope1 = QuantizerConfigScope(place='datalane', k0=0, default_q_type='kif', overflow_mode='wrap', f0=3, i0=3)

exp_table_conf = QuantizerConfig('kif', 'table', k0=0, i0=1, f0=8, overflow_mode='sat_sym')
inv_table_conf = QuantizerConfig('kif', 'table', k0=1, i0=4, f0=4, overflow_mode='sat_sym')
# Layer scope will over formal one. When using scope0, scope1, 'datalane' config will be overriden with config in scope1

use_softmax = False
# QSoftmax is bit-accurate, but it can give exactly zero now: xentropy will diverge and thus USE WITH CAUTION
# For classification, it is recommended to NOT to use softmax in the model, but to use it in the loss function (see below)
def build_model(use_softmax=False, beta0=1e-5):
    with scope0, scope1:
        iq_conf = QuantizerConfig(place='datalane', k0=1)
        oq_conf = QuantizerConfig(place='datalane', k0=1, fr=MonoL1(1e-3))
        layers = [
            QDense(64, beta0=beta0, iq_conf=iq_conf, activation='relu', name='dense_0'),
            QDense(32, beta0=beta0, activation='relu', name='dense_1'),
            QDense(32, beta0=beta0, activation='relu', name='dense_2'),
            QDense(5, beta0=beta0, enable_oq=not use_softmax, name='dense_3', oq_conf=oq_conf),
            # QEinsumDense('...c,oc->...o', 64, bias_axes='o', beta0=beta0, iq_conf=iq_conf, activation='relu', bame='dense_0'),
            # QEinsumDense('...c,oc->...o', 32, bias_axes='o', beta0=beta0, activation='relu', name='dense_1'),
            # QEinsumDense('...c,oc->...o', 32, bias_axes='o', beta0=beta0, activation='relu', name='dense_2'),
            # QEinsumDense('...c,oc->...o', 5, bias_axes='o', beta0=beta0, enable_oq=not use_softmax, name='dense_3'),
        ]
        if use_softmax:
            layers.append(QSoftmax(exp_oq_conf=exp_table_conf, inv_oq_conf=inv_table_conf))

    model = keras.models.Sequential(layers)
    return model


model = build_model(use_softmax=use_softmax, beta0=0.5e-5)
if not use_softmax:
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)
else:
    loss = keras.losses.CategoricalHinge()
opt = keras.optimizers.Adam(learning_rate=5e-3)

model.compile(opt, loss, metrics=['accuracy'], jit_compile=True, steps_per_execution=4)

pbar = PBar('loss: {loss:.3f}/{val_loss:.3f} - acc: {accuracy:.3f}/{val_accuracy:.3f}')
ebops = FreeEBOPs()
nan_terminate = keras.callbacks.TerminateOnNaN()
callbacks = [ebops, pbar, nan_terminate]

history = model.fit(dataset_train, epochs=3000, batch_size=33200, validation_data=dataset_test, verbose=0, callbacks=callbacks)

model.evaluate(dataset_test)


plot_dir = ":/dfs8/sbarwick_lab/ariannaproject/tangch3/HGQ2/test_smalljet/"
os.makedirs(os.path.join(plot_dir, "accuracy"), exist_ok=True)
os.makedirs(os.path.join(plot_dir, "loss"), exist_ok=True)
os.makedirs(os.path.join(plot_dir, "hgq2_results"), exist_ok=True)

train_acc = history.history.get("accuracy")
val_acc   = history.history.get("val_accuracy")
train_loss = history.history.get("loss")
val_loss   = history.history.get("val_loss")
ebops = history.history.get("ebops")

plt.figure(figsize=(8,6))
plt.plot(train_acc, label="HGQ2 train_acc")
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.title("Training Accuracy (HGQ2)")
plt.legend()
plt.savefig(os.path.join(plot_dir, "accuracy", "train_accuracy_hgq2.png"))
plt.close()

plt.figure(figsize=(8,6))
plt.plot(train_loss, label="HGQ2 train_loss")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss (HGQ2)")
plt.legend()
plt.savefig(os.path.join(plot_dir, "loss", "train_loss_hgq2.png"))
plt.close()

plt.figure(figsize=(8,6))
plt.plot(val_acc, label="HGQ2 val_acc")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy (HGQ2)")
plt.legend()
plt.savefig(os.path.join(plot_dir, "accuracy", "val_accuracy_hgq2.png"))
plt.close()

plt.figure(figsize=(8,6))
plt.plot(val_loss, label="HGQ2 val_loss")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss (HGQ2)")
plt.legend()
plt.savefig(os.path.join(plot_dir, "loss", "val_loss_hgq2.png"))
plt.close()

plt.figure(figsize=(8,6))
plt.plot(ebops, '.')
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("EBOPs")
plt.title("HGQ2 EBOPs per Epoch")
plt.savefig(os.path.join(plot_dir, "hgq2_results", "ebops_per_epoch.png"))
plt.close()

plt.figure(figsize=(8,6))
plt.plot(ebops, val_acc, '.')
plt.xscale("log")
plt.xlabel("EBOPs")
plt.ylabel("Validation Accuracy")
plt.title("HGQ2 EBOPs vs Validation Accuracy")
plt.savefig(os.path.join(plot_dir, "hgq2_results", "ebops_vs_val_accuracy.png"))
plt.close()

summary_df = pd.DataFrame({
    "Metric": ["Final Train Accuracy", "Final Val Accuracy", "Final EBOPs"],
    "HGQ2": [train_acc[-1], val_acc[-1], ebops[-1]]
})

summary_file = os.path.join(plot_dir, "hgq2_results", "summary_table.csv")
summary_df.to_csv(summary_file, index=False)

print(summary_df)
print(f"Saved summary table to {summary_file}")


