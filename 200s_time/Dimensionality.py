import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from refactor_train_and_run import load_and_prep_data_for_training
from A0_Utilities import load_config

# Load configuration and data
config = load_config()
data = load_and_prep_data_for_training(config)
training_rcr = data['training_rcr']
training_backlobe = data['training_backlobe']

# Combine data
X = np.vstack([training_rcr, training_backlobe])
labels = np.array([0]*len(training_rcr) + [1]*len(training_backlobe))  # 0 = RCR, 1 = Backlobe

# Reshape: (num_samples, 4, 256) → (num_samples, 1024)
X_flat = X.reshape(X.shape[0], -1)

# Normalize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flat)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.7)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('PCA of Flattened Waveform Data (RCR vs Backlobe)')
plt.legend(*scatter.legend_elements(), title="Class")
plt.grid(True)
plt.show()
