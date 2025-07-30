import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Combine data and labels
X = np.vstack([training_rcr, training_backlobe])
labels = np.array([0]*len(training_rcr) + [1]*len(training_backlobe))  # 0 = RCR, 1 = Backlobe

# Flatten (num_samples, 4, 256) → (num_samples, 1024)
X_flat = X.reshape(X.shape[0], -1)

# Normalize
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
plt.title('PCA of Flattened (4×256) Waveforms')
plt.legend(*scatter.legend_elements(), title="Class")
plt.grid(True)
print('saving')
plt.savefig('/pub/tangch3/ARIANNA/DeepLearning/refactor/tests/test_pca.png')