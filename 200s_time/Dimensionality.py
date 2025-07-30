import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from refactor_train_and_run import load_and_prep_data_for_training
from A0_Utilities import load_config

# Combine your data

config = load_config()
data = load_and_prep_data_for_training(config)
training_rcr = data['training_rcr']
training_backlobe = data['training_backlobe']
X = np.vstack([training_rcr, training_backlobe])
labels = np.array([0]*len(training_rcr) + [1]*len(training_backlobe))  # 0 = RCR, 1 = Backlobe

# Optionally normalize or standardize features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)  # 2D for visualization
X_pca = pca.fit_transform(X_scaled)

# Plot
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.7)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('PCA of Training Data (RCR vs Backlobe)')
plt.legend(*scatter.legend_elements(), title="Class")
plt.grid(True)
plt.save('/pub/tangch3/ARIANNA/DeepLearning/refactor/tests/test_pca.png')