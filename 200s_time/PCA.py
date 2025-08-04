import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

from refactor_train_and_run import load_and_prep_data_for_training
from A0_Utilities import load_config, load_sim


def run_pca(X_list, labels, label_names, out_prefix, n_components=2, input_shape=(4, 256)):
    # Flatten and normalize
    X = np.array(X_list)        # shape: (N, 4, 256)
    X_flat = X.reshape(10, -1)  # shape: (N, 1024)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    # Fit PCA, full for scree plot, and X for visualizing clusters
    pca_full = PCA()
    pca_full.fit(X_scaled)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # --- Plot PCA scatter ---
    plt.figure(figsize=(10, 8))
    if n_components == 2:
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis', alpha=0.8, s=50, edgecolor='w')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    elif n_components == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for i in np.unique(labels):
            ax.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], X_pca[labels == i, 2], label=label_names[i], alpha=0.7)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
        ax.legend()
    else:
        raise ValueError("n_components must be 2 or 3")

    plt.title(f'PCA Visualization ({n_components}D)')
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_pca_{n_components}d.png', dpi=300)


    # --- Find and Plot certain events on scatter plot ---
    # With the scatter plot, we can find events in a closed ball around some coordinate
    def find_points_in_radius(X_pca, center, radius):
        distances = np.linalg.norm(X_pca - center, radius)
        return np.where(distances <= radius)[0]
    
    center = np.array([1.0, 0.0])
    radius = 0.3
    indices = find_points_in_radius(X_pca - center, radius)

    # for idx in indices:
    #     trace = X[idx].reshape(4, 256)

    print(len(indices))

    # --- Scree plot ---
    plt.figure(figsize=(10, 6))
    explained_var = pca_full.explained_variance_ratio_
    plt.plot(np.arange(1, len(explained_var) + 1), explained_var, 'o-', color='black')
    for i in range(min(n_components, len(explained_var))):
        plt.text(i + 1, explained_var[i] + 0.005, f"{explained_var[i]*100:.1f}%", ha='center', fontsize=10, color='black')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_scree.png', dpi=300)

    # --- PC Weights Heatmap ---
    for i in range(n_components):
        pc_weights = pca.components_[i].reshape(input_shape)
        plt.figure(figsize=(8, 4))
        sns.heatmap(pc_weights, cmap='coolwarm', center=0,
                    xticklabels=50, yticklabels=[f'Ch {j}' for j in range(input_shape[0])])
        plt.title(f'PC{i+1} Loadings Heatmap')
        plt.xlabel('Time')
        plt.ylabel('Channels')
        plt.tight_layout()
        plt.savefig(f'{out_prefix}_pc{i+1}_weights.png', dpi=300)


if __name__ == "__main__":
    config = load_config()
    data = load_and_prep_data_for_training(config)

    path = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/'
    amp = config['amp']
    RCR_path = f'simulatedRCRs/{amp}_2.9.24/'
    backlobe_path = f'simulatedBacklobes/{amp}_2.9.24/'
    rcr_sim, backlobe_sim = load_sim(path, RCR_path, backlobe_path, amp)

    # Choose which sets to include
    all_types = ['sim_rcr', 'data_bl_2016', 'data_bl_rcr', 'sim_bl', 'confirmed_bl_2016', 'coincidence_event']
    input_types = ['sim_rcr', 'data_bl_2016']
    n_components = 2  # Change to 3 for 3D

    X_list = []
    label_list = []
    label_names = {}
    label_idx = 0

    if 'sim_rcr' in input_types:
        training_rcr = data['training_rcr']
        X_list.append(training_rcr)
        label_list.extend([label_idx]*len(training_rcr))
        label_names[label_idx] = 'sim RCR'
        label_idx += 1

    if 'sim_bl' in input_types:
        X_list.append(backlobe_sim)
        label_list.extend([label_idx]*len(backlobe_sim))
        label_names[label_idx] = 'Sim Backlobe'
        label_idx += 1

    if 'data_bl_2016' in input_types:
        X_data_bl_2016 = data['data_backlobe_traces2016']
        X_list.append(X_data_bl_2016)
        label_list.extend([label_idx]*len(X_data_bl_2016))
        label_names[label_idx] = 'data Backlobe 2016'
        label_idx += 1

    if 'data_bl_rcr' in input_types:
        X_data_bl_rcr = data['data_backlobe_tracesRCR']
        X_list.append(X_data_bl_rcr)
        label_list.extend([label_idx]*len(X_data_bl_rcr))
        label_names[label_idx] = 'data Backlobe RCR'
        label_idx += 1


    labels = np.array(label_list)
    out_prefix = f'/pub/tangch3/ARIANNA/DeepLearning/refactor/tests/pca_{"_".join(input_types)}'
    run_pca(X_list, labels, label_names, out_prefix, n_components=n_components)
