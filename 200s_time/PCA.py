import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from refactor_train_and_run import load_and_prep_data_for_training
from A0_Utilities import load_config, load_sim
from sklearn.preprocessing import StandardScaler

config = load_config()
data = load_and_prep_data_for_training(config)
training_rcr = data['training_rcr']

# --- load sim BL ---
path = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/'    #Edit path to properly point to folder
amp = config['amp']                                                                 #Set which amplifier to run on
RCR_path = f'simulatedRCRs/{amp}_2.9.24/'
backlobe_path = f'simulatedBacklobes/{amp}_2.9.24/'
rcr, Backlobe = load_sim(path, RCR_path, backlobe_path, amp)

possibilities = ['data_backlobe_traces2016', 'data_backlobe_tracesRCR', 'sim_Backlobe']
for poss in possibilities:

    if poss == 'sim_Backlobe':
        print('using sim BL')
        training_backlobe = Backlobe
    else:
        training_backlobe = data[poss]
        
    X = np.vstack([training_backlobe, training_rcr])
    labels = np.array([0]*len(training_backlobe) + [1]*len(training_rcr))

    X_flat = X.reshape(X.shape[0], -1) # flatten since my inputs are 2D, (4, 256)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat) # normalize

    # To get a scree plot
    pca_full = PCA()
    pca_full.fit(X_scaled)

    # To visualize PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # --- 2D scatter plot of PC1 and PC2 ---
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis', alpha=0.8, s=50, edgecolor='w', legend='full')

    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.title('PCA: sim RCR vs Backlobe', fontsize=16)

    handles, plot_labels = plt.gca().get_legend_handles_labels()
    custom_labels = ['Backlobe', 'RCR']
    plt.legend(handles=handles, labels=custom_labels, title="Waveform Type", loc='best')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    print('saving')
    plt.savefig(f'/pub/tangch3/ARIANNA/DeepLearning/refactor/tests/pca_{poss}.png', dpi=300)

    # --- Scree plot that shows variance of each PC ---
    plt.figure(figsize=(10, 6))
    explained_var = pca_full.explained_variance_ratio_
    plt.plot(np.arange(1, len(explained_var) + 1), explained_var, 'o-', color='black')

    for i in range(min(3, len(explained_var))):
        plt.text(i + 1, explained_var[i] + 0.005, f"{explained_var[i]*100:.1f}%", ha='center', fontsize=10, color='black')

    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'/pub/tangch3/ARIANNA/DeepLearning/refactor/tests/scree_plot_{poss}.png', dpi=300)

    # --- plot weights (linear combimation of each feature that forms the principal component, an eigenvector)
    pc1_weights = pca.components_[0].reshape(4, 256) # reshape
    pc2_weights = pca.components_[1].reshape(4, 256)

    plt.figure(figsize=(16, 6))

    # PC1 heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(pc1_weights, cmap='coolwarm', center=0,
                xticklabels=50, yticklabels=[f'Channel {i+1}' for i in range(4)])
    plt.title('PC1 Loadings Heatmap')
    plt.xlabel('Time Points')
    plt.ylabel('Channels')

    # PC2 heatmap
    plt.subplot(1, 2, 2)
    sns.heatmap(pc2_weights, cmap='coolwarm', center=0,
                xticklabels=50, yticklabels=[f'Channel {i+1}' for i in range(4)])
    plt.title('PC2 Loadings Heatmap')
    plt.xlabel('Time Points')
    plt.ylabel('Channels')

    plt.tight_layout()
    plt.savefig(f'/pub/tangch3/ARIANNA/DeepLearning/refactor/tests/pca_pc1_and_pc2_weights_{poss}.png', dpi=300)

