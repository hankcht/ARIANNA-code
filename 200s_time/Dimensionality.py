import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from refactor_train_and_run import load_and_prep_data_for_training
from A0_Utilities import load_config
from sklearn.preprocessing import StandardScaler

config = load_config()
data = load_and_prep_data_for_training(config)
training_rcr = data['training_rcr']

possibilities = ['data_backlobe_tracesRCR_all', 'data_backlobe_all']
for poss in possibilities:
    training_backlobe = data[poss]

    X = np.vstack([training_backlobe, training_rcr])
    labels = np.array([0]*len(training_backlobe) + [1]*len(training_rcr))

    X_flat = X.reshape(X.shape[0], -1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))

    sns.scatterplot(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        hue=labels,
        palette='viridis',
        alpha=0.8,
        s=50,
        edgecolor='w',
        legend='full'
    )

    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.title('PCA of Flattened Waveform Data: RCR vs Backlobe', fontsize=16)

    handles, plot_labels = plt.gca().get_legend_handles_labels()
    custom_labels = ['Backlobe', 'RCR']
    plt.legend(handles=handles, labels=custom_labels, title="Waveform Type", loc='best')

    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    print('saving')
    plt.savefig(f'/pub/tangch3/ARIANNA/DeepLearning/refactor/tests/test_pca_{poss}.png', dpi=300)
    plt.show()