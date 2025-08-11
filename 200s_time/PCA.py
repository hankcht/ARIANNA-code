import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from glob import glob
import os

from refactor_train_and_run import load_and_prep_data_for_training
from A0_Utilities import load_config, load_sim, pT


def run_pca(X_list, labels, label_names, out_prefix, n_components=2, input_shape=(4, 256), region_filter=None):
    # Flatten and normalize
    X = np.vstack(X_list)
    X_flat = X.reshape(X.shape[0], -1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    # Fit PCA, full for scree plot, and X for visualizing clusters
    pca_full = PCA()
    pca_full.fit(X_scaled)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # --- Find and print indices in region (with optional label filter) ---
    def find_points_in_radius(X_pca, center, radius, labels=None, target_label=None):
        distances = np.linalg.norm(X_pca - center, axis=1)
        in_radius = distances <= radius
        if labels is not None and target_label is not None:
            in_label = labels == target_label
            return np.where(in_radius & in_label)[0]
        else:
            return np.where(in_radius)[0]

    if region_filter is not None:
        center = np.array(region_filter['center'])
        radius = region_filter['radius']
        target_label = region_filter['target_label']
        indices = find_points_in_radius(X_pca, center, radius, labels=labels, target_label=target_label)
        print(f"\nFound {len(indices)} points within radius {radius} of {center} for label '{label_names[target_label]}'")
        print(f"Indices: {indices}")
        # Optional: Save or analyze these traces
        for idx in indices:
            trace = X[idx].reshape(input_shape)
            pT(trace, f'test plot pot. RCR {idx}', f'/pub/tangch3/ARIANNA/DeepLearning/refactor/tests/pca_test_plot_potrcr_{idx}.png')

    # --- Plot PCA scatter ---
    plt.figure(figsize=(10, 8))

    unique_labels = np.unique(labels)

    if n_components == 2:
        if region_filter is not None:
            circle = plt.Circle(region_filter['center'], region_filter['radius'], color='red', fill=False, linestyle='--', linewidth=2)
            plt.gca().add_patch(circle)
        palette = sns.color_palette('viridis', n_colors=len(unique_labels))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette=palette, hue_order=unique_labels, alpha=0.8, s=50, edgecolor='w')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        handles, plot_labels = plt.gca().get_legend_handles_labels()
        custom_labels = [label_names[i] for i in unique_labels]
        plt.legend(handles=handles, labels=custom_labels, title="Waveform Type", loc='best')
        
        plt.title(f'PCA Visualization ({n_components}D)')
        plt.tight_layout()
        plt.grid(True)
        print(f'saving to {out_prefix}_pca_{n_components}d.png')
        plt.savefig(f'{out_prefix}_pca_{n_components}d.png', dpi=300)

    elif n_components == 3:
        palette = sns.color_palette('viridis', n_colors=len(unique_labels))
        label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}

        # Define different POVs you want to save images for
        povs = [(30, -60), (0, 0), (0, 90), (90, -90)]

        for idx, (elev, azim) in enumerate(povs):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            for i in unique_labels:
                ax.scatter(
                    X_pca[labels == i, 0],
                    X_pca[labels == i, 1],
                    X_pca[labels == i, 2],
                    color=label_to_color[i],
                    label=label_names[i],
                    alpha=0.7, s=50, edgecolor='w'
                )

            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
            ax.legend()

            ax.view_init(elev=elev, azim=azim)
            plt.title(f'PCA Visualization ({n_components}D) - elev={elev}, azim={azim}')
            plt.tight_layout()
            plt.grid(True)
            ax.set_box_aspect([1, 1, 1]) 

            filename = f'{out_prefix}_pca_{n_components}d_view_{idx}_elev_{elev}_azim_{azim}.png'
            print(f'saving to {filename}')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)

    else:
        raise ValueError("n_components must be 2 or 3")


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
    print(f'{out_prefix}_scree.png')
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
        print(f'saving {out_prefix}_pc{i+1}_weights.png')
        plt.savefig(f'{out_prefix}_pc{i+1}_weights.png', dpi=300)


if __name__ == "__main__":
    config = load_config()
    data = load_and_prep_data_for_training(config)

    path = f'/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/'
    amp = config['amp']
    RCR_path = f'simulatedRCRs/{amp}_2.9.24/'
    backlobe_path = f'simulatedBacklobes/{amp}_2.9.24/'
    rcr_sim, backlobe_sim = load_sim(path, RCR_path, backlobe_path, amp)

    from refactor_checks import load_all_coincidence_traces, load_2016_backlobe_templates
    pkl_path = "/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/station_data/6.11.25_CoincidenceDatetimes_with_all_params_recalcZenAzi_calcPol.pkl"
    all_coincidence_events, _ = load_all_coincidence_traces(pkl_path)

    template_dir = "/dfs8/sbarwick_lab/ariannaproject/rricesmi/numpy_arrays/templates/confirmed2016Templates"
    template_paths = sorted(glob(os.path.join(template_dir, "Event2016_Stn*.npy")))
    all_2016_backlobes, dict_2016 = load_2016_backlobe_templates(template_paths, amp_type=amp)

    all_possible_types = ['sim_rcr', 'sim_bl', 'data_bl_2016', 'data_bl_rcr', 'confirmed_2016_bl', 'coincidence'] # list of all types of data I want to examine
    input_types = ['coincidence', 'confirmed_2016_bl']
    n_components = 3  # Change to 3 for 3D

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
        label_names[label_idx] = 'sim Backlobe'
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
    
    if 'confirmed_2016_bl' in input_types:
        X_list.append(all_2016_backlobes)
        label_list.extend([label_idx]*len(all_2016_backlobes))
        label_names[label_idx] = '2016 Backlobes'
        label_idx += 1

    if 'coincidence' in input_types:
        X_list.append(all_coincidence_events)
        label_list.extend([label_idx]*len(all_coincidence_events))
        label_names[label_idx] = 'coincidence events'
        label_idx += 1

    labels = np.array(label_list)
    plot_path = '/pub/tangch3/ARIANNA/DeepLearning/refactor/tests/'
    out_prefix = f'{plot_path}pca_{"_".join(input_types)}'

    # Define region filter parameters (optional)
    use_region_filter = False
    if use_region_filter == True:
        target_label_name = 'data Backlobe RCR'
        target_label_idx = [k for k, v in label_names.items() if v == target_label_name][0]
        region_filter = {
            'center': [14, 0], # check dimensions of center with n_components
            'radius': 8,
            'target_label': target_label_idx
        }


    run_pca(X_list, labels, label_names, out_prefix, n_components=n_components, region_filter=region_filter if use_region_filter else None)
