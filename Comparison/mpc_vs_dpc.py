# mpc_vs_dpc.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import pairwise_distances, adjusted_rand_score
from mass_based_distance import MeDissimilarity
from CFSFDP import CFSFDP
from tqdm import tqdm
import seaborn as sns

def plot_clusters(X, labels, title):
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(sorted(unique_labels), colors):
        if k == -1:
            col = [0, 0, 0, 1]
        class_member_mask = labels == k
        xy = X[class_member_mask]
        plt.scatter(
            xy[:, 0], xy[:, 1], color=col, edgecolors='k', label=f'Cluster {k}'
        )
    plt.title(title)
    plt.legend()

def display_heatmaps(mdist_euclidean, mdist_mass):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(mdist_euclidean)
    plt.title("Matrice de distances Euclidiennes")
    plt.subplot(1, 2, 2)
    sns.heatmap(mdist_mass)
    plt.title("Matrice de distances Mass-based")
    plt.show()

def run_mpc_vs_dpc(X, y=None, display_heatmaps_flag=False, display_matrices_flag=False):
    D_scaled = X
    N = D_scaled.shape[0]
    print("Calcul de la matrice de distances euclidiennes...")
    mdist_euclidean = pairwise_distances(D_scaled, metric='euclidean')
    print("Calcul de la matrice de distances mass-based (peut être long)...")
    me_dissim = MeDissimilarity(D_scaled)
    dissim_func = me_dissim.get_dissim_func(num_itrees=100)#inturile pr linstant
    mdist_mass = np.zeros((N, N))
    for i in tqdm(range(N), desc="Calcul des distances mass-based"):
        for j in range(i, N):
            distance = me_dissim.mass_based_dissimilarity(D_scaled[i], D_scaled[j])
            mdist_mass[i, j] = distance
            mdist_mass[j, i] = distance

    if display_heatmaps_flag:
        display_heatmaps(mdist_euclidean, mdist_mass)

    if display_matrices_flag:
        np.set_printoptions(precision=3, suppress=True)
        print("\nMatrice de distances Euclidiennes :")
        print(mdist_euclidean)
        print("\nMatrice de distances Mass-based :")
        print(mdist_mass)
        np.set_printoptions()
    min_density_values = [3, 5]
    max_dist_values = [ 0.1,0.5, 1.0,]
    r_values = [0.1, 0.3, 0.5, 1.0, 1.5]
    results = []
    for r in r_values:
        for min_density in min_density_values:
            for max_dist in max_dist_values:
                print(f"\nDPC/MPC avec r={r}, min_density={min_density}, max_dist={max_dist}")
                clt_euclidean, rho_euclidean, delta_euclidean, seeds_euclidean = CFSFDP(
                    D_scaled, mdist_euclidean, r, min_density, max_dist, True
                )
                labels_euclidean = clt_euclidean
                clt_mass, rho_mass, delta_mass, seeds_mass = CFSFDP(
                    D_scaled, mdist_mass, r, min_density, max_dist, True
                )
                labels_mass = clt_mass
                if y is not None:
                    ari_euclidean = adjusted_rand_score(y, labels_euclidean)
                    ari_mass = adjusted_rand_score(y, labels_mass)
                else:
                    ari_euclidean = ari_mass = None

                n_clusters_euclidean = len(set(labels_euclidean)) - (1 if -1 in labels_euclidean else 0)
                n_noise_euclidean = list(labels_euclidean).count(-1)

                n_clusters_mass = len(set(labels_mass)) - (1 if -1 in labels_mass else 0)
                n_noise_mass = list(labels_mass).count(-1)

                print(f"Euclidienne - Clusters: {n_clusters_euclidean}, Bruit: {n_noise_euclidean}, ARI: {ari_euclidean}")
                print(f"Masse-based - Clusters: {n_clusters_mass}, Bruit: {n_noise_mass}, ARI: {ari_mass}")

                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                title_euclidean = f'Euclidienne r={r} min_density={min_density} max_dist={max_dist}\nARI={ari_euclidean}'
                plot_clusters(D_scaled, labels_euclidean, title=title_euclidean)

                plt.subplot(1, 2, 2)
                title_mass = f'Masse-based r={r} min_density={min_density} max_dist={max_dist}\nARI={ari_mass}'
                plot_clusters(D_scaled, labels_mass, title=title_mass)
                plt.show()

                results.append({
                    'r': r,
                    'min_density': min_density,
                    'max_dist': max_dist,
                    'n_clusters_euclidean': n_clusters_euclidean,
                    'n_noise_euclidean': n_noise_euclidean,
                    'n_clusters_mass': n_clusters_mass,
                    'n_noise_mass': n_noise_mass,
                    'ari_euclidean': ari_euclidean,
                    'ari_mass': ari_mass
                })

    df_results = pd.DataFrame(results)
    print("\nRésultats :")
    print(df_results)
