import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mass_based_distance import MeDissimilarity
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from tqdm import tqdm 
import seaborn as sns

def plot_clusters(X, labels, title):
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(sorted(unique_labels), colors):
        if k == -1:
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.scatter(
            xy[:, 0], xy[:, 1], c=[col], edgecolors='k', label=f'Cluster {k}'
        )
    plt.title(title)
    plt.legend()

def compute_mass_based_distance_matrix(me_dissim, X):
    N = X.shape[0]
    mdist_mass = np.zeros((N, N))
    for i in tqdm(range(N), desc="Calcul des distances mass-based"):
        for j in range(i, N):
            distance = me_dissim.mass_based_dissimilarity(X[i], X[j])
            mdist_mass[i, j] = distance
            mdist_mass[j, i] = distance
    return mdist_mass

def display_heatmaps(mdist_euclidean, mdist_mass):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(mdist_euclidean)
    plt.title("Matrice de distances Euclidiennes")
    plt.subplot(1, 2, 2)
    sns.heatmap(mdist_mass)
    plt.title("Matrice de distances Mass-based")
    plt.show()
def run_mbscan_vs_dbscan(X, y, display_heatmaps_flag=False, display_matrices_flag=False):
    X_scaled = X
    me_dissim = MeDissimilarity(X_scaled)
    dissim_func = me_dissim.get_dissim_func(num_itrees=100)

    print("Calcul de la matrice de distances euclidiennes...")
    mdist_euclidean = pairwise_distances(X_scaled, metric='euclidean')

    print("Calcul de la matrice de distances mass-based (peut être long)...")
    mdist_mass = compute_mass_based_distance_matrix(me_dissim, X_scaled)

    if display_heatmaps_flag:
        display_heatmaps(mdist_euclidean, mdist_mass)

    if display_matrices_flag:
        np.set_printoptions(precision=3, suppress=True)
        print("\nMatrice de distances Euclidiennes :")
        print(mdist_euclidean)
        print("\nMatrice de distances Mass-based :")
        print(mdist_mass)
        np.set_printoptions()

    eps_values = [0.3, 0.35, 0.4, 0.45, 0.5]
    min_samples_values = [3, 5, 7]
    results = []
    for eps in eps_values:
        for min_samples in min_samples_values:
            print(f"\nMBSCAN/DBSCAN avec eps={eps} et min_samples={min_samples}")
            dbscan_euclidean = DBSCAN(eps=eps, min_samples=min_samples)
            labels_euclidean = dbscan_euclidean.fit_predict(X_scaled)
            dbscan_mass = DBSCAN(
                eps=eps, min_samples=min_samples, metric='precomputed'
            )
            labels_mass = dbscan_mass.fit_predict(mdist_mass)
            ari_euclidean = adjusted_rand_score(y, labels_euclidean)
            ari_mass = adjusted_rand_score(y, labels_mass)

            n_clusters_euclidean = len(set(labels_euclidean)) - (
                1 if -1 in labels_euclidean else 0
            )
            n_noise_euclidean = list(labels_euclidean).count(-1)
            n_clusters_mass = len(set(labels_mass)) - (
                1 if -1 in labels_mass else 0
            )
            n_noise_mass = list(labels_mass).count(-1)

            print(
                f"Euclidienne - Clusters: {n_clusters_euclidean}, Bruit: {n_noise_euclidean}, ARI: {ari_euclidean:.4f}"
            )
            print(
                f"Masse-based - Clusters: {n_clusters_mass}, Bruit: {n_noise_mass}, ARI: {ari_mass:.4f}"
            )
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            title_euclidean = f'Euclidienne eps={eps} min_samples={min_samples}\nARI={ari_euclidean:.4f}'
            plot_clusters(
                X_scaled,
                labels_euclidean,
                title=title_euclidean,
            )

            plt.subplot(1, 2, 2)
            title_mass = f'Masse-based eps={eps} min_samples={min_samples}\nARI={ari_mass:.4f}'
            plot_clusters(
                X_scaled,
                labels_mass,
                title=title_mass,
            )
            plt.show()

            results.append({
                'eps': eps,
                'min_samples': min_samples,
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
