import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mass_based_distance import MeDissimilarity
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances, davies_bouldin_score, silhouette_score
from tqdm import tqdm
import seaborn as sns

def plot_clusters(X, labels, title):
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(sorted(unique_labels), colors):
        if k == -1:
            col = [0, 0, 0, 1]  # nori pour le bruit
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.scatter(
            xy[:, 0], xy[:, 1], c=[col], edgecolors='k', label=f'Cluster {k}'
        )
    plt.title(title)
    plt.legend()


def compute_mass_based_distance_matrix(me_dissim, X_orig):
    N = X_orig.shape[0]
    mdist_mass = np.zeros((N, N))
    for i in tqdm(range(N), desc="Calcul des distances mass-based"):
        for j in range(i, N):
            distance = me_dissim.mass_based_dissimilarity(X_orig[i], X_orig[j])
            mdist_mass[i, j] = distance
            mdist_mass[j, i] = distance
    return mdist_mass

def display_heatmaps(mdist_euclidean, mdist_mass):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(mdist_euclidean, cmap='viridis')
    plt.title("Matrice de distances Euclidiennes")
    plt.subplot(1, 2, 2)
    sns.heatmap(mdist_mass, cmap='viridis')
    plt.title("Matrice de distances Mass-based")
    plt.show()


def run_mbscan_vs_dbscan(X_normalized, X_orig, y, display_heatmaps_flag=False, display_matrices_flag=False,
                         wbcd_param=False, wine_param=False, custom_param=False, digits_param=False, iris_param=False):
    X_scaled = X_normalized
    X_mass = X_orig

    me_dissim = MeDissimilarity(X_mass)
    dissim_func = me_dissim.get_dissim_func(num_itrees=100)

    print("Calcul de la matrice de distances Euclidiennes...")
    mdist_euclidean = pairwise_distances(X_scaled, metric='euclidean')

    print("Calcul de la matrice de distances Mass-based (peut être long)...")
    mdist_mass = compute_mass_based_distance_matrix(me_dissim, X_mass)

    if display_heatmaps_flag:
        display_heatmaps(mdist_euclidean, mdist_mass)

    if display_matrices_flag:
        np.set_printoptions(precision=3, suppress=True)
        print("\nMatrice de distances Euclidiennes :")
        print(mdist_euclidean)
        print("\nMatrice de distances Mass-based :")
        print(mdist_mass)
        np.set_printoptions()

    if wbcd_param:
        eps_values = [0.3, 0.35, 0.4, 0.45]
        min_samples_values = [3, 4, 5]
    elif wine_param:
        eps_values = [0.4, 0.45, 0.5]
        min_samples_values = [3, 5, 7]
    elif custom_param:
        eps_values = [0.2, 0.25, 0.3]
        min_samples_values = [2, 4, 6]
    elif digits_param:
        eps_values = [0.15, 0.2, 0.25]
        min_samples_values = [2, 3, 5]
    elif iris_param:
        eps_values = [0.3, 0.35, 0.4]
        min_samples_values = [3, 4]
    else:
        eps_values = [0.2, 0.25, 0.3]
        min_samples_values = [2, 3]

    results = []
    y_adjusted = y + 1

    for eps in eps_values:
        for min_samples in min_samples_values:
            print(f"\nMBSCAN/DBSCAN avec eps={eps} et min_samples={min_samples}")
            dbscan_euclidean = DBSCAN(eps=eps, min_samples=min_samples)
            labels_euclidean = dbscan_euclidean.fit_predict(X_scaled)
            dbscan_mass = DBSCAN(
                eps=eps, min_samples=min_samples, metric='precomputed'
            )
            labels_mass = dbscan_mass.fit_predict(mdist_mass)

            labels_euclidean = np.where(labels_euclidean != -1, labels_euclidean + 1, labels_euclidean)
            labels_mass = np.where(labels_mass != -1, labels_mass + 1, labels_mass)

            kmeans = KMeans(n_clusters=3, random_state=0)
            labels_kmeans = kmeans.fit_predict(X_scaled) + 1

            n_clusters_euclidean = len(set(labels_euclidean)) - (1 if -1 in labels_euclidean else 0)
            n_noise_euclidean = list(labels_euclidean).count(-1)
            n_clusters_mass = len(set(labels_mass)) - (1 if -1 in labels_mass else 0)
            n_noise_mass = list(labels_mass).count(-1)
            n_clusters_kmeans = len(set(labels_kmeans)) - (1 if -1 in labels_kmeans else 0)
            n_noise_kmeans = list(labels_kmeans).count(-1)

            if n_clusters_euclidean > 1:
                silhouette_euclidean = silhouette_score(X_scaled, labels_euclidean)
                db_euclidean = davies_bouldin_score(X_scaled, labels_euclidean)
            else:
                silhouette_euclidean = np.nan
                db_euclidean = np.nan

            if n_clusters_mass > 1:
                silhouette_mass = silhouette_score(X_mass, labels_mass)
                db_mass = davies_bouldin_score(X_mass, labels_mass)
            else:
                silhouette_mass = np.nan
                db_mass = np.nan

            if n_clusters_kmeans > 1:
                silhouette_kmeans = silhouette_score(X_scaled, labels_kmeans)
                db_kmeans = davies_bouldin_score(X_scaled, labels_kmeans)
            else:
                silhouette_kmeans = np.nan
                db_kmeans = np.nan

            print(
                f"Euclidienne - Clusters: {n_clusters_euclidean}, Bruit: {n_noise_euclidean}, "
                f"Davies-Bouldin Index: {db_euclidean:.4f}, Coefficient de silhouette: {silhouette_euclidean:.4f}"
            )
            print(
                f"Masse-based - Clusters: {n_clusters_mass}, Bruit: {n_noise_mass}, "
                f"Davies-Bouldin Index: {db_mass:.4f}, Coefficient de silhouette: {silhouette_mass:.4f}"
            )
            print(
                f"K-means - Clusters: {n_clusters_kmeans}, Bruit: {n_noise_kmeans}, "
                f"Davies-Bouldin Index: {db_kmeans:.4f}, Coefficient de silhouette: {silhouette_kmeans:.4f}"
            )

            plt.figure(figsize=(15, 8))
            plt.subplot(2, 3, 1)
            title_euclidean = f'Euclidienne eps={eps} min_samples={min_samples}'
            plot_clusters(
                X_scaled,
                labels_euclidean,
                title_euclidean,
            )

            plt.subplot(2, 3, 2)
            title_mass = f'Masse-based eps={eps} min_samples={min_samples}'
            plot_clusters(
                X_mass,
                labels_mass,
                title_mass,
            )

            plt.subplot(2, 3, 3)
            title_kmeans = f'K-means (k=3)'
            plot_clusters(
                X_scaled,
                labels_kmeans,
                title_kmeans,
            )

            if y is not None:
                df_euclidean = pd.DataFrame({'Labels': y_adjusted, 'Clusters': labels_euclidean})
                df_mass = pd.DataFrame({'Labels': y_adjusted, 'Clusters': labels_mass})
                df_kmeans = pd.DataFrame({'Labels': y_adjusted, 'Clusters': labels_kmeans})

                labels_classes = np.unique(y_adjusted)
                labels_clusters_euclidean = np.unique(labels_euclidean)
                labels_clusters_mass = np.unique(labels_mass)
                labels_clusters_kmeans = np.unique(labels_kmeans)

                all_labels = np.sort(labels_classes)
                all_clusters_euclidean = np.sort(labels_clusters_euclidean)
                all_clusters_mass = np.sort(labels_clusters_mass)
                all_clusters_kmeans = np.sort(labels_clusters_kmeans)

                ct_euclidean = pd.crosstab(df_euclidean['Labels'], df_euclidean['Clusters'])
                ct_euclidean = ct_euclidean.reindex(index=all_labels, columns=all_clusters_euclidean, fill_value=0)
                ct_euclidean['Total'] = ct_euclidean.sum(axis=1)
                total_row_euclidean = ct_euclidean.sum(axis=0)
                total_row_euclidean.name = 'Total'
                ct_euclidean = pd.concat([ct_euclidean, total_row_euclidean.to_frame().T])

                ct_mass = pd.crosstab(df_mass['Labels'], df_mass['Clusters'])
                ct_mass = ct_mass.reindex(index=all_labels, columns=all_clusters_mass, fill_value=0)
                ct_mass['Total'] = ct_mass.sum(axis=1)
                total_row_mass = ct_mass.sum(axis=0)
                total_row_mass.name = 'Total'
                ct_mass = pd.concat([ct_mass, total_row_mass.to_frame().T])

                ct_kmeans = pd.crosstab(df_kmeans['Labels'], df_kmeans['Clusters'])
                ct_kmeans = ct_kmeans.reindex(index=all_labels, columns=all_clusters_kmeans, fill_value=0)
                ct_kmeans['Total'] = ct_kmeans.sum(axis=1)
                total_row_kmeans = ct_kmeans.sum(axis=0)
                total_row_kmeans.name = 'Total'
                ct_kmeans = pd.concat([ct_kmeans, total_row_kmeans.to_frame().T])

                plt.subplot(2, 3, 4)
                sns.heatmap(ct_euclidean, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 8})
                plt.title(f'Matrice de confusion (Euclidienne)\neps={eps} min_samples={min_samples}')
                plt.xlabel("Clusters")
                plt.ylabel("Labels réels")

                plt.subplot(2, 3, 5)
                sns.heatmap(ct_mass, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 8})
                plt.title(f'Matrice de confusion (Mass-based)\neps={eps} min_samples={min_samples}')
                plt.xlabel("Clusters")
                plt.ylabel("Labels réels")

                plt.subplot(2, 3, 6)
                sns.heatmap(ct_kmeans, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 8})
                plt.title('Matrice de confusion (K-means)')
                plt.xlabel("Clusters")
                plt.ylabel("Labels réels")
                plt.tight_layout()
                plt.show()

            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters_euclidean': n_clusters_euclidean,
                'n_noise_euclidean': n_noise_euclidean,
                'db_euclidean': db_euclidean,
                'silhouette_euclidean': silhouette_euclidean,
                'n_clusters_mass': n_clusters_mass,
                'n_noise_mass': n_noise_mass,
                'db_mass': db_mass,
                'silhouette_mass': silhouette_mass,
                'n_clusters_kmeans': n_clusters_kmeans,
                'n_noise_kmeans': n_noise_kmeans,
                'db_kmeans': db_kmeans,
                'silhouette_kmeans': silhouette_kmeans
            })
    df_results = pd.DataFrame(results)
    print("\nRésultats :")
    print(df_results)