import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import pairwise_distances, silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()


def display_heatmaps(mdist_euclidean, mdist_mass):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(mdist_euclidean)
    plt.title("Euclidean Distance Matrix")
    plt.subplot(1, 2, 2)
    sns.heatmap(mdist_mass)
    plt.title("Mass-based Distance Matrix")
    plt.show()


def run_mpc_vs_dpc(X_normalized, X_orig, y=None, display_heatmaps_flag=False, display_matrices_flag=False,
                   wbcd_param=False, wine_param=False, custom_param=False, digits_param=False, iris_param=False):
    D_euclidean = X_normalized
    D_euclidean = X_normalized
    D_mass = X_orig

    N = D_euclidean.shape[0]

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(D_euclidean)
    explained_variance = pca.explained_variance_ratio_.sum()
    print("Variance expliquée par les deux premières composantes principales : {:.2%}".format(explained_variance))

    print("Calcul de la matrice de distances euclidiennes...")
    mdist_euclidean = pairwise_distances(D_euclidean, metric='euclidean')

    print("Calcul de la matrice de distances mass-based (peut être long)...")
    me_dissim = MeDissimilarity(D_mass)
    dissim_func = me_dissim.get_dissim_func(num_itrees=100)
    mdist_mass = np.zeros((N, N))
    for i in tqdm(range(N), desc="Calcul des distances mass-based"):
        for j in range(i, N):
            distance = me_dissim.mass_based_dissimilarity(D_mass[i], D_mass[j])
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
    # Définition des paramètres en fonction du dataset choisi
    if wbcd_param:
        r_values = [0.4, 0.45, 0.5, 0.55]
        min_density_values = [3, 4]
        max_dist_values = [0.2, 0.4, 0.5]
    elif wine_param:
        r_values = [0.5, 0.6, 0.7]
        min_density_values = [2, 5]
        max_dist_values = [0.3, 0.45]
    elif custom_param:
        r_values = [1, 2, 3, 4]
        min_density_values = [3, 6]
        max_dist_values = [0.2, 0.4, 0.5]
    elif digits_param:
        r_values = [0.1, 0.2, 0.3]
        min_density_values = [1, 2]
        max_dist_values = [0.1, 0.2]
    elif iris_param:
        r_values = [0.3, 0.35, 0.4]
        min_density_values = [2, 3]
        max_dist_values = [0.25, 0.35]
    else:
        r_values = [0.4, 0.45]
        min_density_values = [3]
        max_dist_values = [0.2]


    results = []

    if y is not None:
        y = y + 1

    for r in r_values:
        for min_density in min_density_values:
            for max_dist in max_dist_values:
                print(f"\nDPC/MPC avec r={r}, min_density={min_density}, max_dist={max_dist}")
                clt_euclidean, rho_euclidean, delta_euclidean, seeds_euclidean = CFSFDP(
                    D_euclidean, mdist_euclidean, r, min_density, max_dist, True
                )
                labels_euclidean = clt_euclidean
                clt_mass, rho_mass, delta_mass, seeds_mass = CFSFDP(
                    D_mass, mdist_mass, r, min_density, max_dist, True
                )
                labels_mass = clt_mass

                kmeans = KMeans(n_clusters=3, random_state=0)
                labels_kmeans = kmeans.fit_predict(D_euclidean) + 1

                n_clusters_euclidean = len(set(labels_euclidean)) - (1 if -1 in labels_euclidean else 0)
                n_noise_euclidean = list(labels_euclidean).count(-1)

                n_clusters_mass = len(set(labels_mass)) - (1 if -1 in labels_mass else 0)
                n_noise_mass = list(labels_mass).count(-1)

                n_clusters_kmeans = len(set(labels_kmeans)) - (1 if -1 in labels_kmeans else 0)
                n_noise_kmeans = list(labels_kmeans).count(-1)

                if n_clusters_euclidean > 1:
                    silhouette_euclidean = silhouette_score(D_euclidean, labels_euclidean)
                    db_euclidean = davies_bouldin_score(D_euclidean, labels_euclidean)
                else:
                    silhouette_euclidean = np.nan
                    db_euclidean = np.nan

                if n_clusters_mass > 1:
                    silhouette_mass = silhouette_score(D_mass, labels_mass)
                    db_mass = davies_bouldin_score(D_mass, labels_mass)
                else:
                    silhouette_mass = np.nan
                    db_mass = np.nan

                if n_clusters_kmeans > 1:
                    silhouette_kmeans = silhouette_score(D_euclidean, labels_kmeans)
                    db_kmeans = davies_bouldin_score(D_euclidean, labels_kmeans)
                else:
                    silhouette_kmeans = np.nan
                    db_kmeans = np.nan

                print(
                    f"Euclidienne - Clusters: {n_clusters_euclidean}, Bruit: {n_noise_euclidean}, "
                    f"Coefficient de silhouette: {silhouette_euclidean}, Indice de Davies-Bouldin: {db_euclidean}"
                )
                print(
                    f"Masse-based - Clusters: {n_clusters_mass}, Bruit: {n_noise_mass}, "
                    f"Coefficient de silhouette: {silhouette_mass}, Indice de Davies-Bouldin: {db_mass}"
                )
                print(
                    f"K-means - Clusters: {n_clusters_kmeans}, Bruit: {n_noise_kmeans}, "
                    f"Coefficient de silhouette: {silhouette_kmeans}, Indice de Davies-Bouldin: {db_kmeans}"
                )

                plt.figure(figsize=(15, 8))

                plt.subplot(2, 3, 1)
                title_euclidean = (
                    f'Clusters CFSFDP\n'
                    f'Explained Variance: {explained_variance:.2%}\n'
                    f'r={r}, min_density={min_density}, max_dist={max_dist}'
                )
                plot_clusters(X_pca, labels_euclidean, title=title_euclidean)

                if y is not None:
                    df_euclidean = pd.DataFrame({'Labels': y, 'Clusters': labels_euclidean})
                    labels_classes = np.unique(y)
                    labels_clusters_euclidean = np.unique(labels_euclidean)
                    all_labels = np.sort(labels_classes)
                    all_clusters_euclidean = np.sort(labels_clusters_euclidean)
                    ct_euclidean = pd.crosstab(df_euclidean['Labels'], df_euclidean['Clusters'])
                    ct_euclidean = ct_euclidean.reindex(index=all_labels, columns=all_clusters_euclidean, fill_value=0)
                    ct_euclidean['Total'] = ct_euclidean.sum(axis=1)
                    total_row_euclidean = ct_euclidean.sum(axis=0)
                    total_row_euclidean.name = 'Total'
                    ct_euclidean = pd.concat([ct_euclidean, total_row_euclidean.to_frame().T])
                    plt.subplot(2, 3, 4)
                    sns.heatmap(ct_euclidean, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 8})
                    plt.title('Confusion matrix (CFSFDP) with totals')
                    plt.xlabel('Clusters')
                    plt.ylabel('True labels')
                else:
                    print("Les vraies étiquettes ne sont pas disponibles. Impossible de calculer la matrice de confusion.")

                plt.subplot(2, 3, 2)
                title_mass = (
                    f'm-DPC\n'
                    f'Explained Variance: {explained_variance:.2%}\n'
                    f'r={r}, min_density={min_density}, max_dist={max_dist}'
                )
                plot_clusters(X_pca, labels_mass, title=title_mass)

                if y is not None:
                    df_mass = pd.DataFrame({'Labels': y, 'Clusters': labels_mass})
                    labels_clusters_mass = np.unique(labels_mass)
                    all_clusters_mass = np.sort(labels_clusters_mass)
                    ct_mass = pd.crosstab(df_mass['Labels'], df_mass['Clusters'])
                    ct_mass = ct_mass.reindex(index=all_labels, columns=all_clusters_mass, fill_value=0)
                    ct_mass['Total'] = ct_mass.sum(axis=1)
                    total_row_mass = ct_mass.sum(axis=0)
                    total_row_mass.name = 'Total'
                    ct_mass = pd.concat([ct_mass, total_row_mass.to_frame().T])
                    plt.subplot(2, 3, 5)
                    sns.heatmap(ct_mass, annot=True, fmt='d', cmap='YlGnBu', cbar=False, annot_kws={"size": 8})
                    plt.title('Confusion matrix (m-DPC) with totals')
                    plt.xlabel('Clusters')
                    plt.ylabel('True labels')
                else:
                    print("Les vraies étiquettes ne sont pas disponibles. Impossible de calculer la matrice de confusion.")

                plt.subplot(2, 3, 3)
                title_kmeans = (
                    f'K-means Clustering (with k=3)\n'
                    f'Explained Variance: {explained_variance:.2%}'
                )
                plot_clusters(X_pca, labels_kmeans, title=title_kmeans)

                if y is not None:
                    df_kmeans = pd.DataFrame({'Labels': y, 'Clusters': labels_kmeans})
                    labels_clusters_kmeans = np.unique(labels_kmeans)
                    all_clusters_kmeans = np.sort(labels_clusters_kmeans)
                    ct_kmeans = pd.crosstab(df_kmeans['Labels'], df_kmeans['Clusters'])
                    ct_kmeans = ct_kmeans.reindex(index=all_labels, columns=all_clusters_kmeans, fill_value=0)
                    ct_kmeans['Total'] = ct_kmeans.sum(axis=1)
                    total_row_kmeans = ct_kmeans.sum(axis=0)
                    total_row_kmeans.name = 'Total'
                    ct_kmeans = pd.concat([ct_kmeans, total_row_kmeans.to_frame().T])
                    plt.subplot(2, 3, 6)
                    sns.heatmap(ct_kmeans, annot=True, fmt='d', cmap='OrRd', cbar=False, annot_kws={"size": 8})
                    plt.title('Confusion matrix (K-means) with totals')
                    plt.xlabel('Clusters')
                    plt.ylabel('True labels')
                else:
                    print("Les vraies étiquettes ne sont pas disponibles. Impossible de calculer la matrice de confusion.")

                plt.tight_layout()
                plt.show()

                results.append({
                    'r': r,
                    'min_density': min_density,
                    'max_dist': max_dist,
                    'n_clusters_euclidean': n_clusters_euclidean,
                    'n_noise_euclidean': n_noise_euclidean,
                    'silhouette_euclidean': silhouette_euclidean,
                    'db_euclidean': db_euclidean,
                    'n_clusters_mass': n_clusters_mass,
                    'n_noise_mass': n_noise_mass,
                    'silhouette_mass': silhouette_mass,
                    'db_mass': db_mass,
                    'n_clusters_kmeans': n_clusters_kmeans,
                    'n_noise_kmeans': n_noise_kmeans,
                    'silhouette_kmeans': silhouette_kmeans,
                    'db_kmeans': db_kmeans
                })

    df_results = pd.DataFrame(results)
    print("\nRésultats :")
    print(df_results)
