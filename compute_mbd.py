import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from functools import partial
from sklearn.metrics import f1_score, pairwise_distances
from scipy.optimize import linear_sum_assignment
import pandas as pd
import numpy as np
from functools import partial
import pdb
import seaborn as sns
from matplotlib import pyplot as plt

np.random.seed(42)
class It_node:#la classe permet de pouvoir définir  un arbre binaire(un noeud et ses 2 enfants)

    def __init__(self, l, r, split_attr, split_val, level, mass_comp=0):
        self.l = l                    # le noeud gauche
        self.r = r                    # le noeud droit
        self.split_attr = split_attr  # split attribute
        self.split_val = split_val    # split value/split point
        self.level = level            # hauteur du noeud
        self.mass = 0                 # masse du noeud
class MeDissimilarity: #la classe permet de pouvoir définir un ensemble de fonctions (méthodes) que l'on va pouvoir appliquer au dataset

    def __init__(self, data):
        self.data = data # on peux demander à l'objet quel est son dataset

    def get_random_itree(self, data_sub, current_height=0, lim=10): #création d'un arbre, méthode qui n'est pas appelée directement sur l'objet mais qui sera utilisée dans une autre méthode (méthode interne)
        """
        Objectif : construire un arbre binaire complet
        """
        if current_height >= lim or data_sub.shape[0] <= 1: # cas de base de récursivité : si la hauteur du noeud est superieur à la limite ou alors si il y a une obs ou moins
            return It_node(None, None, None, None, current_height) #alors on crée un leaf node (pas de l et de r ni de split)
        q = np.random.randint(data_sub.shape[1])#le split attribut est choisi aléatoirement parmi tt les col de data_sub
        p = np.random.uniform(data_sub[:, q].min(), data_sub[:, q].max()) #le splitpoint est choisi aléatoirement entre la valeur min et max de l'attribut parmi tt les obs
        xl, xr = data_sub[data_sub[:, q] < p], data_sub[data_sub[:, q] >= p] #l'enfant gauche à les données inférieures et l'enfant droit à les données supérieures
        return It_node(#on renvoie un noeudqui a comme caractéristique (attributs d'instance définis dans __init__)
            l=self.get_random_itree(data_sub=xl, current_height=current_height + 1, lim=lim), # le noeud gauche qui lui même appelle la fonction de division de manière récursivejusqua atteindre le noeud de profondeur maximale
            r=self.get_random_itree(data_sub=xr, current_height=current_height + 1, lim=lim), # le noeud droit qui lui même appelle la fonction de division de manière récursive jusqua atteindre le noeud de profondeur maximale
            split_attr=q, split_val=p, level=current_height #q comme split attribut, p comme split point, et la hauteur comme actuel comme hauteur du neoud
        )

    def get_n_random_itrees(self, n, subs_size): #objectif : créer un stockage de n root nodes
        self.root_nodes = np.array([ # on crée un tableau qui stocke les root nodes (en self car utilisé après)
            #il genere n root nodes en utilisant la méthode adéquate
            self.get_random_itree(data_sub=self.data[np.random.choice(self.data.shape[0], subs_size, replace=False)])#on donne un seul paramètre : une partie des observations (de la taille désirée par l'user)
            for _ in range(n) #compréhension de liste
        ], dtype=object)
        self.subs_size = subs_size  # #on utilise self ici non pas parce que c'est un attribut d'instance (auquel cas il serait dans __init__) mais parce que'il sera réutilisé dans une autre méthode commepour random_itrees


    def get_node_masses(self):#méthode pour obtenir la masse des noeuds
        for root_node in self.root_nodes: #pour chaque noeud racine de la foret
            for example in self.data: #pour chaque observation du dataset
                node = root_node #on appele node le noeud racine
                while node: #tant que node est pas False i.e tant qu'il existe des root nodes
                    node.mass += 1 #on ajoute 1 à la masse du noeud
                    if node.split_val is None: #si le noeud est une leaf node
                        break # on sort de la boucle while
                    node = node.l if example[node.split_attr] < node.split_val else node.r #node devient le noeud enfant droit ou le noeud enfant gauche donc on reste dans la boucle while

    def get_lowest_common_node_mass(self, root_node, x1, x2): # méthode pour obtenir la masse de l'ancètre commun de deux observations
        if root_node.split_val is None: #si le noeud est une leaf node
            return root_node.mass #alors on renvoie la masse du noeud
        cond1= x1[root_node.split_attr] < root_node.split_val #condition 1 : la valeur de l'observation pour l'attribut est inferieur au split point
        cond2=x2[root_node.split_attr] < root_node.split_val #condition 2 : la valeur de l'observation pour l'attribut est inferieur au split point
        if cond1 != cond2: #cas de base (récursif)
            return root_node.mass #on renvoi la masse du noeud

        return self.get_lowest_common_node_mass(root_node.l if cond1 else root_node.r, x1, x2) #appel du noeud enfant gauche ou droit

    def mass_based_dissimilarity(self, x1, x2):#calcul de la dissimilarité entre 2 observations
        #formule :
        masses = [
            self.get_lowest_common_node_mass(root_node, x1, x2) / self.subs_size
            for root_node in self.root_nodes
        ]
        return np.mean(masses)  # moyenne de toute les mbd calculées

    def get_dissim_func(self, num_itrees):
        self.get_n_random_itrees(num_itrees, self.data.shape[0])
        self.get_node_masses()

        def dissim_func(x1, x2):
            if x1.ndim == 1 and x2.ndim == 1:
                return self.mass_based_dissimilarity(x1, x2)
            elif x1.ndim == 1:
                return np.apply_along_axis(partial(self.mass_based_dissimilarity, x1), 1, x2)
            elif x2.ndim == 1:
                return np.apply_along_axis(partial(self.mass_based_dissimilarity, x2), 1, x1)
            elif x1.shape[0] == x2.shape[0]:
                return np.array([self.mass_based_dissimilarity(r1, r2) for r1, r2 in zip(x1, x2)])
            else:
                raise ValueError("les deux matrices doivent avoir le même nombre de lignes pour la dissimilarité paire à paire.")

        return dissim_func

def plot_clusters(X, labels, title):
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(sorted(unique_labels), colors):
        if k == -1:
            col = [0, 0, 0, 1]
        class_member_mask = labels == k
        xy = X[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], color=col, edgecolors='k', label=f'Cluster {k}')
    plt.title(title)
    plt.legend()

def main():
    iris = load_iris()
    X = iris.data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    me_dissim = MeDissimilarity(X_scaled)
    dissim_func = me_dissim.get_dissim_func(num_itrees=100)
    eps_values = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.7, 1]
    min_samples_values = [3, 5, 7, 9, 11]
    results = []
    for eps in eps_values:
        for min_samples in min_samples_values:
            print(f"\n  MBSCAN/DBSCAN avec eps={eps} et min_samples={min_samples}")
            dbscan_euclidean = DBSCAN(eps=eps, min_samples=min_samples)
            labels_euclidean = dbscan_euclidean.fit_predict(X_scaled)
            dbscan_mass = DBSCAN(eps=eps, min_samples=min_samples, metric=dissim_func)
            labels_mass = dbscan_mass.fit_predict(X_scaled)
            n_clusters_euclidean = len(set(labels_euclidean)) - (1 if -1 in labels_euclidean else 0)
            n_noise_euclidean = list(labels_euclidean).count(-1)
            n_clusters_mass = len(set(labels_mass)) - (1 if -1 in labels_mass else 0)
            n_noise_mass = list(labels_mass).count(-1)
            print(f"Euclidienne - Clusters: {n_clusters_euclidean}, Bruit: {n_noise_euclidean}")
            print(f"Masse-based - Clusters: {n_clusters_mass}, Bruit: {n_noise_mass}")
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plot_clusters(X_scaled, labels_euclidean, title=f'Euclidienne eps={eps} min_samples={min_samples}')
            plt.subplot(1, 2, 2)
            plot_clusters(X_scaled, labels_mass, title=f'Masse-based eps={eps} min_samples={min_samples}')
            plt.show()
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters_euclidean': n_clusters_euclidean,
                'n_noise_euclidean': n_noise_euclidean,
                'n_clusters_mass': n_clusters_mass,
                'n_noise_mass': n_noise_mass
            })
    df_results = pd.DataFrame(results)
    print("\nRésultat  :")
    print(df_results)
if __name__ == '__main__':
    main()
