# mass_based_distance.py

import numpy as np
from functools import partial
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