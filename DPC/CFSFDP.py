import numpy as np

from search_cluster import SearchCluster
def CFSFDP(X, dist, r, min_density, max_dist, noise=True):
    rho = np.zeros(len(X), dtype=int)     # tableau de 0 de la taille du dataset, pour stocker la densité de chaque point
    clust = np.zeros(len(X), dtype=int)   # tableau de 0 de la taille du dataset, stocke l'assignation de cluster pour chaque point (0 signifie non assigné).
    delta = np.zeros(len(X), dtype=float) # tableau de 0 de la taille du dataset
    NN = np.zeros(len(X), dtype=int)      # tableau de 0 de la taille du dataset, stocke l'indice du voisin de densité supérieure le plus proche pour chaque point.
    MaxRho = 0                            # densité maximale
    k = 0                                 # compteur du nombre de clusters

    #calcul de densité : rho
    for i in range(len(X)): #on boucle sur toutes les données
        for j in range(len(X)): #on boucle sur toutes les données
            if dist[i][j] <= r: #dist est la matrice des DE, on fixe x_i en ligne et on parcour tous les colonne, si la distance entre les points x_i et x_j est inf ou égale à la distance seuil "r"
                rho[i] += 1 # alors la densité de l'observation x_i est incrémentée de 1
        if rho[i] > MaxRho: #si la densité de i est supérieur à la densité maximale
            MaxRho = rho[i] # alors la densité maximale devient la densité de l'obs i
    #calcul de delta
    for i in range(len(rho)): # on boucle sur chaque donnée (pq taille de rho et pas du dataset (c la meme))
        NN[i] = i # l'indice du point le plus proche de x_i est i
        delta[i] = np.max(dist[i]) # la distance relative du point x_i est égale à la valeur maximale de distance de x_i parmi toutes les colonnes
        if rho[i] < MaxRho: # si densité de x_i est inferieur à la densité maximale donc si le point x_i n'est le point le plus dense alors on utilise le min
            for j in range(len(rho)): #on boucle sur toutes les colonnes
                if rho[j] > rho[i]: # si la densité d'un point x_j est ssupérieur à celle d'un pint x_i
                    if delta[i] > dist[i][j]: #si la distnace relative de x_i est supérieure à la distance entre x_i et nimporte quel x_j, cette condition est toujours vérifié car delta_i représente à cette étape la disance maximale entr x_i et nimpprte quel point x_j
                        delta[i] = dist[i][j] #alors la distance relative de x_i devient la distance la plus petite entre x_i et n'importe quel x_j
                        NN[i] = j # lindice du point le plius proche de x_i est celui du point qui est à la distance minimale


    #**AFFECTATION DES POINTS DANS DES CLUSTERS**
    # np.argsort donne tableau contenant les indices qui trieraient le tableau d'entrée tableau en ordre croissant.ex :tableau = np.array([50, 20, 30, 10]), indices_tris = np.argsort(tableau); donne Indices triés : [3 1 2 0] car le plus petit nombre est en indice 3 du tableau initial
    sorted_set = np.argsort(rho * delta) # sorted[i] donne l'indice dans le dataset de la i-ème plus grande valeur, taille du nombre d'observations
    for m in range(len(X)): #on parcourt tous les points du dataset
        i = sorted_set[len(sorted_set) - m - 1]  #TODO ajouter commentaire tablette, en gros on traite les points de celui qui a la valeur la plus grande à celui qui a la valeur la plus petite
        if clust[i] == 0: #si le point x_i n'a pas été assigné à un cluster
            #Cas 1 le point est du bruit
            if rho[i] < min_density: #si le densité de x_i est inferieur à la densité minimale
                if noise: #si  noise est true(il est tt le temps true (paramètre défini))
                    clust[i] = -1 #alorsle point x_i n'est pas assigné à un cluster il est donc considére comm du brouit
           #cas 2 : le pont est un centre de cluster potentiel ou alors  il est associé à un cluster
            elif NN[i] == i or delta[i] > max_dist: #si l'indice du point le plus proche de x_i est i( = pics de densité) (cas1) OU si la distance relative de x_i est superieur à max_dist (=points isolés de densité élevée, même si x_i n'est pas de densité maximale il est suffisament éloigné pour être le centre d'un cluster)(cas 2) donc si x_i est un centre de cluster potentiel
                newseed = True #flag, s newseed reste True après vérification, x_i sera déclaré comme un nouveau centre de cluster
                for l in range(m): #on boucle sur les points déjà traités pour voir si x_i peut être assigné à un cluster existant au lieu de devenir un nouveau centre de cluster.
                    j = sorted_set[len(sorted_set) - l - 1] # TODO
                   #Cas 2-souscas1 : on trouve un cluster que x_i peut rejoindre
                    if j != i and clust[j] > 0 and dist[i][j] <= r:  #si x_j n'est pas x_i et que x_j est dans un cluster et que les deux points sont proche
                        clust[i] = clust[j] #alors x_i est dans le même cluster que x_j
                        NN[i] = j #et le voisin le plus proche de x_i est x_j
                        newseed = False #x_i ne devient pas un centre de cluster
                        break #sort de la boucle car on a trouvé un cluster pour x_j
                # Cas 2-souscas2 : le point ne peut pas rejoindre de cluster existant donc il devient un centre de cluster
                if newseed: # on a pas trouvé de cluster que x_i peut rejoindre alors x_i devienet un centre
                    k += 1 # on incrémente le nombre de cluster
                    clust[i] = k #x_i rejoint le cluster numéro k

    # cas où x_i n'a pas été affecté à un cluster
    for i in range(len(clust)): #on boucle sur chaque point
        clust = SearchCluster(int(i), NN, clust) #on appelle la fonction  (seulement dans le cas où x_i n'a pas été affecté à un cluster

    # Extraction des centres de clusters (aspect esthétique)
    cluster_seeds = np.zeros(k + 1, dtype=int) #tableau de 0 qui stocke l'indice du point ayant la densité maximale pour chaque cluster
    rho_max = np.zeros(k + 1, dtype=float) #tableau de 0 pour stocker  p la densité maximale trouvée dans chaque cluster.
    for i in range(len(X)): #on parcourt tous les points
        if clust[i] > 0: #si x_i à été assigné à un cluster
            if rho[i] > rho_max[clust[i]]: #si la densité du point x_i est supérieur à la densité maximale du cluster où il appartient
                rho_max[clust[i]] = rho[i] #alors la densité maximale du cluster est la densité de x_i
                cluster_seeds[clust[i]] = i #l'indice du point ayant la densité maximale est i
    cluster_seeds = np.delete(cluster_seeds, 0) # on supprime l'indice du point pour le cluster 0 car ils commencent à 1 (d'où le k+1)

    return clust, rho, delta, cluster_seeds
