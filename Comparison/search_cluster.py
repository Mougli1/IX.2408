def SearchCluster(i, NN, clust):
    if clust[i] == 0: #si le point x_i n'a pas été assigné à un cluster
        if clust[NN[i]] > 0: #si le voisin de densité supérieur à x_i à un cluster assigné
            clust[i] = clust[NN[i]] #alors x_i rejoint le cluster de son voisiin
        else: #si même le voisin n'a pas de cluster
            if i != NN[i]: # si le voisin le plus proche de x_i est  pas lui même
                clust = SearchCluster(int(NN[i]), NN, clust) #alors on recherche un cluster récursivement pour le voisin (on remonte donc jusqu'à trouver un point qui est dans un cluster)
                clust[i] = clust[NN[i]] #on affecte x_i au cluster du voisin
    return clust

