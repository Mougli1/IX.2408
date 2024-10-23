from sklearn.metrics import pairwise_distances
#calcul de la distance euclidienne avec D le data set
def compute_distance_matrix(D):
    return pairwise_distances(D, metric='euclidean')
