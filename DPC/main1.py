import numpy as np

from CFSFDP import CFSFDP
from CFSFDPprint import CFSFDPprint
from compute_distance_matrix import compute_distance_matrix
import matplotlib

np.random.seed(33)  #for reproductibility
a=np.random.multivariate_normal( [5, 0], [[0, 15], [15, 0]], 200)
b=np.random.multivariate_normal( [0, 0], [[0, 1], [1, 0]], 250)
c=np.random.multivariate_normal( [10, -5], [[0, 1.5], [1.5, 0]], 250)
d=np.random.multivariate_normal( [10, 10], [[0, 0.5], [0.5, 0]], 350)
e=np.random.multivariate_normal( [-5, 10], [[0, 0.4], [0.4, 0]], 25)
D = np.concatenate((a, b, c, d, e),)
r=3.5
min_density=5
max_dist=5.0
mdist = compute_distance_matrix(D)
clt, rho, delta, seeds = CFSFDP(D, mdist,r,min_density,max_dist,True)
#clt, rho, delta, seeds = NPCFSFDP(D,mdist,div=3,noise=False)
CFSFDPprint(D, clt, rho, delta, seeds)