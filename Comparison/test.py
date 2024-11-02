from scipy.io import loadmat

mat = loadmat('data.mat')
print("Clés disponibles dans 'data.mat':", mat.keys())