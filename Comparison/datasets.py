from sklearn.datasets import load_iris, make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from scipy.io import loadmat
def load_iris_dataset():
    iris = load_iris()
    X = iris.data
    y = iris.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def generate_blobs_dataset():
    X, y = make_blobs(
        n_samples=1000, centers=4, cluster_std=0.60, random_state=42
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def load_custom_dataset():
    X = np.random.rand(100, 4)
    y = None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def load_matlab_dataset():
    try:
        mat = loadmat('data.mat')
        print("keys disponibles dans 'data.mat':", mat.keys())
        X = mat['data']
        y = mat['class'].ravel() if 'class' in mat else None
        print(f"Shape de X: {X.shape}")
        if y is not None:
            print(f"Shape de y: {y.shape}")
        if y is not None and y.ndim > 1 and y.shape[1] == 1:
            y = y.ravel()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y
    except FileNotFoundError:
        print("not found")
        return None, None
    except KeyError as e:
        print(f"error : {e}")
        return None, None
    except Exception as e:
        print(f"erreur produite : {e}")
        return None, None
