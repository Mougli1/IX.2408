from sklearn.datasets import load_breast_cancer, load_wine, make_blobs, load_digits
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.io import loadmat

def load_breast_cancer_dataset():
    breast_cancer = load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized, y

def load_wine_dataset():
    wine = load_wine()
    X = wine.data
    y = wine.target
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized, y

def generate_custom_dataset():
    a = np.random.multivariate_normal([5, 0], [[15, 0], [0, 15]], 200)
    b = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 250)
    c = np.random.multivariate_normal([10, -5], [[1.5, 0], [0, 1.5]], 250)
    d = np.random.multivariate_normal([10, 10], [[0.5, 0], [0, 0.5]], 350)
    e = np.random.multivariate_normal([-5, 10], [[0.4, 0], [0, 0.4]], 25)
    D = np.concatenate((a, b, c, d, e), )
    
    labels_a = np.zeros(200)  # Label 0 pour 'a'
    labels_b = np.ones(250)  # Label 1 pour 'b'
    labels_c = np.full(250, 2)  # Label 2 pour 'c'
    labels_d = np.full(350, 3)  # Label 3 pour 'd'
    labels_e = np.full(25, 4)  # Label 4 pour 'e'
    y = np.concatenate((labels_a, labels_b, labels_c, labels_d, labels_e))
    return D, y

def load_digits_dataset():
    digits = load_digits()
    X = digits.data
    y = digits.target
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized, y


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
        scaler = MinMaxScaler()
        X_normalized = scaler.fit_transform(X)
        return X_normalized, y
    except FileNotFoundError:
        print("not found")
        return None, None
    except KeyError as e:
        print(f"error : {e}")
        return None, None
    except Exception as e:
        print(f"erreur produite : {e}")
        return None, None