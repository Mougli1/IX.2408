from sklearn.datasets import load_breast_cancer, load_wine, load_digits, load_iris
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def load_breast_cancer_dataset(normalize=True):
    breast_cancer = load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target
    if normalize:
        scaler = MinMaxScaler()
        X_normalized = scaler.fit_transform(X)
        return X_normalized, y
    else:
        return X, y


def load_wine_dataset(normalize=True):
    wine = load_wine()
    X = wine.data
    y = wine.target
    if normalize:
        scaler = MinMaxScaler()
        X_normalized = scaler.fit_transform(X)
        return X_normalized, y
    else:
        return X, y


def generate_custom_dataset(normalize=True):
    a = np.random.multivariate_normal([5, 0], [[15, 0], [0, 15]], 200)
    b = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 250)
    c = np.random.multivariate_normal([10, -5], [[1.5, 0], [0, 1.5]], 250)
    d = np.random.multivariate_normal([10, 10], [[0.5, 0], [0, 0.5]], 350)
    e = np.random.multivariate_normal([-5, 10], [[0.4, 0], [0, 0.4]], 25)
    D = np.concatenate((a, b, c, d, e), )

    labels_a = np.zeros(200)
    labels_b = np.ones(250)
    labels_c = np.full(250, 2)
    labels_d = np.full(350, 3)
    labels_e = np.full(25, 4)
    y = np.concatenate((labels_a, labels_b, labels_c, labels_d, labels_e))

    if normalize:
        scaler = MinMaxScaler()
        D = scaler.fit_transform(D)

    return D, y


def load_digits_dataset(normalize=True):
    digits = load_digits()
    X = digits.data
    y = digits.target
    if normalize:
        scaler = MinMaxScaler()
        X_normalized = scaler.fit_transform(X)
        return X_normalized, y
    else:
        return X, y


def load_iris_dataset(normalize=True):
    iris = load_iris()
    X = iris.data
    y = iris.target
    if normalize:
        scaler = MinMaxScaler()
        X_normalized = scaler.fit_transform(X)
        return X_normalized, y
    else:
        return X, y
