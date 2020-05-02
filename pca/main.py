import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def pca(X, p=-1):
    n, m = X.shape
    if p == -1:
        p = m
    # make sure all variables are zero mean
    assert np.allclose(X.mean(axis=0), np.zeros(m))
    # covariance matrix
    C = np.dot(X.T, X) / (n - 1)
    # eigen-decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    # projection
    proj = np.dot(X, eigen_vecs[:, :p])
    return proj


def main():
    pass


if __name__ == '__main__':
    main()
