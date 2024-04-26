import numpy as np


def rbf_kernel(X, Y=None, gamma=None):
    X = np.array(X)
    if Y is not None:
        Y = np.array(Y)
    else:
        Y = X

    if X.shape[1] != Y.shape[1]:
        max_cols = max(X.shape[1], Y.shape[1])
        X = np.pad(X, ((0, 0), (0, max_cols - X.shape[1])), 'constant')
        Y = np.pad(Y, ((0, 0), (0, max_cols - Y.shape[1])), 'constant')

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
    dist = X_norm + Y_norm - 2 * np.dot(X, Y.T)

    K = np.exp(-gamma * dist)
    return K