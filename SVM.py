import numpy as np
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

def SoftMarg(X, y, gamma):
    d, n = X.shape
    P = matrix(np.matmul(y, y.T) * np.matmul(X.T, X))
    q = matrix(-np.ones((n, 1)))
    G = matrix(np.vstack((-np.identity(n), np.identity(n))))    
    h = matrix(np.vstack((np.zeros((n, 1)), np.ones((n, 1)) * gamma)))
    b = matrix(0.0)
    A = matrix(y.T)

    ans = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(ans['x'])

    # only consider the support vectors
    idx = np.where(alphas > 1e-6)[0]
    X_supp = X[:,idx]
    y_supp = y[idx]
    alphas_supp = alphas[idx]
    beta_T = np.sum(y_supp * alphas_supp * X_supp.T, axis=0, keepdims=True)
    bias = np.average(y_supp - np.transpose(np.matmul(beta_T, X_supp)))
    return np.transpose(beta_T), bias

def classify(X, b, b0):
    pred = np.transpose(np.matmul(np.transpose(b), X) + b0)
    return np.sign(pred)

def classification_acc(y_predicted, y_expected):
    return 1 - (np.count_nonzero(y_predicted==y_expected) / len(y_expected))