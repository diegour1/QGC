from functools import partial
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm

import tensorcircuit as tc
from tensorcircuit import keras

tc.set_backend("tensorflow")
tc.set_dtype("complex128")


def create_U_train(x_train_param, seed=0):
    r"""
    Given the eigenvalues this function return a Unitary gate which converts :math:`|0\rangle` into :math:`| \psi \rangle`.

    :param kind: Optional "kind" of ingredients.
    :type kind: list[str] or None
    :return: The ingredients list.
    :rtype: list[str]

    """
    np.random.seed(seed)
    size_U = len(x_train_param)
    U_train = np.zeros((size_U, size_U), dtype = np.complex128)
    x_1 = x_train_param
    U_train[:, 0] = x_1
    for i in range(1, size_U):
        x_i =  np.complex128(np.random.randn(size_U) + 1j*np.random.randn(size_U))
        for j in range(0, i):
            x_i -= x_i.dot(np.conjugate(U_train[:, j])) * U_train[:, j]
        x_i = x_i/np.linalg.norm(x_i)
        U_train[:, i] = x_i

    return U_train

def predict_features(X, var, gamma):
    """"""
    X_feat = np.ones((len(X), len(var)), dtype = np.complex128)
    X_feat[:, :] = np.cos(np.sqrt(gamma)*X*var) + 1j*np.sin(np.sqrt(gamma)*X*var)
    X_feat *= np.sqrt(1/(len(var)))
    return X_feat


class Model():
    def __init__(self, U_conjtrans_sample_param, var_pure_state_param):
        self.circuit = tc.Circuit(n_total_qubits_temp)
        self.gamma = 1.



