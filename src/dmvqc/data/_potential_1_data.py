import numpy as np
import os

def load_potential_1(train_size, test_size, dimension=2):

    print(f"Loading potential_1 train_size: {train_size} test_size: {test_size} dimension: {dimension}")

    X = np.loadtxt(os.path.join(os.path.dirname(__file__), "raw", "potential_1", "NF1_1M.csv")).astype(np.float32)
    X_densities = np.loadtxt(os.path.join(os.path.dirname(__file__), "raw", "potential_1", "NF1_1M_densities.csv")).astype(np.float32)

    X_train = X[:train_size, :]
    X_train_densities = X_densities[:train_size]
    X_test = X[train_size: train_size + test_size, :]
    X_test_densities = X_densities[train_size: train_size + test_size]

    return X_train, X_train_densities, X_test, X_test_densities
