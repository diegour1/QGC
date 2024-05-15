
import os
import numpy as np


def _load_potential_1(train_size, test_size, dimension=2):
    print(f"Loading potential_1\ntrain_size: {train_size}\ntest_size: {test_size}\ndimension: {dimension}")

    X = np.loadtxt(os.path.join(os.path.dirname(__file__), "raw", "potential_1", "NF1_1M.csv")).astype(np.float32)
    X_densities = np.loadtxt(os.path.join(os.path.dirname(__file__), "raw", "potential_1", "NF1_1M_densities.csv")).astype(np.float32)

    X_train = X[:train_size, :]
    X_train_densities = X_densities[:train_size]
    X_test = X[train_size: train_size + test_size, :]
    X_test_densities = X_densities[train_size: train_size + test_size]

    return X_train, X_train_densities, X_test, X_test_densities


def _load_potential_2(train_size, test_size, dimension=2):
    print(f"Loading potential_2\ntrain_size: {train_size}\ntest_size: {test_size}\ndimension: {dimension}")

    X = np.load(os.path.join(os.path.dirname(__file__), "raw", "potential_2", "nf2.npy")).astype(np.float32)
    X_densities = np.loadtxt(os.path.join(os.path.dirname(__file__), "raw", "potential_2", "NF2_densities.csv")).astype(np.float32)

    X_train = X[:train_size, :]
    X_train_densities = X_densities[:train_size]
    X_test = X[train_size: train_size + test_size, :]
    X_test_densities = X_densities[train_size: train_size + test_size]

    return X_train, X_train_densities, X_test, X_test_densities


def _load_star_eight(train_size, test_size, dimension=2):
    print(f"Loading star_eight\ntrain_size:{train_size}\ntest_size: {test_size}\ndimension: {dimension}")

    X_train = np.load(os.path.join(os.path.dirname(__file__), "raw", "star_eight", "star_eight_train.npy")).astype(np.float32)[:train_size, :dimension]
    X_train_densities = np.load(os.path.join(os.path.dirname(__file__), "raw", "star_eight", "star_eight_train_density.npy")).astype(np.float32) [:train_size]
    X_test = np.load(os.path.join(os.path.dirname(__file__), "raw", "star_eight", "star_eight_test.npy")).astype(np.float32)[:test_size, :dimension]
    X_test_densities = np.load(os.path.join(os.path.dirname(__file__), "raw", "star_eight", "star_eight_test_density.npy")).astype(np.float32)[:test_size]

    return X_train, X_train_densities, X_test, X_test_densities


# Main data loader
def load_data(dataset:str, train_size=10000, test_size=2000, dimension=2):
    match dataset:
        case "arc":
            pass
        case "binomial":
            pass
        case "potential_1":
            X_train, X_train_densities, X_test, X_test_densities = _load_potential_1(train_size, test_size, dimension)
            return X_train, X_train_densities, X_test, X_test_densities
        case "potential_2":
            X_train, X_train_densities, X_test, X_test_densities = _load_potential_2(train_size, test_size, dimension)
            return X_train, X_train_densities, X_test, X_test_densities
        case "star_eight":
            X_train, X_train_densities, X_test, X_test_densities = _load_star_eight(train_size, test_size, dimension)
            return X_train, X_train_densities, X_test, X_test_densities