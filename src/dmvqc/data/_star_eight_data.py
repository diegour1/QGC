import numpy as np
import os

def load_star_eight(train_size, test_size, dimension=2):
    print(f"Loading star_eight\nTrain_size:{train_size}\nTest_size: {test_size}\nDimension: {dimension}")

    X_train = np.load(os.path.join(os.path.dirname(__file__), "raw", "star_eight", "star_eight_train.npy")).astype(np.float32)[:train_size, :dimension]
    X_train_densities = np.load(os.path.join(os.path.dirname(__file__), "raw", "star_eight", "star_eight_train_density.npy")).astype(np.float32) [:train_size]
    X_test = np.load(os.path.join(os.path.dirname(__file__), "raw", "star_eight","star_eight_test.npy")).astype(np.float32)[:test_size, :dimension]
    X_test_densities = np.load(os.path.join(os.path.dirname(__file__), "raw", "star_eight", "star_eight_test_density.npy")).astype(np.float32)[:test_size]

    return X_train, X_train_densities, X_test, X_test_densities