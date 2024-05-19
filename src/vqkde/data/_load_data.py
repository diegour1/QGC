
import os
import numpy as np

from  sklearn import model_selection

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import scipy


def _load_arc(train_size, test_size, dimension=2):

    x2_dist = tfd.Normal(loc=0., scale=4.)
    x2_samples = x2_dist.sample(train_size + test_size)
    x1 = tfd.Normal(loc=.25 * tf.square(x2_samples),
                    scale=tf.ones(train_size + test_size, dtype=tf.float32))
    x1_samples = x1.sample()
    x_samples = tf.stack([x1_samples, x2_samples], axis=1)

    X_densities = x2_dist.prob(x_samples[:,1]) * x1.prob(x_samples[:,0])

    # True densities
    # x2 = tfp.distributions.Normal(loc = 0., scale = 4.)
    #x1 = tfp.distributions.Normal(loc = .25 * tf.square(x_samples[:,1]), scale = tf.ones(12_000, dtype=tf.float32))

    # # Mesh
    # x, y = np.mgrid[-10:40:(50/120), -15:15:(30/120)]
    # pos = np.dstack((x, y))
    # X_plot = pos.reshape([14400,2])

    # real_prob = x2_dist.prob(X_plot[:,1]) * scipy.stats.norm(0.25 * np.square(X_plot[:,1]), 1).pdf(X_plot[:,0])

    # batch_size = 32
    #X_train, X_test = model_selection.train_test_split(x_samples.numpy(), test_size=1/6)

    X_train, X_test = model_selection.train_test_split(x_samples.numpy(), test_size=test_size)

    # train_dataset = tf.data.Dataset.from_tensor_slices(X_train)
    # batched_train_data = train_dataset.batch(batch_size)

    X_train_densities = X_densities[:train_size]
    X_test_densities = X_densities[train_size: train_size + test_size]

    return X_train, X_train_densities, X_test, X_test_densities


def _load_binomial(train_size, test_size, dimension=2):
    mvn = tfd.MultivariateNormalDiag(loc=[1., -1], scale_diag=[1, 2.])
    X = mvn.sample(sample_shape=12000, seed = 1)

    x_samples = mvn.sample(train_size + test_size)

    X_densities = mvn.prob(x_samples)
    X_train, X_test = model_selection.train_test_split(X.numpy(), test_size=0.1666666)

    X_train_densities = X_densities[:train_size]
    X_test_densities = X_densities[train_size: train_size + test_size]

    return X_train, X_train_densities, X_test, X_test_densities



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
            X_train, X_train_densities, X_test, X_test_densities = _load_arc(train_size, test_size, dimension)
        case "binomial":
            X_train, X_train_densities, X_test, X_test_densities = _load_binomial(train_size, test_size, dimension)
        case "potential_1":
            X_train, X_train_densities, X_test, X_test_densities = _load_potential_1(train_size, test_size, dimension)
        case "potential_2":
            X_train, X_train_densities, X_test, X_test_densities = _load_potential_2(train_size, test_size, dimension)
        case "star_eight":
            X_train, X_train_densities, X_test, X_test_densities = _load_star_eight(train_size, test_size, dimension)
        
    return X_train, X_train_densities, X_test, X_test_densities