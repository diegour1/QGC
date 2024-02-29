

import numpy as np
import random
from scipy.stats import norm
import matplotlib.pyplot as plt

def _generate_numbers(k, N, seed=0):
    
    random.seed(seed)

    # Generate k - 1 random integers between 1 and N - k + 1
    numbers = random.sample(range(1, N - k + 2), k - 1)

    # Sort the numbers
    numbers.sort()

    # Calculate the differences between consecutive numbers
    differences = [numbers[0]] + [numbers[i] - numbers[i - 1] for i in range(1, k - 1)] + [N - numbers[-1]]

    return differences

def _predict_features(X, var, gamma):
    r"""
    Add documentation.
    """
    X_feat = np.ones((len(X), len(var)), dtype = np.complex128)
    X_feat[:, :] = np.cos(np.sqrt(gamma)*X*var) + 1j*np.sin(np.sqrt(gamma)*X*var)
    X_feat *= np.sqrt(1/(len(var)))
    return X_feat


def _create_U_train(x_train_param, seed=0):
    r"""
    Given the eigenvalues this function return a Unitary gate which converts :math:`|0\rangle` into :math:`|\psi\rangle`.

    Args:
        x_train_param: Eigenvalues(?).
        seed: Optional seed for random generator.

    Returns:
        The unitary matrix.

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


def dmkde_data(
        means=None, 
        stds=None, 
        sample_sizes=None, 
        k=4, 
        n=1000,
        gamma=2.,
        n_rffs=16,
        domain=(-10, 10),
        seed=0
    ):

    rng = np.random.default_rng(seed)

    if isinstance(means, list) and isinstance(stds, list):
        assert len(means) == len(stds) == len(sample_sizes)
        assert np.isclose(np.sum(sample_sizes), 1.)
        sample_sizes = [int(size*n) for size in sample_sizes]

    else:

        means = rng.uniform(domain[0], domain[1], size=k)
        stds = rng.uniform(0.5, 2, size=k)

        sample_sizes = _generate_numbers(k, n)

    samples = []
    for i in range(k):
        samples.append(rng.normal(means[i], stds[i], size=sample_sizes[i]))

    X = np.concatenate(samples)[:, np.newaxis]
    X_plot = np.linspace(min(means)-4*max(stds), max(means)+4*max(stds), n//2)[:, np.newaxis]

    # Add PDFs
    true_dens = np.sum(
        [(sample_sizes[j]/n) * norm(means[j], stds[j]).pdf(X_plot[:, 0]) for j in range(k)], axis=0
    )

    # Not accessed - To delete ?
    # U_train = np.random.uniform(-7, 14, int(len(x)/(n_rffs-1))).reshape(-1, 1)
    weights_qrff = rng.normal(size=n_rffs)

    X_feat_train = _predict_features(X, weights_qrff, gamma)
    X_feat_test = _predict_features(X_plot, weights_qrff, gamma)
    
    Y_feat_train = np.ones(len(X_feat_train)).reshape(-1, 1)

    U_dagger_train = np.array(
        [np.conjugate(_create_U_train(X_feat_train[i]).T) for i in range(len(X_feat_train))]
        )
    
    U_dagger_test = np.array(
        [np.conjugate(_create_U_train(X_feat_test[i]).T) for i in range(len(X_feat_test))]
        )

    return {'X':X,
            'X_plot': X_plot,
            'true_dens': true_dens,
            'U_dagger_train': U_dagger_train, 
            'Y_feat_train': Y_feat_train,
            'U_dagger_test': U_dagger_test,
            'gamma': gamma,
            'n_rffs': n_rffs}

    
def plot_predict(model, X, X_plot, true_dens, U_dagger_test, n_rffs, **kwargs):
    params = {
        'axes.labelsize': 8,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': False,
        'figure.figsize': [5.5, 4.5]
    }

    plt.rcParams.update(params)

    lw = 2

    predictions = model.predict(U_dagger_test)

    plt.plot(X_plot[:, 0], predictions, color='green', lw=lw,
            linestyle='-', label=f"DMKDE - {n_rffs} QRFF")
    plt.plot(X_plot[:, 0], true_dens, "maroon", label='True pdf')
    indexes = np.random.randint(0, len(X), len(X_plot))
    plt.plot(X[indexes, 0], -0.00125 - 0.00675 * np.random.random(len(indexes)), '+k')

    plt.legend(loc='upper left')
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Probability Density', fontsize=10)





            