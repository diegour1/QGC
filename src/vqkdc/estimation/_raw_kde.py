
import numpy as np

def _calculate_constant_qmkde(gamma=1, dimension = 1):
    sigma = (4*gamma)**(-1/2)
    coefficient = 1 /  (2*np.pi*sigma**2)**(dimension/2)
    return coefficient


def _raw_kde(x_test, x_train, gamma=1):
    sigma = (2*gamma)**(-1/2)
    euclidean_distance = np.sum(((x_test-x_train))**2, axis=1)
    exponential  = np.exp(-euclidean_distance/(2*sigma**2))
    constant_outside = 1/(len(x_train) * (2*np.pi*sigma**2)**(x_train.shape[1]/2))
    return constant_outside * np.sum(exponential)

