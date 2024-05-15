
import numpy as np

def _calculate_constant_qmkde(gamma=1, dimension = 1):
    sigma = (4*gamma)**(-1/2)
    coefficient = 1 /  (2*np.pi*sigma**2)**(dimension/2)
    return coefficient

def _create_U_train(x_train_param, seed=0):
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

def _raw_kde(x_test, x_train, gamma=1):
    sigma = (2*gamma)**(-1/2)
    euclidean_distance = np.sum(((x_test-x_train))**2, axis=1)
    exponential  = np.exp(-euclidean_distance/(2*sigma**2))
    constant_outside = 1/(len(x_train) * (2*np.pi*sigma**2)**(x_train.shape[1]/2))
    return constant_outside * np.sum(exponential)

