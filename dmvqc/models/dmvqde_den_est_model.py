!pip install tensorcircuit

from functools import partial
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm

import tensorcircuit as tc
from tensorcircuit import keras

tc.set_backend("tensorflow")
tc.set_dtype("complex128")

# Given the eigenvalues this function return a Unitary gate which converts the |0> -> |psi_train>
def create_U_train(x_train_param, seed=0):
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

## Data set
N = 1000
np.random.seed(0) # original randomseed(0)
X = np.concatenate(
    (np.random.normal(-5, 1, int(0.3 * N)), np.random.normal(-1, 1, int(0.1 * N)), np.random.normal(5, 1, int(0.4 * N)), np.random.normal(10, 1, int(0.2 * N)))
)[:, np.newaxis]

X_plot = np.linspace(-7, 14, 500)[:, np.newaxis]

true_dens = (0.3 * norm(-5, 1).pdf(X_plot[:, 0]) + 0.1 * norm(-1, 1).pdf(X_plot[:, 0])
             + 0.5 * norm(5, 1).pdf(X_plot[:, 0]) + 0.2 * norm(10, 1).pdf(X_plot[:, 0]))


gamma = 2.0 # best gamma = 1.0
n_rffs = 16

## Uniform points
U_train = np.random.uniform(-7, 14, int(len(X)/(n_rffs-1))).reshape(-1, 1)
X_train_w_uniform = np.concatenate((X, U_train))

np.random.seed(1676) # good performance rs = 12
weights_qrff = np.random.normal(size = n_rffs)

def predict_features(X, var, gamma):
  X_feat = np.ones((len(X), len(var)), dtype = np.complex128)
  X_feat[:, :] = np.cos(np.sqrt(gamma)*X*var) + 1j*np.sin(np.sqrt(gamma)*X*var)
  X_feat *= np.sqrt(1/(len(var)))
  return X_feat

X_feat_train = predict_features(X, weights_qrff, gamma)
X_feat_test = predict_features(X_plot, weights_qrff, gamma)
Y_feat_train = np.ones(len(X_feat_train)).reshape(-1, 1)

## Convert states to unitaries

U_train_conjTrans = np.array([np.conjugate(create_U_train(X_feat_train[i]).T) for i in range(len(X_feat_train))])
U_test_conjTrans = np.array([np.conjugate(create_U_train(X_feat_test[i]).T) for i in range(len(X_feat_test))])

import math as m
pi = tf.constant(m.pi)
var_pure_state_size = 64 ## should be 2**n for some n
var_pure_state_parameters_size = 2*var_pure_state_size - 2

def dmkde_mixed_variational_density_estimation_fixed_qaff(U_conjtrans_sample_param, var_pure_state_param, **kwargs):

    n_rffs_temp = U_conjtrans_sample_param.shape[1]
    n_total_qubits_temp = int(np.log2((len(var_pure_state_param)+2)/2))
    n_qrff_qubits_temp = int(np.log2(n_rffs_temp))
    n_ancilla_qubits_temp = n_total_qubits_temp - n_qrff_qubits_temp
    index_it = iter(np.arange(len(var_pure_state_param)))

    c = tc.Circuit(n_total_qubits_temp)

    # learning pure state

    def circuit_base_ry_n(qc_param, num_qubits_param, target_qubit_param):
      if num_qubits_param == 1:
        qc_param.ry(0, theta = var_pure_state_param[next(index_it)])
      elif num_qubits_param == 2:
        qc_param.ry(target_qubit_param, theta = var_pure_state_param[next(index_it)])
        qc_param.cnot(0, target_qubit_param)
        qc_param.ry(target_qubit_param, theta = var_pure_state_param[next(index_it)])
        return
      else:
        circuit_base_ry_n(qc_param, num_qubits_param-1, target_qubit_param)
        qc_param.cnot(num_qubits_param-2, target_qubit_param)
        circuit_base_ry_n(qc_param, num_qubits_param-1, target_qubit_param)
        target_qubit_param -= 1

    for i in range(1, n_total_qubits_temp+1):
      circuit_base_ry_n(c, i, i - 1)

    def circuit_base_rz_n(qc_param, num_qubits_param, target_qubit_param):
      if num_qubits_param == 1:
        qc_param.rz(0, theta = var_pure_state_param[next(index_it)])
      elif num_qubits_param == 2:
        qc_param.rz(target_qubit_param, theta = var_pure_state_param[next(index_it)])
        qc_param.cnot(0, target_qubit_param)
        qc_param.rz(target_qubit_param, theta = var_pure_state_param[next(index_it)])
        return
      else:
        circuit_base_rz_n(qc_param, num_qubits_param-1, target_qubit_param)
        qc_param.cnot(num_qubits_param-2, target_qubit_param)
        circuit_base_rz_n(qc_param, num_qubits_param-1, target_qubit_param)
        target_qubit_param -= 1

    # learning pure state complex phase

    for i in range(1, n_total_qubits_temp+1):
      circuit_base_rz_n(c, i, i - 1)

    # value to predict

    c.any(*[n for n in range(n_qrff_qubits_temp)], unitary = U_conjtrans_sample_param)

    # trace out ancilla qubits, find probability of [000] state for density estimation
    return tc.backend.real(tf.cast(tf.sqrt(gamma/pi), tf.complex128)*tc.quantum.reduced_density_matrix(c.state(), cut = [m for m in range(n_qrff_qubits_temp, n_total_qubits_temp)])[0, 0])

layer = keras.QuantumLayer(partial(dmkde_mixed_variational_density_estimation_fixed_qaff), [(var_pure_state_parameters_size,)])

# Keras interface with Keras training paradigm

epochs = 50
model = tf.keras.Sequential([layer])

def my_loss_fn(y_true, y_pred):
    return -tf.reduce_sum(tf.math.log(y_pred))

model.compile(
    loss = my_loss_fn,
    optimizer=tf.keras.optimizers.legacy.Adam(0.0005),
    metrics=["binary_accuracy"])

model.fit(U_train_conjTrans, Y_feat_train, batch_size=16, epochs=epochs)

predictions = model.predict(U_test_conjTrans)

params = {
   'axes.labelsize': 8,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [5.5, 4.5]
   }
plt.rcParams.update(params)

colors = ['navy']
kernels = ['gaussian']
lw = 2
plt.plot(X_plot[:, 0], predictions, color='green', lw=lw,
           linestyle='-', label=f"DMKDE \n {n_rffs} QRFF")
plt.plot(X_plot[:, 0], true_dens, "maroon", label='True pdf')
indexes = np.random.randint(0, len(X), len(X_plot))
plt.plot(X[indexes, 0], -0.00125 - 0.00675 * np.random.random(len(indexes)), '+k')
#plt.plot(X, -0.00125 - 0.00675 * np.ones(len(X)), '+k')

plt.legend(loc='upper left')
plt.xlabel('X', fontsize=12)
plt.ylabel('Probability Density', fontsize=10)
