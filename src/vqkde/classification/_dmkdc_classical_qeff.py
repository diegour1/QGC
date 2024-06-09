import numpy as np
import qmc.tf.layers as qmc_layers
import qmc.tf.models as qmc_models
from functools import partial
import tensorflow as tf
import math as m

import tensorcircuit as tc

tc.set_backend("tensorflow")
tc.set_dtype("complex128")

pi = tf.constant(m.pi)

#from ..estimation import QFeatureMapQuantumEnhancedFF

class QFeatureMapQuantumEnhancedFF(tf.keras.layers.Layer):
    """Quantum feature map including the complex part of random Fourier Features.
    Uses `RBFSampler` from sklearn to approximate an RBF kernel using
    complex random Fourier features.

    Input shape:
        (batch_size, dim_in)
    Output shape:
        (batch_size, dim)
    Arguments:
        input_dim: dimension of the input
        dim: int. Number of dimensions to represent a sample.
        gamma: float. Gamma parameter of the RBF kernel to be approximated.
        random_state: random number generator seed.
    """

    def __init__(
            self,
            input_dim: int,
            dim: int = 100,
            gamma: float = 0.5,
            random_state=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.dim = dim
        self.gamma = gamma
        self.random_state = random_state


    def build(self, input_shape):
        self.qeff_weights = tf.random.normal((self.input_dim, int(self.dim*1-1)), mean = 0.0, stddev = 2.0/np.sqrt(self.dim - 1), dtype=tf.dtypes.float64, seed = self.random_state) ## final model self.qeff_weights = tf.random.normal((self.input_dim, int(self.dim*1-1)), mean = 0.0, stddev = 2.0/np.sqrt(self.dim - 1), dtype=tf.dtypes.float64, seed = self.random_state)
        self.built = True

    def call(self, inputs):

        ### build the phases of QEFF
        phases_temp = (tf.cast(tf.sqrt(self.gamma), tf.float64)*tf.linalg.matmul(tf.cast(inputs, tf.float64), self.qeff_weights))[0]

        ### indices qeff for iterator
        index_iter_qeff = iter(np.arange(self.qeff_weights.shape[1]))

        ## build QEFF circuit
        self.n_qeff_qubits = int(np.ceil(np.log2(self.dim)))
        self.circuit = tc.Circuit(self.n_qeff_qubits)

        def circuit_base_rz_qeff_n(qc_param, num_qubits_param, target_qubit_param):
          if num_qubits_param == 1:
            qc_param.rz(0, theta = phases_temp[next(index_iter_qeff)] )
          elif num_qubits_param == 2:
            qc_param.rz(target_qubit_param, theta = phases_temp[next(index_iter_qeff)])
            qc_param.cnot(0, target_qubit_param)
            qc_param.rz(target_qubit_param, theta = phases_temp[next(index_iter_qeff)])
            return
          else:
            circuit_base_rz_qeff_n(qc_param, num_qubits_param-1, target_qubit_param)
            qc_param.cnot(num_qubits_param-2, target_qubit_param)
            circuit_base_rz_qeff_n(qc_param, num_qubits_param-1, target_qubit_param)
            target_qubit_param -= 1

        # Applying the QEFF feature map

        for i in range(0, self.n_qeff_qubits):
          self.circuit.H(i)

        for i in range(1, self.n_qeff_qubits + 1):
          circuit_base_rz_qeff_n(self.circuit, i, i - 1)

        psi = tf.cast(tf.expand_dims(self.circuit.state(), axis=0), tf.complex64)
        return psi

    def get_config(self):
        config = {
            "input_dim": self.input_dim,
            "dim": self.dim,
            "gamma": self.gamma,
            "random_state": self.random_state
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dim)


def classical_dmkdc_qeff(X_train_param, Y_train_oh_param, X_test_param, n_qrffs_param, rs_param, gamma_param = 4.):

  ## Initialize values
  dim_x_temp = X_train_param.shape[1]
  Y_train_oh_param = np.array(Y_train_oh_param)
  num_classes_temp = Y_train_oh_param.shape[1]
  Y_pred_temp = np.zeros((len(X_test_param), num_classes_temp))
  sigma_temp = 1./(np.sqrt(2.*gamma_param))

  ## Training
  dmkdc_temp = []
  fm_x = QFeatureMapQuantumEnhancedFF(dim_x_temp, dim=n_qrffs_param, gamma=gamma_param, random_state= rs_param)
  for j in range(num_classes_temp):
    try:
      dmkdc_temp.append(qmc_models.ComplexQMDensity(fm_x, n_qrffs_param))
    except:
      pass

  ## Prediction
  for j in range(num_classes_temp):
    try:
      dmkdc_temp[j].compile()
      dmkdc_temp[j].fit(X_train_param[Y_train_oh_param[:, j].astype(bool)], epochs=1, batch_size = 1) ### must keep the batch size = 1
      Y_pred_temp[:, j] = ((gamma_param/np.pi)**(dim_x_temp/2))*(dmkdc_temp[j].predict(X_test_param, batch_size = 1)) ### must keep the batch size = 1
    except:
      pass

  return ((Y_train_oh_param.sum(axis=0)/len(Y_train_oh_param)))*Y_pred_temp