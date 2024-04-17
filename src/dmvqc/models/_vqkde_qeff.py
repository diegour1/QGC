

import tensorcircuit as tc
from tensorcircuit import keras
import tensorflow as tf

from functools import partial
import numpy as np
import math as m
from scipy.stats import entropy, spearmanr



tc.set_backend("tensorflow")
tc.set_dtype("complex128")

pi = tf.constant(m.pi)


class VQKDE_QEFF:
    r"""
    Defines the ready-to-use Density Matrix Kernel Density Estimation (DMKDE) model implemented
    in TensorCircuit using the TensorFlow/Keras API. Any additional argument in the methods has to be Keras-compliant.

    Args:
        auto_compile: A boolean to autocompile the model using default settings. (Default True).
        var_pure_state_size:
        gamma:

    Returns:
        An instantiated model ready to train with ad-hoc data.

    """

    def __init__(self, num_ffs_param, dim_x_param, auto_compile=True, var_pure_state_size=64, gamma=2., epochs=15):

        self.circuit = None
        self.gamma = gamma
        self.dim_x = dim_x_param
        self.n_rffs = num_ffs_param
        self.var_pure_state_parameters_size = 2*var_pure_state_size - 2
        self.qeff_weights =  tf.random.normal((dim_x_param, int(num_ffs_param*2-2)), mean=0.0, stddev=1.0/np.sqrt(num_ffs_param-1), dtype=tf.dtypes.float64)
        self.epochs = epochs

        layer = keras.QuantumLayer(
            partial(self.layer),
            [(self.var_pure_state_parameters_size,)]
            )

        self.model = tf.keras.Sequential([layer])

        if auto_compile:
            self.compile()


    def layer(
            self,
            x_sample_param,
            var_pure_state_param,
        ):
        r"""
        Defines a Density Matrix Kernel Density Estimation quantum layer for learning with fixed qaff (Meaning of qaff?). (This function was originally named dmkde_mixed_variational_density_estimation_fixed_qaff)

        Args:
            U_dagger:
            var_pure_state_param:

        Returns:
            The probability of :math:`|000\rangle` state for density estimation.
        """

        n_total_qubits_temp = int(np.log2((len(var_pure_state_param)+2)/2))
        n_qeff_qubits_temp = int(np.log2(self.n_rffs))

        ### indices pure state
        index_it = iter(np.arange(len(var_pure_state_param)))

        ### indices qeff
        index_iter_qeff = iter(np.arange(self.qeff_weights.shape[1]))

        # Instantiate a circuit with the calculated number of qubits.
        self.circuit = tc.Circuit(n_total_qubits_temp)

        def circuit_base_ry_n(qc_param, num_qubits_param, target_qubit_param):
            if num_qubits_param == 1:
                qc_param.ry(0, theta = var_pure_state_param[next(index_it)])
            elif num_qubits_param == 2:
                qc_param.ry(target_qubit_param, theta=var_pure_state_param[next(index_it)])
                qc_param.cnot(0, target_qubit_param)
                qc_param.ry(target_qubit_param, theta=var_pure_state_param[next(index_it)])
                return
            else:
                circuit_base_ry_n(qc_param, num_qubits_param-1, target_qubit_param)
                qc_param.cnot(num_qubits_param-2, target_qubit_param)
                circuit_base_ry_n(qc_param, num_qubits_param-1, target_qubit_param)
                target_qubit_param -= 1

        def circuit_base_rz_n(qc_param, num_qubits_param, target_qubit_param):
            if num_qubits_param == 1:
                qc_param.rz(0, theta = var_pure_state_param[next(index_it)])
            elif num_qubits_param == 2:
                qc_param.rz(target_qubit_param, theta=var_pure_state_param[next(index_it)])
                qc_param.cnot(0, target_qubit_param)
                qc_param.rz(target_qubit_param, theta=var_pure_state_param[next(index_it)])
                return
            else:
                circuit_base_rz_n(qc_param, num_qubits_param-1, target_qubit_param)
                qc_param.cnot(num_qubits_param-2, target_qubit_param)
                circuit_base_rz_n(qc_param, num_qubits_param-1, target_qubit_param)
                target_qubit_param -= 1

        # Learning pure state
        for i in range(1, n_total_qubits_temp+1):
            circuit_base_ry_n(self.circuit, i, i-1)

        # Learning pure state complex phase
        for j in range(1, n_total_qubits_temp+1):
            circuit_base_rz_n(self.circuit, j, j-1)

        # Value to predict

        x_sample_temp = tf.expand_dims(x_sample_param, axis=0)
        phases_temp = tf.cast(tf.sqrt(self.gamma), tf.float64)*tf.linalg.matmul(tf.cast(x_sample_temp, tf.float64), self.qeff_weights)
        init_qubit_qeff_temp = 0 # qubit at which the qaff mapping starts

        def circuit_base_rz_qeff_n(qc_param, num_qubits_param, target_qubit_param, init_qubit_param):
          if num_qubits_param == 1:
            qc_param.rz(init_qubit_param, theta = phases_temp[0][next(index_iter_qeff)])
            qc_param.X(init_qubit_param)
            qc_param.rz(init_qubit_param, theta = phases_temp[0][next(index_iter_qeff)])
          elif num_qubits_param == 2:
            qc_param.rz(target_qubit_param + init_qubit_param, theta = phases_temp[0][next(index_iter_qeff)])
            qc_param.X(target_qubit_param + init_qubit_param)
            qc_param.rz(target_qubit_param + init_qubit_param, theta = phases_temp[0][next(index_iter_qeff)])
            qc_param.cnot(init_qubit_param, target_qubit_param + init_qubit_param)
            qc_param.rz(target_qubit_param + init_qubit_param, theta = phases_temp[0][next(index_iter_qeff)])
            qc_param.X(target_qubit_param + init_qubit_param)
            qc_param.rz(target_qubit_param + init_qubit_param, theta = phases_temp[0][next(index_iter_qeff)])
            return
          else:
            circuit_base_rz_qeff_n(qc_param, num_qubits_param-1, target_qubit_param, init_qubit_param)
            qc_param.cnot(num_qubits_param-2+init_qubit_param, target_qubit_param+init_qubit_param)
            circuit_base_rz_qeff_n(qc_param, num_qubits_param-1, target_qubit_param, init_qubit_param)
            target_qubit_param -= 1

        # Applying the QEFF feature map

        for i in range( n_qeff_qubits_temp - init_qubit_qeff_temp + 1 - 1, 1 - 1, -1):
          circuit_base_rz_qeff_n(self.circuit, i, i - 1, init_qubit_qeff_temp)

        for i in range(init_qubit_qeff_temp, n_qeff_qubits_temp):
          self.circuit.H(i)

        # Trace out ancilla qubits, find probability of [000] state for density estimation

        return (1./(self.epochs))*\
                tc.backend.real(
            tc.quantum.reduced_density_matrix(
                self.circuit.state(),
                cut=[m for m in range(n_qeff_qubits_temp, n_total_qubits_temp)]
            )[0, 0]
        )

    def loss(self, y_train, y_pred):
        return -tf.reduce_sum(tf.math.log(y_pred)) # this loss function works relatively well

    def compile(
            self,
            optimizer=tf.keras.optimizers.legacy.Adam(0.0005), # originally 0.0005
            **kwargs):
        r"""
        Method to compile the model.

        Args:
            optimizer:
            **kwargs: Any additional argument.

        Returns:
            None.
        """
        self.model.compile(
            loss = self.loss,
            optimizer=optimizer,
            metrics=[tf.keras.metrics.KLDivergence()],
            **kwargs
        )

    def fit(self, x_train, y_train, batch_size=16, **kwargs):
        r"""
        Method to fit (train) the model using the ad-hoc dataset.

        Args:
            x_train:
            y_train:
            batch_size:
            epochs:
            **kwargs: Any additional argument.

        Returns:
            None.
        """

        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=self.epochs, **kwargs)

    def predict(self, x_test):
        r"""
        Method to make predictions with the trained model.

        Args:
            x_test:

        Returns:
            The predictions of the PDF of the input data.
        """
        return (tf.math.pow((self.gamma/(pi)), self.dim_x/2)*\
            self.model.predict(x_test)).numpy()
    

