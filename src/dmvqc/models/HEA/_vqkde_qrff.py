### DMKDE mixed

import tensorcircuit as tc
from tensorcircuit import keras
import tensorflow as tf

from functools import partial
import numpy as np
import math as m



tc.set_backend("tensorflow")
tc.set_dtype("complex128")

pi = tf.constant(m.pi)


class VQKDE_QRFF_HEA:
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

    def __init__(self, dim_x_param, auto_compile=True, var_hea_ansatz_size_param=64, num_layers_hea_param = 3, gamma=2., epochs=15):

        self.circuit = None
        self.gamma = gamma
        self.dim_x = dim_x_param
        self.num_layers_hea = num_layers_hea_param
        self.var_hea_ansatz_size = var_hea_ansatz_size_param
        self.epochs = epochs

        layer = keras.QuantumLayer(
            partial(self.layer),
            [(self.var_hea_ansatz_size,)]
            )

        self.model = tf.keras.Sequential([layer])

        if auto_compile:
            self.compile()


    def layer(
            self,
            U_dagger,
            var_hea_ansatz_param,
        ):
        r"""
        Defines a Density Matrix Kernel Density Estimation quantum layer for learning with fixed qaff (Meaning of qaff?). (This function was originally named dmkde_mixed_variational_density_estimation_fixed_qaff)

        Args:
            U_dagger:
            var_pure_state_param:

        Returns:
            The probability of :math:`|000\rangle` state for density estimation.
        """

        n_rffs_temp = U_dagger.shape[1]
        n_total_qubits_temp = int(self.var_hea_ansatz_size/((self.num_layers_hea+1)*2))
        n_qrff_qubits_temp = int(np.log2(n_rffs_temp))

        index_iter_hea  = iter(np.arange(len(var_hea_ansatz_param)))

        # Instantiate a circuit with the calculated number of qubits.
        self.circuit = tc.Circuit(n_total_qubits_temp)

        def hea_ansatz(qc_param, num_qubits_param, num_layers_param):
        # encoding
          for i in range (0, num_qubits_param):
            qc_param.ry(i, theta = var_hea_ansatz_param[next(index_iter_hea)])
            qc_param.rz(i, theta = var_hea_ansatz_param[next(index_iter_hea)])
        # layers
          for j in range(num_layers_param):
            for i in range (0, num_qubits_param-1):
              qc_param.CNOT(i, i+1)

            for i in range (0, num_qubits_param):
              qc_param.ry(i, theta= var_hea_ansatz_param[next(index_iter_hea)])
              qc_param.rz(i, theta= var_hea_ansatz_param[next(index_iter_hea)])

        ## learning pure state with HEA

        hea_ansatz(self.circuit, n_total_qubits_temp, self.num_layers_hea)

        # Value to predict
        self.circuit.any(
            *[n for n in range(n_qrff_qubits_temp)], unitary=U_dagger
        )

        # Trace out ancilla qubits, find probability of [000] state for density estimation
        return (1./(self.epochs))*\
                tc.backend.real(
                    tc.quantum.reduced_density_matrix(
                        self.circuit.state(),
                        cut=[m for m in range(n_qrff_qubits_temp, n_total_qubits_temp)]
                    )[0, 0]
                )

    def loss(self, x_pred, y_pred):
        return -tf.reduce_sum(tf.math.log(y_pred))

    def compile(
            self,
            optimizer=tf.keras.optimizers.legacy.Adam(0.0005),
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