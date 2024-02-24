

import tensorcircuit as tc
from tensorcircuit import keras
import tensorflow as tf

from functools import partial
import numpy as np
import math as m

import matplotlib.pyplot as plt


tc.set_backend("tensorflow")
tc.set_dtype("complex128")

pi = tf.constant(m.pi)


class DMKDE:
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

    def __init__(self, auto_compile=True, var_pure_state_size=64, gamma=2.):

        assert m.isqrt(var_pure_state_size)**2 == var_pure_state_size

        self.circuit = None

        self.gamma = gamma
        self.var_pure_state_parameters_size = 2*var_pure_state_size - 2

        layer = keras.QuantumLayer(
            partial(self.layer), 
            [(self.var_pure_state_parameters_size,)]
            )

        self.model = tf.keras.Sequential([layer])

        if auto_compile:
            self.compile()


    def layer(
            self,
            U_dagger, 
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

        n_rffs_temp = U_dagger.shape[1]
        n_total_qubits_temp = int(np.log2((len(var_pure_state_param)+2)/2))
        n_qrff_qubits_temp = int(np.log2(n_rffs_temp))

        # Not accessed - to delete (?)
        # n_ancilla_qubits_temp = n_total_qubits_temp - n_qrff_qubits_temp  

        index_it = iter(np.arange(len(var_pure_state_param)))

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
        self.circuit.any(
            *[n for n in range(n_qrff_qubits_temp)], unitary=U_dagger
        )
        
        # Trace out ancilla qubits, find probability of [000] state for density estimation
        return tc.backend.real(
            tf.cast(tf.sqrt(self.gamma/pi), tf.complex128)*\
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
            metrics=["binary_accuracy"],
            **kwargs
        )

    def fit(self, x_train, y_train, batch_size=16, epochs=50, **kwargs):
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

        # x_train == U_train_conjTrans
        # y=train == Y_feat_train
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, **kwargs)

    def predict(self, x_test):
        r"""
        Method to make predictions with the trained model. 

        Args:
            x_test:
        
        Returns:
            The predictions of the PDF of the input data.
        """

        # x_test == U_dagger_test
        return self.model.predict(x_test)
    
    def plot_predict(self, X, X_plot, true_dens, U_dagger_test, n_rffs, **kwargs):
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

        predictions = self.predict(U_dagger_test)

        plt.plot(X_plot[:, 0], predictions, color='green', lw=lw,
                linestyle='-', label=f"DMKDE - {n_rffs} QRFF")
        plt.plot(X_plot[:, 0], true_dens, "maroon", label='True pdf')
        indexes = np.random.randint(0, len(X), len(X_plot))
        plt.plot(X[indexes, 0], -0.00125 - 0.00675 * np.random.random(len(indexes)), '+k')

        plt.legend(loc='upper left')
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Probability Density', fontsize=10)
    
