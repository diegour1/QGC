### Quantum variational KDC with QEFF

import tensorcircuit as tc
from tensorcircuit import keras
import tensorflow as tf

from ...utils.utils import _indices_qubits_classes

from functools import partial
import numpy as np
import math as m
from scipy.stats import entropy, spearmanr




tc.set_backend("tensorflow")
tc.set_dtype("complex128")

pi = tf.constant(m.pi)


class VQKDC_MIXED_QEFF_HEA:
    r"""
    Defines the ready-to-use Quantum measurement classification (QMC) model implemented
    in TensorCircuit using the TensorFlow/Keras API. Any additional argument in the methods has to be Keras-compliant.

    Args:
        auto_compile: A boolean to autocompile the model using default settings. (Default True).
        var_pure_state_size:
        gamma:

    Returns:
        An instantiated model ready to train with ad-hoc data.

    """
    def __init__(self, dim_x_param, n_qeff_qubits, n_ancilla_qubits, num_classes_qubits, num_classes_param, gamma, n_training_data, reduction = "none", training_type = "generative", num_layers_hea = 3, batch_size = 16, learning_rate = 0.0005, random_state = 15, auto_compile=True):

        self.circuit = None
        self.gamma = gamma
        self.dim_x = dim_x_param
        self.num_layers_hea = num_layers_hea
        self.num_classes = num_classes_param
        self.num_classes_qubits = num_classes_qubits
        self.n_qeff_qubits = n_qeff_qubits
        self.n_ancilla_qubits = n_ancilla_qubits
        self.n_total_qubits_temp = self.num_classes_qubits + self.n_qeff_qubits + self.n_ancilla_qubits
        self.num_ffs = 2**self.n_qeff_qubits
        self.n_training_data = n_training_data
        self.var_hea_ansatz_size = int(self.n_total_qubits_temp*(self.num_layers_hea+1)*2)
        self.reduction  = reduction
        self.training_type = training_type
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.qeff_weights = tf.random.normal((dim_x_param, int(self.num_ffs*1-1)), mean = 0.0, stddev = 2.0/np.sqrt(self.num_ffs - 1), dtype=tf.dtypes.float64, seed = random_state)

        layer = keras.QuantumLayer(
            partial(self.layer),
            [(self.var_hea_ansatz_size,)]
            )

        self.model = tf.keras.Sequential([layer])

        if auto_compile:
            self.compile()

    def layer(
            self,
            x_sample_param,
            var_hea_ansatz_param,
        ):
        r"""
        Defines a Density Matrix Kernel Density Estimation quantum layer for learning with fixed qaff (Meaning of qaff?). (This function was originally named dmkde_mixed_variational_density_estimation_fixed_qaff)

        Args:
            U_dagger:
            var_pure_state_param:

        Returns:
            The probabilities of :math:`|k\rangle`, `|1\rangle`, ..., `|k\rangle` state for kernel density classification of the classes.
        """

        ### indices pure state hea
        index_iter_hea  = iter(np.arange(len(var_hea_ansatz_param)))

        ### indices qeff
        index_iter_qeff = iter(np.arange(self.qeff_weights.shape[1]))

        ### indices classes, of ms
        n_qubits_classes_qeff_temp = self.num_classes_qubits + self.n_qeff_qubits
        index_qubit_states = _indices_qubits_classes(n_qubits_classes_qeff_temp, self.num_classes) # extract indices of the bit string of classes


        # Instantiate a circuit with the calculated number of qubits.
        self.circuit = tc.Circuit(self.n_total_qubits_temp)

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
        hea_ansatz(self.circuit, self.n_total_qubits_temp, self.num_layers_hea)

        # Value to predict

        x_sample_temp = tf.expand_dims(x_sample_param, axis=0)
        phases_temp = (tf.cast(tf.sqrt(self.gamma), tf.float64)*tf.linalg.matmul(tf.cast(x_sample_temp, tf.float64), self.qeff_weights))[0]
        init_qubit_qeff_temp = self.num_classes_qubits # qubit at which the qaff mapping starts it starts after the qubits of the classes

        def circuit_base_rz_qeff_n(qc_param, num_qubits_param, target_qubit_param, init_qubit_param):
          if num_qubits_param == 1:
            qc_param.rz(init_qubit_param, theta = phases_temp[next(index_iter_qeff)] )
          elif num_qubits_param == 2:
            qc_param.rz(target_qubit_param + init_qubit_param, theta = phases_temp[next(index_iter_qeff)])
            qc_param.cnot(init_qubit_param, target_qubit_param + init_qubit_param)
            qc_param.rz(target_qubit_param + init_qubit_param, theta = phases_temp[next(index_iter_qeff)])
            return
          else:
            circuit_base_rz_qeff_n(qc_param, num_qubits_param-1, target_qubit_param, init_qubit_param)
            qc_param.cnot(num_qubits_param-2 + init_qubit_param, target_qubit_param + init_qubit_param)
            circuit_base_rz_qeff_n(qc_param, num_qubits_param-1, target_qubit_param, init_qubit_param)
            target_qubit_param -= 1

        # Applying the QEFF feature map

        for i in reversed(range(1, self.n_qeff_qubits + 1)):
          circuit_base_rz_qeff_n(self.circuit, i, i - 1, init_qubit_qeff_temp)

        for i in range(init_qubit_qeff_temp, init_qubit_qeff_temp + self.n_qeff_qubits):
          self.circuit.H(i)

        # Trace out ancilla qubits, find probability of [000] state for density estimation
        measurement_state = tc.quantum.reduced_density_matrix(
                        self.circuit.state(),
                        cut=[m for m in range(n_qubits_classes_qeff_temp, self.n_total_qubits_temp)])
        measurements_results = tc.backend.real(tf.stack([measurement_state[index_qubit_states[i], index_qubit_states[i]] for i in range(self.num_classes)]))
        if self.training_type == "discriminative":
          measurements_results = measurements_results / tf.reduce_sum(measurements_results, axis = -1)
        return measurements_results

    def custom_categorical_crossentropy(self, y_true, y_pred):
      ## code generated with the aid of chat gpt
      """
      Compute the categorical cross-entropy loss with mean reduction.

      Args:
      y_true: Tensor of true labels, shape (batch_size, num_classes).
      y_pred: Tensor of predicted probabilities, shape (batch_size, num_classes).

      Returns:
      Scalar tensor representing the mean loss over the batch.
      """
      # Ensure predictions are clipped to avoid log(0)
      epsilon_two = 1e-7  # small constant to avoid division by zero
      y_pred = tf.clip_by_value(y_pred, epsilon_two, np.inf)  # clip values to avoid log(0) originaly 1.0 - epsilon

      # Compute the categorical cross-entropy loss for each sample
      loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

      if self.reduction == "none":
        return loss
      elif self.reduction == "mean":
        # Compute the mean loss over the batch
        mean_loss = tf.reduce_mean(loss)
        return mean_loss
      elif self.reduction == "sum":
        # Compute the sum loss over the batch
        sum_loss = tf.reduce_sum(loss)
        return sum_loss
      else:
        return loss

    def compile(
            self,
            optimizer=tf.keras.optimizers.Adam,
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
            loss = self.custom_categorical_crossentropy,
            optimizer=optimizer(self.learning_rate),
            metrics=["accuracy"],
            **kwargs
        )

    def fit(self, x_train, y_train, batch_size=16, epochs = 30, **kwargs):
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

        self.model.fit(x_train, y_train, batch_size = self.batch_size, epochs = epochs, **kwargs)

    def predict(self, x_test):
      r"""
      Method to make predictions with the trained model.

      Args:
          x_test:

      Returns:
          The predictions of the conditional density estimation of the input data.
      """
      return (tf.experimental.numpy.power((self.gamma/(pi)), self.dim_x/2.)*\
          self.model.predict(x_test)).numpy()
