
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from scipy.stats import entropy
from scipy.stats import spearmanr
import tensorflow as tf

from tabulate import tabulate

def _indices_qubits_classes(num_qubits_param, num_classes_param):
    num_qubits_classes_temp = int(np.ceil(np.log2(num_classes_param)))
    a = [np.binary_repr(i, num_qubits_param) for i in range(2**num_qubits_param)]
    b = [(np.binary_repr(i, num_qubits_classes_temp) + "0"*(num_qubits_param - num_qubits_classes_temp)) for i in range(num_classes_param)]
    indices_temp = []
    for i in range(len(a)):
        if a[i] in b:
            indices_temp.append(i)

    return indices_temp

def create_U_train(x_train_param, seed=0):
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


def predict_features(X, var, gamma):
    r"""
    Add documentation.
    """
    X_feat = np.ones((len(X), var.shape[1]), dtype = np.complex128)
    X_feat[:, :] = np.cos(np.sqrt(gamma)*(X @ var)) - 1j*np.sin(np.sqrt(gamma)*(X @ var))
    X_feat *= np.sqrt(1/(var.shape[1]))
    return X_feat


## evaluate accuracy and conditional density estimation

def evaluate_class_model(y_true_test_param, y_true_train_param, y_pred_test_kdc_param,  y_pred_train_dmkdc_param, y_pred_train_model_param, y_pred_test_model_param, model_name_param, gamma_param, grid_volume_param, dim_x_param = 1.0):

  # Accuracy kdc
  accuracy_kdc_temp = accuracy_score(y_true_test_param, np.argmax(y_pred_test_kdc_param, axis = 1))

  # Accuracy model
  accuracy_model_temp = accuracy_score(y_true_test_param, np.argmax(y_pred_test_model_param, axis = 1))

  # kullback-Leibler divergence test
  y_test_kdc_temp = y_pred_test_kdc_param.reshape(-1,)
  y_test_model_temp = y_pred_test_model_param.reshape(-1,)
  kldiv_kdc_vs_model = tf.keras.backend.sum(y_test_kdc_temp * tf.keras.backend.log(y_test_kdc_temp / (y_test_model_temp)), axis=-1).numpy()

  # mean average error
  absolute_differences_temp = np.abs(y_test_kdc_temp - y_test_model_temp)
  average_error_temp = np.mean(absolute_differences_temp)

  # Calculate spearmann correlation per class, i.e., the ranks of the density values
  # spearman class 0
  ranks1_class0 = np.array(y_pred_test_kdc_param[:, 0]).argsort().argsort()
  ranks2_class0 = y_pred_test_model_param[:, 0].argsort().argsort()
  # spearman class 1
  ranks1_class1 = np.array(y_pred_test_kdc_param[:, 1]).argsort().argsort()
  ranks2_class1 = y_pred_test_model_param[:, 1].argsort().argsort()

  # Calculate the Spearman correlation
  spearman_corr_class0, _ = spearmanr(ranks2_class0, ranks1_class0)
  spearman_corr_class1, _ = spearmanr(ranks2_class1, ranks1_class1)

  # Calculate relative entropy
  preds_train_dmkdc_loss = np.zeros(len(y_pred_train_model_param))
  preds_train_model_loss = np.zeros(len(y_pred_train_model_param))
  for i in range(len(y_pred_train_model_param)):
    if y_true_train_param[i] == 0:
      preds_train_model_loss[i] = y_pred_train_model_param[i, 0]
      preds_train_dmkdc_loss[i] = y_pred_train_dmkdc_param[i, 0]
    else:
      preds_train_model_loss[i] = y_pred_train_model_param[i, 1]
      preds_train_dmkdc_loss[i] = y_pred_train_dmkdc_param[i, 1]

  relative_entropy_train = (1./len(y_pred_train_dmkdc_param))*tf.reduce_sum(tf.math.log(((gamma_param/np.pi)**((dim_x_param/2.0)))*preds_train_dmkdc_loss))-(1./len(y_pred_train_model_param))*tf.reduce_sum(tf.math.log(((gamma_param/np.pi)**((dim_x_param/2.0)))*preds_train_model_loss))

  # build table
  table = [["Accuracy KDC:", np.round(accuracy_kdc_temp,3)], [f"Accuracy {model_name_param}:", np.round(accuracy_model_temp,3)], [f"KL-div KDC vs {model_name_param}:", np.round(kldiv_kdc_vs_model,3)], [f"MAE KDC vs {model_name_param}:", np.round(average_error_temp,3)], [f"Spearmann class 0 KDC vs {model_name_param}:", np.round(spearman_corr_class0,3)], [f"Spearmann class 1 KDC vs {model_name_param}:", np.round(spearman_corr_class1, 3)], [f"relative_entropy_train DMKDC vs {model_name_param}:", np.round(relative_entropy_train, 3)]]
  headers = ['Metrics', f"KDC vs {model_name_param}"]
  print(tabulate(table, headers), "\n")


def plot_data(X, y):
    """Function to visualize a 2D dataset"""
    y_unique = np.unique(y)
    colors = plt.cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
    for this_y, color in zip(y_unique, colors):
        this_X = X[y == this_y]
        plt.scatter(this_X[:, 0], this_X[:, 1],  c=color,
                    alpha=0.5, edgecolor='k',
                    label="Class %s" % this_y)
    plt.legend(loc="best")
    plt.title("Data")

def plot_decision_region(X_param, preds_plot_param):
    """Function to visualize the decision surface of a classifier"""
    min_x = np.min(X_param[:, 0])
    max_x = np.max(X_param[:, 0])
    min_y = np.min(X_param[:, 1])
    max_y = np.max(X_param[:, 1])
    min_x = min_x - (max_x - min_x) * 0.05
    max_x = max_x + (max_x - min_x) * 0.05
    min_y = min_y - (max_y - min_y) * 0.05
    max_y = max_y + (max_y - min_y) * 0.05
    x_vals = np.linspace(min_x, max_x, 20)
    y_vals = np.linspace(min_y, max_y, 20)
    XX, YY = np.meshgrid(x_vals, y_vals)
    grid_r, grid_c = XX.shape

    ZZ = np.reshape(preds_plot_param[:, 1]/preds_plot_param.sum(axis=1), (grid_r, grid_c))

    plt.contourf(XX, YY, ZZ, 100, cmap = plt.cm.coolwarm, vmin= 0, vmax=1)
    plt.colorbar()
    #CS = plt.contour(XX, YY, ZZ, 100, levels = [0.125*i for i in range(1,8)])
    CS = plt.contour(XX, YY, ZZ, 100, levels = [0.5])
    plt.clabel(CS, inline=1, fontsize=10)
    plt.xlabel("x")
    plt.ylabel("y")