
import qmc.tf.layers as qmc_layers
import qmc.tf.models as qmc_models

import tensorcircuit as tc

tc.set_backend("tensorflow")
tc.set_dtype("complex128")

import numpy as np

from ..estimation import QFeatureMapQuantumEnhancedFF

GAMMA = float(2**2) ### original 2**2
N_FFS = 32 ### original 32

def classical_dmkdc_qeff(X_train_param, Y_train_oh_param, X_test_param, n_qrffs_param, rs_param, gamma_param = 4.):

  ## Initialize values
  dim_x_temp = X_train_param.shape[1]
  Y_train_oh_param = np.array(Y_train_oh_param)
  num_classes_temp = Y_train_oh_param.shape[1]
  Y_pred_temp = np.zeros((len(X_test_param), num_classes_temp))
  sigma_temp = 1./(np.sqrt(2.*gamma_param)) ## Not accesed

  ## Training
  dmkdc_temp = []
  fm_x = QFeatureMapQuantumEnhancedFF(dim_x_temp, dim=n_qrffs_param, gamma=gamma_param, random_state= rs_param)
  for j in range(num_classes_temp):
    dmkdc_temp.append(qmc_models.ComplexQMDensity(fm_x, n_qrffs_param))

  ## Prediction
  for j in range(num_classes_temp):
    dmkdc_temp[j].compile()
    dmkdc_temp[j].fit(X_train_param[Y_train_oh_param[:, j].astype(bool)], epochs=1, batch_size = 1) ### must keep the batch size = 1
    Y_pred_temp[:, j] = ((gamma_param/np.pi)**(dim_x_temp/2))*(dmkdc_temp[j].predict(X_test_param, batch_size = 1)) ### must keep the batch size = 1

  return ((Y_train_oh_param.numpy().sum(axis=0)/len(Y_train_oh_param)))*Y_pred_temp